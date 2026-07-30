"""Convert a HuggingFace checkpoint that stores MoE experts in the *unfused*
per-expert layout into a Megatron torch-dist checkpoint.

Background
----------
The default ``tools/convert_hf_to_torch_dist.py`` relies on ``mbridge`` +
``slime_plugins.mbridge.qwen3_5.Qwen3_5Bridge``. That bridge expects the
**fused** expert layout, i.e. a single 3D tensor per layer:

    model.language_model.layers.{L}.mlp.experts.gate_up_proj   # [E, 2*moe_ffn, hidden]
    model.language_model.layers.{L}.mlp.experts.down_proj      # [E, hidden,    moe_ffn]

Some exported checkpoints instead store experts **unfused**, one tensor per
expert:

    model.language_model.layers.{L}.mlp.experts.{E}.gate_proj.weight  # [moe_ffn, hidden]
    model.language_model.layers.{L}.mlp.experts.{E}.up_proj.weight    # [moe_ffn, hidden]
    model.language_model.layers.{L}.mlp.experts.{E}.down_proj.weight  # [hidden,  moe_ffn]

With such a checkpoint mbridge raises::

    KeyError: 'model.language_model.layers.15.mlp.experts.gate_up_proj'

because the fused name is not present in ``model.safetensors.index.json``.

This script fixes that *without modifying any existing file*. It subclasses
``mbridge``'s ``SafeTensorIO`` so that, for the regular decoder layers, it:

  1. advertises the synthetic fused keys (``...gate_up_proj`` / ``...down_proj``)
     so ``Bridge.load_weights`` does not skip the expert params, and
  2. synthesizes the fused 3D tensors on the fly from the per-expert tensors,
     matching exactly the layout the bridge consumes:

         gate_up_proj = stack_E( cat([gate_proj[E], up_proj[E]], dim=0) )
         down_proj    = stack_E( down_proj[E] )

MTP layers (``mtp.layers.*``) are intentionally left untouched because the
bridge already expects them in the unfused per-expert layout.

Usage (identical to the original script, just a different entrypoint)::

    source scripts/models/qwen3.5-35B-A3B.sh
    PYTHONPATH=/root/Megatron-LM torchrun \
        --nproc_per_node=8 \
        tools/convert_hf_to_torch_dist_unfused_experts.py \
        "${MODEL_ARGS[@]}" \
        --hf-checkpoint /path/to/unfused_hf_ckpt \
        --save /path/to/output_torch_dist
"""

import gc
import os
import re
import shutil
from collections import OrderedDict, defaultdict

import torch
import torch.distributed as dist
from megatron.core.enums import ModelType
from megatron.training.arguments import parse_args, validate_args
from megatron.training.checkpointing import get_checkpoint_name, get_checkpoint_tracker_filename, save_checkpoint
from megatron.training.training import get_model
from safetensors import safe_open

import slime_plugins.mbridge  # noqa: F401
from mbridge import AutoBridge
from mbridge.core.safetensor_io import SafeTensorIO
from slime.backends.megatron_utils.arguments import set_default_megatron_args
from slime.backends.megatron_utils.initialize import init
from slime.backends.megatron_utils.model_provider import get_model_provider_func
from slime.utils.logging_utils import configure_logger
from slime.utils.memory_utils import print_memory

# Matches per-expert weights of the *regular* decoder layers only (not MTP).
#   group(1) = prefix up to and including ".mlp.experts"
#   group(2) = expert id
#   group(3) = gate_proj | up_proj | down_proj
_EXPERT_RE = re.compile(
    r"^(model\.language_model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)

# Sentinel filename used for the synthetic fused keys we inject into the index.
_FUSED_SENTINEL = "__FUSED_SYNTHESIZED__"


class FusedExpertSafeTensorIO(SafeTensorIO):
    """SafeTensorIO that exposes fused expert tensors built from unfused per-expert weights."""

    def __init__(self, hf_dir: str, cache_size: int = 4):
        super().__init__(hf_dir)
        # ``self.index`` currently maps every real HF key (including the per-expert
        # ones) to its safetensors shard. Keep a copy so we can still read the raw
        # per-expert tensors from disk while synthesizing.
        self._orig_index = dict(self.index)

        # fused_key -> list of (gate_key, up_key) ordered by expert id
        self._gate_up_members: dict[str, list[tuple[str, str]]] = {}
        # fused_key -> list of down_key ordered by expert id
        self._down_members: dict[str, list[str]] = {}

        self._cache_size = max(1, cache_size)
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        self._setup_fused_index()

    def _setup_fused_index(self) -> None:
        # prefix -> {expert_id -> {proj -> hf_key}}
        members: dict[str, dict[int, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
        for key in self._orig_index.keys():
            m = _EXPERT_RE.match(key)
            if m is None:
                continue
            prefix, eid, proj = m.group(1), int(m.group(2)), m.group(3)
            members[prefix][eid][proj] = key

        for prefix, experts in members.items():
            eids = sorted(experts.keys())
            # Sanity: experts should be contiguous 0..N-1.
            expected = list(range(len(eids)))
            if eids != expected:
                raise ValueError(
                    f"Non-contiguous expert ids for '{prefix}': got {eids[:8]}... "
                    f"(count={len(eids)}). Cannot build fused tensor."
                )

            gate_up_key = f"{prefix}.gate_up_proj"
            down_key = f"{prefix}.down_proj"

            gate_up_list = []
            down_list = []
            for e in eids:
                projs = experts[e]
                for required in ("gate_proj", "up_proj", "down_proj"):
                    if required not in projs:
                        raise ValueError(f"Missing '{required}' for expert {e} of '{prefix}'.")
                gate_up_list.append((projs["gate_proj"], projs["up_proj"]))
                down_list.append(projs["down_proj"])

            self._gate_up_members[gate_up_key] = gate_up_list
            self._down_members[down_key] = down_list

            # Advertise the fused keys so Bridge.load_weights does not skip the
            # expert params (it filters out params whose HF names are absent from
            # the index). The per-expert keys remain in the index too, which is
            # harmless since the bridge only ever requests the fused names.
            self.index[gate_up_key] = _FUSED_SENTINEL
            self.index[down_key] = _FUSED_SENTINEL

        if self._gate_up_members:
            num_fused_layers = len(self._gate_up_members)
            num_experts = len(next(iter(self._gate_up_members.values())))
            print(
                f"[FusedExpertSafeTensorIO] Synthesizing fused experts for {num_fused_layers} "
                f"layer(s), {num_experts} experts each."
            )

    def _is_fused_key(self, name: str) -> bool:
        return name in self._gate_up_members or name in self._down_members

    def _read_raw(self, key: str) -> torch.Tensor:
        filename = self._orig_index[key]
        path = os.path.join(self.hf_dir, filename)
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def _build_fused_tensor(self, fused_key: str) -> torch.Tensor:
        cached = self._cache.get(fused_key)
        if cached is not None:
            self._cache.move_to_end(fused_key)
            return cached

        if fused_key in self._gate_up_members:
            per_expert = []
            for gate_key, up_key in self._gate_up_members[fused_key]:
                gate = self._read_raw(gate_key)
                up = self._read_raw(up_key)
                per_expert.append(torch.cat([gate, up], dim=0))
            tensor = torch.stack(per_expert, dim=0).contiguous()
        else:
            per_expert = [self._read_raw(down_key) for down_key in self._down_members[fused_key]]
            tensor = torch.stack(per_expert, dim=0).contiguous()

        self._cache[fused_key] = tensor
        self._cache.move_to_end(fused_key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return tensor

    def load_some_hf_weight(self, hf_weight_names: list) -> dict:
        fused = [n for n in hf_weight_names if self._is_fused_key(n)]
        normal = [n for n in hf_weight_names if not self._is_fused_key(n)]
        ret = {}
        if normal:
            ret.update(super().load_some_hf_weight(normal))
        for name in fused:
            ret[name] = self._build_fused_tensor(name)
        return ret

    def load_one_hf_weight(self, hf_weight_name: str) -> torch.Tensor:
        if self._is_fused_key(hf_weight_name):
            return self._build_fused_tensor(hf_weight_name)
        return super().load_one_hf_weight(hf_weight_name)


def add_convertion_args(parser):
    """Add conversion arguments to the parser"""
    parser.add_argument("--hf-checkpoint", type=str, required=True, help="HuggingFace model path")
    parser.add_argument(
        "--megatron-to-hf-mode",
        choices=["raw", "bridge"],
        default="raw",
        help="The method to convert megatron weights to hugging face weights for SGLang.",
    )
    parser.add_argument(
        "--fused-expert-cache-size",
        type=int,
        default=4,
        help="Number of synthesized fused expert tensors to keep cached in RAM during loading.",
    )
    try:
        parser.add_argument("--padded-vocab-size", type=int, default=None)
    except Exception:
        pass
    return parser


def get_args():
    args = parse_args(add_convertion_args)
    args = set_default_megatron_args(args)

    # set to pass megatron validate_args
    args.save_interval = 1
    args.micro_batch_size = 1
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    args.global_batch_size = int(os.environ.get("WORLD_SIZE", "1"))

    assert world_size <= args.num_layers, (
        f"World size {world_size} must be less than or equal to number of layers {args.num_layers}. "
        "You are using too many GPUs for this conversion."
    )

    def ceildiv(a, b):
        return -(a // -b)

    if args.pipeline_model_parallel_size == 1 and world_size > 1:
        pp_size = world_size
        while True:
            args.pipeline_model_parallel_size = pp_size
            args.decoder_last_pipeline_num_layers = args.num_layers - ceildiv(
                args.num_layers, args.pipeline_model_parallel_size
            ) * (args.pipeline_model_parallel_size - 1)

            if args.decoder_last_pipeline_num_layers > 0:
                break

            if pp_size % 2 == 0:
                pp_size //= 2
            else:
                raise ValueError(
                    f"Cannot find a valid pipeline model parallel size for {args.num_layers} layers and {world_size} GPUs."
                )
    print(
        f"Using pipeline model parallel size: {args.pipeline_model_parallel_size}, decoder last pipeline num layers: {args.decoder_last_pipeline_num_layers}"
    )

    validate_args(args)
    return args


def main():
    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module
        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    configure_logger()

    # Initialize distributed environment
    world_size = int(os.getenv("WORLD_SIZE") or os.getenv("SLURM_NTASKS") or 1)
    local_rank = int(os.getenv("LOCAL_RANK") or os.getenv("SLURM_LOCALID") or 0)
    global_rank = int(os.getenv("RANK") or os.getenv("SLURM_PROCID") or 0)

    torch.cuda.set_device(local_rank)
    os.environ.setdefault("WORLD_SIZE", str(world_size))
    os.environ.setdefault("RANK", str(global_rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        world_size=world_size,
        rank=global_rank,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    args = get_args()
    init(args)

    # if using AMD gpus, we have to do the conversion in cpu
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        assert args.use_cpu_initialization, "AMD GPU requires --use_cpu_initialization=True"

    model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

    # Load model
    hf_model_path = args.hf_checkpoint
    bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)

    # Inject our fused-expert-aware SafeTensorIO. This only affects this process
    # and does not modify any installed/existing file.
    cache_size = getattr(args, "fused_expert_cache_size", 4)

    def _patched_get_safetensor_io(weights_path):
        return FusedExpertSafeTensorIO(bridge._get_actual_hf_path(weights_path), cache_size=cache_size)

    bridge._get_safetensor_io = _patched_get_safetensor_io

    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"Model loaded: {hf_model_path}")

    if args.use_cpu_initialization:
        model[0] = model[0].cpu()

    print_memory("after loading model")
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    save_checkpoint(1, model, None, None, 0)

    if dist.get_rank() == 0:
        # change to release ckpt
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, "w") as f:
            f.write("release")
        source_dir = get_checkpoint_name(args.save, 1, False, return_base_dir=True)
        target_dir = get_checkpoint_name(args.save, -1, True, return_base_dir=True)
        shutil.move(source_dir, target_dir)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
