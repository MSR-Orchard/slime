import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config

model_id = "/data/users/shared/models/openai/gpt-oss-120b"
output_dir = "/data/users/shared/models/openai/gpt-oss-120b-bf16"

if os.path.exists(output_dir):
    print(f"Model already exists at {output_dir}, skipping conversion.")
else:
    print(f"Converting model from {model_id} to {output_dir}...")
    
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        use_cache=False,
        device_map="auto",
    )
    
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    
    # Patch config
    model.config.attn_implementation = "eager"
    
    model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
    print("Conversion done.")