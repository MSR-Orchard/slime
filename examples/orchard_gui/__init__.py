"""
Tau2-Bench Integration for slime Training

This package provides an extensible framework for integrating gym-like
environments with slime's training pipeline.

Main Entry Points:
    - generate: Main function for agent-environment interaction
    - ENV_REGISTRY: Registry for environment adapters
    - set_active_env: Configure active environment

Usage (New API):
    from examples.tau2_bench.generate_new import generate, set_active_env, ENV_REGISTRY
    
    # Configure environment
    set_active_env("tau2", {"domain": "retail"})
    
    # Run interaction
    sample = await generate(args, sample, sampling_params)

Usage (Legacy API - backward compatible):
    from examples.tau2_bench.generate import generate, get_task_prompts, TAU2_CONFIGS

    # Configure the environment
    TAU2_CONFIGS["domain"] = "retail"
    TAU2_CONFIGS["task_split"] = "train"

    # Get task prompts for training
    prompts = get_task_prompts()

    # Generate trajectories (called by slime training pipeline)
    sample = await generate(args, sample, sampling_params)

Extending:
    from examples.tau2_bench.base import BaseGymEnvAdapter
    from examples.tau2_bench.generate_new import ENV_REGISTRY
    
    class MyAdapter(BaseGymEnvAdapter):
        # Implement required methods
        pass
    
    ENV_REGISTRY.register("my_env", MyAdapter)
"""

import os
import sys

# Core types from base module
from examples.orchard_gui.base.types import (
    Status,
    InteractionResult,
    EnvConfig,
    ToolCall,
    ParsedToolResult,
)

# Base adapter class for extensions
from examples.orchard_gui.base.adapter import BaseGymEnvAdapter

# Registry
from examples.orchard_gui.base.registry import EnvRegistry

# Utilities
from examples.orchard_gui.base.utils import (
    ToolParser,
    TokenHandler,
    tools_to_openai_format,
    calls_to_action,
    TOOL_INSTRUCTION,
)

# New extensible interface (uses registry from base)
# from examples.orchard_gui.generate_new import (
#     ENV_REGISTRY,
#     set_active_env,
#     get_task_prompts as get_task_prompts_new,
#     get_task_ids as get_task_ids_new,
# )

# Legacy interface (for backward compatibility)
from examples.orchard_gui.generate_browser import (
    generate_trajectory_sample,
    generate_turn_sample,
    # get_task_prompts,
    # get_task_ids,
    # res_to_sample,
    # TAU2_CONFIGS,
)

__all__ = [
    # Core types
    "Status",
    "InteractionResult",
    "EnvConfig",
    "ToolCall",
    "ParsedToolResult",
    # Base class for extensions
    "BaseGymEnvAdapter",
    # Registry
    "EnvRegistry",
    # Utilities
    "ToolParser",
    "TokenHandler",
    "tools_to_openai_format",
    "calls_to_action",
    "TOOL_INSTRUCTION",
    # New interface
    "ENV_REGISTRY",
    # "set_active_env",
    # "get_task_prompts_new",
    # "get_task_ids_new",
    # Legacy interface
    # "generate",
    "generate_trajectory_sample",
    "generate_turn_sample"
    # "get_task_prompts",
    # "get_task_ids",
    # "res_to_sample",
]
