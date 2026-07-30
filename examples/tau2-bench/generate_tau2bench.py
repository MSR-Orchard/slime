"""
Extensible Gym Environment Integration for slime Training

This module provides a standardized interface for training agents in various
gym-like environments using the slime framework. It uses an adapter pattern
to support multiple environments through a common interface.

Key Features:
- Pluggable environment adapters via registry
- Standardized multi-turn interaction loop
- Unified token tracking and loss mask computation
- Support for tool calling and multi-step reasoning

Usage:
    # Configure environment via registry
    from generate import ENV_REGISTRY, generate, get_task_prompts
    
    # Use tau2-bench (default)
    sample = await generate(args, sample, sampling_params)
    
    # Register custom environment
    ENV_REGISTRY.register("my_env", MyEnvAdapter)
    
    # Use custom environment
    sample = await generate(args, sample, sampling_params, env_name="my_env")

Extending:
    To add support for a new environment:
    1. Create adapter class extending BaseGymEnvAdapter
    2. Implement required methods: create_env, get_initial_messages, 
       parse_env_info, process_observation
    3. Register adapter: ENV_REGISTRY.register("env_name", YourAdapter)
"""

import os
import sys
from typing import Any, Type

# Ensure package root is in path for imports
_pkg_root = os.path.dirname(os.path.abspath(__file__))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

# Re-setup logging after tau2-bench imports (which may clear handlers)
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

from slime.utils.types import Sample

from base.types import (
    InteractionResult,
    TurnResult,
    Status,
    EnvConfig,
)
from base.adapter import BaseGymEnvAdapter
from base.registry import EnvRegistry


# ============================================================================
# Global Registry Instance
# ============================================================================

# Global registry instance (can also use base.registry.DEFAULT_REGISTRY)
ENV_REGISTRY = EnvRegistry()


# ============================================================================
# Register Built-in Environments
# ============================================================================

def _register_tau2():
    """Register tau2-bench environment."""
    # Use relative import within package context
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adapters.tau2_adapter import Tau2Adapter, Tau2EnvConfig
    
    ENV_REGISTRY.register(
        name="tau2",
        adapter_class=Tau2Adapter,
        config_class=Tau2EnvConfig,
        default_config={
            "domain": "retail",
            "task_split": "base",
            "solo_mode": False,
            "user_llm": "gpt-4.1",
            "user_llm_args": {"temperature": 1.0},
            "max_steps": 30,
        }
    )

# Register on import
_register_tau2()


# ============================================================================
# Current Environment Configuration
# ============================================================================

# Active environment name (can be changed at runtime)
ACTIVE_ENV = "tau2"

# Environment-specific overrides
ENV_CONFIG_OVERRIDES: dict[str, Any] = {}


def set_active_env(
    env_name: str, 
    config_overrides: dict[str, Any] | None = None
) -> None:
    """
    Set the active environment for generate calls.
    
    Args:
        env_name: Environment name from registry
        config_overrides: Optional config overrides
    """
    global ACTIVE_ENV, ENV_CONFIG_OVERRIDES
    
    if env_name not in ENV_REGISTRY.list_environments():
        available = ", ".join(ENV_REGISTRY.list_environments())
        raise ValueError(f"Unknown environment '{env_name}'. Available: {available}")
    
    ACTIVE_ENV = env_name
    ENV_CONFIG_OVERRIDES = config_overrides or {}


# ============================================================================
# Main Generate Interface
# ============================================================================

def _result_to_sample(result: InteractionResult, task_index: int) -> Sample:
    """
    Convert InteractionResult to Sample format for slime training.
    
    Args:
        result: InteractionResult from adapter
        task_index: Index of the task
        
    Returns:
        Sample object for slime training
    """
    status_mapping = {
        Status.PENDING: "pending",
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }
    
    sample = Sample(
        index=task_index,
        prompt=result.prompt,
        tokens=result.tokens,
        response=result.response,
        reward=result.reward,
        loss_mask=result.loss_mask,
        status=status_mapping.get(result.status, "aborted"),
        metadata=result.info,
    )
    sample.response_length = result.response_length
    
    return sample


def _turn_to_sample(
    turn: TurnResult, 
    task_index: int, 
    turn_index: int,
    episode_reward: float,
    episode_status: Status,
    total_turns: int,
) -> Sample:
    """
    Convert a single TurnResult to Sample format for turn-level training.
    
    Args:
        turn: TurnResult from adapter
        task_index: Index of the task
        turn_index: Index of the turn within the episode
        episode_reward: Final episode reward (used for the last turn)
        episode_status: Final episode status
        total_turns: Total number of turns in the episode
        
    Returns:
        Sample object for slime training
    """
    status_mapping = {
        Status.PENDING: "pending",
        Status.COMPLETED: "completed",
        Status.TRUNCATED: "truncated",
        Status.ABORTED: "aborted",
    }
    
    # For the last turn, use episode reward; otherwise use turn reward
    is_last_turn = (turn_index == total_turns - 1)
    reward = episode_reward if is_last_turn else turn.turn_reward
    status = episode_status if is_last_turn else Status.PENDING
    
    sample = Sample(
        index=task_index,
        group_index=turn_index,  # Use group_index to track turn within episode
        prompt=turn.prompt,
        tokens=turn.tokens,
        response=turn.response,
        reward=reward,
        loss_mask=turn.loss_mask,
        status=status_mapping.get(status, "pending"),
        metadata={
            **turn.info,
            "turn_index": turn_index,
            "total_turns": total_turns,
            "turn_reward": turn.turn_reward,
            "episode_reward": episode_reward,
            "is_last_turn": is_last_turn,
            "messages": turn.messages,  # Messages up to this turn
        },
    )
    sample.response_length = turn.response_length
    
    return sample


async def generate(
    args: Any, 
    sample: Sample, 
    sampling_params: dict,
    env_name: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    turn_level: bool = False,
) -> Sample | list[Sample]:
    """
    Generate a complete agent-environment interaction trajectory.
    
    This is the main entry point for slime training. It creates the appropriate
    gym environment adapter, executes a full interaction trajectory, and returns
    the result in slime's Sample format.
    
    Args:
        args: Rollout arguments from slime training pipeline
        sample: Sample containing task info in prompt field
        sampling_params: LLM sampling parameters
        env_name: Environment name (defaults to ACTIVE_ENV)
        config_overrides: Optional environment config overrides
        turn_level: If True, return list of Samples (one per turn);
                   If False, return single Sample for entire trajectory
        
    Returns:
        If turn_level=False: Sample object containing the complete interaction trajectory
        If turn_level=True: List of Sample objects, one per turn
        
    Raises:
        AssertionError: If partial rollout is requested (not supported)
    """
    # Validate arguments
    assert not getattr(args, 'partial_rollout', False), \
        "Partial rollout is not supported for gym environment interactions."
    
    # Determine environment
    env_name = env_name or ACTIVE_ENV
    overrides = config_overrides or ENV_CONFIG_OVERRIDES
    
    # Create adapter
    adapter = ENV_REGISTRY.create_adapter(env_name, args, overrides)
    
    # Parse task info from sample prompt
    prompt = str(sample.prompt)
    
    if ":" in prompt:
        # Format: "domain:task_id" or "env:task_id"
        _, task_id = prompt.split(":", 1)
    else:
        # Treat prompt as task index
        task_index = int(prompt)
        
        # For tau2-bench, convert index to task_id
        if hasattr(adapter, 'task_id_from_index'):
            task_id = adapter.task_id_from_index(task_index)
        else:
            task_id = prompt
    
    # Get task index for sample
    task_idx = int(prompt) if prompt.isdigit() else 0
    
    # Run interaction (same for both modes)
    result = await adapter.run_interaction(task_id, sampling_params)
    
    if turn_level:
        # Split trajectory into turns
        turns = result.split_into_turns(adapter.token_handler, adapter._tools_info)
        
        # Convert each turn to a sample
        samples = []
        for turn_idx, turn in enumerate(turns):
            turn_sample = _turn_to_sample(
                turn=turn,
                task_index=task_idx,
                turn_index=turn_idx,
                episode_reward=result.reward,
                episode_status=result.status,
                total_turns=len(turns),
            )
            samples.append(turn_sample)
        
        return samples
    else:
        # Return single sample for entire trajectory
        return _result_to_sample(result, task_idx)


# ============================================================================
# Task Prompt Helpers
# ============================================================================

def get_task_prompts(
    env_name: str | None = None,
    **kwargs
) -> list[str]:
    """
    Get list of task prompts for training.
    
    Args:
        env_name: Environment name (defaults to ACTIVE_ENV)
        **kwargs: Environment-specific arguments
        
    Returns:
        List of task prompt strings
    """
    env_name = env_name or ACTIVE_ENV
    
    if env_name == "tau2":
        from tau2.registry import registry
        
        domain = kwargs.get("domain", ENV_CONFIG_OVERRIDES.get("domain", "retail"))
        task_split = kwargs.get("task_split", ENV_CONFIG_OVERRIDES.get("task_split", "base"))
        
        tasks = registry.get_tasks_loader(domain)(task_split)
        return [str(i) for i in range(len(tasks))]
    
    # Default: return empty list (adapter should override)
    return []


def get_task_ids(
    env_name: str | None = None,
    **kwargs
) -> list[str]:
    """
    Get list of task IDs for an environment.
    
    Args:
        env_name: Environment name (defaults to ACTIVE_ENV)
        **kwargs: Environment-specific arguments
        
    Returns:
        List of task ID strings
    """
    env_name = env_name or ACTIVE_ENV
    
    if env_name == "tau2":
        from tau2.registry import registry
        
        domain = kwargs.get("domain", ENV_CONFIG_OVERRIDES.get("domain", "retail"))
        task_split = kwargs.get("task_split", ENV_CONFIG_OVERRIDES.get("task_split", "base"))
        
        tasks = registry.get_tasks_loader(domain)(task_split)
        return [task.id for task in tasks]
    
    return []


# ============================================================================
# Backward Compatibility
# ============================================================================

# These are provided for backward compatibility with existing code

# Legacy config dict (deprecated, use set_active_env instead)
TAU2_CONFIGS = {
    "domain": "retail",
    "task_split": "base",
    "solo_mode": False,
    "user_llm": "gpt-4.1",
    "user_llm_args": {"temperature": 1.0},
    "max_steps": 30,
}


def update_tau2_config(**kwargs) -> None:
    """
    Update tau2-bench configuration.
    
    This is a convenience function for backward compatibility.
    Prefer using set_active_env() for new code.
    """
    global TAU2_CONFIGS
    TAU2_CONFIGS.update(kwargs)
    set_active_env("tau2", TAU2_CONFIGS)


# Initialize with defaults
set_active_env("tau2", TAU2_CONFIGS)
