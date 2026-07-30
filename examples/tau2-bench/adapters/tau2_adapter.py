"""
Tau2-Bench environment adapter.

This module provides the adapter for integrating tau2-bench with slime's
training pipeline using the standard gym interface.
"""

import os
from dataclasses import dataclass, field
from typing import Any

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base.adapter import BaseGymEnvAdapter
from base.types import EnvConfig, ParsedToolResult
from base.utils import tools_to_openai_format


@dataclass
class Tau2EnvConfig(EnvConfig):
    """Configuration specific to tau2-bench environment."""
    domain: str = "retail"
    task_split: str = "base"
    solo_mode: bool = False
    user_llm: str = "gpt-4.1"
    max_steps: int = 30
    user_llm_args: dict[str, Any] = field(default_factory=lambda: {"temperature": 1.0})


class Tau2Adapter(BaseGymEnvAdapter):
    """
    Adapter for tau2-bench gymnasium environment.
    
    Tau2-bench provides multi-domain task-oriented dialogue benchmarks
    with a gymnasium-compatible interface for agent training.
    
    Supported domains: retail, airline, telecom, mock
    """
    
    def __init__(self, config: Tau2EnvConfig, rollout_args: Any):
        """
        Initialize the tau2 adapter.
        
        Args:
            config: Tau2-specific configuration
            rollout_args: Rollout arguments from slime
        """
        super().__init__(config, rollout_args)
        self.tau2_config = config
        self._registered = False
    
    def _ensure_registered(self):
        """Ensure tau2-bench gym environments are registered."""
        if not self._registered:
            from tau2.gym import register_gym_agent
            register_gym_agent()
            self._registered = True
    
    def create_env(self, task_id: str) -> Any:
        """
        Create tau2-bench gym environment.
        
        Args:
            task_id: Task identifier (task ID within domain)
            
        Returns:
            AgentGymEnv instance
        """
        import gymnasium as gym
        from tau2.gym import TAU_BENCH_ENV_ID
        
        self._ensure_registered()
        
        return gym.make(
            TAU_BENCH_ENV_ID,
            domain=self.tau2_config.domain,
            task_id=task_id,
            max_steps=self.tau2_config.max_steps,
            solo_mode=self.tau2_config.solo_mode,
            user_llm=self.tau2_config.user_llm,
            user_llm_args=self.tau2_config.user_llm_args,
        )
    
    def get_initial_messages(
        self, 
        policy: str, 
        initial_observation: str | None
    ) -> list[dict[str, Any]]:
        """
        Build initial conversation messages for tau2-bench.
        
        Args:
            policy: Domain policy from environment
            initial_observation: Initial user query
            
        Returns:
            List of system and user messages
        """
        messages = [{"role": "system", "content": policy}]
        if initial_observation:
            messages.append({"role": "user", "content": initial_observation})
        return messages
    
    def parse_env_info(self, info: dict[str, Any]) -> tuple[list[dict], str]:
        """
        Extract tools and policy from tau2-bench env info.
        
        Args:
            info: Info dict from env.reset()
            
        Returns:
            Tuple of (tools_info, policy)
        """
        tools = info.get("tools", [])
        policy = info.get("policy", "")
        tools_info = tools_to_openai_format(tools)
        return tools_info, policy
    
    def process_observation(
        self, 
        observation: str | None, 
        parsed_result: ParsedToolResult,
        messages: list[dict],
    ) -> dict[str, Any] | None:
        """
        Convert tau2-bench observation to message.
        
        Args:
            observation: Observation from env.step()
            parsed_result: Parsed tool call from LLM response
            messages: Current conversation
            
        Returns:
            Tool or user message dict
        """
        if not observation:
            return None
        
        # If the LLM made a tool call, this is a tool response
        if parsed_result.calls:
            return {
                "role": "tool",
                "name": parsed_result.calls[0].name,
                "content": observation,
            }
        else:
            # User response (from user simulator)
            return {"role": "user", "content": observation}
    
    def get_tasks(self) -> list[str]:
        """
        Get all task IDs for the configured domain.
        
        Returns:
            List of task ID strings
        """
        from tau2.registry import registry
        
        tasks = registry.get_tasks_loader(self.tau2_config.domain)(
            self.tau2_config.task_split
        )
        return [task.id for task in tasks]
    
    def get_task_indices(self) -> list[str]:
        """
        Get task indices as strings for training prompts.
        
        Returns:
            List of task index strings
        """
        from tau2.registry import registry
        
        tasks = registry.get_tasks_loader(self.tau2_config.domain)(
            self.tau2_config.task_split
        )
        return [str(i) for i in range(len(tasks))]
    
    def task_id_from_index(self, index: int) -> str:
        """
        Get task ID from task index.
        
        Args:
            index: Task index
            
        Returns:
            Task ID string
        """
        from tau2.registry import registry
        
        tasks = registry.get_tasks_loader(self.tau2_config.domain)(
            self.tau2_config.task_split
        )
        if index >= len(tasks):
            raise ValueError(f"Task index {index} out of range for domain {self.tau2_config.domain}")
        return tasks[index].id


# Factory function for easy creation
def create_tau2_adapter(
    domain: str = "retail",
    task_split: str = "base",
    solo_mode: bool = False,
    user_llm: str = "gpt-4.1",
    user_llm_args: dict[str, Any] | None = None,
    max_steps: int = 30,
    rollout_args: Any = None,
) -> Tau2Adapter:
    """
    Create a Tau2Adapter with the given configuration.
    
    Args:
        domain: Domain name (retail, airline, telecom, mock)
        task_split: Task split (train, test, base)
        solo_mode: Whether agent works alone or with user simulator
        user_llm: LLM for user simulator
        user_llm_args: LLM arguments for user simulator
        max_steps: Maximum interaction steps
        rollout_args: Rollout arguments from slime
        
    Returns:
        Configured Tau2Adapter instance
    """
    config = Tau2EnvConfig(
        domain=domain,
        task_split=task_split,
        solo_mode=solo_mode,
        user_llm=user_llm,
        user_llm_args=user_llm_args or {"temperature": 1.0},
        max_steps=max_steps,
    )
    return Tau2Adapter(config, rollout_args)
