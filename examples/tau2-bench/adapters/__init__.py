"""
Adapters for various gym-like environments.

This module contains concrete implementations of BaseGymEnvAdapter
for different environments.

Built-in Adapters:
    - Tau2Adapter: tau2-bench multi-domain task-oriented dialogue

Adding New Adapters:
    1. Create your adapter in this package (e.g., my_env_adapter.py)
    2. Import and export from this __init__.py
    3. Register with ENV_REGISTRY in generate_new.py
"""

from .tau2_adapter import Tau2Adapter, Tau2EnvConfig, create_tau2_adapter

__all__ = [
    "Tau2Adapter",
    "Tau2EnvConfig",
    "create_tau2_adapter",
    "MockEnvConfig",
    "register_mock_adapter",
]
