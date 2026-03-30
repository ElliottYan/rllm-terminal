"""Trainer module for rLLM.

This module contains the training infrastructure for RL training of language agents.
"""

from .env_agent_mappings import *

__all__ = []

try:
    from .agent_trainer import AgentTrainer

    __all__.append("AgentTrainer")
except ImportError:
    # AgentTrainer pulls in optional training dependencies such as Ray and dataset
    # backends. Keep lightweight imports like env/agent mappings available even when
    # those optional packages are not installed.
    AgentTrainer = None
