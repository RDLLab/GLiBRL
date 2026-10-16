from metaworld_algorithms.config.rl import AlgorithmConfig

from .base import Algorithm, OffPolicyAlgorithm
from .mtsac import MTSAC, MTSACConfig
from .ppo import PPOConfig, PPO
from .sac import SAC, SACConfig
from .mamltrpo import MAMLTRPO, MAMLTRPOConfig
from .rl2 import RL2, RL2Config
from .glibrl import GLiBRL, GLiBRLConfig
from .glibrl_sac import GLiBRLSAC, GLiBRLSACConfig
from .varibad import VariBad, VariBadConfig

def get_algorithm_for_config(config: AlgorithmConfig) -> type[Algorithm]:
    if type(config) is MTSACConfig:
        return MTSAC
    elif type(config) is PPOConfig:
        return PPO
    elif type(config) is SACConfig:
        return SAC
    elif type(config) is MAMLTRPOConfig:
        return MAMLTRPO
    elif type(config) is RL2Config:
        return RL2
    elif type(config) is GLiBRLConfig:
        return GLiBRL
    elif type(config) is GLiBRLSACConfig:
        return GLiBRLSAC
    elif type(config) is VariBadConfig:
        return VariBad
    else:
        raise ValueError(f"Invalid config type: {type(config)}")


__all__ = [
    "Algorithm",
    "OffPolicyAlgorithm",
    "MTSAC",
    "PPO",
    "MTSACConfig",
    "PPOConfig",
]
