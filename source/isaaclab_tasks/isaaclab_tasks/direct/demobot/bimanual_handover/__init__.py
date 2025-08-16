"""
Author: Yucheng Xu
Date: 29 Apr 2025
Description: 
    Gym registration for bimanual hammer assembly direct RL env
"""

import gymnasium as gym

from . import agents

task_entry = "isaaclab_tasks.direct.demobot.bimanual_handover"


gym.register(
    id="Bimanual-Handover-Async-v0",
    entry_point=f"{task_entry}.env:HandoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HandoverEnvCfg_vel_wref_async_kpts",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandoverPPORunnerCfg_vel_wref_async_kpts",
    },
)

gym.register(
    id="Bimanual-Handover-Async-Asymmetric-v0",
    entry_point=f"{task_entry}.env:HandoverEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HandoverEnvCfg_vel_wref_async_kpts_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HandoverPPORunnerCfg_vel_wref_async_kpts_asymmetric",
    },
)
