"""
Author: Yucheng Xu
Date: 29 Apr 2025
Description: 
    Gym registration for bimanual hammer assembly direct RL env
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

inhand_task_entry = "isaaclab_tasks.direct.demobot.bimanual_assembly.hammer_assembly"


gym.register(
    id="Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-v0",
    entry_point=f"{inhand_task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Asymmetric-v0",
    entry_point=f"{inhand_task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_asymmetric",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-GRU-Vel-WRef-Async-Kpts-Asymmetric-v0",
    entry_point=f"{inhand_task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_gru_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_asymmetric",
    },
)

gym.register(
    id="Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Right-v0",
    entry_point=f"{inhand_task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_right",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Right-Asymmetric-v0",
    entry_point=f"{inhand_task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_right_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right_asymmetric",
    },
)



gym.register(
    id="Bimanual-Hammer-Assembly-GRU-Vel-WRef-Async-Kpts-Right-Asymmetric-v0",
    entry_point=f"{inhand_task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_right_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_gru_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right_asymmetric",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Left-v0",
    entry_point=f"{inhand_task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_left",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Left-Asymmetric-v0",
    entry_point=f"{inhand_task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_left_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left_asymmetric",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-GRU-Vel-WRef-Async-Kpts-Left-Asymmetric-v0",
    entry_point=f"{inhand_task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_left_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_gru_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left_asymmetric",
    },
)



gym.register(
    id="Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-v0-Eval",
    entry_point=f"{inhand_task_entry}.env_eval:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Right-v0-Eval",
    entry_point=f"{inhand_task_entry}.env_eval:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_right",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Vel-WRef-Async-Kpts-Left-v0-Eval",
    entry_point=f"{inhand_task_entry}.env_eval:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_left",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left",
    },
)
