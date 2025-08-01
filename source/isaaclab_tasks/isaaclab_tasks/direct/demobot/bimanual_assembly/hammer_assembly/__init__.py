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

task_entry = "isaaclab_tasks.direct.demobot.bimanual_assembly.hammer_assembly"

###############
# bimanual envs
###############
gym.register(
    id="Bimanual-Hammer-Assembly-Sync-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Asymmetric-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_asymmetric",
    },
)

gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Asymmetric-v1",
    entry_point=f"{task_entry}.env_v1:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg_v1:HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_asymmetric",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Asymmetric-v1",
    entry_point=f"{task_entry}.env_v1:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg_v1:HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_asymmetric",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Asymmetric-Pos-v1",
    entry_point=f"{task_entry}.env_v1:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg_v1:BaselineCfg_pos",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselineCfg_pos",
    },
)

gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Asymmetric-Posv2-v1",
    entry_point=f"{task_entry}.env_v1:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg_v1:BaselineCfg_posv2",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselineCfg_posv2",
    },
)

gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Asymmetric-Pos-Noref-v1",
    entry_point=f"{task_entry}.env_v1:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg_v1:BaselineCfg_pos_noref",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg_pos_noref",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Asymmetric-RND-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric_rnd",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_asymmetric_rnd",
    },
)


###############
# right hand envs
###############
gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Right-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_right",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Right-Asymmetric-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_right_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right_asymmetric",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Right-Asymmetric-v1",
    entry_point=f"{task_entry}.env_v1:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg_v1:HammerAssemblyEnvCfg_vel_wref_async_kpts_right_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right_asymmetric",
    },
)

gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Right-Asymmetric-v1",
    entry_point=f"{task_entry}.env_v1:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg_v1:HammerAssemblyEnvCfg_vel_wref_async_kpts_right_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right_asymmetric",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Right-Asymmetric-RND-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_right_asymmetric_rnd",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right_asymmetric_rnd",
    },
)




###############
# left hand envs
###############
gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Left-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_left",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Left-Asymmetric-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_left_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left_asymmetric",
    },
)

gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Left-Asymmetric-v1",
    entry_point=f"{task_entry}.env_v1:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg_v1:HammerAssemblyEnvCfg_vel_wref_async_kpts_left_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left_asymmetric",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Left-Asymmetric-RND-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_left_asymmetric_rnd",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left_asymmetric_rnd",
    },
)


###############
# eval envs
###############

gym.register(
    id="Bimanual-Hammer-Assembly-Sync-v0-Eval",
    entry_point=f"{task_entry}.env_eval:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Right-v0-Eval",
    entry_point=f"{task_entry}.env_eval:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_right",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Left-v0-Eval",
    entry_point=f"{task_entry}.env_eval:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_left",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left",
    },
)


gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Right-Asymmetric-Eval",
    entry_point=f"{task_entry}.env_eval:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_right_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right_asymmetric",
    },
)

gym.register(
    id="Bimanual-Hammer-Assembly-Sync-Left-Asymmetric-Eval",
    entry_point=f"{task_entry}.env_eval:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_left_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left_asymmetric",
    },
)