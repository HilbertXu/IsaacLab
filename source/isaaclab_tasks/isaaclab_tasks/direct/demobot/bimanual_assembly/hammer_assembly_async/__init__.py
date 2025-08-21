"""
Author: Yucheng Xu
Date: 29 Apr 2025
Description: 
    Gym registration for bimanual hammer assembly direct RL env
"""

import gymnasium as gym

from . import agents

task_entry = "isaaclab_tasks.direct.demobot.bimanual_assembly.hammer_assembly_async"


gym.register(
    id="Bimanual-Hammer-Assembly-Async-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts",
    },
)

gym.register(
    id="Bimanual-Hammer-Assembly-Async-Asymmetric-v0",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_asymmetric",
    },
)


# Baseline comparison experiments
################################################################
# with reference, velocity control, with/without reaching stage
################################################################
gym.register(
    id="Async-Assembly-Baseline-wref-vel-wreach",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:ours_wref_vel_wreach",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:OursPPORunnerCfg_wref_vel_wreach",
    },
)

gym.register(
    id="Async-Assembly-Baseline-wref-vel-woreach",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:Baselines_wref_vel_woreach",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg_wref_vel_woreach",
    },
)


################################################################
# with reference, position control, with/without reaching stage
################################################################
gym.register(
    id="Async-Assembly-Baseline-wref-pos-wreach",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:Baselines_wref_pos_wreach",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg_wref_pos_wreach",
    },
)

gym.register(
    id="Async-Assembly-Baseline-wref-pos-woreach",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:Baselines_wref_pos_woreach",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg_wref_pos_woreach",
    },
)

################################################################
# without reference, velocity control, with/without reaching stage
################################################################
gym.register(
    id="Async-Assembly-Baseline-woref-vel-wreach",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:Baselines_woref_vel_wreach",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg_woref_vel_wreach",
    },
)

gym.register(
    id="Async-Assembly-Baseline-woref-vel-woreach",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:Baselines_woref_vel_woreach",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg_woref_vel_woreach",
    },
)


################################################################
# without reference, position control, with/without reaching stage
################################################################
gym.register(
    id="Async-Assembly-Baseline-woref-pos-wreach",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:Baselines_woref_pos_wreach",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg_woref_pos_wreach",
    },
)

gym.register(
    id="Async-Assembly-Baseline-woref-pos-woreach",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:Baselines_woref_pos_woreach",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg_woref_pos_woreach",
    },
)



##############################################################################
# with reference, velocity control, with reaching stage, disable chunk-split #
##############################################################################
gym.register(
    id="Async-Assembly-Baseline-wref-vel-wreach-nochunk",
    entry_point=f"{task_entry}.env_nochunk:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:Baselines_wref_vel_wreach_nochunk",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg_wref_vel_wreach_nochunk",
    },
)


######################################################################################
# with reference, velocity control, with reaching stage, disable success-gated reset #
######################################################################################
gym.register(
    id="Async-Assembly-Baseline-wref-vel-wreach-noreset",
    entry_point=f"{task_entry}.env:HammerAssemblyEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cfg:Baselines_wref_vel_wreach_noreset",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BaselinePPORunnerCfg_wref_vel_wreach_noreset",
    },
)