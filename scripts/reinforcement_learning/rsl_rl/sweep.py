# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL and Optuna for hyperparameter sweeping."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import torch
from datetime import datetime
import optuna
from optuna.pruners import MedianPruner

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL and Optuna.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)

# optuna parameters
parser.add_argument('--study_name', type=str, default='')
parser.add_argument('--num_trials', type=int, default=10)
parser.add_argument('--storage', type=str, default='')

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def objective(trial: optuna.Trial, env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Objective function for Optuna to optimize."""
    # -- Hyperparameters to sweep --
    # env parameters to be swept
    env_cfg.reset_to_last_success_ratio = trial.suggest_categorical("reset_to_last_success_ratio", [0.0, 0.25, 0.5, 0.75])
    
    # PPO parameters to be swept
    agent_cfg.algorithm.value_loss_coef = trial.suggest_categorical("value_loss_coef", [0.25, 0.5, 0.75, 1.0])
    agent_cfg.algorithm.entropy_coef = trial.suggest_categorical("entropy_coef", [0.0, 0.00005, 0.0001])
    agent_cfg.algorithm.num_learning_epochs = trial.suggest_categorical("num_learning_epochs", [4, 8, 16, 32])
    agent_cfg.algorithm.num_mini_batches = trial.suggest_categorical("num_mini_batches", [4, 8, 16, 32, 64])
    agent_cfg.num_steps_per_env = trial.suggest_categorical("num_mini_batches", [16, 24, 32])

    # Environment Parameters (example for a locomotor environment)
    # You might need to adjust these based on the specific task
    if hasattr(env_cfg, "rewards"):
        if hasattr(env_cfg.rewards, "lin_vel_xy_exp"):
             env_cfg.rewards.lin_vel_xy_exp["weight"] = trial.suggest_float("reward_lin_vel_xy_weight", 0.5, 2.0)
        if hasattr(env_cfg.rewards, "alive"):
            env_cfg.rewards.alive["weight"] = trial.suggest_float("reward_alive_weight", 0.1, 1.0)

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", "optuna_sweep", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_trial_{trial.number}"
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)
    
    # run training with the pruning callback
    try:
        final_reward, is_prune = runner.learn(
            num_learning_iterations=agent_cfg.max_iterations, 
            init_at_random_ep_len=False,
            trial=trial
        )
    except optuna.exceptions.TrialPruned:
        print(f"Trial {trial.number} pruned.")
        env.close()
        # Return a value for pruned trials. Optuna handles this, but a high negative value can be a clear indicator
        return -1.0

    return final_reward


if __name__ == "__main__":
    
    pruner = MedianPruner(
        n_startup_trials=5,  # Allow first 5 trials to complete without pruning
        n_warmup_steps=2000,  # Warmup for 2000 iterations before pruning
        interval_steps=100   # Check for pruning every 100 iterations
    )
    sampler = optuna.samplers.TPESampler(n_startup_trials=args_cli.num_trials)

    
    # create the optuna study
    study = optuna.create_study(
        storage=args_cli.storage,
        sampler=sampler,
        pruner=pruner,
        study_name=args_cli.study_name,
        direction="maximize",
        load_if_exists=True,
    )
    
    # run the main function
    study.optimize(objective, n_trials=args_cli.num_trials)  # n_trials is the number of hyperparameter combinations to test

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # close sim app
    simulation_app.close()