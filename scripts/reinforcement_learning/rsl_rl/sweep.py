# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL and Optuna for hyperparameter sweeping."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import os
import copy
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
def launch_sweep(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """
    Main function decorated by Hydra to load base configs and launch the Optuna sweep.
    """
    
    def objective(trial: optuna.Trial, base_env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, base_agent_cfg: RslRlOnPolicyRunnerCfg):
        """Objective function for Optuna to optimize."""
        print(f"Starting trial: {trial.number}")
        
        # Create deep copies from the passed-in base configs.
        trial_agent_cfg = copy.deepcopy(base_agent_cfg)
        trial_env_cfg = copy.deepcopy(base_env_cfg)


        # env parameters to be swept
        reset_to_last_success_ratio = trial.suggest_categorical("reset_to_last_success_ratio", [0.0, 0.25, 0.5, 0.75])
        trial_env_cfg.reset_to_last_success_ratio = reset_to_last_success_ratio
        
        # PPO parameters to be swept
        value_loss_coef = trial.suggest_categorical("value_loss_coef", [0.25, 0.5, 0.75, 1.0])
        entropy_coef = trial.suggest_categorical("entropy_coef", [0.0, 0.00005, 0.0001])
        num_learning_epochs = trial.suggest_categorical("num_learning_epochs", [4, 8, 16, 32])
        num_mini_batches = trial.suggest_categorical("num_mini_batches", [4, 8, 16, 32, 64])
        num_steps_per_env = trial.suggest_categorical("num_steps_per_env", [16, 24, 32])
        
        
        trial_agent_cfg.algorithm.value_loss_coef = value_loss_coef
        trial_agent_cfg.algorithm.entropy_coef = entropy_coef
        trial_agent_cfg.algorithm.num_learning_epochs = num_learning_epochs
        trial_agent_cfg.algorithm.num_mini_batches = num_mini_batches
        trial_agent_cfg.num_steps_per_env = num_steps_per_env

        # override configurations with non-hydra CLI arguments
        trial_agent_cfg = cli_args.update_rsl_rl_cfg(trial_agent_cfg, args_cli)
        trial_env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else trial_env_cfg.scene.num_envs
        trial_agent_cfg.max_iterations = (
            args_cli.max_iterations if args_cli.max_iterations is not None else trial_agent_cfg.max_iterations
        )

        # set the environment seed
        trial_env_cfg.seed = trial_agent_cfg.seed
        trial_env_cfg.sim.device = args_cli.device if args_cli.device is not None else trial_env_cfg.sim.device
        
        run_name = f"ratio{reset_to_last_success_ratio}-vcoef{value_loss_coef}-ecoef{entropy_coef}-nlep{num_learning_epochs}-miniep{num_mini_batches}-rollout{num_steps_per_env}"
        trial_agent_cfg.wandb_kwargs['run_name'] = run_name
        
        trial_log_string = (
            f"""reset_to_last_success_ratio: {trial_env_cfg.reset_to_last_success_ratio}\n"""
            f"""value_loss_coef: {trial_agent_cfg.algorithm.value_loss_coef}\n"""
            f"""entropy_coef: {trial_agent_cfg.algorithm.entropy_coef}\n"""
            f"""num_learning_epochs: {trial_agent_cfg.algorithm.num_learning_epochs}\n"""
            f"""num_mini_batches: {trial_agent_cfg.algorithm.num_mini_batches}\n"""
            f"""num_steps_per_env: {trial_agent_cfg.num_steps_per_env}\n\n"""
            
        )
        
        print(trial_log_string)

        # multi-gpu training configuration
        if args_cli.distributed:
            trial_env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            trial_agent_cfg.device = f"cuda:{app_launcher.local_rank}"
            seed = trial_agent_cfg.seed + app_launcher.local_rank
            trial_env_cfg.seed = seed
            trial_agent_cfg.seed = seed

        # specify directory for logging experiments
        log_root_path = os.path.join("logs", "rsl_rl", "optuna_sweep", trial_agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_trial_{trial.number}"
        if trial_agent_cfg.run_name:
            log_dir += f"_{trial_agent_cfg.run_name}"
        log_dir = os.path.join(log_root_path, log_dir)

        # create isaac environment
        env = gym.make(args_cli.task, cfg=trial_env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        if trial_agent_cfg.resume or trial_agent_cfg.algorithm.class_name == "Distillation":
            resume_path = get_checkpoint_path(log_root_path, trial_agent_cfg.load_run, trial_agent_cfg.load_checkpoint)

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

        env = RslRlVecEnvWrapper(env, clip_actions=trial_agent_cfg.clip_actions)

        runner = OnPolicyRunner(env, trial_agent_cfg.to_dict(), log_dir=log_dir, device=trial_agent_cfg.device)
        runner.add_git_repo_to_log(__file__)
        if trial_agent_cfg.resume or trial_agent_cfg.algorithm.class_name == "Distillation":
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)

        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), trial_env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), trial_agent_cfg)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), trial_env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), trial_agent_cfg)
        
        # run training with the pruning callback
        try:
            final_reward, is_prune = runner.learn(
                num_learning_iterations=trial_agent_cfg.max_iterations, 
                init_at_random_ep_len=False,
                trial=trial
            )
        except optuna.exceptions.TrialPruned:
            print(f"Trial {trial.number} pruned.")
            env.close()
            # Return a value for pruned trials. Optuna handles this, but a high negative value can be a clear indicator
            return -1.0

        return final_reward

    pruner = MedianPruner(
        n_startup_trials=5,  # Allow first 5 trials to complete without pruning
        n_warmup_steps=2000,  # Warmup for 2000 iterations before pruning
        interval_steps=100   # Check for pruning every 100 iterations
    )
    sampler = optuna.samplers.TPESampler(n_startup_trials=args_cli.num_trials, seed=3107)

    
    # create the optuna study
    os.makedirs(os.path.join("logs", "rsl_rl", "optuna_sweep", agent_cfg.experiment_name), exist_ok=True)
    db_path = os.path.join("logs", "rsl_rl", "optuna_sweep", agent_cfg.experiment_name, "sweep.db")
    study = optuna.create_study(
        storage=f"sqlite:///{db_path}",
        sampler=sampler,
        pruner=pruner,
        study_name=args_cli.study_name,
        direction="maximize",
        load_if_exists=True,
    )
    
    # Change the wandb project of the wandb writer
    agent_cfg.wandb_kwargs['project'] = 'demobot-sweep'
    
    # run the main function
    study.optimize(
        lambda trial: objective(
                trial, base_env_cfg=env_cfg, base_agent_cfg=agent_cfg
            ),
        n_trials=args_cli.num_trials)  # n_trials is the number of hyperparameter combinations to test

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    launch_sweep()
    # close sim app
    simulation_app.close()
