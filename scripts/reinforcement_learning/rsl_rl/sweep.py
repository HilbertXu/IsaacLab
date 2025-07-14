"""
Author: Yucheng Xu
Date: 12 June 2025
Description: 
    Script for sweeping hyperparameters with RSL_RL
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import optuna

# add argparse arguments
parser = argparse.ArgumentParser(description="Sweep PPO hyperparameters with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--study", type=str, default="default", help="study name")

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

import gymnasium as gym
import os
import torch
from datetime import datetime

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

class OptunaRunner:
    def __init__(
            self, 
            storage: str,
            study_name: str, 
            n_startup_trials: int = 3, 
            n_warmup_steps: int = 100, 
            interval_steps: int = 10
    ):
        self.sampler = optuna.samplers.TPESampler(n_startup_trials=n_startup_trials)

        self.pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
            interval_steps=interval_steps
        )

        self.study = optuna.create_study(
            storage=storage,
            sampler=self.sampler,
            pruner=self.pruner,
            study_name=study_name,
            direction="maximize",
            load_if_exists=True,
        )

    
    def objective(self, trial: optuna.Trial, env, env_cfg, agent_cfg):
        print(f"Starting trial: {trial.number}")

        # https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
        # CHOOSE YOUR OWN HPARAMS TO OPTIMISE
        mini_batches = trial.suggest_categorical("mini_batches", [4, 8, 16, 32, 64])
        learning_rate = trial.suggest_float("learning_rate", low=1e-5, high=1e-3, log=True)
        learning_epochs = trial.suggest_categorical("learning_epochs", [4, 8, 16, 32])
        rollouts = trial.suggest_categorical("rollouts", [16, 32, 64])
    

    def run(self, env, env_cfg, agent_cfg, n_trials=50):
        self.study.optimize(
            lambda trial: self.objective(
                trial, env=env, env_cfg=env_cfg, agent_cfg=agent_cfg
            ),
            n_trials=n_trials,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        # Antonin's code
        print(f"Number of finished trials: {len(self.study.trials)}")
        print("Best trial:")
        trial = self.study.best_trial
        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("  User attrs:")
        for key, value in trial.user_attrs.items():
            print(f"    {key}: {value}")
        
        return self.study.best_trial