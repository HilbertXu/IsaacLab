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
def launch_sweep(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """
    Main function decorated by Hydra to load base configs and launch the Optuna sweep.
    This version creates the environment ONCE and reconfigures it for each trial.
    """
    
    # --- 1. Create the Environment and Runner ONCE, outside the objective function ---
    
    # Override configurations with non-hydra CLI arguments that do not change between trials
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sweep = True # set envs to sweep mode, to disable curriculum
    
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    agent_cfg.sweep = True # set runner to sweep mode, use torch.no_grad() in stead of torch.inference_mode()
    
    # Multi-gpu training configuration (if applicable)
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed
    else:
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        
    # Set the base environment seed
    env_cfg.seed = agent_cfg.seed
        
    # Create the single, persistent Isaac environment instance
    print("[INFO] Creating the persistent Isaac Lab environment...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap the environment if needed (e.g., for multi-agent)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        
    # Wrap the environment for RSL-RL
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    
    
    # --- 2. Define the Objective Function ---
    
    def objective(trial: optuna.Trial):
        """Objective function for Optuna to optimize."""
        print(f"\n-------------------- Starting Trial: {trial.number} --------------------")
        
        # NOTE: We do NOT deepcopy the configs. We will modify the existing ones.
        # This is safe because we are reconfiguring a running environment.
        
        # --- A. Suggest and Reconfigure Environment Hyperparameters ---
        
        # Get the unwrapped env to access its properties directly
        unwrapped_env = env.unwrapped.unwrapped
        
        reset_to_last_success_ratio = trial.suggest_categorical("reset_to_last_success_ratio", [0.75])
        # IMPORTANT: Directly set the attribute on the running environment instance
        unwrapped_env.reset_to_last_success_ratio = reset_to_last_success_ratio
        
        # --- B. Suggest and Create a New Agent/Runner Config for this trial ---
        trial_agent_cfg = copy.deepcopy(agent_cfg) # The agent config can be safely copied
        
        value_loss_coef = trial.suggest_categorical("value_loss_coef", [0.5, 0.75, 1.0])
        entropy_coef = trial.suggest_categorical("entropy_coef", [0.00005, 0.0001, 0])
        num_learning_epochs = trial.suggest_categorical("num_learning_epochs", [4, 8, 16, 32])
        num_mini_batches = trial.suggest_categorical("num_mini_batches", [4, 8, 16, 32])
        num_steps_per_env = trial.suggest_categorical("num_steps_per_env", [16, 24, 32])
        
        trial_agent_cfg.algorithm.value_loss_coef = value_loss_coef
        trial_agent_cfg.algorithm.entropy_coef = entropy_coef
        trial_agent_cfg.algorithm.num_learning_epochs = num_learning_epochs
        trial_agent_cfg.algorithm.num_mini_batches = num_mini_batches
        trial_agent_cfg.num_steps_per_env = num_steps_per_env
        
        # Create a unique run name for logging
        run_name = f"ratio{reset_to_last_success_ratio}-vcoef{value_loss_coef}-ecoef{entropy_coef}-nlep{num_learning_epochs}-miniep{num_mini_batches}-rollout{num_steps_per_env}"
        trial_agent_cfg.wandb_kwargs['project'] = 'demobot-sweep'
        trial_agent_cfg.wandb_kwargs['run_name'] = run_name
        
        print(f"Trial {trial.number} Parameters:")
        print(f"  - reset_to_last_success_ratio: {unwrapped_env.reset_to_last_success_ratio}")
        print(f"  - value_loss_coef: {trial_agent_cfg.algorithm.value_loss_coef}")
        print(f"  - entropy_coef: {trial_agent_cfg.algorithm.entropy_coef}")
        print(f"  - num_learning_epochs: {trial_agent_cfg.algorithm.num_learning_epochs}")
        print(f"  - num_mini_batches: {trial_agent_cfg.algorithm.num_mini_batches}")
        print(f"  - num_steps_per_env: {trial_agent_cfg.num_steps_per_env}")
        print()
        # ... print other params ...
        
        # --- C. Create a New Runner for Each Trial ---
        
        # Specify a unique directory for this trial's logs
        log_root_path = os.path.join("logs", "rsl_rl", "optuna_sweep", trial_agent_cfg.experiment_name)
        log_root_path = os.path.abspath(log_root_path)
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_trial_{trial.number}"
        log_dir = os.path.join(log_root_path, log_dir)
        
        # Create a new runner with the trial-specific agent config.
        # It will use the *same* persistent `env` object.
        runner = OnPolicyRunner(env, trial_agent_cfg.to_dict(), log_dir=log_dir, device=trial_agent_cfg.device)
        
        # Dump the params for this specific trial
        # Note: We dump the env's current config, which might have been modified
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), unwrapped_env.cfg)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), trial_agent_cfg)
        
        final_reward = 0.0
        try:
            # The runner's learn method will reset the environment internally before starting
            final_reward, _ = runner.learn(
                num_learning_iterations=trial_agent_cfg.max_iterations, 
                init_at_random_ep_len=False,
                trial=trial
            )
        except optuna.exceptions.TrialPruned:
            print(f"Trial {trial.number} pruned.")
            # No need to close the env, just return
            return -1.0 # Return a value to signify pruning

        runner.writer.stop()

        return final_reward

    # --- 3. Set up and Run the Optuna Study ---
    
    known_good_params = {
		"value_loss_coef": 0.5,
		"entropy_coef": 5e-5,
		"num_learning_epochs": 4,
		"num_mini_batches": 32,
		"num_steps_per_env": 32,
		"reset_to_last_success_ratio": 0.75,
	}
    
    pruner = MedianPruner(
        n_startup_trials=5,  # Allow first 5 trials to complete without pruning
        n_warmup_steps=4000,  # Warmup for 2000 iterations before pruning
        interval_steps=100   # Check for pruning every 100 iterations
    )
    sampler = optuna.samplers.TPESampler(n_startup_trials=args_cli.num_trials)
    
    os.makedirs(os.path.join("source", "rsl_rl", "optuna_sweep", agent_cfg.experiment_name), exist_ok=True)
    study = optuna.create_study(
        storage=f'sqlite:///{os.path.join("source", "rsl_rl", "optuna_sweep", agent_cfg.experiment_name, "sweep.db")}',
        sampler=sampler,
        pruner=pruner,
        study_name=args_cli.study_name,
        direction="maximize",
        load_if_exists=True,
    )
    study.enqueue_trial(known_good_params)
    
    # IMPORTANT: The lambda now only passes the trial object
    study.optimize(objective, n_trials=args_cli.num_trials)
    
    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # --- 4. Clean up the single environment at the very end ---
    print("[INFO] Sweep finished. Closing the environment.")
    env.close()


if __name__ == "__main__":
    launch_sweep()
    # close sim app at the very end of the script
    simulation_app.close()

