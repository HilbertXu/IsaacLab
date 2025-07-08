# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg, RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab.utils import configclass


@configclass
class HammerAssemblyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 500
    experiment_name = "bimanual_hammer_assembly_gru"
    empirical_normalization = True
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_size=256,
        rnn_num_layers=1,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
    )

    # set logger to wandb
    logger = 'wandb'
    experiment_name = "base"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly-GRU',
        'run_name': 'base'
    }



@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts(HammerAssemblyPPORunnerCfg):
    experiment_name = "gru_vel_wref_async_kpts"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly-GRU',
        'run_name': 'gru_vel_wref_async_kpts'
    }

# use object keypoint
@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_asymmetric(HammerAssemblyPPORunnerCfg):
    experiment_name = "gru_vel_wref_async_kpts_asymmetric"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly-GRU',
        'run_name': 'gru_vel_wref_async_kpts_asymmetric'
    }


@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right_asymmetric(HammerAssemblyPPORunnerCfg):
    experiment_name = "gru_vel_wref_async_kpts_right_asymmetric"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly-GRU',
        'run_name': 'gru_vel_wref_async_kpts_right_asymmetric'
    }

@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left_asymmetric(HammerAssemblyPPORunnerCfg):
    experiment_name = "gru_vel_wref_async_kpts_left_asymmetric"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly-GRU',
        'run_name': 'gru_vel_wref_async_kpts_left_asymmetric'
    }