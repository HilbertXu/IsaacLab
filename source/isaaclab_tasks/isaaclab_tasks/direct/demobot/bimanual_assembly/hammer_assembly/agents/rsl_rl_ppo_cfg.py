# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlRndCfg

from isaaclab.utils import configclass


@configclass
class HammerAssemblyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # device = "cpu"
    num_steps_per_env = 16
    max_iterations = 100000
    save_interval = 1000
    empirical_normalization = True
    num_eval_envs = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        zero_init=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
        value_normalization=True,
        enable_amp=True,
    )
    # set logger to wandb
    logger = 'wandb'
    experiment_name = "base"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly',
        'run_name': 'base'
    }

    debug = 1
    sweep = False


# use object keypoint
@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts(HammerAssemblyPPORunnerCfg):
    experiment_name = "bimanual_kpts"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly',
        'run_name': 'bimanual_kpts'
    }


# use object keypoint
@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_asymmetric(HammerAssemblyPPORunnerCfg):
    experiment_name = "bimanual_kpts_asymmetric"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly',
        'run_name': 'bimanual_kpts_asymmetric'
    }


@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_asymmetric_rnd(HammerAssemblyPPORunnerCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
        rnd_cfg=RslRlRndCfg(
            weight=5,
            weight_schedule=RslRlRndCfg.LinearWeightScheduleCfg(
                final_value=0.5,
                initial_step=30000,
                final_step=22500
            ),
            reward_normalization=False,
            state_normalization=True,
            learning_rate=3e-4
        )
    )

    experiment_name = "bimanual_kpts_asymmetric_rnd"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly',
        'run_name': 'bimanual_kpts_asymmetric_rnd'
    }
    
    

@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right(HammerAssemblyPPORunnerCfg):
    experiment_name = "right_kpts"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly',
        'run_name': 'right_kpts'
    }


@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right_asymmetric(HammerAssemblyPPORunnerCfg):
    experiment_name = "right_kpts_asymmetric"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly',
        'run_name': 'right_kpts_asymmetric'
    }

@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_right_asymmetric_rnd(HammerAssemblyPPORunnerCfg):

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
        rnd_cfg=RslRlRndCfg(
            weight=5,
            weight_schedule=RslRlRndCfg.LinearWeightScheduleCfg(
                final_value=0.5,
                initial_step=30000,
                final_step=22500
            ),
            reward_normalization=False,
            state_normalization=True,
            learning_rate=3e-4
        )
    )
    experiment_name = "right_kpts_asymmetric_rnd"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly',
        'run_name': 'right_kpts_asymmetric_rnd'
    }
    

@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left(HammerAssemblyPPORunnerCfg):
    experiment_name = "kpts_left"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly',
        'run_name': 'kpts_left'
    }


@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left_asymmetric(HammerAssemblyPPORunnerCfg):
    experiment_name = "left_kpts_asymmetric"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly',
        'run_name': 'left_kpts_asymmetric'
    }


@configclass
class HammerAssemblyPPORunnerCfg_vel_wref_async_kpts_left_asymmetric_rnd(HammerAssemblyPPORunnerCfg):
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.00,
        num_learning_epochs=8,
        num_mini_batches=8,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
        rnd_cfg=RslRlRndCfg(
            weight=5,
            weight_schedule=RslRlRndCfg.LinearWeightScheduleCfg(
                final_value=0.5,
                initial_step=30000,
                final_step=22500
            ),
            reward_normalization=False,
            state_normalization=True,
            learning_rate=3e-4
        )
    )
    experiment_name = "left_kpts_asymmetric_rnd"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-hammer-assembly',
        'run_name': 'left_kpts_asymmetric_rnd'
    }
