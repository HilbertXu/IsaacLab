# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlRndCfg

from isaaclab.utils import configclass


@configclass
class HandoverPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    # device = "cpu"
    num_steps_per_env = 24
    max_iterations = 100000
    save_interval = 1000
    empirical_normalization = True
    num_eval_envs = 100
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256],
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
        zero_init=True,
        noise_std_type='log'
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=5.0e-5,
        num_learning_epochs=4,
        num_mini_batches=32,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.008,
        max_grad_norm=1.0,
        value_normalization=True,
        enable_amp=False,
    )
    # set logger to wandb
    logger = 'wandb'
    experiment_name = "base"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-handover-async',
        'run_name': 'base'
    }

    debug = 1
    sweep = False


# use object keypoint
@configclass
class HandoverPPORunnerCfg_vel_wref_async_kpts(HandoverPPORunnerCfg):
    experiment_name = "handover_bimanual_kpts_async"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-handover-async',
        'run_name': 'bimanual_kpts'
    }


# use object keypoint
@configclass
class HandoverPPORunnerCfg_vel_wref_async_kpts_asymmetric(HandoverPPORunnerCfg):
    experiment_name = "handover_bimanual_kpts_asymmetric_sync"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-handover-async',
        'run_name': 'bimanual_kpts_asymmetric'
    }


@configclass
class HandoverPPORunnerCfg_vel_wref_async_kpts_asymmetric_rnd(HandoverPPORunnerCfg):
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

    experiment_name = "handover_bimanual_kpts_asymmetric_rnd_async"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-handover-async',
        'run_name': 'bimanual_kpts_asymmetric_rnd'
    }
    
    

@configclass
class HandoverPPORunnerCfg_vel_wref_async_kpts_right(HandoverPPORunnerCfg):
    experiment_name = "handover_right_kpts_async"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-handover-async',
        'run_name': 'right_kpts'
    }


@configclass
class HandoverPPORunnerCfg_vel_wref_async_kpts_right_asymmetric(HandoverPPORunnerCfg):
    experiment_name = "handover_right_kpts_asymmetric_async"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-handover-async',
        'run_name': 'right_kpts_asymmetric'
    }

@configclass
class HandoverPPORunnerCfg_vel_wref_async_kpts_right_asymmetric_rnd(HandoverPPORunnerCfg):

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
    experiment_name = "handover_right_kpts_asymmetric_rnd_async"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-handover-async',
        'run_name': 'right_kpts_asymmetric_rnd'
    }
    

@configclass
class HandoverPPORunnerCfg_vel_wref_async_kpts_left(HandoverPPORunnerCfg):
    experiment_name = "handover_kpts_left_async"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-handover-async',
        'run_name': 'kpts_left'
    }


@configclass
class HandoverPPORunnerCfg_vel_wref_async_kpts_left_asymmetric(HandoverPPORunnerCfg):
    experiment_name = "handover_left_kpts_asymmetric_async"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-handover-async',
        'run_name': 'left_kpts_asymmetric'
    }


@configclass
class HandoverPPORunnerCfg_vel_wref_async_kpts_left_asymmetric_rnd(HandoverPPORunnerCfg):
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
    experiment_name = "handover_left_kpts_asymmetric_rnd_async"
    wandb_kwargs = {
        'entity': 'hilbertxu',
        'project': 'demobot-handover-async',
        'run_name': 'left_kpts_asymmetric_rnd'
    }
