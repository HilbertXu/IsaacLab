"""
Author: Yucheng Xu
Date: 4 Mar 2025
Description: 
    This script contains configclass for franka_allegro hand tasks
"""


from __future__ import annotations

import math
import torch
import pickle
from collections.abc import Sequence
from dataclasses import MISSING

# isaaclab
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG  # isort:skip

from isaaclab.sensors import (
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    TiledCamera,
    TiledCameraCfg,
    ContactSensor,
    ContactSensorCfg
)

    

@configclass
class HammerAssemblyEnvBaseCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 100.0
    action_scale = 15  # [N]
    action_space = 23 * 2
    observation_space = 0
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    right_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HammerHandle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"devel/assets/hammer_v2/hammer_handle.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.0)),
            scale=(1., 1., 1.),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )

    left_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HammerHead",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"devel/assets/hammer_v2/hammer_head.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.0)),
            scale=(1., 1., 1.),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    )
    
    # robot
    robot_cfg: ArticulationCfg = FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    
    robot_entity_cfg = SceneEntityCfg(
        "robot", 
        joint_names=[
            "right_panda_joint_1", "right_panda_joint_2", "right_panda_joint_3",
            "right_panda_joint_4", "right_panda_joint_5", "right_panda_joint_6", "right_panda_joint_7",
            "right_index_joint_0", "right_index_joint_1", "right_index_joint_2", "right_index_joint_3", 
            "right_middle_joint_0", "right_middle_joint_1", "right_middle_joint_2", "right_middle_joint_3", 
            "right_ring_joint_0", "right_ring_joint_1", "right_ring_joint_2", "right_ring_joint_3", 
            "right_thumb_joint_0", "right_thumb_joint_1", "right_thumb_joint_2", "right_thumb_joint_3",
            "left_panda_joint_1", "left_panda_joint_2", "left_panda_joint_3",
            "left_panda_joint_4", "left_panda_joint_5", "left_panda_joint_6", "left_panda_joint_7",
            "left_index_joint_0", "left_index_joint_1", "left_index_joint_2", "left_index_joint_3", 
            "left_middle_joint_0", "left_middle_joint_1", "left_middle_joint_2", "left_middle_joint_3", 
            "left_ring_joint_0", "left_ring_joint_1", "left_ring_joint_2", "left_ring_joint_3", 
            "left_thumb_joint_0", "left_thumb_joint_1", "left_thumb_joint_2", "left_thumb_joint_3",
        ], 
        body_names=[
            "left_palm",
            "right_palm",
            # "right_index_link_tip",
            # "right_middle_link_tip",
            # "right_ring_link_tip",
            # "right_thumb_link_tip",
            # "left_index_link_tip",
            # "left_middle_link_tip",
            # "left_ring_link_tip",
            # "left_thumb_link_tip",
        ] 
    )

    right_ee_config: FrameTransformerCfg = FrameTransformerCfg(
        # source frame
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path='/Visuals/right_ee',
            markers={
                'sphere': sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            }
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/right_palm",
                name="right_end_effector",
                offset=OffsetCfg(
                    pos=[0.1, 0.02, 0.08],
                ),
            ),
        ],
    )

    right_palm_config: FrameTransformerCfg = FrameTransformerCfg(
        # source frame
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        # visualizer_cfg=VisualizationMarkersCfg(
        #     prim_path='/Visuals/right_palm',
        #     markers={
        #         'sphere': sim_utils.SphereCfg(
        #             radius=0.01,
        #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        #         ),
        #     }
        # ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_palm",
                name="left_end_effector",
                offset=OffsetCfg(
                    pos=[0.1, 0.0, 0.08]
                ),
            ),
        ],
    )

    left_ee_config: FrameTransformerCfg = FrameTransformerCfg(
        # source frame
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=True,
        visualizer_cfg=VisualizationMarkersCfg(
            prim_path='/Visuals/left_ee',
            markers={
                'sphere': sim_utils.SphereCfg(
                    radius=0.01,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            }
        ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_palm",
                name="left_end_effector",
                offset=OffsetCfg(
                    pos=[0.1, -0.01, 0.08],
                ),
            ),
        ],
    )

    left_palm_config: FrameTransformerCfg = FrameTransformerCfg(
        # source frame
        prim_path="/World/envs/env_.*/Robot/base_link",
        debug_vis=False,
        # visualizer_cfg=VisualizationMarkersCfg(
        #     prim_path='/Visuals/left_palm',
        #     markers={
        #         'sphere': sim_utils.SphereCfg(
        #             radius=0.01,
        #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        #         ),
        #     }
        # ),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/left_palm",
                name="left_end_effector",
                offset=OffsetCfg(
                    pos=[0.1, 0.0, 0.08],
                ),
            ),
        ],
    )

    right_goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/right_goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"devel/assets/hammer_v2/hammer_handle.usd",
                scale=(1., 1., 1.),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.0, 0.0)),
            )
        },
    )

    left_goal_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/left_goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"devel/assets/hammer_v2/hammer_head.usd",
                scale=(1., 1., 1.),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.7, 0.0, 0.0)),
            )
        },
    )

    
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, 
        env_spacing=4.0, 
        replicate_physics=True
    )

    num_eval_envs = 100
    
    # replay buffer
    replay_buffer = 'devel/data/realsense/hammer/bimanual_assembly_separate_fps30/trajectory_franka_allegro.pkl'
    right_object_keypoint = 'devel/assets/hammer_v2/hammer_handle.npz'
    left_object_keypoint = 'devel/assets/hammer_v2/hammer_head.npz'

    # reward scales
    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_rotation_noise = 0.1
    reset_dof_pos_noise = 0.05  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = 1.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.001
    reach_goal_bonus = 4000
    object_lifted_bonus = 50
    object_moved_bonus = 5
    fall_penalty = 0
    out_of_reach_dist = 0.24
    vel_obs_scale = 0.2
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 0.75
    force_torque_obs_scale = 10.0

    pos_success_tolerance = 0.0125
    rot_success_tolerance = 0.2
    finger_pos_success_tolerance = 0.15
    hand_dof_speed_scale = 1.0
    action_res_scale = 1.0
    early_complete_threshold = 5
    
    # Observations
    obs_type = 'reduced'
    asymmetric_obs = False
    rnd_obs = False
    control_mode = 'vel'
    use_ref = True
    reward_type = 'async'
    reach_only = False
    use_right_side_reward = True
    use_left_side_reward = True
    enable_gravity = True
    use_object_keypoint = False
    use_coordination_bonus = True
    distance_function = 'mse'
    
    # debug
    debug = False
    plot_delta_qpos = False
    sweep = False
    
    

# use object keypoint
@configclass
class HammerAssemblyEnvCfg_vel_wref_async_kpts(HammerAssemblyEnvBaseCfg):
    control_mode = 'vel'
    use_ref = True
    reward_type = 'async'
    use_object_keypoint = True


# use object keypoint
@configclass
class HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric(HammerAssemblyEnvBaseCfg):
    control_mode = 'vel'
    use_ref = True
    reward_type = 'async'
    use_object_keypoint = True
    
    asymmetric_obs = True
    

@configclass
class HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric_rnd(HammerAssemblyEnvBaseCfg):
    control_mode = 'vel'
    use_ref = True
    reward_type = 'async'
    use_object_keypoint = True
    
    asymmetric_obs = True
    rnd_obs = True
    


@configclass
class HammerAssemblyEnvCfg_vel_wref_async_kpts_right(HammerAssemblyEnvBaseCfg):
    action_space = 23

    control_mode = 'vel'
    use_ref = True
    reward_type = 'async'
    use_object_keypoint = True
    use_coordination_bonus = False

    use_left_side_reward = False
    use_right_side_reward = True


@configclass
class HammerAssemblyEnvCfg_vel_wref_async_kpts_right_asymmetric(HammerAssemblyEnvBaseCfg):
    action_space = 23

    control_mode = 'vel'
    use_ref = True
    reward_type = 'async'
    use_object_keypoint = True
    use_coordination_bonus = False

    use_left_side_reward = False
    use_right_side_reward = True

    asymmetric_obs = True


@configclass
class HammerAssemblyEnvCfg_vel_wref_async_kpts_right_asymmetric_rnd(HammerAssemblyEnvBaseCfg):
    action_space = 23

    control_mode = 'vel'
    use_ref = True
    reward_type = 'async'
    use_object_keypoint = True
    use_coordination_bonus = False

    use_left_side_reward = False
    use_right_side_reward = True

    asymmetric_obs = True
    rnd_obs = True


@configclass
class HammerAssemblyEnvCfg_vel_wref_async_kpts_left(HammerAssemblyEnvBaseCfg):
    action_space = 23
    
    control_mode = 'vel'
    use_ref = True
    reward_type = 'async'
    use_object_keypoint = True
    use_coordination_bonus = False

    use_left_side_reward = True
    use_right_side_reward = False


@configclass
class HammerAssemblyEnvCfg_vel_wref_async_kpts_left_asymmetric(HammerAssemblyEnvBaseCfg):
    action_space = 23
    
    control_mode = 'vel'
    use_ref = True
    reward_type = 'async'
    use_object_keypoint = True
    use_coordination_bonus = False

    use_left_side_reward = True
    use_right_side_reward = False

    asymmetric_obs = True


@configclass
class HammerAssemblyEnvCfg_vel_wref_async_kpts_left_asymmetric_rnd(HammerAssemblyEnvBaseCfg):
    action_space = 23
    
    control_mode = 'vel'
    use_ref = True
    reward_type = 'async'
    use_object_keypoint = True
    use_coordination_bonus = False

    use_left_side_reward = True
    use_right_side_reward = False

    asymmetric_obs = True
    rnd_obs = True
