"""
Author: Yucheng Xu
Date: 4 Mar 2025
Description: 
    Franka allegro RL env
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import pickle
import math
import time
from typing import Optional, Union

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBase, RigidObject, RigidObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import DirectRLEnv
from isaaclab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, spawn_from_usd
from isaaclab.sim.spawners.meshes import spawn_mesh_cuboid, MeshCuboidCfg
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate, matrix_from_quat, quat_apply
from isaacsim.core.simulation_manager import SimulationManager

from collections import deque
import statistics

from isaaclab.sensors import (
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    TiledCamera,
    TiledCameraCfg,
    ContactSensor,
    ContactSensorCfg
)

from .cfg import (
    HammerAssemblyEnvCfg_vel_wref_async_kpts,
    HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric,
    HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric_rnd,
)
import matplotlib.pyplot as plt



def plot(delta_qpos, joint_names, output_f):
    tgt_joint_names = [
        "right_index_joint_3",
        "right_middle_joint_3",
        "right_ring_joint_3", 
        "right_thumb_joint_3",
        "left_index_joint_3",
        "left_middle_joint_3",
        "left_ring_joint_3", 
        "left_thumb_joint_3",
    ]
    for i, jname in enumerate(joint_names):
        if jname not in tgt_joint_names:
            continue
            
        idx = joint_names.index(jname)
        qpos = delta_qpos[:, idx]

        fig, ax = plt.subplots(figsize=(18, 12))
        ax.plot(qpos, label='delta_q')
        ax.set_title(f'{jname}')
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', labelsize=10)

        plt.tight_layout()
        plt.savefig(f'./{jname}.pdf', dpi=300)
        plt.close(fig)  # Close the figure to avoid memory issues


def plot_curve(log_episode):
    from glob import glob
    num_episode = len(glob("./episode_log_*.png"))
    num_figures = len(log_episode.keys())
    nrow = 4
    ncol = num_figures // 4 + 1
    fig, axes = plt.subplots(ncol, nrow, figsize=(16, 8), sharey=True)
    axes = axes.flatten()

    for ax, (name, values) in zip(axes, log_episode.items()):
        x = list(range(len(values)))
        ax.plot(x, values)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.grid(True)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'./episode_log_{num_episode}.png')


class HammerAssemblyEnv(DirectRLEnv):
    # env initialization
    #   |-- super().__init__()
    #      |-- _setup_scene()
    #   |-- reset of __init__()
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()
    cfg: Optional[
        Union[
            HammerAssemblyEnvCfg_vel_wref_async_kpts,
            HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric,
            HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric_rnd,
        ]
    ]
    
    def __init__(
        self, 
        cfg: Optional[
            Union[
                HammerAssemblyEnvCfg_vel_wref_async_kpts,
                HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric,
                HammerAssemblyEnvCfg_vel_wref_async_kpts_asymmetric_rnd,
            ]
        ], 
        render_mode: str | None = None, 
        **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)
        ######################################################################
        # the scene entities is already created in the _setup_scene() called 
        # in the __init__ function of the parent class
        ######################################################################

        # Simulation parameters
        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.debug = self.cfg.debug

        # Resolving the scene entities
        self.cfg.robot_entity_cfg.resolve(self.scene)
        self.robot_joint_ids = self.cfg.robot_entity_cfg.joint_ids
        self.robot_joint_names = self.cfg.robot_entity_cfg.joint_names
        self.right_palm_body_idx = self.cfg.robot_entity_cfg.body_ids[1]
        self.left_palm_body_idx = self.cfg.robot_entity_cfg.body_ids[0]

        # Basic task settings
        self.control_mode = self.cfg.control_mode
        self.with_reach_stage = self.cfg.with_reach_stage
        self.enable_chunk_split = self.cfg.enable_chunk_split
        self.use_ref = self.cfg.use_ref
        self.reward_type = self.cfg.reward_type
        self.reach_only = self.cfg.reach_only
        self.use_right_side_reward = self.cfg.use_right_side_reward
        self.use_left_side_reward = self.cfg.use_left_side_reward
        self.use_object_keypoint = self.cfg.use_object_keypoint
        self.distance_function = self.cfg.distance_function
        self.sweep = self.cfg.sweep

        ################################################
        # one-time indicator for getting the env states
        ################################################
        self.init_step = True
        
        ###########################################
        # Get mecessary indices for robot entities
        ###########################################
        right_finger_tip_names = [
            "right_index_link_tip",
            "right_middle_link_tip",
            "right_ring_link_tip",
            "right_thumb_link_tip",
        ]
        self.right_finger_tip_indices = list()
        for body_name in right_finger_tip_names:
            self.right_finger_tip_indices.append(self.robot.body_names.index(body_name))
        self.right_finger_tip_indices.sort()
        self.num_right_finger_tips = len(self.right_finger_tip_indices)
        right_joint_names = [n for n in self.robot.joint_names if 'right' in n]
        self.right_joint_indices = [self.robot.joint_names.index(n) for n in right_joint_names]

        left_finger_tip_names = [
            "left_index_link_tip",
            "left_middle_link_tip",
            "left_ring_link_tip",
            "left_thumb_link_tip",
        ]
        self.left_finger_tip_indices = list()
        for body_name in left_finger_tip_names:
            self.left_finger_tip_indices.append(self.robot.body_names.index(body_name))
        self.left_finger_tip_indices.sort()
        self.num_left_finger_tips = len(self.left_finger_tip_indices)
        left_joint_names = [n for n in self.robot.joint_names if 'left' in n]
        self.left_joint_indices = [self.robot.joint_names.index(n) for n in left_joint_names]

        if self.cfg.use_left_side_reward and not self.cfg.use_right_side_reward:
            arm_joint_names = [n for n in self.robot.joint_names if 'panda' in n and 'left' in n]
        if self.cfg.use_right_side_reward and not self.cfg.use_left_side_reward:
            arm_joint_names = [n for n in self.robot.joint_names if 'panda' in n and 'right' in n]
        if self.cfg.use_left_side_reward and self.cfg.use_right_side_reward:
            arm_joint_names = [n for n in self.robot.joint_names if 'panda' in n]
        self.arm_joint_indices = [self.robot.joint_names.index(n) for n in arm_joint_names]

        if self.cfg.use_left_side_reward and not self.cfg.use_right_side_reward:
            hand_joint_names = [n for n in self.robot.joint_names if 'panda' not in n and 'left' in n]
        if self.cfg.use_right_side_reward and not self.cfg.use_left_side_reward:
            hand_joint_names = [n for n in self.robot.joint_names if 'panda' not in n and 'right' in n]
        if self.cfg.use_left_side_reward and self.cfg.use_right_side_reward:
            hand_joint_names = [n for n in self.robot.joint_names if 'panda' not in n]
        self.hand_joint_indices = [self.robot.joint_names.index(n) for n in hand_joint_names]
        

        ####################################################################################
        # create auxiliary variables for computing applied action, observations and rewards
        ####################################################################################
        self.robot_dof_lower_limits = self.robot.data.soft_joint_pos_limits[..., 0].to(device=self.device)
        self.robot_dof_upper_limits = self.robot.data.soft_joint_pos_limits[..., 1].to(device=self.device)
        self.robot_dof_targets = torch.zeros((self.num_envs, self.robot.num_joints), device=self.device)
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)

        self.num_robot_dofs = self.robot.num_joints
        self.pred_actions = torch.zeros((self.num_envs, self.cfg.action_space), dtype=torch.float, device=self.device)
        self.prev_targets = self.robot.data.default_joint_pos[:].clone()
        self.cur_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.last_success_prev_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)
        self.last_success_cur_targets = torch.zeros((self.num_envs, self.num_robot_dofs), dtype=torch.float, device=self.device)

        # default goal positions
        self.right_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.right_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.left_goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.left_goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)

        if hasattr(self.cfg, 'right_object_keypoint') and hasattr(self.cfg, 'left_object_keypoint'):
            self.right_object_keypoints = torch.tensor(np.load(self.cfg.right_object_keypoint)['object_keypoints'][:], dtype=torch.float, device=self.device)
            self.left_object_keypoints = torch.tensor(np.load(self.cfg.left_object_keypoint)['object_keypoints'][:], dtype=torch.float, device=self.device)

            self.right_object_grasp_keypoints = torch.tensor(np.load(self.cfg.right_object_keypoint)['grasp_keypoints'][:], dtype=torch.float, device=self.device)
            self.left_object_grasp_keypoints = torch.tensor(np.load(self.cfg.left_object_keypoint)['grasp_keypoints'][:], dtype=torch.float, device=self.device)

        # initialize right object
        right_object_init_pos = self.right_object_init_pose[:3]
        # right_object_init_pos[0] += 0.05
        right_object_init_quat = self.right_object_init_pose[3:7]
        self.right_object_init_pos = torch.tensor(right_object_init_pos+self.right_offset, dtype=torch.float, device=self.sim.device)
        self.right_object_keypoint_init = torch.einsum(
                'nij,nmj->nmi', 
                matrix_from_quat(torch.tensor(right_object_init_quat, dtype=torch.float, device=self.sim.device).unsqueeze(0).repeat(self.num_envs, 1)), 
                self.right_object_keypoints.clone().unsqueeze(0).repeat(self.num_envs, 1, 1)
            ) + (self.right_object_init_pos).unsqueeze(0).repeat(self.num_envs, 1).unsqueeze(1) # [num_envs, N, 3]
        self.right_object_keypoint_cur = torch.zeros_like(self.right_object_keypoint_init, dtype=torch.float, device=self.device)
        self.right_object_keypoint_goal = torch.zeros_like(self.right_object_keypoint_init, dtype=torch.float, device=self.device)

        # initialize left object
        left_object_init_pos = self.left_object_init_pose[:3]
        # left_object_init_pos[0] -= 0.05
        left_object_init_quat = self.left_object_init_pose[3:7]
        self.left_object_init_pos = torch.tensor(left_object_init_pos+self.left_offset, dtype=torch.float, device=self.sim.device)
        self.left_object_keypoint_init = torch.einsum(
                'nij,nmj->nmi', 
                matrix_from_quat(torch.tensor(left_object_init_quat, dtype=torch.float, device=self.sim.device).unsqueeze(0).repeat(self.num_envs, 1)), 
                self.left_object_keypoints.clone().unsqueeze(0).repeat(self.num_envs, 1, 1)
            ) + (self.left_object_init_pos).unsqueeze(0).repeat(self.num_envs, 1).unsqueeze(1) # [num_envs, N, 3]
        self.left_object_keypoint_cur = torch.zeros_like(self.left_object_keypoint_init, dtype=torch.float, device=self.device)
        self.left_object_keypoint_goal = torch.zeros_like(self.left_object_keypoint_init, dtype=torch.float, device=self.device)

        if self.init_side == 'right':
            object_height = (torch.max(self.right_object_keypoint_init[:, :, 2], dim=1).values - torch.min(self.right_object_keypoint_init[:, :, 2], dim=1).values) / 2
            self.right_ref_targets[self.switch_idx:, 2] = self.table_height + object_height[0]
        else:
            object_height = (torch.max(self.left_object_keypoint_init[:, :, 2], dim=1).values - torch.min(self.left_object_keypoint_init[:, :, 2], dim=1).values) / 2
            self.left_ref_targets[self.switch_idx-1:, 2] = self.table_height + object_height[0]


        ###################
        # state indicators
        ###################
        """ Indicators for sparse bonus
        
        The basic idea is to make sure each bonus will only be gained once,
        so that we need indicators to trace whether each bonus has been gained in an episode
        """
        self.right_object_not_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.left_object_not_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_not_sync_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.right_object_not_lifted = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.left_object_not_lifted = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_not_sync_lifted = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.right_goal_not_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.left_goal_not_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_not_sync_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.object_not_stable_placed = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)


        """ Indicators for current state
        
        These parameters are updated in _compute_intermediate_values()
        """
        # default value
        # 0 - left
        # 1 - right
        self.last_active_side = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.cur_active_side = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.last_chunk_step_idx = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)

        self.right_object_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.left_object_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_sync_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.right_object_lifted = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.left_object_lifted = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.object_sync_lifted = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        self.right_goal_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.left_goal_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.goal_sync_reached = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)


        
        """task state tracker
        
        reward_stage_indicator: tracker of the reward stage: [0: reaching, 1: reached, 2: lifted, 3: goaled]
        successes: tracker of task consecutive successes
        """
        self.reward_stage_indicator = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)


        """reset condition buffers
        
        reset_out_of_reach: failed to reach the current goal
        terminate: treminated because of fall
        max_success_reached: completed all goals
        reset_goal_buf: completed the current goal
        """
        self.truncated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.terminate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.max_success_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)


        ###################
        # counters
        ###################
        self.lift_episode_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.right_goal_reach_episode_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.left_goal_reach_episode_counter = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.step_to_lift_right = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.step_to_lift_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.step_to_goal_right = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.step_to_goal_left = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        
        #####################
        # mask for eval envs
        #####################
        self.num_eval_envs = self.cfg.num_eval_envs if self.num_envs > self.cfg.num_eval_envs else self.num_envs
        self.eval_env_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.eval_env_mask[:self.num_eval_envs] = torch.tensor(True)

        
        #######################
        # threshold parameters
        #######################
        self.right_object_lift_thresh = 0.003 * torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.right_object_pos_tolerance = 0.05 * torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.right_object_rot_tolerance = 99.0 #@TODO start with pos tolerance only
        self.right_finger_dist_tolerance = 0.15 * torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        
        self.left_object_lift_thresh = 0.003 * torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.left_object_pos_tolerance = 0.05 * torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.left_object_rot_tolerance = 99.0 #@TODO start with pos tolerance only
        self.left_finger_dist_tolerance = 0.15 * torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        self.pos_tolerance_curriculum_step = 100 if self.cfg.use_left_side_reward and self.cfg.use_right_side_reward else 50
        self.pos_tolerance_reduce = 0.0025 if self.cfg.use_left_side_reward and self.cfg.use_right_side_reward else 0.005
        self.reset_to_last_success_ratio = getattr(self.cfg, "reset_to_last_success_ratio", 0.0)
        print("reset_to_last_success_ratio: ", self.reset_to_last_success_ratio)

        self.reset_dof_pos_noise = self.cfg.reset_dof_pos_noise * torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_dof_vel_noise = self.cfg.reset_dof_vel_noise * torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_position_noise = self.cfg.reset_position_noise * torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_rotation_noise = self.cfg.reset_rotation_noise * torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))


        #####################
        # setting up loggers
        #####################
        self.episode_log = {}
        self.eval_consecutive_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.csbuffer = deque(maxlen=self.num_eval_envs*2)
        self.extras["log"] = {
            "common_step_counter": None,
            "consecutive_successes": None,
            "lift_thresh": None,
            "lift_episode_counter": None,
            "reward_stage_indicator": None,
            "switch_bonus": None,
            "stable_placement_bonus": None,
            "completion_bonus": None,
        }
        
        if self.use_left_side_reward:
            self.extras["log"].update(
                {   
                    "step_to_lift_left": None,
                    "step_to_goal_left": None,
                    "left_ee2o_dist": None,
                    "left_h2o_dist": None,
                    "left_goal_dist": None,
                    "left_ee2o_dist_reward": None,
                    "left_h2o_dist_reward": None,
                    "left_goal_dist_reward": None,
                    "left_object_reached_bonus": None,
                    "left_lift_bonus": None,
                    "left_goal_reached_bonus": None,
                    "left_goal_reached_speed_penalty": None,
                    "left_goal_reach_thresh": None,
                    "left_goal_reach_episode_counter": None,
                }
            )
        
        if self.use_right_side_reward:
            self.extras["log"].update(
                {
                    "step_to_lift_right": None,
                    "step_to_goal_right": None,
                    "right_ee2o_dist": None,
                    "right_h2o_dist": None,
                    "right_goal_dist": None,
                    "right_ee2o_dist_reward": None,
                    "right_h2o_dist_reward": None,
                    "right_goal_dist_reward": None,
                    "right_object_reached_bonus": None,
                    "right_lift_bonus": None,
                    "right_goal_reached_bonus": None,
                    "right_goal_reached_speed_penalty": None,
                    "right_goal_reach_thresh": None,
                    "right_goal_reach_episode_counter": None,
                }
            )
        
        
        #########################################
        # setting up debug visualizers if needed
        #########################################
        if self.debug:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
            self.palm_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
            self.object_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/object_current"))
            self.finger_markers = [VisualizationMarkers(frame_marker_cfg.replace(prim_path=f"/Visuals/finger_{i}")) for i in range(len(self.right_finger_tip_indices))]

            self.kpts_init_markers = [VisualizationMarkers(frame_marker_cfg.replace(prim_path=f"/Visuals/kpts_init_{i}")) for i in range(8)]
            self.kpts_markers = [VisualizationMarkers(frame_marker_cfg.replace(prim_path=f"/Visuals/kpts_{i}")) for i in range(8)]
            

    def _setup_action_chunks(self, action_buffer, right_offset, left_offset):
        def _check_goal_pose_change(pose1, pose2):
            pos1 = pose1[:3]
            rot1 = pose1[3:]

            pos2 = pose2[:3]
            rot2 = pose2[3:]
            pos_moved = torch.norm(pos1-pos2, p=2, dim=-1) > 0.03
            rot_moved = rotation_distance(rot1.unsqueeze(0), rot2.unsqueeze(0))

            return (pos_moved > 0.05)

        chunk_ids = list(action_buffer.keys())
        num_chunks = int(len(chunk_ids))
        ref_actions = torch.zeros((num_chunks, 500, 46), dtype=torch.float, device=self.device) # [num_chunks, num_steps, action_space]
        right_ref_targets = torch.zeros((num_chunks, 7), dtype=torch.float, device=self.device) # [num_chunks, 7]
        left_ref_targets = torch.zeros((num_chunks, 7), dtype=torch.float, device=self.device) # [num_chunks, 7]
        ref_chunk_step_idx = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device) # [num_envs, {chunk_id, step_id in chunk}]
        ref_chunk_max_steps = torch.zeros(num_chunks, dtype=torch.long, device=self.device)
        
        right_side_init_ids = -1
        left_side_init_ids = -1
        
        print(f"{len(chunk_ids)} chunks in total")
        for i in range(len(chunk_ids)):
            ref_actions[i, :, :] = torch.tensor(action_buffer[chunk_ids[i]]['qpos'][-1, :], dtype=torch.float, device=self.device)
            ref_actions[i, :action_buffer[chunk_ids[i]]['qpos'].shape[0], :] = torch.tensor(action_buffer[chunk_ids[i]]['qpos'], dtype=torch.float, device=self.device)
            right_object_goal_pose = action_buffer[chunk_ids[i]]['goal_object_pose.right']
            right_object_goal_pose[:3] += right_offset
            left_object_goal_pose = action_buffer[chunk_ids[i]]['goal_object_pose.left']
            left_object_goal_pose[:3] += left_offset
            right_ref_targets[i, :] = torch.tensor(right_object_goal_pose, dtype=torch.float, device=self.device) # [x, y, z, w, x, y, z]
            left_ref_targets[i, :] = torch.tensor(left_object_goal_pose, dtype=torch.float, device=self.device) # [x, y, z, w, x, y, z]
            ref_chunk_max_steps[i] = action_buffer[chunk_ids[i]]['qpos'].shape[0]

            if i == 1: # find the side to start
                right_moved = _check_goal_pose_change(right_ref_targets[i, :], right_ref_targets[i-1, :])
                left_moved = _check_goal_pose_change(left_ref_targets[i, :], left_ref_targets[i-1, :])

                if right_moved:
                    print("start with right side")
                    self.cur_active_side = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
                    right_side_init_ids = 0
                    self.init_side = 'right'
                    self.right_init_id = 0

                if left_moved:
                    print("start with left side")
                    self.cur_active_side = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    left_side_init_ids = 0
                    self.init_side = 'left'
                    self.left_init_id = 0

            elif i > 1: # find where is the switch point
                right_moved = _check_goal_pose_change(right_ref_targets[i, :], right_ref_targets[i-1, :])
                left_moved = _check_goal_pose_change(left_ref_targets[i, :], left_ref_targets[i-1, :])

                if right_side_init_ids == -1 and right_moved:
                    self.switch_idx = i
                    right_side_init_ids = i
                    print(f"switch to right side at {self.switch_idx}-th goal")
                    self.right_init_id = i
                
                if left_side_init_ids == -1 and left_moved:
                    self.switch_idx = i
                    left_side_init_ids = i
                    print(f"switch to left side at {self.switch_idx}-th goal")
                    self.left_init_id = i

        max_consecutive_success = num_chunks

        return num_chunks, ref_actions, right_ref_targets, left_ref_targets, ref_chunk_step_idx, ref_chunk_max_steps, max_consecutive_success


    def _setup_scene(self):
        """
            Setup scene entities here
        """
        demo = pickle.load(open(self.cfg.replay_buffer, 'rb'))
        self.action_buffer = demo['action_chunks']

        self.robot_init_qpos = demo['init_robot_qpos']
        self.robot_init_qpos['left_thumb_joint_0'] = 0.4
        self.robot_init_qpos['right_thumb_joint_0'] = 0.4
        
        self.right_object_init_pose = demo['init_object_pose.right']
        self.right_offset = demo['object_offset.right']
        self.left_object_init_pose = demo['init_object_pose.left']
        self.left_offset = demo['object_offset.left']

        self.num_chunks, self.ref_actions, self.right_ref_targets, self.left_ref_targets, \
            self.ref_chunk_step_idx, self.ref_chunk_max_steps, \
                self.max_consecutive_success= self._setup_action_chunks(self.action_buffer, self.right_offset, self.left_offset)
        self.cur_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # initialize robot
        self.cfg.robot_cfg.init_state.joint_pos = self.robot_init_qpos
        self.robot = Articulation(self.cfg.robot_cfg)
        
        # initialize objects
        self.cfg.right_object_cfg.init_state = RigidObjectCfg.InitialStateCfg(
            pos=self.right_object_init_pose[:3]+self.right_offset,
            rot=self.right_object_init_pose[3:7]
        )
        self.right_object = RigidObject(self.cfg.right_object_cfg)

        
        self.cfg.left_object_cfg.init_state = RigidObjectCfg.InitialStateCfg(
            pos=self.left_object_init_pose[:3]+self.left_offset,
            rot=self.left_object_init_pose[3:7]
        )
        self.left_object = RigidObject(self.cfg.left_object_cfg)

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["right_object"] = self.right_object
        self.scene.rigid_objects["left_object"] = self.left_object

        # setup frames from FrameTransform
        self.right_ee_frame = FrameTransformer(self.cfg.right_ee_config)
        self.left_ee_frame = FrameTransformer(self.cfg.left_ee_config)
        self.right_palm_frame = FrameTransformer(self.cfg.right_palm_config)
        self.left_palm_frame = FrameTransformer(self.cfg.left_palm_config)

        self.scene.sensors["right_ee_frame"] = self.right_ee_frame
        self.scene.sensors["left_ee_frame"] = self.left_ee_frame
        self.scene.sensors["right_palm_frame"] = self.right_palm_frame
        self.scene.sensors["left_palm_frame"] = self.left_palm_frame


        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # add table
        self.table_height = 0.11
        table_cfg = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd", scale=(1.5, 1.5, 1.5)
        )
        table_cfg.func("/World/envs/env_.*/Table", table_cfg, translation=(0.6, 0, 0), orientation=(0.7071068, 0, 0, 0.7071068))

        operate_space_cfg = sim_utils.CuboidCfg(
                    size=(0.75, 1, self.table_height),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.75), metallic=0.2),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                )
        operate_space_cfg.func(
            "/World/envs/env_.*/ObjectPlate", operate_space_cfg, translation=(0.6, 0.0, 0.055)
        )

        fake_operate_space_cfg = sim_utils.CuboidCfg(
                    size=(0.75, 1, self.table_height),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.75, 0.75, 0.75), metallic=0.2),
                    mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                )
        fake_operate_space_cfg.func(
            "/World/envs/env_.*/FakeObjectPlate", operate_space_cfg, translation=(0.6, 0.0, 0.055)
        )

        self.right_goal_marker = VisualizationMarkers(self.cfg.right_goal_marker_cfg)
        self.left_goal_marker = VisualizationMarkers(self.cfg.left_goal_marker_cfg)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        self.scene.filter_entities_collisions(
            entities_prim_path=[
                "/World/envs/env_0/ObjectPlate",
                "/World/envs/env_0/Table",
                "/World/envs/env_0/Robot"
            ]
        )

    
    def set_reset_ratio(self, ratio):
        print(f"reset_to_last_success_ratio from {self.reset_to_last_success_ratio} to {ratio}: ")
        self.reset_to_last_success_ratio = ratio
    

    def _retrieve_ref_actions(self, chunk_step_idx, ref_actions, normalize=False):
        chunk_ids = chunk_step_idx[:, 0]
        step_ids = chunk_step_idx[:, 1]

        step_ids = torch.clamp(step_ids, max=self.ref_chunk_max_steps[chunk_ids]-1)

        ref_action = ref_actions[chunk_ids, step_ids, :].clone()

        if normalize:
            ref_action = unscale(
                ref_action,
                self.robot_dof_lower_limits,
                self.robot_dof_upper_limits,
            ) # normalized to [-1, 1]

            return ref_action
        else:
            return ref_action

    
    def _compute_action(self, pred_actions, ref_actions=None):
        def _compute_one_side(qpos, ref_actions, pred_actions, indices, control_mode):
            out = qpos.clone()
            if ref_actions is not None:
                if control_mode == 'vel':
                    out[:, indices] = ref_actions[:, indices] + \
                                        self.robot_dof_speed_scales[:, indices] * self.dt * pred_actions[:, indices] * self.cfg.action_scale
                elif control_mode == 'pos':
                    out[:, indices] = ref_actions[:, indices] + \
                        scale(pred_actions[:, indices], self.robot_dof_lower_limits[:, indices], self.robot_dof_upper_limits[:, indices])
            else:
                if control_mode == 'vel':
                    out[:, indices] = self.prev_targets[:, indices] + \
                                        self.robot_dof_speed_scales[:, indices] * self.dt * pred_actions[:, indices] * self.cfg.action_scale
                elif control_mode == 'pos':
                    out[:, indices] = scale(pred_actions[:, indices], self.robot_dof_lower_limits[:, indices], self.robot_dof_upper_limits[:, indices])
                
            return out
        
        out = self.robot.data.default_joint_pos[:].clone()
        if self.use_ref and ref_actions is not None:
            right_side_qpos = _compute_one_side(out, ref_actions=ref_actions, pred_actions=pred_actions, indices=self.right_joint_indices, control_mode=self.control_mode)
            left_side_qpos = _compute_one_side(out, ref_actions=ref_actions, pred_actions=pred_actions, indices=self.left_joint_indices, control_mode=self.control_mode)
            
            out = torch.where(
                (self.cur_active_side).unsqueeze(-1).repeat(1, self.num_robot_dofs) == 0,
                left_side_qpos,
                right_side_qpos
            )
        else:
            right_side_qpos = _compute_one_side(out, ref_actions=None, pred_actions=pred_actions, indices=self.right_joint_indices, control_mode=self.control_mode)
            left_side_qpos = _compute_one_side(out, ref_actions=None, pred_actions=pred_actions, indices=self.left_joint_indices, control_mode=self.control_mode)
            
            out = torch.where(
                (self.cur_active_side).unsqueeze(-1).repeat(1, self.num_robot_dofs) == 0,
                left_side_qpos,
                right_side_qpos
            )
        
        if self.with_reach_stage:
            # with ref trajectory, we keep the hand open until the object is reached
            out_fix_hand = out.clone()
            out_fix_hand[:, self.hand_joint_indices] = self.robot.data.default_joint_pos[:, self.hand_joint_indices]

            object_reached = torch.where(
                self.cur_active_side == 0,
                ~self.left_object_not_reached,
                ~self.right_object_not_reached
            )
            out = torch.where(
                (object_reached).unsqueeze(-1).repeat(1, self.num_robot_dofs),
                out,
                out_fix_hand
            )
        
            
        return out

    
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        '''
        1. start from the first chunk, set the target object pose
        2. Execute the ref_qpos + action in the first chunk
        3. calculate rewards
        4. if the object in i-th env is lifted to the target pose, then reset 'step_idx=0', 'ref_idx+=1', set the target object pose of next chunk to the i-th env
        5. if the object in i-th env failed to reach the target pose of this chunk, marked as failure and reset i-th env
        
        NOTE: be careful about the object pose in reference trajectory, it should be the object pose in each env's world frame
        in the current reference trajectory, an offset is applied to the object pose, check the replay.py script
        '''
        
        self.pred_actions[:] = actions.clone().clamp(-1.0, 1.0)
        
        if self.use_ref:
            # # Read reference trajectory
            ref_action = self._retrieve_ref_actions(
                self.ref_chunk_step_idx, 
                self.ref_actions, 
                normalize=False
            )
        else:
            ref_action = None # no ref, start from init qpos.
        
        self.robot_dof_targets[:] = self._compute_action(
            ref_actions=ref_action, 
            pred_actions=self.pred_actions, 
        )

        self.cur_targets = self.robot_dof_targets[:].clone()
        
        self.cur_targets = (
            self.cfg.act_moving_average * self.cur_targets
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets
        )
        self.cur_targets[:] = torch.clamp(self.cur_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        
        self.prev_targets[:] = self.cur_targets[:].clone()
        
    
    def _pre_physics_step_pos(self, actions: torch.Tensor) -> None:
        '''
        1. start from the first chunk, set the target object pose
        2. Execute the ref_qpos + action in the first chunk
        3. calculate rewards
        4. if the object in i-th env is lifted to the target pose, then reset 'step_idx=0', 'ref_idx+=1', set the target object pose of next chunk to the i-th env
        5. if the object in i-th env failed to reach the target pose of this chunk, marked as failure and reset i-th env
        
        NOTE: be careful about the object pose in reference trajectory, it should be the object pose in each env's world frame
        in the current reference trajectory, an offset is applied to the object pose, check the replay.py script
        '''
        
        self.pred_actions[:] = torch.nn.functional.tanh(actions).clone() # normalized to [-1, 1]
        
        pred_actions = scale(
            self.pred_actions,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits
        )

        self.cur_targets = self._compute_action(
            pred_actions=pred_actions,
            ref_actions=None
        )
        
        self.cur_targets = (
            self.cfg.act_moving_average * self.cur_targets
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets
        )
        self.cur_targets[:] = torch.clamp(self.cur_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        self.prev_targets[:] = self.cur_targets[:].clone()


    def _pre_physics_step_pos_v2(self, actions: torch.Tensor) -> None:
        '''
        1. start from the first chunk, set the target object pose
        2. Execute the ref_qpos + action in the first chunk
        3. calculate rewards
        4. if the object in i-th env is lifted to the target pose, then reset 'step_idx=0', 'ref_idx+=1', set the target object pose of next chunk to the i-th env
        5. if the object in i-th env failed to reach the target pose of this chunk, marked as failure and reset i-th env
        
        NOTE: be careful about the object pose in reference trajectory, it should be the object pose in each env's world frame
        in the current reference trajectory, an offset is applied to the object pose, check the replay.py script
        '''
        
        self.pred_actions[:] = torch.nn.functional.tanh(actions).clone() # normalized to [-1, 1]

        ref_action = self._retrieve_ref_actions(
            self.ref_chunk_step_idx, 
            self.ref_actions, 
            normalize=True
        )

        cur_targets = scale(
            torch.clamp(ref_action+self.pred_actions, min=-1, max=1),
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits
        )

        self.cur_targets = self._compute_action(
            pred_actions=cur_targets,
            ref_actions=None
        )
        
        self.cur_targets = (
            self.cfg.act_moving_average * self.cur_targets
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets
        )
        self.cur_targets[:] = torch.clamp(self.cur_targets, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        
        self.prev_targets[:] = self.cur_targets[:].clone()



    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(
            self.cur_targets, joint_ids=self.robot_joint_ids
        )
    

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model.apply(action)

        # process actions
        self._pre_physics_step(action)
        # if self.control_mode == 'vel':
        #     self._pre_physics_step_vel(action)
        # if self.control_mode == 'pos':
        #     self._pre_physics_step_pos(action)
        # if self.control_mode == 'pos_v2':
        #     self._pre_physics_step_pos_v2(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self._apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        self.ref_chunk_step_idx[:, 1] += 1 # in-chunk step counter


        cur_chunk = self.ref_chunk_step_idx[:, 0].clone()
        self.ref_chunk_step_idx[:, 0] = torch.where(
            self.ref_chunk_step_idx[:, 1] >= self.ref_chunk_max_steps[cur_chunk],
            self.ref_chunk_step_idx[:, 0] + 1,
            self.ref_chunk_step_idx[:, 0]
        )

        self.ref_chunk_step_idx[:, 1] = torch.where(
            self.ref_chunk_step_idx[:, 1] >= self.ref_chunk_max_steps[cur_chunk],
            0,
            self.ref_chunk_step_idx[:, 1]
        )

        self.cur_active_side = torch.where(
            self.ref_chunk_step_idx[:, 0] >= self.switch_idx,
            ~self.cur_active_side,
            self.cur_active_side
        )

        self._compute_intermediate_values()

        self.reward_buf = self._get_rewards()

        self.terminate[:], self.max_success_reached[:], self.truncated[:] = self._get_dones()
        
        self.reset_buf = self.terminate | self.max_success_reached | self.truncated

        # #NOTE test only
        # self.reset_buf = self.successes >= self.max_consecutive_success
        # self.terminate[:] = self.successes >= self.max_consecutive_success
        # self.max_success_reached[:] = self.successes >= self.max_consecutive_success
        # self.truncated[:] = self.successes >= self.max_consecutive_success

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
        if len(reset_env_ids) > 0:
            self._reset_envs(
                self.terminate | self.max_success_reached, 
                self.truncated
            )

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model.apply(self.obs_buf["policy"])

        if self.init_step:
            # Create a buffer for saving the last success state of all envs
            # @NOTE a little hacky
            # This only executed for the very first step
            # the entities need to be initialized (env step once) so that we can get data from them
            self.last_success_state = self.scene.get_state(is_relative=False)
            self.init_step = False

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.terminate | self.max_success_reached, self.truncated, self.extras


    def _compute_intermediate_values(self):
        def _compute_object_values(obj):
            object_pos = obj.data.root_pos_w.clone() - self.scene.env_origins
            object_rot = obj.data.root_quat_w.clone()
            object_velocities = obj.data.root_vel_w.clone()
            object_linvel = obj.data.root_lin_vel_w.clone()
            object_angvel = obj.data.root_ang_vel_w.clone()

            return object_pos, object_rot, object_velocities, object_linvel, object_angvel


        def _compute_sync_state(right_state, left_state, cur_active_side):
            state = torch.where(
                cur_active_side==0,
                left_state,
                right_state
            )

            return state

        # stage encoding
        self.successes_onehot = F.one_hot(self.successes.clone().long(), num_classes=self.max_consecutive_success+1)
        self.cur_active_side_onehot = F.one_hot(self.cur_active_side.clone().long(), num_classes=2)
        
        # data for robot
        self.robot_dof_pos = self.robot.data.joint_pos.clone()
        self.robot_dof_vel = self.robot.data.joint_vel.clone()

        self.right_fingertip_pos = self.robot.data.body_pos_w[:, self.right_finger_tip_indices].clone()
        self.right_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_right_finger_tips)).reshape(
            self.num_envs, self.num_right_finger_tips, 3
        )
        self.right_fingertip_quat = self.robot.data.body_quat_w[:, self.right_finger_tip_indices].clone()
        self.right_fingertip_velocities = self.robot.data.body_vel_w[:, self.right_finger_tip_indices].clone()
        
        self.left_fingertip_pos = self.robot.data.body_pos_w[:, self.left_finger_tip_indices].clone()
        self.left_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_left_finger_tips)).reshape(
            self.num_envs, self.num_left_finger_tips, 3
        )
        self.left_fingertip_quat = self.robot.data.body_quat_w[:, self.left_finger_tip_indices].clone()
        self.left_fingertip_velocities = self.robot.data.body_vel_w[:, self.left_finger_tip_indices].clone()
        
        self.right_palm_pos = self.right_palm_frame.data.target_pos_source[..., 0, :]
        self.left_palm_pos = self.left_palm_frame.data.target_pos_source[..., 0, :]

        self.right_ee_pos = self.right_ee_frame.data.target_pos_source[..., 0, :]
        self.right_ee_quat = self.right_ee_frame.data.target_quat_source[..., 0, :]

        self.left_ee_pos = self.left_ee_frame.data.target_pos_source[..., 0, :]
        self.left_ee_quat = self.left_ee_frame.data.target_quat_source[..., 0, :]

        self.ref_action_norm = self._retrieve_ref_actions(
            self.ref_chunk_step_idx, 
            self.ref_actions, 
            normalize=True
        )

        # data for object
        self.right_object_pos, self.right_object_rot, self.right_object_vel, \
            self.right_object_linvel, self.right_object_angvel = _compute_object_values(self.right_object)
        
        self.left_object_pos, self.left_object_rot, self.left_object_vel, \
            self.left_object_linvel, self.left_object_angvel = _compute_object_values(self.left_object)
        
        if self.use_object_keypoint:
            self.right_object_keypoint_cur = torch.einsum(
                'nij,nmj->nmi', 
                matrix_from_quat(self.right_object_rot), 
                self.right_object_keypoints.clone().unsqueeze(0).repeat(self.num_envs, 1, 1)
            ) + self.right_object_pos.unsqueeze(1) # [num_envs, N, 3]

            self.left_object_keypoint_cur = torch.einsum(
                'nij,nmj->nmi', 
                matrix_from_quat(self.left_object_rot), 
                self.left_object_keypoints.clone().unsqueeze(0).repeat(self.num_envs, 1, 1)
            ) + self.left_object_pos.unsqueeze(1) # [num_envs, N, 3]


            self.right_object_keypoint_goal = torch.einsum(
                'nij,nmj->nmi', 
                matrix_from_quat(self.right_goal_rot), 
                self.right_object_keypoints.clone().unsqueeze(0).repeat(self.num_envs, 1, 1)
            ) + self.right_goal_pos.unsqueeze(1) # [num_envs, N, 3]

            self.left_object_keypoint_goal = torch.einsum(
                'nij,nmj->nmi', 
                matrix_from_quat(self.left_goal_rot), 
                self.left_object_keypoints.clone().unsqueeze(0).repeat(self.num_envs, 1, 1)
            ) + self.left_goal_pos.unsqueeze(1) # [num_envs, N, 3]
            
            self.right_object_grasp_keypoint_cur = torch.einsum(
                'nij,nmj->nmi', 
                matrix_from_quat(self.right_object_rot), 
                self.right_object_grasp_keypoints.clone().unsqueeze(0).repeat(self.num_envs, 1, 1)
            ) + self.right_object_pos.unsqueeze(1) # [num_envs, N, 3]

            self.left_object_grasp_keypoint_cur = torch.einsum(
                'nij,nmj->nmi', 
                matrix_from_quat(self.left_object_rot), 
                self.left_object_grasp_keypoints.clone().unsqueeze(0).repeat(self.num_envs, 1, 1)
            ) + self.left_object_pos.unsqueeze(1) # [num_envs, N, 3]


        # zero out the reward_stage indicator for each step
        self.reward_stage_indicator = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # compute distance values for reward calculation

        # For reaching stage
        self.right_ee2o_dist = torch.clamp(torch.norm(self.right_ee_pos - self.right_object_pos, p=2, dim=-1), min=-9.0, max=9.0)
        self.left_ee2o_dist = torch.clamp(torch.norm(self.left_ee_pos - self.left_object_pos, p=2, dim=-1), min=-9.0, max=9.0)
        self.right_h2o_dist = torch.clamp((torch.cdist(self.right_fingertip_pos, self.right_object_grasp_keypoint_cur).min(dim=-1).values).max(dim=-1).values, min=-9.0, max=9.0)
        self.left_h2o_dist = torch.clamp((torch.cdist(self.left_fingertip_pos, self.left_object_grasp_keypoint_cur).min(dim=-1).values).max(dim=-1).values, min=-9.0, max=9.0)

        self.right_h2o_min_dist = torch.clamp((torch.cdist(self.right_fingertip_pos, self.right_object_grasp_keypoint_cur).min(dim=-1).values).min(dim=-1).values, min=-9.0, max=9.0)
        self.left_h2o_min_dist = torch.clamp((torch.cdist(self.left_fingertip_pos, self.left_object_grasp_keypoint_cur).min(dim=-1).values).min(dim=-1).values, min=-9.0, max=9.0)

        # For lifting stage, we only care about the z-axis translation
        # 1. successes == 0, objects are not lifted, use keypoints to make sure that the agent won't try to 'flip' the object
        # 2. successes > 0, object already lifted to the first goal, use object mass center height for relaxation
        
        self.right_lift_height = torch.where(
            self.successes == self.right_init_id,
            torch.clamp((self.right_object_keypoint_cur[:, :, 2] - self.right_object_keypoint_init[:, :, 2]).min(dim=-1).values, min=-9.0, max=9.0),
            torch.clamp(self.right_object_pos[:, 2] - self.right_object_init_pos[2], min=-9.0, max=9.0),
        )
        self.left_lift_height = torch.where(
            self.successes == self.left_init_id,
            torch.clamp((self.left_object_keypoint_cur[:, :, 2] - self.left_object_keypoint_init[:, :, 2]).min(dim=-1).values, min=-9.0, max=9.0),
            torch.clamp(self.left_object_pos[:, 2] - self.left_object_init_pos[2], min=-9.0, max=9.0),
        )

        # For goaling stage
        self.right_goal_dist = torch.clamp(torch.norm(self.right_object_keypoint_cur - self.right_object_keypoint_goal, p=2, dim=-1).max(dim=-1).values, min=-9.0, max=9.0)
        self.left_goal_dist = torch.clamp(torch.norm(self.left_object_keypoint_cur - self.left_object_keypoint_goal, p=2, dim=-1).max(dim=-1).values, min=-9.0, max=9.0)

        # compute reaching state
        # 1. successes == 0, objects are not reached, use a strict condition for accurate reaching
        # 2. successes > 0, object already lifted to the first goal, use a relaxed condition to allow more in-hand adjustment
        self.right_object_reached = torch.where(
            self.successes == self.right_init_id,
            (self.right_ee2o_dist < 0.04) & (self.right_ee_pos[:, 2] - self.right_object_pos[:, 2] <= 0.03),
            (self.right_ee2o_dist < 0.07)
        )
        self.left_object_reached = torch.where(
            self.successes == self.left_init_id,
            (self.left_ee2o_dist < 0.04) & (self.left_ee_pos[:, 2] - self.left_object_pos[:, 2] <= 0.03),
            (self.left_ee2o_dist < 0.07)
        )
        self.object_sync_reached = _compute_sync_state(
            right_state=self.right_object_reached, 
            left_state=self.left_object_reached, 
            cur_active_side=self.cur_active_side
        )

        # update reward_stage_indicator for reaching stage
        self.reward_stage_indicator = torch.where(
            self.object_sync_reached,
            1.0,
            self.reward_stage_indicator
        )
    
        # compute lifting state
        self.right_object_lifted = (self.right_lift_height > self.right_object_lift_thresh) & (self.right_h2o_dist < self.right_finger_dist_tolerance)
        self.left_object_lifted = (self.left_lift_height > self.left_object_lift_thresh) & (self.left_h2o_dist < self.left_finger_dist_tolerance)
        self.object_sync_lifted = _compute_sync_state(
            right_state=self.right_object_lifted,
            left_state=self.left_object_lifted,
            cur_active_side=self.cur_active_side
        )
        # update reward_stage_indicator for lifting stage
        self.reward_stage_indicator = torch.where(
            (self.object_sync_reached & self.object_sync_lifted) | \
                ((self.successes > 0) & (self.successes != self.switch_idx)),
            2.0,
            self.reward_stage_indicator
        )

        # compute goaling stage
        self.right_goal_reached = (self.right_goal_dist < self.right_object_pos_tolerance) & (self.right_h2o_min_dist < 0.075)
        self.left_goal_reached = (self.left_goal_dist < self.left_object_pos_tolerance) & (self.left_h2o_min_dist < 0.075)
        
        self.right_goal_reached_speed_penalty = torch.where(
            self.right_goal_dist < (self.right_object_pos_tolerance + 0.01),
            2 * torch.norm(self.left_object_vel, dim=-1, p=2),
            0.0
        )
        self.left_goal_reached_speed_penalty = torch.where(
            self.left_goal_dist < (self.left_object_pos_tolerance + 0.01),
            2 * torch.norm(self.left_object_vel, dim=-1, p=2),
            0.0
        )

        self.goal_sync_reached = _compute_sync_state(
            right_state=self.right_goal_reached,
            left_state=self.left_goal_reached,
            cur_active_side=self.cur_active_side
        )
        if self.init_side == 'left':
            self.object_stable_placed = torch.where(
                self.successes > self.switch_idx,
                self.left_goal_reached,
                torch.tensor(False)
            )
        else:
            self.object_stable_placed = torch.where(
                self.successes > self.switch_idx,
                self.right_goal_reached,
                torch.tensor(False)
            )

        # update reward_stage_indicator for goaling stage
        self.reward_stage_indicator = torch.where(
            self.goal_sync_reached,
            3.0,
            self.reward_stage_indicator
        )

        self.object_dropped = torch.where(
            self.cur_active_side == 0,
            (self.left_h2o_min_dist > 0.01) & ((self.successes > 0) & (self.successes != self.switch_idx)),
            (self.right_h2o_min_dist > 0.01) & ((self.successes > 0) & (self.successes != self.switch_idx))
        )

        self.object_dropped_penalty = torch.where(
            self.object_dropped,
            200, 
            0.0
        )

        # compute penalty items
        self.action_penalty = torch.where(
            self.cur_active_side == 0, # if left side is activated
            1.0 * torch.norm(self.pred_actions[:, self.right_joint_indices], p=2, dim=-1) + 0.0001* torch.norm(self.pred_actions[:, self.left_joint_indices], p=2, dim=-1),
            1.0 * torch.norm(self.pred_actions[:, self.left_joint_indices], p=2, dim=-1) + 0.0001* torch.norm(self.pred_actions[:, self.right_joint_indices], p=2, dim=-1)
        )

        if self.init_side == 'left':
            self.moving_penalty = torch.where(
                self.successes >= self.switch_idx,
                5 * (self.left_goal_dist + torch.norm(self.left_object_vel, dim=-1, p=2)),
                0.0
            )
        else:
            self.moving_penalty = torch.where(
                self.successes >= self.switch_idx,
                5 * (self.right_goal_dist + torch.norm(self.right_object_vel, dim=-1, p=2)),
                0.0
            )


        # print("*"*100)
        # print(self.reward_stage_indicator)
        # print("reach condition: ")
        # print(self.right_object_reached, self.left_object_reached)
        # print("lifting condition: ")
        # print("right object cur height (kpts): ", self.right_object_keypoint_cur[:, :, 2])
        # print("left object cur height (kpts): ", self.left_object_keypoint_cur[:, :, 2])
        # print("right object init height (kpts): ", self.right_object_keypoint_init[:, :, 2])
        # print("left object init height (kpts): ", self.left_object_keypoint_init[:, :, 2])

        # print("right object cur height: ", self.right_object_pos[:, 2])
        # print("left object cur height: ", self.left_object_pos[:, 2])
        # print("right object init height: ", self.right_object_init_pos[2])
        # print("left object init height: ", self.left_object_init_pos[2])
        # print("right object height: ", self.right_lift_height)
        # print("left object height: ", self.left_lift_height)
        # print("right object height: ", self.right_lift_height > self.right_object_lift_thresh)
        # print("left object height: ", self.left_lift_height > self.left_object_lift_thresh)
        # print("right finger distance: ", self.right_h2o_dist)
        # print("left finger distance: ", self.left_h2o_dist)
        # print("right finger distance: ", self.right_h2o_dist < self.right_finger_dist_tolerance)
        # print("left finger distance: ", self.left_h2o_dist < self.left_finger_dist_tolerance)
        # print(self.right_object_lifted, self.left_object_lifted)
        # print()

    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            self.left_fingertip_force_sensors = self.robot.root_physx_view.get_link_incoming_joint_force()[
                :, self.left_finger_tip_indices
            ]
            self.right_fingertip_force_sensors = self.robot.root_physx_view.get_link_incoming_joint_force()[
                :, self.right_finger_tip_indices
            ]

        if self.cfg.obs_type == "reduced":
            obs = self.compute_reduced_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        if self.cfg.rnd_obs:
            rnd_state = self.compute_rnd_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations.update({"critic": states})
        
        if self.cfg.rnd_obs:
            observations.update({"rnd_state": rnd_state})

        return observations

    
    def compute_reduced_observations(self):
        
        obs = [
            # stage condition
            self.successes_onehot,
            # current active side
            self.cur_active_side_onehot,
            # last predicted action residual 
            self.pred_actions,

        ]
        right_base_obs = [
            # robot joint pos & vel
            unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)[:, self.right_joint_indices],
            (self.cfg.vel_obs_scale * self.robot_dof_vel)[:, self.right_joint_indices],
            # hand to object
            self.right_ee_pos - self.right_object_pos, # right ee to right object translation
            (self.right_fingertip_pos - self.right_object_pos.unsqueeze(1)).view(self.num_envs, self.num_right_finger_tips * 3),
        ]
        right_ref_act_obs = [
            self.ref_action_norm[:, self.right_joint_indices]
        ]
        
        left_base_obs = [
            # robot joint pos & vel
            unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)[:, self.left_joint_indices],
            (self.cfg.vel_obs_scale * self.robot_dof_vel)[:, self.left_joint_indices],
            self.left_ee_pos - self.left_object_pos, # left ee to left object translation
            (self.left_fingertip_pos - self.left_object_pos.unsqueeze(1)).view(self.num_envs, self.num_right_finger_tips * 3),
        ]
        left_ref_act_obs = [
            self.ref_action_norm[:, self.left_joint_indices]
        ]

        right_object_obs = [
            self.right_object_keypoint_cur.reshape(self.num_envs, -1),
            self.right_object_keypoint_goal.reshape(self.num_envs, -1),
            (self.right_object_keypoint_cur - self.right_object_keypoint_goal).reshape(self.num_envs, -1),
        ]
        left_object_obs = [
            self.left_object_keypoint_cur.reshape(self.num_envs, -1),
            self.left_object_keypoint_goal.reshape(self.num_envs, -1),
            (self.left_object_keypoint_cur - self.left_object_keypoint_goal).reshape(self.num_envs, -1),
        ]
        
        if self.use_left_side_reward:
            obs = obs + left_base_obs + left_object_obs
            if self.use_ref:
                obs = obs + left_ref_act_obs
        
        if self.use_right_side_reward:
            obs = obs + right_base_obs + right_object_obs
            if self.use_ref:
                obs = obs + right_ref_act_obs
        
        obs =  torch.cat(obs, dim=-1)

        return obs

    
    def compute_full_state(self):
        
        states = [
            # stage condition
            self.successes_onehot,
            # current active side
            self.cur_active_side_onehot,
            # last predicted action residual 
            self.pred_actions,
        ]
        right_base_states = [
            # robot joint pos & vel
            unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)[:, self.right_joint_indices],
            (self.cfg.vel_obs_scale * self.robot_dof_vel)[:, self.right_joint_indices],
            # hand to object
            self.right_ee_pos - self.right_object_pos, # right ee to right object translation
            # right finger tips
            self.right_fingertip_pos.view(self.num_envs, self.num_right_finger_tips * 3),
            self.right_fingertip_quat.view(self.num_envs, self.num_right_finger_tips * 4),
            self.right_fingertip_velocities.view(self.num_envs, self.num_right_finger_tips * 6),
            self.cfg.force_torque_obs_scale * self.right_fingertip_force_sensors.view(self.num_envs, self.num_right_finger_tips * 6),
            (self.right_fingertip_pos - self.right_object_pos.unsqueeze(1)).view(self.num_envs, self.num_right_finger_tips * 3),
        ]
        right_ref_act_states = [
            self.ref_action_norm[:, self.right_joint_indices]
        ]
        
        left_base_states = [
            # robot joint pos & vel
            unscale(self.robot_dof_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)[:, self.left_joint_indices],
            (self.cfg.vel_obs_scale * self.robot_dof_vel)[:, self.left_joint_indices],
            self.left_ee_pos - self.left_object_pos, # left ee to left object translation
            # right finger tips
            self.left_fingertip_pos.view(self.num_envs, self.num_left_finger_tips * 3),
            self.left_fingertip_quat.view(self.num_envs, self.num_left_finger_tips * 4),
            self.left_fingertip_velocities.view(self.num_envs, self.num_left_finger_tips * 6),
            self.cfg.force_torque_obs_scale * self.left_fingertip_force_sensors.view(self.num_envs, self.num_left_finger_tips * 6),
            (self.left_fingertip_pos - self.left_object_pos.unsqueeze(1)).view(self.num_envs, self.num_left_finger_tips * 3),
        ]
        left_ref_act_states = [
            self.ref_action_norm[:, self.left_joint_indices]
        ]

        right_object_states = [
            # object linvel angvel
            self.right_object_linvel,
            self.cfg.vel_obs_scale * self.right_object_angvel,
            self.right_object_keypoint_cur.reshape(self.num_envs, -1),
            self.right_object_keypoint_goal.reshape(self.num_envs, -1),
            (self.right_object_keypoint_cur - self.right_object_keypoint_goal).reshape(self.num_envs, -1),
        ]
        left_object_states = [
            # object linvel angvel
            self.left_object_linvel,
            self.cfg.vel_obs_scale * self.left_object_angvel,
            self.left_object_keypoint_cur.reshape(self.num_envs, -1),
            self.left_object_keypoint_goal.reshape(self.num_envs, -1),
            (self.left_object_keypoint_cur - self.left_object_keypoint_goal).reshape(self.num_envs, -1),
        ]
        
        if self.use_left_side_reward:
            states = states + left_base_states + left_object_states
            if self.use_ref:
                states = states + left_ref_act_states
        
        if self.use_right_side_reward:
            states = states + right_base_states + right_object_states
            if self.use_ref:
                states = states + right_ref_act_states
        
        states =  torch.cat(states, dim=-1)

        return states
    

    def compute_rnd_state(self):
        
        states = [
            # current stage condition
            self.successes_onehot,
        ]
        right_base_states = [
            # right hand states
            self.right_ee_pos, 
            self.right_fingertip_pos.view(self.num_envs, self.num_right_finger_tips * 3),
        ]
        left_base_states = [
            # left hand state
            self.left_ee_pos,
            self.left_fingertip_pos.view(self.num_envs, self.num_left_finger_tips * 3),
        ]

        right_object_states = [
            self.right_object_keypoint_cur.reshape(self.num_envs, -1),
            self.right_object_keypoint_goal.reshape(self.num_envs, -1),
            (self.right_object_keypoint_cur - self.right_object_keypoint_goal).reshape(self.num_envs, -1),
        ]
        left_object_states = [
            self.left_object_keypoint_cur.reshape(self.num_envs, -1),
            self.left_object_keypoint_goal.reshape(self.num_envs, -1),
            (self.left_object_keypoint_cur - self.left_object_keypoint_goal).reshape(self.num_envs, -1),
        ]
        
        if self.use_left_side_reward:
            states = states + left_base_states + left_object_states
            
        if self.use_right_side_reward:
            states = states + right_base_states + right_object_states
        
        states =  torch.cat(states, dim=-1)

        return states
    

    def _update_conuter(self):
        #################################
        # update sparse bonus indicators
        #################################

        # reaching
        self.right_object_not_reached = torch.where(
            self.right_object_reached & (self.cur_active_side == 1),
            torch.tensor(False),
            self.right_object_not_reached
        )

        self.left_object_not_reached = torch.where(
            self.left_object_reached & (self.cur_active_side == 0),
            torch.tensor(False),
            self.left_object_not_reached
        )

        # lifting
        self.right_object_not_lifted = torch.where(
            self.right_object_lifted & (self.cur_active_side == 1),
            torch.tensor(False),
            self.right_object_not_lifted
        )

        self.left_object_not_lifted = torch.where(
            self.left_object_lifted & (self.cur_active_side == 0),
            torch.tensor(False),
            self.left_object_not_lifted
        )

        # goaling
        self.right_goal_not_reached = torch.where(
            self.right_goal_reached & (self.cur_active_side == 1),
            torch.tensor(False),
            self.right_goal_not_reached
        )

        self.left_goal_not_reached = torch.where(
            self.left_goal_reached & (self.cur_active_side == 0),
            torch.tensor(False),
            self.left_goal_not_reached
        )

        # placing
        self.object_not_stable_placed = torch.where(
            self.object_stable_placed & (self.successes > self.switch_idx),
            torch.tensor(False),
            self.object_not_stable_placed
        )

        #################
        # update counter
        #################
        self.step_to_lift_right = torch.where(
            self.right_object_lifted & (self.cur_active_side == 1),
            self.episode_length_buf,
            self.step_to_lift_right
        )

        self.step_to_lift_left = torch.where(
            self.left_object_lifted & (self.cur_active_side == 0),
            self.episode_length_buf,
            self.step_to_lift_left
        )

        self.step_to_goal_right = torch.where(
            self.right_goal_reached & (self.cur_active_side == 1),
            self.episode_length_buf,
            self.step_to_goal_right
        )

        self.step_to_goal_left = torch.where(
            self.left_goal_reached & (self.cur_active_side == 0),
            self.episode_length_buf,
            self.step_to_goal_left
        )


    def _get_rewards(self) -> torch.Tensor:
        
        (
            total_reward,
            self.reset_goal_buf,
            self.successes,
            # reset_goal_buf,
            # successes,
            log_dict,
        ) = compute_rewards_async(
            reset_goal_buf=self.reset_goal_buf,
            successes=self.successes,
            reward_stage_indicator=self.reward_stage_indicator,
            # task states
            right_ee2o_dist=self.right_ee2o_dist,
            right_h2o_dist=self.right_h2o_dist,
            right_goal_dist=self.right_goal_dist,
            right_object_reached=self.right_object_reached,
            right_object_lifted=self.right_object_lifted,
            right_goal_reached=self.right_goal_reached,
            left_ee2o_dist=self.left_ee2o_dist,
            left_h2o_dist=self.left_h2o_dist,
            left_goal_dist=self.left_goal_dist,
            left_object_reached=self.left_object_reached,
            left_object_lifted=self.left_object_lifted,
            left_goal_reached=self.left_goal_reached,
            object_stable_placed=self.object_stable_placed,
            # sparse bonus indicator
            right_object_not_reached=self.right_object_not_reached,
            right_object_not_lifted=self.right_object_not_lifted,
            right_goal_not_reached=self.right_goal_not_reached,
            left_object_not_reached=self.left_object_not_reached,
            left_object_not_lifted=self.left_object_not_lifted,
            left_goal_not_reached=self.left_goal_not_reached,
            object_not_stable_placed=self.object_not_stable_placed,
            # penalty
            action_penalty=self.action_penalty,
            moving_penalty=self.moving_penalty,
            object_dropped_penalty=self.object_dropped_penalty,
            right_goal_reached_speed_penalty=self.right_goal_reached_speed_penalty,
            left_goal_reached_speed_penalty=self.left_goal_reached_speed_penalty,
            # which side rewards to be used
            cur_active_side=self.cur_active_side,
            switch_point=self.switch_idx,
            max_consecutive_success=self.max_consecutive_success
        )
        
        if self.debug:
            # self.palm_marker.visualize(self.right_ee_pos+self.scene.env_origins, self.right_ee_quat)
            # self.object_marker.visualize(self.right_object_pos+self.scene.env_origins, self.right_object_rot) 
            # for idx, marker in enumerate(self.finger_markers):
            #     marker.visualize(self.right_fingertip_pos[:, idx, :] + self.scene.env_origins, self.right_fingertip_quat[:, idx, :])

            kpts = self.left_object_keypoint_init
            for idx in range(kpts.shape[1]):
                self.kpts_init_markers[idx].visualize(kpts[:, idx, :] + self.scene.env_origins, self.right_object_rot)
            
            kpts = self.left_object_keypoint_cur
            for idx in range(kpts.shape[1]):
                self.kpts_markers[idx].visualize(kpts[:, idx, :] + self.scene.env_origins, self.right_object_rot)
        
        self._update_conuter()

            
        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["common_step_counter"] = float(self.common_step_counter)
        self.extras["log"]["reset_to_last_success_ratio"] = float(self.reset_to_last_success_ratio)
        self.extras["log"]["consecutive_successes"] = self.successes.clone().detach()
        self.extras["log"]["lift_thresh"] = self.right_object_lift_thresh.mean().clone().detach()
        self.extras["log"]["lift_episode_counter"] = self.lift_episode_counter.mean().clone().detach()
        self.extras["log"]["reward_stage_indicator"] = self.reward_stage_indicator.mean().clone().detach()
        self.extras["log"]["switch_bonus"] = log_dict["switch_bonus"].clone().detach()
        self.extras["log"]["stable_placement_bonus"] = log_dict["stable_placement_bonus"].clone().detach()
        self.extras["log"]["completion_bonus"] = log_dict["completion_bonus"]

        self.extras["log"]["right_goal_reach_thresh"] = self.right_object_pos_tolerance.mean().clone().detach()
        self.extras["log"]["right_goal_reach_episode_counter"] = self.right_goal_reach_episode_counter.mean().clone().detach()
        self.extras["log"]["left_goal_reach_thresh"] = self.left_object_pos_tolerance.mean().clone().detach()
        self.extras["log"]["left_goal_reach_episode_counter"] = self.left_goal_reach_episode_counter.mean().clone().detach()

        self.extras["log"]["step_to_lift_left"] = self.step_to_lift_left.mean().clone().detach()
        self.extras["log"]["step_to_goal_left"] = self.step_to_goal_left.mean().clone().detach()
        self.extras["log"]["step_to_goal_right"] = self.step_to_goal_right.mean().clone().detach()
        self.extras["log"]["step_to_lift_right"] = self.step_to_lift_right.mean().clone().detach()
        

        # keep tracking of the max consecutive successes for each eval envs
        self.eval_consecutive_successes = torch.maximum(self.successes, self.eval_consecutive_successes)
        
        for k in sorted(list(log_dict.keys())):
            if k not in self.extras["log"].keys():
                continue

            if self.use_left_side_reward and 'left' in k:
                self.extras["log"][k] = log_dict[k].clone().detach()
            
            if self.use_right_side_reward and 'right' in k:
                self.extras["log"][k] = log_dict[k].clone().detach()
            
        
        for k in self.extras["log"].keys():
            if self.extras["log"][k] is None:
                print(k, " is None in log!")

        # reset goals if the goal has been reached
        # self.reset_goal_buf = (self.ref_chunk_step_idx[:, 1] >= (self.ref_chunk_max_steps[self.ref_chunk_step_idx[:, 0]] - 1)) #NOTE test only
        # self.successes = self.successes + self.reset_goal_buf
        
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward


    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    #     # termination condition: 
    #     #   |-- the object falls off the table
    #     #   |-- the object dropped after being lifted
    #     #   |-- we reached the last sub-goal
    #     fall_terminated = (self.right_object_pos[:, 2] < 0.1) | (self.left_object_pos[:, 2] < 0.1)
    #     # drop_terminated = (~self.right_object_lifted | ~self.left_object_lifted) & (self.successes > 1)

    #     # drop_terminated = torch.where(
    #     #     self.cur_active_side == 0,
    #     #     ~self.left_object_lifted & (self.successes > self.left_init_id+1),
    #     #     ~self.right_object_lifted & (self.successes > self.right_init_id+1),
    #     # )
        
    #     succeed_terminated = self.successes >= self.max_consecutive_success

    #     # all timeout: the current action chunk reaches to its end but not reached goal yet
    #     all_timeout = (self.reward_stage_indicator < 3) & (self.ref_chunk_step_idx[:, 1] >= self.ref_chunk_max_steps[self.ref_chunk_step_idx[:, 0]])

    #     bad_placement_terminated = (self.successes == self.switch_idx) & (~getattr(self, f'{self.init_side}_goal_reached'))
    #     truncated = all_timeout  & (~bad_placement_terminated)


    #     return fall_terminated | bad_placement_terminated, succeed_terminated, truncated
    

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # termination condition: 
        #   |-- the object falls off the table
        #   |-- the object dropped after being lifted
        #   |-- we reached the last sub-goal
        fall_terminated = (self.right_object_pos[:, 2] < 0.1) | (self.left_object_pos[:, 2] < 0.1)

        right_h2o_min_dist = torch.clamp((torch.cdist(self.right_fingertip_pos, self.right_object_grasp_keypoint_cur).min(dim=-1).values).min(dim=-1).values, min=-9.0, max=9.0)
        left_h2o_min_dist = torch.clamp((torch.cdist(self.left_fingertip_pos, self.left_object_grasp_keypoint_cur).min(dim=-1).values).min(dim=-1).values, min=-9.0, max=9.0)
        
        drop_terminated = torch.where(
            (self.cur_active_side == 0),
            (left_h2o_min_dist > 0.01) & ((self.successes > 0) & (self.successes != self.switch_idx)),
            (right_h2o_min_dist > 0.01) & ((self.successes > 0) & (self.successes != self.switch_idx))
        )

        succeed_terminated = self.successes >= self.max_consecutive_success

        # all timeout: the current action chunk reaches to its end but not reached goal yet
        truncated = (self.reward_stage_indicator < 3) & \
            (self.ref_chunk_step_idx[:, 0] == (self.max_consecutive_success-1)) & \
                (self.ref_chunk_step_idx[:, 1] == (self.ref_chunk_max_steps[self.ref_chunk_step_idx[:, 0]]-1))

        return fall_terminated | drop_terminated, succeed_terminated, truncated


    def _reset_idx(self, env_ids: Sequence[int] | None):
        def _reset_object(obj, add_noise=False):
            object_default_state = obj.data.default_root_state.clone()[env_ids]
            if add_noise:
                pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
                pos_noise[:, 2] = 0.0
                # global object positions
                object_default_state[:, 0:3] = (
                    object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
                )

                rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids)), device=self.device) * self.cfg.reset_rotation_noise # noise for X and Y rotation
                rot_noise_quat_z = quat_from_angle_axis(rot_noise * np.pi, self.z_unit_tensor[env_ids])
                object_default_state[:, 3:7] = quat_mul(
                    rot_noise_quat_z, object_default_state[:, 3:7]
                )
            else:
                object_default_state[:, 0:3] = (
                    object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
                )

            object_default_state[:, 7:] = torch.zeros_like(obj.data.default_root_state[env_ids, 7:])
            obj.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
            obj.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

            return object_default_state

        def _reset_robot(robot, add_noise=False):
            if add_noise:
                delta_max = self.robot_dof_upper_limits[env_ids] - self.robot.data.default_joint_pos[env_ids]
                delta_min = self.robot_dof_lower_limits[env_ids] - self.robot.data.default_joint_pos[env_ids]
                
                dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
                rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
                dof_pos = self.robot.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

                dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_robot_dofs), device=self.device)
                dof_vel = self.robot.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise
            else:
                # reset robot
                dof_pos = self.robot.data.default_joint_pos[env_ids]
                dof_vel = self.robot.data.default_joint_vel[env_ids]

            self.prev_targets[env_ids] = dof_pos.clone()
            self.cur_targets[env_ids] = dof_pos.clone()

            self.robot.set_joint_position_target(dof_pos, env_ids=env_ids)
            self.robot.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)
        
        self.lift_episode_counter[env_ids] = torch.where(
            self.object_sync_lifted[env_ids],
            self.lift_episode_counter[env_ids] + 1,
            self.lift_episode_counter[env_ids]
        )

        # Update the number of consecutive success for the reset eval envs
        eval_env_ids = env_ids[env_ids < self.num_eval_envs]
        if len(eval_env_ids) > 0:
            self.csbuffer.extend(self.eval_consecutive_successes[eval_env_ids].cpu().numpy().tolist())

        # In sweep mode, the curriculum is not activated
        if not self.sweep:
            self._compute_curriculum(env_ids)

        # reset in chunk steps
        self.ref_chunk_step_idx[env_ids, 1] = 0
        self.ref_chunk_step_idx[env_ids, 0] = 0

        self.successes[env_ids] = 0
        self.cur_goal_idx[env_ids] = 0
        self.eval_consecutive_successes[eval_env_ids] = 0.0
        self.reset_goal_buf[env_ids] = 0
        if self.init_side == 'right':
            self.cur_active_side[env_ids] = True
        elif self.init_side == 'left':
            self.cur_active_side[env_ids] = False

        # sparse bonus indicator
        self.right_object_not_reached[env_ids] = torch.tensor(True)
        self.left_object_not_reached[env_ids] = torch.tensor(True)
        self.object_not_sync_reached[env_ids] = torch.tensor(True)

        self.right_object_not_lifted[env_ids] = torch.tensor(True)
        self.left_object_not_lifted[env_ids] = torch.tensor(True)
        self.object_not_sync_lifted[env_ids] = torch.tensor(True)

        self.right_goal_not_reached[env_ids] = torch.tensor(True)
        self.left_goal_not_reached[env_ids] = torch.tensor(True)

        self.object_not_stable_placed[env_ids] = torch.tensor(True)


        self.step_to_lift_right[env_ids] = 0
        self.step_to_lift_left[env_ids] = 0
        self.step_to_goal_right[env_ids] = 0
        self.step_to_goal_left[env_ids] = 0

        # update goal pose and markers
        self.right_goal_rot[env_ids] = self.right_ref_targets[self.cur_goal_idx[env_ids], 3:7]
        self.right_goal_pos[env_ids] = self.right_ref_targets[self.cur_goal_idx[env_ids], :3]
        self.left_goal_rot[env_ids] = self.left_ref_targets[self.cur_goal_idx[env_ids], 3:7]
        self.left_goal_pos[env_ids] = self.left_ref_targets[self.cur_goal_idx[env_ids], :3]

        self.right_goal_marker.visualize(self.right_goal_pos+self.scene.env_origins, self.right_goal_rot)
        self.left_goal_marker.visualize(self.left_goal_pos+self.scene.env_origins, self.left_goal_rot)

        # reset object
        _reset_object(self.right_object, add_noise=True)
        _reset_object(self.left_object, add_noise=True)

        # reset hand
        _reset_robot(self.robot, add_noise=True)

    def _reset_to(
        self,
        state: dict[str, dict[str, dict[str, torch.Tensor]]],
        env_ids: Sequence[int] | None,
        seed: int | None = None,
        is_relative: bool = False,
    ) -> None:
        """Resets specified environments to known states.

        Note that this is different from reset() function as it resets the environments to specific states

        Args:
            state: The state to reset the specified environments to.
            env_ids: The environment ids to reset. Defaults to None, in which case all environments are reset.
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            is_relative: If set to True, the state is considered relative to the environment origins. Defaults to False.
        """
        # set the seed
        if seed is not None:
            self.seed(seed)
            
            
        self.ref_chunk_step_idx[env_ids, :] = self.last_chunk_step_idx[env_ids, :].clone()
        self.episode_length_buf[env_ids] = 0.0
        self.reset_goal_buf[env_ids] = 0

        # reset goal counter only
        self.step_to_goal_right[env_ids] = 0
        self.step_to_goal_left[env_ids] = 0

        # reset the action history
        self.prev_targets[env_ids] = self.last_success_prev_targets[env_ids]
        self.cur_targets[env_ids] = self.last_success_cur_targets[env_ids]
        
        # reset goal bonus indicator 
        self.right_goal_not_reached[env_ids] = torch.tensor(True)
        self.left_goal_not_reached[env_ids] = torch.tensor(True)
        
        self.right_goal_rot[env_ids] = self.right_ref_targets[self.cur_goal_idx[env_ids], 3:7]
        self.right_goal_pos[env_ids] = self.right_ref_targets[self.cur_goal_idx[env_ids], :3]

        self.left_goal_rot[env_ids] = self.left_ref_targets[self.cur_goal_idx[env_ids], 3:7]
        self.left_goal_pos[env_ids] = self.left_ref_targets[self.cur_goal_idx[env_ids], :3]

        self.cur_active_side[env_ids] = self.last_active_side[env_ids]

        # set the state
        self.scene.reset_to(state, env_ids, is_relative=is_relative)
        
    # def _reset_envs(
    #     self,
    #     terminate: torch.Tensor,
    #     truncated: torch.Tensor,
    # ):
    #     """
    #     Resets environments based on termination, timeout, or max success conditions.

    #     This method creates a clean partition of environments to be reset:
    #     1.  A small subset of training environments that timed out but had prior success
    #         may be reset to their last successful state.
    #     2.  All other environments that need a reset (due to termination, max success,
    #         or timeout) are reset to their initial state.
    #     """
    #     # Determine the complete set of environments that need any form of reset
    #     all_reset_mask = terminate | truncated
    #     all_reset_ids = all_reset_mask.nonzero(as_tuple=False).squeeze(-1)

    #     # If no environments need resetting, return early.
    #     if all_reset_ids.numel() == 0:
    #         return

    #     # --- Partitioning Logic ---

    #     # 1. Identify candidates to be reset to their LAST successful state.
    #     # These are training envs that timed out, had prior success, and did NOT terminate or reach max success.
    #     # This is the most specific category, so we identify it first.
    #     reset_to_last_succ_candidate_mask = (
    #         truncated &                               # Timed out without reaching goal
    #         ~self.eval_env_mask &                     # Is a training environment
    #         (self.successes > 0) &                    # Has had at least one success
    #         ~terminate                                # Did NOT terminate (e.g., fall off table)
    #     )
        
    #     last_succ_candidate_ids = reset_to_last_succ_candidate_mask.nonzero(as_tuple=False).squeeze(-1)

    #     # From the candidates, sample a portion based on the configured ratio
    #     final_last_succ_ids = torch.tensor([], dtype=torch.long, device=self.device)
    #     if last_succ_candidate_ids.numel() > 0:
    #         num_to_sample = int(math.ceil(last_succ_candidate_ids.numel() * self.reset_to_last_success_ratio))
    #         if num_to_sample > 0:
    #             perm = torch.randperm(last_succ_candidate_ids.numel(), device=self.device)
    #             final_last_succ_ids = last_succ_candidate_ids[perm[:num_to_sample]]

    #     # 2. All other environments needing a reset go to the INITIAL state.
    #     # We create a mask for the final "last success" envs to easily exclude them
    #     # from the set of envs to be reset to their initial state.
    #     final_last_succ_mask = torch.zeros_like(all_reset_mask)
    #     if final_last_succ_ids.numel() > 0:
    #         final_last_succ_mask[final_last_succ_ids] = True

    #     reset_to_init_mask = all_reset_mask & ~final_last_succ_mask
    #     final_init_ids = reset_to_init_mask.nonzero(as_tuple=False).squeeze(-1)
        
    #     # --- Perform Resets ---
    #     # make sure reset is performed for all needed envs
    #     assert all_reset_ids.numel() == (final_init_ids.numel() + final_last_succ_ids.numel())

    #     if final_init_ids.numel() > 0:
    #         # print(f"Resetting {final_init_ids.numel()} env(s) to initial state.")
    #         self._reset_idx(final_init_ids)

    #     if final_last_succ_ids.numel() > 0:
    #         # print(f"Resetting {final_last_succ_ids.numel()} env(s) to last successful state.")
    #         self._reset_to(
    #             self.last_success_state,
    #             final_last_succ_ids,
    #             seed=None,
    #             is_relative=False
    #         )
        
    #     # update articulation kinematics
    #     self.scene.write_data_to_sim()
    #     self.sim.forward()

    #     # if sensors are added to the scene, make sure we render to reflect changes in reset
    #     if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
    #         self.sim.render()

    #     self._compute_intermediate_values()



    def _reset_envs(
        self,
        terminate: torch.Tensor,
        truncated: torch.Tensor,
    ):
        """
        Resets environments with a specialized curriculum for the task switch.

        This method creates a clean partition of environments to be reset:
        1.  (NEW) Envs that fail immediately after switching tasks are ALWAYS reset
            to their last successful state to enforce practice on the switch.
        2.  A small subset of other training environments that timed out but had prior success
            may be reset to their last successful state based on a probability ratio.
        3.  All other environments that need a reset are reset to their initial state.
        """
        # Determine the complete set of environments that need any form of reset
        all_reset_mask = terminate | truncated
        all_reset_ids = all_reset_mask.nonzero(as_tuple=False).squeeze(-1)

        # If no environments need resetting, return early.
        if all_reset_ids.numel() == 0:
            return

        # --- NEW Partitioning Logic with a Mandatory Curriculum ---

        ## NEW ##: Identify envs that failed right AT the switch point. These get a mandatory reset-to-success.
        # Condition: The env is resetting AND its success count is exactly the switch point.
        # We also exclude evaluation environments from this specific curriculum.
        reset_at_switch_mask = all_reset_mask & (self.successes == self.switch_idx) & (~self.eval_env_mask) & (~terminate)
        final_switch_reset_ids = reset_at_switch_mask.nonzero(as_tuple=False).squeeze(-1)

        ## NEW ##: Identify candidates for the ORIGINAL probabilistic reset-to-success.
        # This is the same logic as before...
        probabilistic_candidate_mask = (
            truncated &                               # Timed out without reaching goal
            ~self.eval_env_mask &                     # Is a training environment
            (self.successes > 0) &                    # Has had at least one success
            ~terminate                                # Did NOT terminate (e.g., fall off table)
        )
        ## NEW ##: ...BUT, we must exclude the envs that are already being reset by our mandatory rule.
        probabilistic_candidate_mask = probabilistic_candidate_mask & ~reset_at_switch_mask
        probabilistic_candidate_ids = probabilistic_candidate_mask.nonzero(as_tuple=False).squeeze(-1)

        # Sample a portion of the probabilistic candidates based on the configured ratio
        final_probabilistic_reset_ids = torch.tensor([], dtype=torch.long, device=self.device)
        if probabilistic_candidate_ids.numel() > 0:
            num_to_sample = int(math.ceil(probabilistic_candidate_ids.numel() * self.reset_to_last_success_ratio))
            if num_to_sample > 0:
                perm = torch.randperm(probabilistic_candidate_ids.numel(), device=self.device)
                final_probabilistic_reset_ids = probabilistic_candidate_ids[perm[:num_to_sample]]
                
        ## NEW ##: Combine both groups (mandatory and probabilistic) that will be reset to last success.
        # torch.unique is used to be safe, although the sets should already be disjoint.
        final_last_succ_ids = torch.unique(torch.cat([final_switch_reset_ids, final_probabilistic_reset_ids]))

        ## NEW ##: All other resetting environments will be reset to the initial state.
        # We create a mask for all "last success" envs to easily exclude them.
        final_last_succ_mask = torch.zeros_like(all_reset_mask)
        if final_last_succ_ids.numel() > 0:
            final_last_succ_mask[final_last_succ_ids] = True
        
        reset_to_init_mask = all_reset_mask & ~final_last_succ_mask
        final_init_ids = reset_to_init_mask.nonzero(as_tuple=False).squeeze(-1)
        
        # --- Perform Resets ---
        # Sanity check: our partitioning should cover all resetting envs exactly once.
        assert all_reset_ids.numel() == (final_init_ids.numel() + final_last_succ_ids.numel())

        if final_init_ids.numel() > 0:
            self._reset_idx(final_init_ids)

        if final_last_succ_ids.numel() > 0:
            # Note: This relies on `last_success_state` being correctly updated when the
            # agent completes the first side's task, which your `_reset_target_pose` and
            # `update_last_success_state` functions already handle.
            self._reset_to(
                self.last_success_state,
                final_last_succ_ids,
                seed=None,
                is_relative=False
            )
        
        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        self._compute_intermediate_values()
    

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[VecEnvObs, dict]:
        """Resets all the environments and returns observations.

        This function calls the :meth:`_reset_idx` function to reset all the environments.
        However, certain operations, such as procedural terrain generation, that happened during initialization
        are not repeated.

        Args:
            seed: The seed to use for randomization. Defaults to None, in which case the seed is not set.
            options: Additional information to specify how the environment is reset. Defaults to None.

                Note:
                    This argument is used for compatibility with Gymnasium environment definition.

        Returns:
            A tuple containing the observations and extras.
        """
        # set the seed
        if seed is not None:
            self.seed(seed)

        # full reset state of scene
        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self._reset_idx(indices)

        # update articulation kinematics
        self.scene.write_data_to_sim()
        self.sim.forward()

        self._compute_intermediate_values()

        # if sensors are added to the scene, make sure we render to reflect changes in reset
        if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
            self.sim.render()

        if self.cfg.wait_for_textures and self.sim.has_rtx_sensors():
            while SimulationManager.assets_loading():
                self.sim.render()

        # return observations
        return self._get_observations(), self.extras
    

    def update_last_success_state(self, env_ids):
        new_state = self.scene.get_state(is_relative=False)

        assert list(new_state['articulation'].keys()) == list(self.last_success_state['articulation'].keys()), \
            f"Entities mismatch between the current env state and last success state under the articulation class \n" \
            f"Current entities: {new_state['articulation'].keys()} \n" \
            f"Last success entities: {self.last_success_state['articulation'].keys()}"
        
        for name in self.last_success_state['articulation']:
            self.last_success_state['articulation'][name]['root_pose'][env_ids] = \
                new_state['articulation'][name]['root_pose'][env_ids].clone()
            
            self.last_success_state['articulation'][name]['root_velocity'][env_ids] = \
                new_state['articulation'][name]['root_velocity'][env_ids].clone()
            
            self.last_success_state['articulation'][name]['joint_position'][env_ids] = \
                new_state['articulation'][name]['joint_position'][env_ids].clone()
            
            self.last_success_state['articulation'][name]['joint_velocity'][env_ids] = \
                new_state['articulation'][name]['joint_velocity'][env_ids].clone()


        for name in self.last_success_state['deformable_object']:
            self.last_success_state['deformable_object'][name]['nodal_position'][env_ids] = \
                new_state['deformable_object'][name]['nodal_position'][env_ids].clone()
            self.last_success_state['deformable_object'][name]['nodal_velocity'][env_ids] = \
                new_state['deformable_object'][name]['nodal_velocity'][env_ids].clone()
        

        for name in self.last_success_state['rigid_object']:
            self.last_success_state['rigid_object'][name]['root_pose'][env_ids] = \
                new_state['rigid_object'][name]['root_pose'][env_ids].clone()
            self.last_success_state['rigid_object'][name]['root_velocity'][env_ids] = \
                new_state['rigid_object'][name]['root_velocity'][env_ids].clone()
    


    def _reset_target_pose(self, env_ids):
        # update chunk ids for success envs
        
        # self.ref_chunk_step_idx[env_ids, 0] = torch.where(
        #     self.successes[env_ids] < self.max_consecutive_success,
        #     self.ref_chunk_step_idx[env_ids, 0] + 1,
        #     torch.zeros_like(self.ref_chunk_step_idx[env_ids, 0])
        # )
        # self.ref_chunk_step_idx[env_ids, 1] = 0

        self.cur_goal_idx[env_ids] = torch.where(
            self.successes[env_ids] < self.max_consecutive_success,
            self.cur_goal_idx[env_ids] + 1,
            torch.zeros_like(self.cur_goal_idx[env_ids])
        )

        # update goal pose and markers
        self.right_goal_rot[env_ids] = self.right_ref_targets[self.cur_goal_idx[env_ids], 3:7]
        self.right_goal_pos[env_ids] = self.right_ref_targets[self.cur_goal_idx[env_ids], :3]

        self.left_goal_rot[env_ids] = self.left_ref_targets[self.cur_goal_idx[env_ids], 3:7]
        self.left_goal_pos[env_ids] = self.left_ref_targets[self.cur_goal_idx[env_ids], :3]

        self.right_goal_marker.visualize(self.right_goal_pos+self.scene.env_origins, self.right_goal_rot)
        self.left_goal_marker.visualize(self.left_goal_pos+self.scene.env_origins, self.left_goal_rot)

        
        self.reset_goal_buf[env_ids] = 0
        self.right_goal_not_reached[env_ids] = torch.tensor(True)
        self.left_goal_not_reached[env_ids] = torch.tensor(True)

        self.last_success_prev_targets[env_ids] = self.prev_targets[env_ids]
        self.last_success_cur_targets[env_ids] = self.cur_targets[env_ids]
        
        # Update the content of last success state buffer for the envs 
        # that reached new success
        self.update_last_success_state(env_ids=env_ids)

        

        self.last_active_side[env_ids] = self.cur_active_side[env_ids].clone()
        self.last_chunk_step_idx[env_ids, :] = self.last_chunk_step_idx[env_ids, :].clone()


    def _compute_curriculum(self, env_ids):

        # object lift counter
        self.right_object_lift_thresh[env_ids] = torch.where(
            (self.lift_episode_counter[env_ids] > 20) & (self.cur_active_side[env_ids] == 1),
            torch.clamp(self.right_object_lift_thresh[env_ids] + 0.005, max=0.075),
            self.right_object_lift_thresh[env_ids]
        )
        
        self.left_object_lift_thresh[env_ids] = torch.where(
            (self.lift_episode_counter[env_ids] > 20) & (self.cur_active_side[env_ids] == 0),
            torch.clamp(self.left_object_lift_thresh[env_ids] + 0.005, max=0.075),
            self.left_object_lift_thresh[env_ids]
        )

        # finger contact distance
        self.right_finger_dist_tolerance[env_ids] = torch.where(
            (self.lift_episode_counter[env_ids] > 20) & (self.cur_active_side[env_ids] == 1),
            torch.clamp(self.right_finger_dist_tolerance[env_ids] - 0.005, min=0.05),
            self.right_finger_dist_tolerance[env_ids]
        )

        self.left_finger_dist_tolerance[env_ids] = torch.where(
            (self.lift_episode_counter[env_ids] > 20) & (self.cur_active_side[env_ids] == 0),
            torch.clamp(self.left_finger_dist_tolerance[env_ids] - 0.005, min=0.05),
            self.left_finger_dist_tolerance[env_ids]
        )

        self.lift_episode_counter[env_ids] = torch.where(
            self.lift_episode_counter[env_ids] > 20,
            0,
            self.lift_episode_counter[env_ids]
        )
        

        # reduce the reset_to_last_success_ratio
        # max_steps = 16 * 40000 # rollout lenght = 16, max_iterations = 40000
        # warm_up_steps = 16 * 10000 if self.use_left_side_reward and self.use_right_side_reward else 16 * 2000
        # interval = 16 * 2500 if self.use_left_side_reward and self.use_right_side_reward else 16 * 1000

        # if (self.common_step_counter > warm_up_steps) and ((self.common_step_counter+1) % (16 * 5000) == 0):
        #     self.reset_to_last_success_ratio = max(self.reset_to_last_success_ratio-0.01, 0.05)

        # reduce the threshold for reaching the sub goal
        # @TODO
        
        if len(self.csbuffer) > self.num_eval_envs:
            if statistics.mean(self.csbuffer) > self.switch_idx:
                if self.init_side == 'left':
                    self.left_goal_reach_episode_counter[env_ids] += 1
                else:
                    self.right_goal_reach_episode_counter[env_ids] += 1

            if statistics.mean(self.csbuffer) > self.max_consecutive_success - 1.5:
                if self.init_side == 'left':
                    self.right_goal_reach_episode_counter[env_ids] += 1
                else:
                    self.left_goal_reach_episode_counter[env_ids] += 1

        # start the second arm curriculum when the first arm curriculum is finished
        if self.init_side == 'right':
            right_condition = (self.right_goal_reach_episode_counter[env_ids] > self.pos_tolerance_curriculum_step)
        else:
            right_condition = (self.right_goal_reach_episode_counter[env_ids] > self.pos_tolerance_curriculum_step) & (self.left_object_pos_tolerance[env_ids] < 0.0075)
        self.right_object_pos_tolerance[env_ids] = torch.where(
            right_condition,
            torch.clamp(self.right_object_pos_tolerance[env_ids] - self.pos_tolerance_reduce, min=0.005),
            self.right_object_pos_tolerance[env_ids]
        )

        if self.init_side == 'left':
            left_condition = (self.left_goal_reach_episode_counter[env_ids] > self.pos_tolerance_curriculum_step)
        else:
            left_condition = (self.left_goal_reach_episode_counter[env_ids] > self.pos_tolerance_curriculum_step) & (self.right_goal_reach_episode_counter[env_ids] < 0.0075)
        self.left_object_pos_tolerance[env_ids] = torch.where(
            left_condition,
            torch.clamp(self.left_object_pos_tolerance[env_ids] - self.pos_tolerance_reduce, min=0.005),
            self.left_object_pos_tolerance[env_ids]
        )

        self.right_goal_reach_episode_counter[env_ids] = torch.where(
            self.right_goal_reach_episode_counter[env_ids] > self.pos_tolerance_curriculum_step,
            0,
            self.right_goal_reach_episode_counter[env_ids]
        )

        self.left_goal_reach_episode_counter[env_ids] = torch.where(
            self.left_goal_reach_episode_counter[env_ids] > self.pos_tolerance_curriculum_step,
            0,
            self.left_goal_reach_episode_counter[env_ids]
        )

        #@TODO
        # maybe add a curriculum for the noise level of the object position and rotation? 



@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower



@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower + 1e-6)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def check_goal_reached(
    cur_pos: torch.Tensor, tgt_pos: torch.Tensor, 
    cur_rot: torch.Tensor, tgt_rot: torch.Tensor, 
    pos_tolerance: torch.Tensor, rot_tolerance: float
):
    pos_dist = torch.norm(cur_pos - tgt_pos, p=2, dim=-1)
    rot_dist = rotation_distance(cur_rot, tgt_rot)

    return (pos_dist < pos_tolerance) & (rot_dist < rot_tolerance), pos_dist, rot_dist

@torch.jit.script
def check_goal_reached_keypoint(cur_keypoint: torch.Tensor, tgt_keypoint: torch.Tensor, pos_tolerance: torch.Tensor):
    dist = torch.norm(cur_keypoint - tgt_keypoint, p=2, dim=-1).max(dim=1).values

    return (dist < pos_tolerance), dist

@torch.jit.script
def _mse_distance_reward(dist: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    rew = (1.0 / (dist + 1.0))
    rew *= rew

    return scale * rew

@torch.jit.script
def _tanh_distance_reward(dist: torch.Tensor, std: float = 0.1, scale: float = 1.0) -> torch.Tensor:
    rew = 1 - torch.tanh(dist / std)

    return scale * rew

@torch.jit.script
def update_rewards(reward, cond):
    return torch.where(cond, reward, 0)


# @torch.jit.script
def compute_rewards_async(
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    reward_stage_indicator: torch.Tensor,
    # task states
    right_ee2o_dist: torch.Tensor,
    right_h2o_dist: torch.Tensor,
    right_goal_dist: torch.Tensor,
    right_object_reached: torch.Tensor,
    right_object_lifted: torch.Tensor,
    right_goal_reached: torch.Tensor,
    left_ee2o_dist: torch.Tensor,
    left_h2o_dist: torch.Tensor,
    left_goal_dist: torch.Tensor,
    left_object_reached: torch.Tensor,
    left_object_lifted: torch.Tensor,
    left_goal_reached: torch.Tensor,
    object_stable_placed: torch.Tensor,
    # sparse bonus indicator
    right_object_not_reached: torch.Tensor,
    right_object_not_lifted: torch.Tensor,
    right_goal_not_reached: torch.Tensor,
    left_object_not_reached: torch.Tensor,
    left_object_not_lifted: torch.Tensor,
    left_goal_not_reached: torch.Tensor,
    object_not_stable_placed: torch.Tensor,
    # penalty
    action_penalty: torch.Tensor,
    moving_penalty: torch.Tensor,
    object_dropped_penalty: torch.Tensor,
    right_goal_reached_speed_penalty: torch.Tensor,
    left_goal_reached_speed_penalty: torch.Tensor,
    # which side rewards to be used
    cur_active_side: torch.Tensor,
    switch_point: int,
    max_consecutive_success: int
):
    
    """
    In this reward function, we separate the lift bonus and goal reaching bonus for right and left hands
    """
    log_dict = {}
    rewards = 0.0
    
    ####################
    # Stage 1, reaching
    ####################
    right_ee2o_reward = _tanh_distance_reward(right_ee2o_dist, std=0.3, scale=1.0)
    right_object_reached_bonus = torch.where(
        right_object_reached & right_object_not_reached,
        torch.zeros_like(successes)+50,
        torch.zeros_like(successes)
    ).float()

    left_ee2o_reward = _tanh_distance_reward(left_ee2o_dist, std=0.3, scale=1.0)
    left_object_reached_bonus = torch.where(
        left_object_reached & left_object_not_reached,
        torch.zeros_like(successes)+50,
        torch.zeros_like(successes)
    ).float()

    stage1_reward = torch.where(
        cur_active_side == 0,
        left_ee2o_reward + left_object_reached_bonus,
        right_ee2o_reward + right_object_reached_bonus
    )

    # print(cur_active_side)
    # print(stage1_reward)
    # print(left_ee2o_reward + left_object_reached_bonus)
    # print(right_ee2o_reward + right_object_reached_bonus)
    # print()

    rewards = rewards + stage1_reward


    log_dict['right_ee2o_dist'] = right_ee2o_dist
    log_dict['right_ee2o_dist_reward'] = right_ee2o_reward
    log_dict["right_object_reached"] = right_object_reached
    log_dict["right_object_reached_bonus"] = right_object_reached_bonus
    log_dict['left_ee2o_dist'] = left_ee2o_dist
    log_dict['left_ee2o_dist_reward'] = left_ee2o_reward
    log_dict["left_object_reached"] = left_object_reached
    log_dict["left_object_reached_bonus"] = left_object_reached_bonus
    
    ####################
    # stage 1 ends here
    ####################

    

    # ##############################
    # stage 2, grasping and lifting
    ################################
    right_h2o_dist_reward = _tanh_distance_reward(right_h2o_dist, std=0.05, scale=2.0)
    left_h2o_dist_reward = _tanh_distance_reward(left_h2o_dist, std=0.05, scale=2.0)

    right_lift_bonus = torch.where(
        right_object_lifted & right_object_not_lifted,
        torch.zeros_like(successes)+100,
        torch.zeros_like(successes)
    ).float()
    
    left_lift_bonus = torch.where(
        left_object_lifted & left_object_not_lifted, # assign the lift-up bonus once
        torch.zeros_like(successes)+100,
        torch.zeros_like(successes)
    ).float()


    stage2_rewards = torch.where(
        cur_active_side == 0,
        left_h2o_dist_reward + left_lift_bonus,
        right_h2o_dist_reward + right_lift_bonus
    )

    # only the agent that completes stage 1 could gain rewards from stage 2
    rewards = torch.where(
        reward_stage_indicator >= 1,
        rewards + stage2_rewards,
        rewards
    )

    log_dict['right_h2o_dist'] = right_h2o_dist
    log_dict['right_object_lifted'] = right_object_lifted
    log_dict['left_h2o_dist'] = left_h2o_dist
    log_dict['left_object_lifted'] = left_object_lifted

    # update rewards based on the current stage for logging
    log_dict['right_h2o_dist_reward'] = update_rewards(right_h2o_dist_reward, (reward_stage_indicator >= 1) & (cur_active_side == 1))
    log_dict['right_lift_bonus'] = update_rewards(right_lift_bonus, (reward_stage_indicator >= 1) & (cur_active_side == 1))
    log_dict['left_h2o_dist_reward'] = update_rewards(left_h2o_dist_reward, (reward_stage_indicator >= 1) & (cur_active_side == 0))
    log_dict['left_lift_bonus'] = update_rewards(left_lift_bonus, (reward_stage_indicator >= 1) & (cur_active_side == 0))

    ####################
    # stage 2 ends here
    ####################


    ##################################
    # Stage 3, reaching the current sub-goals
    ##################################
    

    right_goal_dist_reward = 15 * _tanh_distance_reward(right_goal_dist, std=0.1)
    left_goal_dist_reward = 15 * _tanh_distance_reward(left_goal_dist, std=0.1)

    right_goal_reached_bonus = torch.where(
        right_goal_reached & right_goal_not_reached,
        torch.zeros_like(successes) + 1000,
        torch.zeros_like(successes)
    ).float()
    
    left_goal_reached_bonus = torch.where(
        left_goal_reached & left_goal_not_reached,
        torch.zeros_like(successes) + 1000,
        torch.zeros_like(successes)
    ).float()


    stage3_rewards = torch.where(
        cur_active_side == 0,
        left_goal_dist_reward + left_goal_reached_bonus - left_goal_reached_speed_penalty,
        right_goal_dist_reward + right_goal_reached_bonus - right_goal_reached_speed_penalty
    )
    
    rewards = torch.where(
        reward_stage_indicator >= 2,
        rewards + stage3_rewards,
        rewards
    )


    log_dict['right_goal_dist'] = right_goal_dist
    log_dict['right_goal_reached'] = right_goal_reached
    log_dict['left_goal_dist'] = left_goal_dist
    log_dict['left_goal_reached'] = left_goal_reached
    log_dict['object_stable_placed'] = object_stable_placed
    
    log_dict['right_goal_dist_reward'] = update_rewards(right_goal_dist_reward, (reward_stage_indicator >= 2) & (cur_active_side == 1))
    log_dict['right_goal_reached_bonus'] = update_rewards(right_goal_reached_bonus, (reward_stage_indicator >= 2) & (cur_active_side == 1))
    log_dict["right_goal_reached_speed_penalty"] = update_rewards(right_goal_reached_speed_penalty, (reward_stage_indicator >= 2) & (cur_active_side == 1))
    log_dict['left_goal_dist_reward'] = update_rewards(left_goal_dist_reward, (reward_stage_indicator >= 2) & (cur_active_side == 0))
    log_dict['left_goal_reached_bonus'] = update_rewards(left_goal_reached_bonus, (reward_stage_indicator >= 2) & (cur_active_side == 0))
    log_dict["left_goal_reached_speed_penalty"] = update_rewards(left_goal_reached_speed_penalty, (reward_stage_indicator >= 2) & (cur_active_side == 0))
    

    ####################
    # stage 3 ends here
    ####################

    # Give a bonus if the agent made a stable and good placement
    stable_placement_bonus = torch.where(
        object_stable_placed & object_not_stable_placed,
        2000,
        0.0
    )

    # if torch.any(stable_placement_bonus > 0):
    #     print("gain placement rewards")

    # Give a huge bonus if the agent finish one side job
    is_switch_goal_completed = (reward_stage_indicator == 3) & (successes == (switch_point - 1))
    switch_bonus = torch.where(
        is_switch_goal_completed,
        4000,
        0.0
    )

    rewards = rewards + stable_placement_bonus + switch_bonus
    rewards = rewards - action_penalty - moving_penalty - object_dropped_penalty# try to limit the output of inactive side close to 0.0

    log_dict['stable_placement_bonus'] = stable_placement_bonus
    log_dict['switch_bonus'] = switch_bonus

    rewards = rewards * torch.exp(successes / 2)

    # # Scale up rewards for later stage    
    # rewards = torch.where(
    #     successes >= switch_point,
    #     rewards * (successes+1),
    #     rewards
    # )
    # rewards = rewards * (successes+1) - 0.001 * action_penalty

    goal_resets = torch.where(
        reward_stage_indicator == 3,
        torch.ones_like(reset_goal_buf), 
        reset_goal_buf
    )
    successes = successes + goal_resets # success count


    completion_bonus = torch.where(
        successes == max_consecutive_success,
        100000,
        0.0
    )
    rewards = rewards + completion_bonus
    
    log_dict['completion_bonus'] = completion_bonus

    return (
        rewards / 100., goal_resets, successes, log_dict, 
    )


# @TODO
# add constraint relaxtion? Add linvel and angvel to the object, to make it move towards the goal pose? 
# Check object pos, palm pos are correct? 
# add stable condition for reaching the sub-goals
# add a counter for counting how many step it takes for reaching the right goal and left goal