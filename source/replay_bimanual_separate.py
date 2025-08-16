# """
# Author: Yucheng Xu
# Date: 4 Mar 2025
# Description: 
#     This script receive the retargeted robot hand motions and object motions from FoundationPose.
#     These motions are replayed in simulation to record robot joint angles with synchorinized right & left robots
# Usage:
#     ./isaaclab.sh -p devel/replay.py --robot_usd /path/to/robot use --retarget /path/to/retarget file --camera /path/to/camera extrinsics

# Before running this script, we need to convert the robot urdf and object model to usd format.
# For converting robot urdf, check scripts/tools/convert_urdf.py for details.
# For converting object model, import it in the IsaacLab simulator and add physics on it (RigidObject, Mass), then export it as usd.
# """
# import os
# import argparse
# import numpy as np
# from glob import glob
# import time
# import pickle

# from isaaclab.app import AppLauncher

# import logging
# logger = logging.getLogger()
# logger.setLevel(logging.ERROR)
# for hndlr in logger.handlers:
#     logger.removeHandler(hndlr)

# # add argparse arguments
# parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
# parser.add_argument("--robot", type=str, default="franka_allegro", help="Name of the robot_right.")
# parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# parser.add_argument("--data_dir", type=str, default="devel/data")
# parser.add_argument("--assets_dir", type=str, default="devel/assets")
# parser.add_argument("--seq_name", type=str, default="realsense/hammer/bimanual_assembly_separate_fps30")
# parser.add_argument("--object_mesh_right", type=str, default="hammer_v2/hammer_handle.usd", help="path to the right-side object usd in scene")
# parser.add_argument("--object_mesh_left", type=str, default="hammer_v2/hammer_head.usd", help="path to the left-side object usd in scene")
# parser.add_argument("--trajectory_output_f", type=str, default="trajectory_franka_allegro.pkl", help="path to save the output trajectory")

# parser.add_argument("--move_thres_r", type=float, default=0.1, help="threshold for lifting right object")
# parser.add_argument("--move_thres_l", type=float, default=0.1, help="threshold for lifting left object")
# parser.add_argument("--ik_ee_name_r", type=str, default="right_hand_base_link", help="name of the body, use as the end-effector in IK calculation")
# parser.add_argument("--ik_ee_name_l", type=str, default="left_hand_base_link", help="name of the body, use as the end-effector in IK calculation")
# parser.add_argument("--min_chunk_steps", type=int, default=50, help="Minimal action steps in a chunk")
# parser.add_argument("--save_interval", type=int, default=2)
# parser.add_argument("--use_selected_keyframes", type=bool, default=True)
# parser.add_argument("--debug", type=int, default=1, help="turn on debug visualization or not")
# parser.add_argument("--repeat", action='store_true', default=True, help="replay the trajectory repeatly")
# parser.add_argument("--separate", action='store_true', default=True)
# parser.add_argument("--order", type=str, default='left_right')
# parser.add_argument("--kpts_approx", type=str, default='right,')

# # append AppLauncher cli args
# AppLauncher.add_app_launcher_args(parser)
# # parse the arguments
# args_cli = parser.parse_args()

# # launch omniverse app
# app_launcher = AppLauncher(args_cli)
# simulation_app = app_launcher.app


# """Rest everything follows."""

# import torch

# import omni.usd
# from pxr import UsdGeom, Usd

# import isaaclab.sim as sim_utils
# from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
# from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
# from isaaclab.markers.config import FRAME_MARKER_CFG
# from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
# from isaaclab.utils import configclass
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
# from isaaclab.utils.math import subtract_frame_transforms, compute_pose_error, quat_from_matrix
# from isaaclab.sensors.camera import Camera, CameraCfg
# from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
# from isaaclab.assets.articulation import ArticulationCfg

# from isaaclab.sensors import (
#     FrameTransformer,
#     FrameTransformerCfg,
#     OffsetCfg,
#     TiledCamera,
#     TiledCameraCfg,
#     ContactSensor,
#     ContactSensorCfg
# )

# from isaaclab.utils.math import (
#     saturate, quat_conjugate, quat_from_angle_axis, 
#     quat_mul, sample_uniform, saturate, 
#     matrix_from_quat, quat_apply, yaw_quat
# )

# ##
# # Pre-defined robot configs
# ##
# from isaaclab_assets import FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG   # isort:skip

# @configclass
# class ReplaySceneCfg(InteractiveSceneCfg):
#     # ground plane
#     ground = AssetBaseCfg(
#         prim_path="/World/defaultGroundPlane",
#         spawn=sim_utils.GroundPlaneCfg(color=(242/255, 238/255, 203/255)),
#         init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
#     )

#     # lights
#     dome_light = AssetBaseCfg(
#         prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
#     )

#     # mount
#     table_right = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/Table_r",
#         spawn=sim_utils.UsdFileCfg(
#             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, -0.35, 0.0], rot=[1, 0, 0, 0]),
#     )
    
#     table_left = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/Table_l",
#         spawn=sim_utils.UsdFileCfg(
#             usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.35, 0.0], rot=[1, 0, 0, 0]),
#     )
    
#     object_r = RigidObjectCfg(
#         prim_path="{ENV_REGEX_NS}/Object_r",
#         spawn=sim_utils.UsdFileCfg(
#             usd_path=f"{args_cli.assets_dir}/{args_cli.object_mesh_right}", scale=(1.0, 1.0, 1.0),
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
#             mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
#             collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.2, 0.055], rot=[1, 0, 0, 0]),
#     )
    
#     object_l = RigidObjectCfg(
#         prim_path="{ENV_REGEX_NS}/Object_l",
#         spawn=sim_utils.UsdFileCfg(
#             usd_path=f"{args_cli.assets_dir}/{args_cli.object_mesh_left}", scale=(1.0, 1.0, 1.0),
#             rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
#             mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
#             collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
#         ),
#         init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, -0.2, 0.055], rot=[1, 0, 0, 0]),
#     )

#     right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
#         # source frame
#         prim_path="/World/envs/env_.*/Robot/base_link",
#         debug_vis=True,
#         visualizer_cfg=VisualizationMarkersCfg(
#             prim_path='/Visuals/right_ee',
#             markers={
#                 'sphere': sim_utils.SphereCfg(
#                     radius=0.01,
#                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
#                 ),
#             }
#         ),
#         target_frames=[
#             FrameTransformerCfg.FrameCfg(
#                 prim_path="/World/envs/env_.*/Robot/right_palm",
#                 name="right_end_effector",
#                 offset=OffsetCfg(
#                     pos=[0.1, 0.0, 0.08],
#                 ),
#             ),
#         ],
#     )

#     left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
#         # source frame
#         prim_path="/World/envs/env_.*/Robot/base_link",
#         debug_vis=True,
#         visualizer_cfg=VisualizationMarkersCfg(
#             prim_path='/Visuals/left_ee',
#             markers={
#                 'sphere': sim_utils.SphereCfg(
#                     radius=0.01,
#                     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
#                 ),
#             }
#         ),
#         target_frames=[
#             FrameTransformerCfg.FrameCfg(
#                 prim_path="/World/envs/env_.*/Robot/left_palm",
#                 name="left_end_effector",
#                 offset=OffsetCfg(
#                     pos=[0.1, -0.01, 0.08],
#                 ),
#             ),
#         ],
#     )


    
#     # articulation
#     if args_cli.robot == "franka_allegro":
#         print("Load franka allegro")
#         robot = FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
#     elif args_cli.robot == "xarm_leap":
#         print("Load franka allegro")
#         robot = XARM_LEAP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
#     else:
#         raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, franka_allegro, ur10")


# class DemoReplay(object):
#     def __init__(self, scene_cfg, args_cli):
#         self.scene_cfg = scene_cfg
#         self.min_chunk_steps = args_cli.min_chunk_steps
#         self.debug = args_cli.debug
#         self.separate = args_cli.separate
#         self.order = args_cli.order.split("_")
#         self.num_envs = args_cli.num_envs
#         self.kpts_approx = []
#         self.kpts_approx = [k for k in args_cli.kpts_approx.split(',') if k]
#         for key in self.kpts_approx:
#             kpts_f = f"{args_cli.assets_dir}/{getattr(args_cli, f'object_mesh_{key}').replace('.usd', '.npz')}"
#             if not os.path.exists(kpts_f):
#                 print(f"required keypoint file does not exist, skip keypoint approximation on {key} side")
#                 self.kpts_approx.remove(key)
#                 continue
#             setattr(
#                 self, f'{key}_object_kpts', torch.tensor(np.load(kpts_f)['object_keypoints'][:], dtype=torch.float, device='cuda')
#             )
        
#         self.setup_sim(args_cli=args_cli)
#         self.read_data(args_cli=args_cli)

    
#     def setup_sim(self, args_cli):
#         sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
#         self.sim = sim_utils.SimulationContext(sim_cfg)
#         # Set main camera
#         self.sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
#         # Design scene
#         self.scene = InteractiveScene(self.scene_cfg)
#         # Play the simulator
#         self.sim.reset()
        
#         self.sim_dt = self.sim.get_physics_dt()

#         self.stage = omni.usd.get_context().get_stage()
        
#         self.robot = self.scene['robot']

#         self.object_r = self.scene['object_r']
#         self.object_l = self.scene['object_l']
#         self.object_root_state_r = self.object_r.data.default_root_state.clone()
#         self.object_root_state_l = self.object_l.data.default_root_state.clone()

#         joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.sim.device)
#         self.robot_dof_lower_limits = joint_pos_limits[..., 0]
#         self.robot_dof_upper_limits = joint_pos_limits[..., 1]
        
#         diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
#         self.diff_ik_controller_r = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)
#         self.diff_ik_controller_l = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)
        
#         self.ik_commands_r = torch.zeros(self.scene.num_envs, self.diff_ik_controller_r.action_dim, device=self.sim.device)
#         self.ik_commands_l = torch.zeros(self.scene.num_envs, self.diff_ik_controller_l.action_dim, device=self.sim.device)
#         self.obj_ref_pose_r = torch.zeros(self.scene.num_envs, 7, device=self.sim.device)
#         self.obj_ref_pose_l = torch.zeros(self.scene.num_envs, 7, device=self.sim.device)


#         # self.scene.sensors["right_ee_frame"] = self.right_ee_frame
#         # self.scene.sensors["left_ee_frame"] = self.left_ee_frame

#         self.save_interval = args_cli.save_interval
        
#         # Specify robot-specific parameters
#         if args_cli.robot == "franka_allegro":
#             self.right_arm_entity_cfg = SceneEntityCfg(
#                 "robot", 
#                 joint_names=[
#                     "right_panda_joint_1", "right_panda_joint_2", "right_panda_joint_3",
#                     "right_panda_joint_4", "right_panda_joint_5", "right_panda_joint_6", "right_panda_joint_7",
                    
#                 ], 
#                 body_names=["right_panda_link_0", args_cli.ik_ee_name_r]
#             )

#             self.left_arm_entity_cfg = SceneEntityCfg(
#                 "robot", 
#                 joint_names=[
#                     "left_panda_joint_1", "left_panda_joint_2", "left_panda_joint_3",
#                     "left_panda_joint_4", "left_panda_joint_5", "left_panda_joint_6", "left_panda_joint_7",
#                 ], 
#                 body_names=["left_panda_link_0", args_cli.ik_ee_name_l]
#             )

#             self.right_hand_entity_cfg = SceneEntityCfg(
#                 "robot", 
#                 joint_names=[
#                     "right_index_joint_0", "right_index_joint_1", "right_index_joint_2", "right_index_joint_3", 
#                     "right_middle_joint_0", "right_middle_joint_1", "right_middle_joint_2", "right_middle_joint_3", 
#                     "right_ring_joint_0", "right_ring_joint_1", "right_ring_joint_2", "right_ring_joint_3", 
#                     "right_thumb_joint_0", "right_thumb_joint_1", "right_thumb_joint_2", "right_thumb_joint_3",
#                 ], 
#             )

#             self.left_hand_entity_cfg = SceneEntityCfg(
#                 "robot", 
#                 joint_names=[
#                     "left_index_joint_0", "left_index_joint_1", "left_index_joint_2", "left_index_joint_3", 
#                     "left_middle_joint_0", "left_middle_joint_1", "left_middle_joint_2", "left_middle_joint_3", 
#                     "left_ring_joint_0", "left_ring_joint_1", "left_ring_joint_2", "left_ring_joint_3", 
#                     "left_thumb_joint_0", "left_thumb_joint_1", "left_thumb_joint_2", "left_thumb_joint_3", 
#                 ], 
#             )

#             self.right_robot_entity_cfg = SceneEntityCfg(
#                 "robot", 
#                 joint_names=[
#                     "right_panda_joint_1", "right_panda_joint_2", "right_panda_joint_3",
#                     "right_panda_joint_4", "right_panda_joint_5", "right_panda_joint_6", "right_panda_joint_7",
#                     "right_index_joint_0", "right_index_joint_1", "right_index_joint_2", "right_index_joint_3", 
#                     "right_middle_joint_0", "right_middle_joint_1", "right_middle_joint_2", "right_middle_joint_3", 
#                     "right_ring_joint_0", "right_ring_joint_1", "right_ring_joint_2", "right_ring_joint_3", 
#                     "right_thumb_joint_0", "right_thumb_joint_1", "right_thumb_joint_2", "right_thumb_joint_3",
#                 ], 
#             )

#             self.left_robot_entity_cfg = SceneEntityCfg(
#                 "robot", 
#                 joint_names=[
#                     "left_panda_joint_1", "left_panda_joint_2", "left_panda_joint_3",
#                     "left_panda_joint_4", "left_panda_joint_5", "left_panda_joint_6", "left_panda_joint_7",
#                     "left_index_joint_0", "left_index_joint_1", "left_index_joint_2", "left_index_joint_3", 
#                     "left_middle_joint_0", "left_middle_joint_1", "left_middle_joint_2", "left_middle_joint_3", 
#                     "left_ring_joint_0", "left_ring_joint_1", "left_ring_joint_2", "left_ring_joint_3", 
#                     "left_thumb_joint_0", "left_thumb_joint_1", "left_thumb_joint_2", "left_thumb_joint_3", 
#                 ], 
#             )
        
#         else:
#             raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda")
        
        
#         self.init_robot_qpos = None
#         self.object_pre_grasp_reached = False
#         self.object_lifted = False
#         self.move_thres_r = args_cli.move_thres_r
#         self.move_thres_l = args_cli.move_thres_l
#         self.repeat = args_cli.repeat
#         self.use_selected_keyframes = args_cli.use_selected_keyframes
        
#         # Resolving the scene entities
#         self.right_arm_entity_cfg.resolve(self.scene)
#         self.right_hand_entity_cfg.resolve(self.scene)
#         self.right_robot_entity_cfg.resolve(self.scene)

#         self.left_arm_entity_cfg.resolve(self.scene)
#         self.left_hand_entity_cfg.resolve(self.scene)
#         self.left_robot_entity_cfg.resolve(self.scene)

#         self.right_arm_joint_ids = self.right_arm_entity_cfg.joint_ids
#         self.right_arm_joint_names = self.right_arm_entity_cfg.joint_names
#         self.right_hand_joint_ids = self.right_hand_entity_cfg.joint_ids
#         self.right_robot_joint_ids = self.right_robot_entity_cfg.joint_ids
#         self.right_robot_joint_names = self.right_robot_entity_cfg.joint_names

#         self.left_arm_joint_ids = self.left_arm_entity_cfg.joint_ids
#         self.left_arm_joint_names = self.left_arm_entity_cfg.joint_names
#         self.left_hand_joint_ids = self.left_hand_entity_cfg.joint_ids
#         self.left_robot_joint_ids = self.left_robot_entity_cfg.joint_ids
#         self.left_robot_joint_names = self.left_robot_entity_cfg.joint_names

#         # Obtain the frame index of the end-effector
#         # For a fixed base robot, the frame index is one less than the body index. This is because
#         # the root body is not included in the returned Jacobians.
#         # In this replay script, we only setup on body in the robot cfg
#         self.ee_body_id_r = self.right_arm_entity_cfg.body_ids[1]
#         self.ee_body_id_l = self.left_arm_entity_cfg.body_ids[1]
#         self.root_body_id_r = self.right_arm_entity_cfg.body_ids[0]
#         self.root_body_id_l = self.left_arm_entity_cfg.body_ids[0]
#         if self.robot.is_fixed_base:
#             self.ee_jacobi_idx_r = self.ee_body_id_r - 1
#             self.ee_jacobi_idx_l = self.ee_body_id_l - 1
#         else:
#             self.ee_jacobi_idx_r = self.ee_body_id_r
#             self.ee_jacobi_idx_l = self.ee_body_id_l
            
#         self.joint_pos = self.robot.data.default_joint_pos.clone()
#         self.joint_vel = self.robot.data.default_joint_vel.clone()
#         self.robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
#         self.robot.reset()
#         self.right_robot_base_offset = (self.robot.data.body_state_w[:, self.right_arm_entity_cfg.body_ids[0], :3] - self.scene.env_origins)[0, :]
#         self.left_robot_base_offset = (self.robot.data.body_state_w[:, self.left_arm_entity_cfg.body_ids[0], :3] - self.scene.env_origins)[0, :]
        
#         self.scene.write_data_to_sim()
#         # perform step
#         self.sim.step()
        
#         # update buffers
#         self.scene.update(self.sim_dt)

#         if args_cli.debug:
#             frame_marker_cfg = FRAME_MARKER_CFG.copy()
#             frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
#             self.ee_r_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_r_current"))
#             self.goal_r_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_r_goal"))
#             self.ee_l_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_l_current"))
#             self.goal_l_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_l_goal"))
#             self.obj_r_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/object_r"))
#             self.obj_l_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/object_l"))
    
#     def read_data(self, args_cli):
#         retarget_f = f"{args_cli.data_dir}/{args_cli.seq_name}/retarget.npz"
#         retarget_bimanual = np.load(retarget_f, allow_pickle=True)
#         self.right_offset = np.asarray([0.65, 0.0, 0.10])
#         self.left_offset = np.asarray([0.65, 0.0, 0.10])
#         self.right_ee_offset = np.asarray([0.6, 0.0, 0.10])
#         self.left_ee_offset = np.asarray([0.65, 0.0, 0.08])
#         self.retarget_data = {}
#         for key in retarget_bimanual.keys():
#             retarget = retarget_bimanual[key].item()
#             ref_ee_pos = retarget['base_transl']
#             ref_ee_quat = retarget['base_quat']
#             ref_hand_qpos = retarget['qpos'][:, 6:]
            
#             ref_ee_pos += getattr(self, f'{key}_ee_offset')
#             # Since we got xyzw format in retargeting, convert it to wxyc for IsaacLab
#             ref_ee_quat = np.stack([ref_ee_quat[:, 3], ref_ee_quat[:, 0], ref_ee_quat[:, 1], ref_ee_quat[:, 2]], axis=1)
#             ref_ee_pose = np.concatenate([ref_ee_pos, ref_ee_quat], axis=1)

#             # Convert to torch.Tensor   
#             ref_ee_pos = torch.tensor(ref_ee_pos, device=self.sim.device)
#             ref_ee_quat = torch.tensor(ref_ee_quat, device=self.sim.device)
#             ref_ee_pose = torch.tensor(ref_ee_pose, device=self.sim.device)
#             ref_hand_qpos = torch.tensor(ref_hand_qpos, device=self.sim.device).float()

#             # Object motions
#             assert hasattr(args_cli, f"object_mesh_{key}")
#             object_name = getattr(args_cli, f"object_mesh_{key}").split("/")[-1].split(".")[0]
#             assert os.path.exists(f"{args_cli.data_dir}/{args_cli.seq_name}/processed/object/{object_name}_pose_cam.refine.npy")
#             object_poses = np.load(f"{args_cli.data_dir}/{args_cli.seq_name}/processed/object/{object_name}_pose_cam.refine.npy")
#             self.num_waypoints = object_poses.shape[0]

#             self.retarget_data[key] = {
#                 'ref_ee_pose': ref_ee_pose,
#                 'ref_hand_qpos': ref_hand_qpos,
#                 'ref_obj_pose': object_poses
#             }
        
#         # Camera extrinsic parameters
#         camera_pose = np.load(f"{args_cli.data_dir}/{args_cli.seq_name}/camera_extrinsic.npz")
#         extrinsic_matrix = np.eye(4)
#         extrinsic_matrix[:3, :3] = camera_pose['R']
#         extrinsic_matrix[:3, 3] = camera_pose['T'].reshape(3)
#         self.extrinsic_matrix = np.linalg.inv(extrinsic_matrix)

#         # read keyframe indices if exist
#         all_images = sorted(os.listdir(f"{args_cli.data_dir}/{args_cli.seq_name}/rgb"))
#         keyframes = sorted(os.listdir(f"{args_cli.data_dir}/{args_cli.seq_name}/kf_rgb"))
#         self.keyframe_indices = np.asarray([all_images.index(n) for n in keyframes])
#         self.num_keyframes = len(self.keyframe_indices)


#         # in async mode, we record keyframes for left and right separately
#         right_keyframes = sorted(os.listdir(f"{args_cli.data_dir}/{args_cli.seq_name}/right_kf_rgb"))
#         self.right_keyframe_indices = np.asarray([all_images.index(n) for n in right_keyframes])

#         left_keyframes = sorted(os.listdir(f"{args_cli.data_dir}/{args_cli.seq_name}/left_kf_rgb"))
#         self.left_keyframe_indices = np.asarray([all_images.index(n) for n in left_keyframes])

#         self.switch_indices = getattr(self, f'{self.order[0]}_keyframe_indices')[-1]
#         print(f"start with {self.order[0]} side")
#         print(f"Switch to {self.order[1]} side after {self.switch_indices}-th waypoint")


#     def transform_object_pose_cam_to_world(self, pose, offset):
#         # camera -> robot base
#         pose_b = np.dot(self.extrinsic_matrix, pose)
#         transl_b = pose_b[:3, 3]
#         quat_b = quat_from_matrix(
#             torch.tensor(pose_b[:3, :3], device=self.sim.device)
#         ).cpu().numpy()
        
#         # robot base -> sim world
#         transl_w = torch.tensor(transl_b+offset, device=self.sim.device) + self.scene.env_origins
#         quat_w = torch.tensor(quat_b, device=self.sim.device).unsqueeze(0).repeat(transl_w.shape[0], 1)
        
#         return torch.cat([transl_w, quat_w], dim=-1), np.concatenate([transl_b, quat_b], axis=0)
    

#     def check_reached(self, pos_error, ang_error, pos_thres=0.005, ang_thres=0.05):
#         if torch.linalg.norm(pos_error) < pos_thres and torch.abs(ang_error).sum() < ang_thres:
#             return True
#         else:
#             return False
    

#     def calculate_right_qpos_ik(self):
#         jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx_r, :, self.right_arm_joint_ids]
#         ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id_r, :7]
#         root_pose_w = self.robot.data.body_state_w[:, self.root_body_id_r, :7]
#         ee_pos_b, ee_quat_b = subtract_frame_transforms(
#                     root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
#                 )

#         position_error, axis_angle_error = compute_pose_error(
#                     ee_pos_b, ee_quat_b, self.ik_commands_r[:, :3], self.ik_commands_r[:, 3:7], rot_error_type="axis_angle"
#                 )
#         joint_pos = self.robot.data.joint_pos[:, self.right_arm_joint_ids]
#         joint_pos_des = self.diff_ik_controller_r.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
#         joint_pos_des = saturate(
#                     joint_pos_des.clone(),
#                     self.robot_dof_lower_limits[:, self.right_arm_joint_ids],
#                     self.robot_dof_upper_limits[:, self.right_arm_joint_ids]
#                 )
        
#         # print(position_error, axis_angle_error)
#         # For now we use the same target pose for all envs, so just simply mean the errors
#         position_error = position_error.mean(dim=0)
#         axis_angle_error = axis_angle_error.mean(dim=0)

#         reached = self.check_reached(position_error, axis_angle_error)
#         if self.debug:
#             self.ee_r_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
#             self.goal_r_marker.visualize(
#                 self.ik_commands_r[:, 0:3] + self.right_robot_base_offset + self.scene.env_origins, 
#                 self.ik_commands_r[:, 3:7]
#             )
#             self.obj_r_marker.visualize(self.object_root_state_r[:, 0:3], yaw_quat(self.object_root_state_r[:, 3:7]))

#         return joint_pos_des, reached


#     def calculate_left_qpos_ik(self):
#         jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx_l, :, self.left_arm_joint_ids]
#         ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id_l, :7]
#         root_pose_w = self.robot.data.body_state_w[:, self.root_body_id_l, :7]
#         ee_pos_b, ee_quat_b = subtract_frame_transforms(
#                     root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
#                 )

#         position_error, axis_angle_error = compute_pose_error(
#                     ee_pos_b, ee_quat_b, self.ik_commands_l[:, :3], self.ik_commands_l[:, 3:7], rot_error_type="axis_angle"
#                 )
#         joint_pos = self.robot.data.joint_pos[:, self.left_arm_joint_ids]
#         joint_pos_des = self.diff_ik_controller_l.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
#         joint_pos_des = saturate(
#                     joint_pos_des.clone(),
#                     self.robot_dof_lower_limits[:, self.left_arm_joint_ids],
#                     self.robot_dof_upper_limits[:, self.left_arm_joint_ids]
#                 )
        
#         # print(position_error, axis_angle_error)
#         # For now we use the same target pose for all envs, so just simply mean the errors
#         position_error = position_error.mean(dim=0)
#         axis_angle_error = axis_angle_error.mean(dim=0)

#         reached = self.check_reached(position_error, axis_angle_error)

#         if self.debug:
#             self.ee_l_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
#             self.goal_l_marker.visualize(
#                 self.ik_commands_l[:, 0:3] + self.left_robot_base_offset + self.scene.env_origins, 
#                 self.ik_commands_l[:, 3:7]
#             )
#             self.obj_l_marker.visualize(self.object_root_state_l[:, 0:3], yaw_quat(self.object_root_state_l[:, 3:7]))

#         return joint_pos_des, reached

    
#     def move_to_initial_waypoint(
#         self, 
#         right_goal_ee_pose, left_goal_ee_pose,
#         right_object_pose, left_object_pose,
#     ):
        
#         self.diff_ik_controller_r.reset()
#         self.ik_commands_r[:] = right_goal_ee_pose
#         self.ik_commands_r[:, :3] -= self.right_robot_base_offset
#         self.diff_ik_controller_r.set_command(self.ik_commands_r)

#         self.diff_ik_controller_l.reset()
#         self.ik_commands_l[:] = left_goal_ee_pose
#         self.ik_commands_l[:, :3] -= self.left_robot_base_offset
#         self.diff_ik_controller_l.set_command(self.ik_commands_l)

#         success = False
#         while not success:
#             # object pose
#             self.object_root_state_r[:, :7] = right_object_pose
#             self.object_root_state_l[:, :7] = left_object_pose 
            
#             joint_pos_des_r, reached_r = self.calculate_right_qpos_ik()
#             joint_pos_des_l, reached_l = self.calculate_left_qpos_ik()
#             self.joint_pos[:, self.right_arm_joint_ids] = joint_pos_des_r
#             self.joint_pos[:, self.left_arm_joint_ids] = joint_pos_des_l
#             if reached_r and reached_l:
#                 init_robot_qpos_r = joint_pos_des_r[0, :].cpu().numpy()
#                 init_robot_qpos_l = joint_pos_des_l[0, :].cpu().numpy()
#                 init_robot_qpos_dict = {}
#                 init_robot_qpos_dict.update(
#                     {k:v for (k, v) in zip(self.right_arm_joint_names, init_robot_qpos_r)}
#                 )
#                 init_robot_qpos_dict.update(
#                     {k:v for (k, v) in zip(self.left_arm_joint_names, init_robot_qpos_l)}
#                 )


#                 init_robot_qpos = self.robot.data.default_joint_pos.clone()[0, :]
#                 init_robot_qpos[self.right_arm_joint_ids] = joint_pos_des_r[0, :]
#                 init_robot_qpos[self.left_arm_joint_ids] = joint_pos_des_l[0, :]
#                 success = True
#             else:
#                 self.robot.set_joint_position_target(
#                     joint_pos_des_r, 
#                     joint_ids=self.right_arm_joint_ids
#                 )
#                 self.robot.set_joint_position_target(
#                     joint_pos_des_l, 
#                     joint_ids=self.left_arm_joint_ids
#                 )
#                 self.object_r.write_root_pose_to_sim(self.object_root_state_r[:, :7])
#                 self.object_l.write_root_pose_to_sim(self.object_root_state_l[:, :7])
                
#                 self.scene.write_data_to_sim()
#                 # perform step
#                 self.sim.step()
                
#                 # update buffers
#                 self.scene.update(self.sim_dt)
            
        
#         return init_robot_qpos_dict, init_robot_qpos
                

#     def move_to_next_waypoint(
#         self, 
#         right_goal_ee_pose, left_goal_ee_pose,
#         right_object_pose, left_object_pose,
#         right_hand_ref_qpos, left_hand_ref_qpos,
#         inactive_side='right'
#     ):
#         self.diff_ik_controller_r.reset()
#         self.ik_commands_r[:] = right_goal_ee_pose
#         self.ik_commands_r[:, :3] -= self.right_robot_base_offset
#         self.diff_ik_controller_r.set_command(self.ik_commands_r)

#         self.diff_ik_controller_l.reset()
#         self.ik_commands_l[:] = left_goal_ee_pose
#         self.ik_commands_l[:, :3] -= self.left_robot_base_offset

#         self.diff_ik_controller_l.set_command(self.ik_commands_l)

#         right_hand_qpos = self.joint_pos[:, self.right_hand_joint_ids]
#         left_hand_qpos = self.joint_pos[:, self.left_hand_joint_ids]

#         right_hand_qpos[:] = right_hand_ref_qpos * 0.95
#         left_hand_qpos[:] = left_hand_ref_qpos * 0.95

#         success = False
#         actions = []
#         count = 0
#         pre_grasp_flag = False
#         pre_grasp_rot = None

#         while not success:
#             if not self.object_pre_grasp_reached:
#                 if inactive_side == 'right':
#                     if 'left' in self.kpts_approx:
#                         obj_pose = self.calculate_kpts_approx_pose(left_object_pose.float(), self.left_object_kpts)
#                     else:
#                         obj_pose = self.calculate_local_pose(left_object_pose.float(), self.left_robot_base_offset)
                    
#                     ee_pose = torch.zeros_like(obj_pose)
#                     ee_pose[:, :3] = self.scene.sensors["left_ee_frame"].data.target_pos_source[..., 0, :] - self.left_robot_base_offset
#                     ee_pose[:, 3:] = self.scene.sensors["left_ee_frame"].data.target_quat_source[..., 0, :]

#                     # object pose
#                     self.object_root_state_r[:, :7] = right_object_pose
#                     self.object_root_state_l[:, :3] = obj_pose[:, :3] + self.scene.env_origins + self.left_robot_base_offset
#                     self.object_root_state_l[:, 3:7] = obj_pose[:, 3:]

#                 else:
#                     if 'right' in self.kpts_approx:
#                         obj_pose = self.calculate_kpts_approx_pose(right_object_pose.float(), self.right_object_kpts)
#                     else:
#                         obj_pose = self.calculate_local_pose(right_object_pose, self.right_robot_base_offset)

#                     ee_pose = torch.zeros_like(obj_pose)
#                     ee_pose[:, :3] = self.scene.sensors["right_ee_frame"].data.target_pos_source[..., 0, :] - self.right_robot_base_offset
#                     ee_pose[:, 3:] = self.scene.sensors["right_ee_frame"].data.target_quat_source[..., 0, :]

#                     self.object_root_state_r[:, :7] = obj_pose
#                     self.object_root_state_l[:, :7] = left_object_pose

#                 _pre_grasp_flag, _pre_grasp_rot = self.pre_grasp_check(
#                     ee_pose, obj_pose, threshold=0.04
#                 )
                
#                 if _pre_grasp_flag:
#                     pre_grasp_flag = _pre_grasp_flag
#                     pre_grasp_rot = _pre_grasp_rot
#                     self.object_pre_grasp_reached = True

            
            
#             joint_pos_des_r, reached_r = self.calculate_right_qpos_ik()
#             joint_pos_des_l, reached_l = self.calculate_left_qpos_ik()
#             self.joint_pos[:, self.right_arm_joint_ids] = joint_pos_des_r
#             self.joint_pos[:, self.left_arm_joint_ids] = joint_pos_des_l
#             self.joint_pos[:, self.right_hand_joint_ids] = right_hand_ref_qpos
#             self.joint_pos[:, self.left_hand_joint_ids] = left_hand_ref_qpos
            
#             reached = reached_l if inactive_side == 'right' else reached_r
#             if reached:
#                 actions.append(self.robot.data.joint_pos[0, :].cpu().numpy())
#                 success = True
#             else:
#                 self.robot.set_joint_position_target(
#                     joint_pos_des_r, 
#                     joint_ids=self.right_arm_joint_ids
#                 )
#                 self.robot.set_joint_position_target(
#                 right_hand_qpos,
#                 joint_ids=self.right_hand_joint_ids
#                 )
                    
#                 self.robot.set_joint_position_target(
#                     joint_pos_des_l, 
#                     joint_ids=self.left_arm_joint_ids
#                 )
                
#                 self.robot.set_joint_position_target(
#                     left_hand_qpos,
#                     joint_ids=self.left_hand_joint_ids
#                 )

#                 self.object_r.write_root_pose_to_sim(self.object_root_state_r[:, :7])
#                 self.object_l.write_root_pose_to_sim(self.object_root_state_l[:, :7])
#                 if count % self.save_interval == 0:
#                     actions.append(self.robot.data.joint_pos[0, :].cpu().numpy())
                
#                 self.scene.write_data_to_sim()
#                 # perform step
#                 self.sim.step()
                
#                 # update buffers
#                 self.scene.update(self.sim_dt)

#                 count += 1

                
            
        
#         return actions, pre_grasp_flag, pre_grasp_rot
    

#     def calculate_kpts_approx_pose(self, pose, kpts):
#         """
#         Calculates a canonical pose where the Y-axis is aligned with the cylinder's length.
#         """
#         # 1. Transform keypoints to world frame (this part is correct)
#         kpts_w = torch.einsum(
#                 'nij,nmj->nmi', 
#                 matrix_from_quat(pose[:, 3:]), # Assuming quat is [w,x,y,z] or library handles it
#                 kpts.clone().unsqueeze(0).repeat(self.num_envs, 1, 1)
#             ) + pose[:, :3].unsqueeze(1)
        
#         # --- FIX STARTS HERE ---

#         # For clarity, let's use the midpoint for translation (more robust)
#         translation = (kpts_w[:, 0] + kpts_w[:, -1]) / 2.0

#         # 2. Define the new Y-axis as the cylinder's principal axis
#         # Renamed from 'xaxis' to 'yaxis'
#         yaxis = kpts_w[:, 0] - kpts_w[:, -1]
#         yaxis = yaxis / torch.linalg.norm(yaxis, dim=-1, keepdim=True)
        
#         # 3. Define the new X-axis to be perpendicular to the new Y-axis and world Z
#         # This makes the new X-axis point out to the "side" of the cylinder
#         world_up = torch.zeros_like(yaxis)
#         world_up[:, 2] = 1

#         # Use a robustness check for vertical cylinders
#         dot_prod = torch.sum(yaxis * world_up, dim=1) # Batched dot product
#         is_vertical = torch.abs(dot_prod) > 0.999
        
#         # If vertical, use world X as reference. Otherwise, use world Z (up).
#         world_ref_for_vertical = torch.zeros_like(yaxis)
#         world_ref_for_vertical[:, 0] = 1 # World X-axis
#         ref_vec = torch.where(is_vertical.unsqueeze(-1), world_ref_for_vertical, world_up)
        
#         xaxis = torch.cross(yaxis, ref_vec, dim=-1)
#         xaxis = xaxis / torch.linalg.norm(xaxis, dim=-1, keepdim=True)

#         # 4. Define the new Z-axis to complete the right-handed frame
#         # The order is important: cross(X, Y) = Z
#         zaxis = torch.cross(xaxis, yaxis, dim=-1)
#         # This should already be normalized

#         # 5. Assemble the rotation matrix with the correct column assignments
#         rot = torch.eye(3, device=pose.device).unsqueeze(0).repeat(self.num_envs, 1, 1)

#         rot[:, :, 0] = xaxis  # First column is the new X-axis
#         rot[:, :, 1] = yaxis  # Second column is the new Y-axis
#         rot[:, :, 2] = zaxis  # Third column is the new Z-axis
        
#         # 6. Create the final pose in [pos, quat] format (this part is correct)
#         final_pose = torch.zeros_like(pose)
#         final_pose[:, :3] = translation
#         final_pose[:, 3:7] = quat_from_matrix(rot) # Assumes quat is [x,y,z,w]

#         return final_pose



#     def calculate_local_pose(self, pose, offset):
#         local_obj_pose = pose.clone()
#         local_obj_pose[:, :3] -= offset
#         local_obj_pose[:, :3] -= self.scene.env_origins

#         return local_obj_pose

    
#     def update_chunk(self, buffer, chunk_id, right_goal_object_pose, left_goal_object_pose, waypoint_idx):
#         def create_new_chunk(chunk_id, right_obj_pose, left_obj_pose):
#             buffer['action_chunks'][f'chunk_{chunk_id}'] = {
#                         'goal_object_pose.right': right_obj_pose,
#                         'goal_object_pose.left': left_obj_pose,
#                         'qpos': []
#                     }
        
#         def merge_current_chunk(chunk_id, right_obj_pose, left_obj_pose):
#             buffer['action_chunks'][f'chunk_{chunk_id}']['goal_object_pose.right'] = right_obj_pose
#             buffer['action_chunks'][f'chunk_{chunk_id}']['goal_object_pose.left'] = left_obj_pose


#         if waypoint_idx == getattr(self, f'{self.order[0]}_keyframe_indices')[0]: # create the first action chunk for the init arm
#             create_new_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
#         else:
#             if not self.object_lifted: 
#                 init_right_object_pose = buffer['init_object_pose.right']
#                 init_left_object_pose = buffer['init_object_pose.left']
#                 diff_r = np.abs(init_right_object_pose[2] - right_goal_object_pose[2])
#                 diff_l = np.abs(init_left_object_pose[2] - left_goal_object_pose[2])
#                 if not self.separate:
#                     lifted = diff_r > self.move_thres_r and diff_l > self.move_thres_l
#                 else:
#                     lifted = diff_r > self.move_thres_r or diff_l > self.move_thres_l
#                 if lifted:
#                     if not self.lift_chunk_end: # a little hack to make sure the lift-up chunk covers all lift up action
#                         merge_current_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
#                         self.lift_chunk_end = True
#                     else:
#                         chunk_id += 1
#                         self.object_lifted = True
#                         print(f"create {chunk_id} chunk, object lifted")
#                         create_new_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
#                 else:
#                     merge_current_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
#             else:
#                 if not self.use_selected_keyframes:
#                     if len(buffer['action_chunks'][f'chunk_{chunk_id}']['qpos']) > self.min_chunk_steps: # current chunk contains enough steps, create a new action chunk
#                         chunk_id += 1
#                         create_new_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
#                     else: # current chunk does not have enough steps, merge with the next chunk
#                         merge_current_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
#                 else: # use pre-selected keyframes
#                     if (waypoint_idx-1) in self.keyframe_indices and (waypoint_idx-1) != getattr(self, f'{self.order[-1]}_keyframe_indices')[0]:
#                         chunk_id += 1
#                         print(f"create {chunk_id} chunk, keyframe reached")
#                         create_new_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
#                     else: 
#                         merge_current_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)

            
#         return chunk_id
    

#     def pre_grasp_check(self, ee_pose, obj_pose, threshold):
#         """
#         Function to check if the hand ee is close to the object and 
#         then record the relative rotation between hand ee and object
#         """
#         ee2o_dist = torch.norm(ee_pose[:, :3] - obj_pose[:, :3], p=2, dim=-1)
#         if ee2o_dist < threshold:
#             print("Pre-grasp reached, record the relative pre-grasp pose")
#             ee2o_rot = torch.zeros_like(ee_pose[:, 3:])
#             obj_yaw = yaw_quat(obj_pose[:, 3:])
#             ee2o_rot = quat_mul(quat_conjugate(obj_yaw), ee_pose[:, 3:])

#             return True, ee2o_rot
#         else:
#             return False, None            

    
    
#     def replay(self):
#         # Simulation loop
#         start_idx = 0
#         self.lift_chunk_end = False
#         while simulation_app.is_running():
#             chunk_id = 0
#             trajectorys = {
#                     "init_robot_qpos": None,
#                     "init_object_pose.right": None,
#                     "init_object_pose.left": None,
#                     "pre_grasp_pose.right": None,
#                     "pre_grasp_pose.left": None,
#                     "object_offset.right": self.right_offset,
#                     "object_offset.left": self.left_offset,
#                     "action_chunks": {}
#                 }
#             print("Move left robots to the initial pose")
#             init_goal_object_pose_rw, init_goal_object_pose_rb = self.transform_object_pose_cam_to_world(
#                 self.retarget_data['right']['ref_obj_pose'][start_idx, :],
#                 offset=self.right_offset
#             )
#             init_goal_object_pose_lw, init_goal_object_pose_lb = self.transform_object_pose_cam_to_world(
#                 self.retarget_data['left']['ref_obj_pose'][start_idx, :],
#                 offset=self.left_offset
#             )

#             # here we use the ee poses from first right & left keyframe to calculate the initial robot qpos
#             init_robot_qpos_dict, init_robot_qpos = self.move_to_initial_waypoint(
#                 right_goal_ee_pose=self.retarget_data['right']['ref_ee_pose'][self.right_keyframe_indices[0], :], 
#                 left_goal_ee_pose=self.retarget_data['left']['ref_ee_pose'][self.left_keyframe_indices[0], :],
#                 right_object_pose=init_goal_object_pose_rw, 
#                 left_object_pose=init_goal_object_pose_lw,
#             )

#             trajectorys['init_robot_qpos'] = init_robot_qpos_dict
#             trajectorys['init_object_pose.right'] = init_goal_object_pose_rb
#             trajectorys['init_object_pose.left'] = init_goal_object_pose_lb

#             print("Start replaying...")
#             for waypoint_idx in range(start_idx, self.num_waypoints):
#                 if waypoint_idx > getattr(self, f'{self.order[0]}_keyframe_indices')[-1] + 1 and \
#                     waypoint_idx < getattr(self, f'{self.order[1]}_keyframe_indices')[0]:
#                     print('skip: ', waypoint_idx, self.switch_indices, getattr(self, f"{self.order[0]}_keyframe_indices")[-1], getattr(self, f"{self.order[1]}_keyframe_indices")[0], self.keyframe_indices)
#                     continue
#                 else:
#                     print('running: ', waypoint_idx, self.switch_indices, getattr(self, f"{self.order[0]}_keyframe_indices")[-1], getattr(self, f"{self.order[1]}_keyframe_indices")[0], self.keyframe_indices)
                
#                 cur_goal_object_pose_rw, cur_goal_object_pose_rb = self.transform_object_pose_cam_to_world(
#                     self.retarget_data['right']['ref_obj_pose'][waypoint_idx, :],
#                     offset=self.right_offset
#                 )
#                 cur_goal_object_pose_lw, cur_goal_object_pose_lb = self.transform_object_pose_cam_to_world(
#                     self.retarget_data['left']['ref_obj_pose'][waypoint_idx, :],
#                     offset=self.left_offset
#                 )
                
#                 chunk_id = self.update_chunk(
#                     buffer=trajectorys, 
#                     chunk_id=chunk_id, 
#                     right_goal_object_pose=cur_goal_object_pose_rb, 
#                     left_goal_object_pose=cur_goal_object_pose_lb,
#                     waypoint_idx=waypoint_idx
#                 )

#                 active_side = self.order[0] if waypoint_idx <= self.switch_indices+1 else self.order[1]
#                 inactive_side = self.order[1] if waypoint_idx <= self.switch_indices+1 else self.order[0]
#                 if inactive_side == 'right':
#                     right_goal_ee_pose=self.retarget_data['right']['ref_ee_pose'][self.right_keyframe_indices[0], :]
#                     right_object_pose=cur_goal_object_pose_rw
#                     right_hand_ref_qpos=self.retarget_data['right']['ref_hand_qpos'][self.right_keyframe_indices[0], :]

#                     left_goal_ee_pose=self.retarget_data['left']['ref_ee_pose'][waypoint_idx, :]
#                     left_object_pose=cur_goal_object_pose_lw
#                     left_hand_ref_qpos=self.retarget_data['left']['ref_hand_qpos'][waypoint_idx, :]
#                 else:
#                     right_goal_ee_pose=self.retarget_data['right']['ref_ee_pose'][waypoint_idx, :]
#                     right_object_pose=cur_goal_object_pose_rw
#                     right_hand_ref_qpos=self.retarget_data['right']['ref_hand_qpos'][waypoint_idx, :]

#                     left_goal_ee_pose=self.retarget_data['left']['ref_ee_pose'][self.left_keyframe_indices[0], :]
#                     left_object_pose=cur_goal_object_pose_lw
#                     left_hand_ref_qpos=self.retarget_data['left']['ref_hand_qpos'][self.left_keyframe_indices[0], :]
                

#                 actions, pre_grasp_flag, pre_grasp_rot = self.move_to_next_waypoint(
#                     right_goal_ee_pose=right_goal_ee_pose, 
#                     right_object_pose=right_object_pose, 
#                     right_hand_ref_qpos=right_hand_ref_qpos, 
#                     left_goal_ee_pose=left_goal_ee_pose,
#                     left_object_pose=left_object_pose,
#                     left_hand_ref_qpos=left_hand_ref_qpos, 
#                     inactive_side=self.order[1] if waypoint_idx < self.switch_indices else self.order[0]
#                 )
            
#                 trajectorys['action_chunks'][f'chunk_{chunk_id}']['qpos'].extend(actions)

#                 if pre_grasp_flag and trajectorys[f'pre_grasp_pose.{active_side}'] is None:
#                     trajectorys[f'pre_grasp_pose.{active_side}'] = pre_grasp_rot
#                     print(f"{active_side} side pre-grasp pose: {pre_grasp_rot}")

#                 if waypoint_idx == (self.switch_indices+1) and self.object_lifted == True:
#                     print("refresh object lift state")
#                     self.object_lifted = False # fresh state for another side
#                     self.lift_chunk_end = False
#                     self.object_pre_grasp_reached = False
#                     trajectorys['switch_idx'] = chunk_id
#                     print("switch_idx: ", chunk_id)
#                 print()
            
            
            
#             if not self.repeat:
#                 print(trajectorys[f'pre_grasp_pose.right'])
#                 print(trajectorys[f'pre_grasp_pose.left'])
#                 for key in trajectorys["action_chunks"].keys():
#                     print(key, len(trajectorys["action_chunks"][key]['qpos']))
#                     trajectorys["action_chunks"][key]['qpos'] = np.stack(trajectorys["action_chunks"][key]['qpos'], axis=0)
#                 with open(f"{args_cli.data_dir}/{args_cli.seq_name}/{args_cli.trajectory_output_f}", 'wb') as f:
#                     pickle.dump(trajectorys, f)
#                 break
        
#         self.object_lifted = False

# def main():
#     """Main function."""
#     scene_cfg = ReplaySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

#     # Now we are ready!
#     print("[INFO]: Setup complete...")
#     # Run the simulator
#     entry = DemoReplay(scene_cfg=scene_cfg, args_cli=args_cli)
#     entry.replay()


# if __name__ == "__main__":
#     # run the main function
#     main()
#     # close sim app
#     simulation_app.close()



# # @TODO
# # 1. change the replay to reach -> grasp way, similar to the evn
# # 2. change the lift condition from object mass center to keypoints


"""
Author: Yucheng Xu
Date: 4 Mar 2025
Description: 
    This script receive the retargeted robot hand motions and object motions from FoundationPose.
    These motions are replayed in simulation to record robot joint angles with synchorinized right & left robots
Usage:
    ./isaaclab.sh -p devel/replay.py --robot_usd /path/to/robot use --retarget /path/to/retarget file --camera /path/to/camera extrinsics

Before running this script, we need to convert the robot urdf and object model to usd format.
For converting robot urdf, check scripts/tools/convert_urdf.py for details.
For converting object model, import it in the IsaacLab simulator and add physics on it (RigidObject, Mass), then export it as usd.
"""
import os
import argparse
import numpy as np
from glob import glob
import time
import pickle

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_allegro", help="Name of the robot_right.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

parser.add_argument("--data_dir", type=str, default="devel/data")
parser.add_argument("--assets_dir", type=str, default="devel/assets")
parser.add_argument("--seq_name", type=str, default="realsense/hammer/bimanual_assembly_separate_fps30")
parser.add_argument("--object_mesh_right", type=str, default="hammer_v2/hammer_handle.usd", help="path to the right-side object usd in scene")
parser.add_argument("--object_mesh_left", type=str, default="hammer_v2/hammer_head.usd", help="path to the left-side object usd in scene")
parser.add_argument("--trajectory_output_f", type=str, default="trajectory_franka_allegro.pkl", help="path to save the output trajectory")

parser.add_argument("--move_thres_r", type=float, default=0.1, help="threshold for lifting right object")
parser.add_argument("--move_thres_l", type=float, default=0.1, help="threshold for lifting left object")
parser.add_argument("--ik_ee_name_r", type=str, default="right_hand_base_link", help="name of the body, use as the end-effector in IK calculation")
parser.add_argument("--ik_ee_name_l", type=str, default="left_hand_base_link", help="name of the body, use as the end-effector in IK calculation")
parser.add_argument("--min_chunk_steps", type=int, default=50, help="Minimal action steps in a chunk")
parser.add_argument("--save_interval", type=int, default=2)
parser.add_argument("--use_selected_keyframes", type=bool, default=True)
parser.add_argument("--debug", type=int, default=1, help="turn on debug visualization or not")
parser.add_argument("--repeat", action='store_true', default=False, help="replay the trajectory repeatly")
parser.add_argument("--separate", action='store_true', default=True)
parser.add_argument("--switch_point", type=int, default=3)
parser.add_argument("--order", type=str, default='left_right')
parser.add_argument("--kpts_approx", type=str, default='right,')

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms, compute_pose_error, quat_from_matrix
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
from isaaclab.assets.articulation import ArticulationCfg

from isaaclab.sensors import (
    FrameTransformer,
    FrameTransformerCfg,
    OffsetCfg,
    TiledCamera,
    TiledCameraCfg,
    ContactSensor,
    ContactSensorCfg
)

from isaaclab.utils.math import (
    saturate, quat_conjugate, quat_from_angle_axis, 
    quat_mul, sample_uniform, saturate, 
    matrix_from_quat, quat_apply, yaw_quat,
    euler_xyz_from_quat, quat_from_euler_xyz
)

##
# Pre-defined robot configs
##
from isaaclab_assets import FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG   # isort:skip

@configclass
class ReplaySceneCfg(InteractiveSceneCfg):
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(color=(242/255, 238/255, 203/255)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table_r",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, -0.35, 0.0], rot=[1, 0, 0, 0]),
    )
    
    table_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table_l",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.35, 0.0], rot=[1, 0, 0, 0]),
    )
    
    object_r = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_r",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{args_cli.assets_dir}/{args_cli.object_mesh_right}", scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0.2, 0.055], rot=[1, 0, 0, 0]),
    )
    
    object_l = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object_l",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{args_cli.assets_dir}/{args_cli.object_mesh_left}", scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, -0.2, 0.055], rot=[1, 0, 0, 0]),
    )

    right_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
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

    left_ee_frame: FrameTransformerCfg = FrameTransformerCfg(
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
                    pos=[0.1, -0.02, 0.08],
                ),
            ),
        ],
    )

    
    # articulation
    if args_cli.robot == "franka_allegro":
        print("Load franka allegro")
        robot = FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    
    elif args_cli.robot == "xarm_leap":
        print("Load franka allegro")
        robot = XARM_LEAP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, franka_allegro, ur10")


class DemoReplay(object):
    def __init__(self, scene_cfg, args_cli):
        self.scene_cfg = scene_cfg
        self.min_chunk_steps = args_cli.min_chunk_steps
        self.debug = args_cli.debug
        self.separate = args_cli.separate
        self.order = args_cli.order.split("_")
        self.kpts_approx = []
        self.kpts_approx = [k for k in args_cli.kpts_approx.split(',') if k]
        for key in self.kpts_approx:
            kpts_f = f"{args_cli.assets_dir}/{getattr(args_cli, f'object_mesh_{key}').replace('.usd', '.npz')}"
            if not os.path.exists(kpts_f):
                print(f"required keypoint file does not exist, skip keypoint approximation on {key} side")
                self.kpts_approx.remove(key)
                continue
            setattr(
                self, f'{key}_object_kpts', torch.tensor(np.load(kpts_f)['object_keypoints'][:], dtype=torch.float, device='cuda')
            )
        
        self.setup_sim(args_cli=args_cli)
        self.read_data(args_cli=args_cli)

    
    def setup_sim(self, args_cli):
        sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
        self.sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        self.sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
        # Design scene
        self.scene = InteractiveScene(self.scene_cfg)
        # Play the simulator
        self.sim.reset()
        
        self.sim_dt = self.sim.get_physics_dt()

        self.num_envs = args_cli.num_envs
        
        self.robot = self.scene['robot']

        self.object_r = self.scene['object_r']
        self.object_l = self.scene['object_l']
        self.object_root_state_r = self.object_r.data.default_root_state.clone()
        self.object_root_state_l = self.object_l.data.default_root_state.clone()
        
        joint_pos_limits = self.robot.root_physx_view.get_dof_limits().to(self.sim.device)
        self.robot_dof_lower_limits = joint_pos_limits[..., 0]
        self.robot_dof_upper_limits = joint_pos_limits[..., 1]
        
        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller_r = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)
        self.diff_ik_controller_l = DifferentialIKController(diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)
        
        self.ik_commands_r = torch.zeros(self.scene.num_envs, self.diff_ik_controller_r.action_dim, device=self.sim.device)
        self.ik_commands_l = torch.zeros(self.scene.num_envs, self.diff_ik_controller_l.action_dim, device=self.sim.device)
        self.obj_ref_pose_r = torch.zeros(self.scene.num_envs, 7, device=self.sim.device)
        self.obj_ref_pose_l = torch.zeros(self.scene.num_envs, 7, device=self.sim.device)

        self.save_interval = args_cli.save_interval
        
        # Specify robot-specific parameters
        if args_cli.robot == "franka_allegro":
            self.right_arm_entity_cfg = SceneEntityCfg(
                "robot", 
                joint_names=[
                    "right_panda_joint_1", "right_panda_joint_2", "right_panda_joint_3",
                    "right_panda_joint_4", "right_panda_joint_5", "right_panda_joint_6", "right_panda_joint_7",
                    
                ], 
                body_names=["right_panda_link_0", args_cli.ik_ee_name_r]
            )

            self.left_arm_entity_cfg = SceneEntityCfg(
                "robot", 
                joint_names=[
                    "left_panda_joint_1", "left_panda_joint_2", "left_panda_joint_3",
                    "left_panda_joint_4", "left_panda_joint_5", "left_panda_joint_6", "left_panda_joint_7",
                ], 
                body_names=["left_panda_link_0", args_cli.ik_ee_name_l]
            )

            self.right_hand_entity_cfg = SceneEntityCfg(
                "robot", 
                joint_names=[
                    "right_index_joint_0", "right_index_joint_1", "right_index_joint_2", "right_index_joint_3", 
                    "right_middle_joint_0", "right_middle_joint_1", "right_middle_joint_2", "right_middle_joint_3", 
                    "right_ring_joint_0", "right_ring_joint_1", "right_ring_joint_2", "right_ring_joint_3", 
                    "right_thumb_joint_0", "right_thumb_joint_1", "right_thumb_joint_2", "right_thumb_joint_3",
                ], 
            )

            self.left_hand_entity_cfg = SceneEntityCfg(
                "robot", 
                joint_names=[
                    "left_index_joint_0", "left_index_joint_1", "left_index_joint_2", "left_index_joint_3", 
                    "left_middle_joint_0", "left_middle_joint_1", "left_middle_joint_2", "left_middle_joint_3", 
                    "left_ring_joint_0", "left_ring_joint_1", "left_ring_joint_2", "left_ring_joint_3", 
                    "left_thumb_joint_0", "left_thumb_joint_1", "left_thumb_joint_2", "left_thumb_joint_3", 
                ], 
            )

            self.right_robot_entity_cfg = SceneEntityCfg(
                "robot", 
                joint_names=[
                    "right_panda_joint_1", "right_panda_joint_2", "right_panda_joint_3",
                    "right_panda_joint_4", "right_panda_joint_5", "right_panda_joint_6", "right_panda_joint_7",
                    "right_index_joint_0", "right_index_joint_1", "right_index_joint_2", "right_index_joint_3", 
                    "right_middle_joint_0", "right_middle_joint_1", "right_middle_joint_2", "right_middle_joint_3", 
                    "right_ring_joint_0", "right_ring_joint_1", "right_ring_joint_2", "right_ring_joint_3", 
                    "right_thumb_joint_0", "right_thumb_joint_1", "right_thumb_joint_2", "right_thumb_joint_3",
                ], 
            )

            self.left_robot_entity_cfg = SceneEntityCfg(
                "robot", 
                joint_names=[
                    "left_panda_joint_1", "left_panda_joint_2", "left_panda_joint_3",
                    "left_panda_joint_4", "left_panda_joint_5", "left_panda_joint_6", "left_panda_joint_7",
                    "left_index_joint_0", "left_index_joint_1", "left_index_joint_2", "left_index_joint_3", 
                    "left_middle_joint_0", "left_middle_joint_1", "left_middle_joint_2", "left_middle_joint_3", 
                    "left_ring_joint_0", "left_ring_joint_1", "left_ring_joint_2", "left_ring_joint_3", 
                    "left_thumb_joint_0", "left_thumb_joint_1", "left_thumb_joint_2", "left_thumb_joint_3", 
                ], 
            )
        
        else:
            raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda")
        
        
        self.init_robot_qpos = None
        self.object_lifted = False
        self.object_pre_grasp_reached = False
        self.move_thres_r = args_cli.move_thres_r
        self.move_thres_l = args_cli.move_thres_l
        self.repeat = args_cli.repeat
        self.use_selected_keyframes = args_cli.use_selected_keyframes
        
        # Resolving the scene entities
        self.right_arm_entity_cfg.resolve(self.scene)
        self.right_hand_entity_cfg.resolve(self.scene)
        self.right_robot_entity_cfg.resolve(self.scene)

        self.left_arm_entity_cfg.resolve(self.scene)
        self.left_hand_entity_cfg.resolve(self.scene)
        self.left_robot_entity_cfg.resolve(self.scene)

        self.right_arm_joint_ids = self.right_arm_entity_cfg.joint_ids
        self.right_arm_joint_names = self.right_arm_entity_cfg.joint_names
        self.right_hand_joint_ids = self.right_hand_entity_cfg.joint_ids
        self.right_robot_joint_ids = self.right_robot_entity_cfg.joint_ids
        self.right_robot_joint_names = self.right_robot_entity_cfg.joint_names

        self.left_arm_joint_ids = self.left_arm_entity_cfg.joint_ids
        self.left_arm_joint_names = self.left_arm_entity_cfg.joint_names
        self.left_hand_joint_ids = self.left_hand_entity_cfg.joint_ids
        self.left_robot_joint_ids = self.left_robot_entity_cfg.joint_ids
        self.left_robot_joint_names = self.left_robot_entity_cfg.joint_names

        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index. This is because
        # the root body is not included in the returned Jacobians.
        # In this replay script, we only setup on body in the robot cfg
        self.ee_body_id_r = self.right_arm_entity_cfg.body_ids[1]
        self.ee_body_id_l = self.left_arm_entity_cfg.body_ids[1]
        self.root_body_id_r = self.right_arm_entity_cfg.body_ids[0]
        self.root_body_id_l = self.left_arm_entity_cfg.body_ids[0]
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx_r = self.ee_body_id_r - 1
            self.ee_jacobi_idx_l = self.ee_body_id_l - 1
        else:
            self.ee_jacobi_idx_r = self.ee_body_id_r
            self.ee_jacobi_idx_l = self.ee_body_id_l
            
        self.joint_pos = self.robot.data.default_joint_pos.clone()
        self.joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(self.joint_pos, self.joint_vel)
        self.robot.reset()
        self.right_robot_base_offset = (self.robot.data.body_state_w[:, self.right_arm_entity_cfg.body_ids[0], :3] - self.scene.env_origins)[0, :]
        self.left_robot_base_offset = (self.robot.data.body_state_w[:, self.left_arm_entity_cfg.body_ids[0], :3] - self.scene.env_origins)[0, :]
        
        self.scene.write_data_to_sim()
        # perform step
        self.sim.step()
        
        # update buffers
        self.scene.update(self.sim_dt)

        if args_cli.debug:
            frame_marker_cfg = FRAME_MARKER_CFG.copy()
            frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            self.ee_r_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_r_current"))
            self.goal_r_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_r_goal"))
            self.ee_l_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_l_current"))
            self.goal_l_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_l_goal"))
            self.obj_r_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/object_r"))
            self.obj_l_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/object_l"))
    
    def read_data(self, args_cli):
        retarget_f = f"{args_cli.data_dir}/{args_cli.seq_name}/retarget.npz"
        retarget_bimanual = np.load(retarget_f, allow_pickle=True)
        self.right_offset = np.asarray([0.65, 0.0, 0.10])
        self.left_offset = np.asarray([0.65, 0.0, 0.10])
        self.right_ee_offset = np.asarray([0.6, 0.0, 0.10])
        self.left_ee_offset = np.asarray([0.65, 0.0, 0.08])
        self.retarget_data = {}
        for key in retarget_bimanual.keys():
            retarget = retarget_bimanual[key].item()
            ref_ee_pos = retarget['base_transl']
            ref_ee_quat = retarget['base_quat']
            ref_hand_qpos = retarget['qpos'][:, 6:]
            
            ref_ee_pos += getattr(self, f'{key}_ee_offset')
            # Since we got xyzw format in retargeting, convert it to wxyc for IsaacLab
            ref_ee_quat = np.stack([ref_ee_quat[:, 3], ref_ee_quat[:, 0], ref_ee_quat[:, 1], ref_ee_quat[:, 2]], axis=1)
            ref_ee_pose = np.concatenate([ref_ee_pos, ref_ee_quat], axis=1)

            # Convert to torch.Tensor   
            ref_ee_pos = torch.tensor(ref_ee_pos, device=self.sim.device)
            ref_ee_quat = torch.tensor(ref_ee_quat, device=self.sim.device)
            ref_ee_pose = torch.tensor(ref_ee_pose, device=self.sim.device)
            ref_hand_qpos = torch.tensor(ref_hand_qpos, device=self.sim.device).float()

            # Object motions
            assert hasattr(args_cli, f"object_mesh_{key}")
            object_name = getattr(args_cli, f"object_mesh_{key}").split("/")[-1].split(".")[0]
            assert os.path.exists(f"{args_cli.data_dir}/{args_cli.seq_name}/processed/object/{object_name}_pose_cam.refine.npy")
            object_poses = np.load(f"{args_cli.data_dir}/{args_cli.seq_name}/processed/object/{object_name}_pose_cam.refine.npy")
            self.num_waypoints = object_poses.shape[0]

            self.retarget_data[key] = {
                'ref_ee_pose': ref_ee_pose,
                'ref_hand_qpos': ref_hand_qpos,
                'ref_obj_pose': object_poses
            }
        
        # Camera extrinsic parameters
        camera_pose = np.load(f"{args_cli.data_dir}/{args_cli.seq_name}/camera_extrinsic.npz")
        extrinsic_matrix = np.eye(4)
        extrinsic_matrix[:3, :3] = camera_pose['R']
        extrinsic_matrix[:3, 3] = camera_pose['T'].reshape(3)
        self.extrinsic_matrix = np.linalg.inv(extrinsic_matrix)

        # read keyframe indices if exist
        all_images = sorted(os.listdir(f"{args_cli.data_dir}/{args_cli.seq_name}/rgb"))
        keyframes = sorted(os.listdir(f"{args_cli.data_dir}/{args_cli.seq_name}/kf_rgb"))
        self.keyframe_indices = np.asarray([all_images.index(n) for n in keyframes])
        self.num_keyframes = len(self.keyframe_indices)


        # in async mode, we record keyframes for left and right separately
        right_keyframes = sorted(os.listdir(f"{args_cli.data_dir}/{args_cli.seq_name}/right_kf_rgb"))
        self.right_keyframe_indices = np.asarray([all_images.index(n) for n in right_keyframes])

        left_keyframes = sorted(os.listdir(f"{args_cli.data_dir}/{args_cli.seq_name}/left_kf_rgb"))
        self.left_keyframe_indices = np.asarray([all_images.index(n) for n in left_keyframes])

        self.switch_indices = getattr(self, f'{self.order[0]}_keyframe_indices')[-1]
        print(f"start with {self.order[0]} side")
        print(f"Switch to {self.order[1]} side after {self.switch_indices}-th waypoint")


    def transform_object_pose_cam_to_world(self, pose, offset):
        # camera -> robot base
        pose_b = np.dot(self.extrinsic_matrix, pose)
        transl_b = pose_b[:3, 3]
        quat_b = quat_from_matrix(
            torch.tensor(pose_b[:3, :3], device=self.sim.device)
        ).cpu().numpy()
        
        # robot base -> sim world
        transl_w = torch.tensor(transl_b+offset, device=self.sim.device) + self.scene.env_origins
        quat_w = torch.tensor(quat_b, device=self.sim.device).unsqueeze(0).repeat(transl_w.shape[0], 1)
        
        return torch.cat([transl_w, quat_w], dim=-1), np.concatenate([transl_b, quat_b], axis=0)
    

    def check_reached(self, pos_error, ang_error, pos_thres=0.005, ang_thres=0.05):
        if torch.linalg.norm(pos_error) < pos_thres and torch.abs(ang_error).sum() < ang_thres:
            return True
        else:
            return False
    

    def calculate_right_qpos_ik(self):
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx_r, :, self.right_arm_joint_ids]
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id_r, :7]
        root_pose_w = self.robot.data.body_state_w[:, self.root_body_id_r, :7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )

        position_error, axis_angle_error = compute_pose_error(
                    ee_pos_b, ee_quat_b, self.ik_commands_r[:, :3], self.ik_commands_r[:, 3:7], rot_error_type="axis_angle"
                )
        joint_pos = self.robot.data.joint_pos[:, self.right_arm_joint_ids]
        joint_pos_des = self.diff_ik_controller_r.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        joint_pos_des = saturate(
                    joint_pos_des.clone(),
                    self.robot_dof_lower_limits[:, self.right_arm_joint_ids],
                    self.robot_dof_upper_limits[:, self.right_arm_joint_ids]
                )
        
        # print(position_error, axis_angle_error)
        # For now we use the same target pose for all envs, so just simply mean the errors
        position_error = position_error.mean(dim=0)
        axis_angle_error = axis_angle_error.mean(dim=0)

        reached = self.check_reached(position_error, axis_angle_error)
        if self.debug:
            self.ee_r_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            self.goal_r_marker.visualize(
                self.ik_commands_r[:, 0:3] + self.right_robot_base_offset + self.scene.env_origins, 
                self.ik_commands_r[:, 3:7]
            )
            self.obj_r_marker.visualize(self.object_root_state_r[:, 0:3], self.object_root_state_r[:, 3:7])

        return joint_pos_des, reached


    def calculate_left_qpos_ik(self):
        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx_l, :, self.left_arm_joint_ids]
        ee_pose_w = self.robot.data.body_state_w[:, self.ee_body_id_l, :7]
        root_pose_w = self.robot.data.body_state_w[:, self.root_body_id_l, :7]
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
                )

        position_error, axis_angle_error = compute_pose_error(
                    ee_pos_b, ee_quat_b, self.ik_commands_l[:, :3], self.ik_commands_l[:, 3:7], rot_error_type="axis_angle"
                )
        joint_pos = self.robot.data.joint_pos[:, self.left_arm_joint_ids]
        joint_pos_des = self.diff_ik_controller_l.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
        joint_pos_des = saturate(
                    joint_pos_des.clone(),
                    self.robot_dof_lower_limits[:, self.left_arm_joint_ids],
                    self.robot_dof_upper_limits[:, self.left_arm_joint_ids]
                )
        
        # print(position_error, axis_angle_error)
        # For now we use the same target pose for all envs, so just simply mean the errors
        position_error = position_error.mean(dim=0)
        axis_angle_error = axis_angle_error.mean(dim=0)

        reached = self.check_reached(position_error, axis_angle_error)

        if self.debug:
            self.ee_l_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
            self.goal_l_marker.visualize(
                self.ik_commands_l[:, 0:3] + self.left_robot_base_offset + self.scene.env_origins, 
                self.ik_commands_l[:, 3:7]
            )

            self.obj_l_marker.visualize(self.object_root_state_l[:, 0:3], self.object_root_state_l[:, 3:7])

        return joint_pos_des, reached
    

    def calculate_kpts_approx_pose(self, pose, kpts, offset):
        """
        Calculates a canonical pose where the Y-axis is aligned with the cylinder's length.
        """
        # 1. Transform keypoints to world frame (this part is correct)
        kpts_w = torch.einsum(
                'nij,nmj->nmi', 
                matrix_from_quat(pose[:, 3:]), # Assuming quat is [w,x,y,z] or library handles it
                kpts.clone().unsqueeze(0).repeat(self.num_envs, 1, 1)
            ) + pose[:, :3].unsqueeze(1)
        
        # --- FIX STARTS HERE ---

        # For clarity, let's use the midpoint for translation (more robust)
        translation = (kpts_w[:, 0] + kpts_w[:, -1]) / 2.0

        # 2. Define the new Y-axis as the cylinder's principal axis
        # Renamed from 'xaxis' to 'yaxis'
        yaxis = kpts_w[:, 0] - kpts_w[:, -1]
        yaxis = yaxis / torch.linalg.norm(yaxis, dim=-1, keepdim=True)
        
        # 3. Define the new X-axis to be perpendicular to the new Y-axis and world Z
        # This makes the new X-axis point out to the "side" of the cylinder
        world_up = torch.zeros_like(yaxis)
        world_up[:, 2] = 1

        # Use a robustness check for vertical cylinders
        dot_prod = torch.sum(yaxis * world_up, dim=1) # Batched dot product
        is_vertical = torch.abs(dot_prod) > 0.999
        
        # If vertical, use world X as reference. Otherwise, use world Z (up).
        world_ref_for_vertical = torch.zeros_like(yaxis)
        world_ref_for_vertical[:, 0] = 1 # World X-axis
        ref_vec = torch.where(is_vertical.unsqueeze(-1), world_ref_for_vertical, world_up)
        
        xaxis = torch.cross(yaxis, ref_vec, dim=-1)
        xaxis = xaxis / torch.linalg.norm(xaxis, dim=-1, keepdim=True)

        # 4. Define the new Z-axis to complete the right-handed frame
        # The order is important: cross(X, Y) = Z
        zaxis = torch.cross(xaxis, yaxis, dim=-1)
        # This should already be normalized

        # 5. Assemble the rotation matrix with the correct column assignments
        rot = torch.eye(3, device=pose.device).unsqueeze(0).repeat(self.num_envs, 1, 1)

        rot[:, :, 0] = xaxis  # First column is the new X-axis
        rot[:, :, 1] = yaxis  # Second column is the new Y-axis
        rot[:, :, 2] = zaxis  # Third column is the new Z-axis
        
        # 6. Create the final pose in [pos, quat] format (this part is correct)
        final_pose = torch.zeros_like(pose)
        final_pose[:, :3] = translation - self.scene.env_origins - offset
        final_pose[:, 3:7] = quat_from_matrix(rot) # Assumes quat is [x,y,z,w]

        return final_pose
    
    def calculate_local_pose(self, pose, offset):
        local_obj_pose = pose.clone()
        local_obj_pose[:, :3] -= offset
        local_obj_pose[:, :3] -= self.scene.env_origins

        return local_obj_pose
    

    def pre_grasp_check(self, ee_pose, obj_pose, threshold):
        """
        Function to check if the hand ee is close to the object and 
        then record the relative rotation between hand ee and object
        """
        ee2o_dist = torch.norm(ee_pose[:, :3] - obj_pose[:, :3], p=2, dim=-1)
        if ee2o_dist < threshold:
            print("Pre-grasp reached, record the relative pre-grasp pose")
            ee2o_rot = torch.zeros_like(ee_pose[:, 3:])
            obj_yaw = yaw_quat(obj_pose[:, 3:])
            ee2o_rot = quat_mul(quat_conjugate(obj_yaw), ee_pose[:, 3:])

            return True, ee2o_dist, ee2o_rot
        else:
            return False, ee2o_dist, None  

    
    def move_to_initial_waypoint(
        self, 
        right_goal_ee_pose, left_goal_ee_pose,
        right_object_pose, left_object_pose,
    ):
        
        self.diff_ik_controller_r.reset()
        self.ik_commands_r[:] = right_goal_ee_pose
        self.ik_commands_r[:, :3] -= self.right_robot_base_offset
        self.diff_ik_controller_r.set_command(self.ik_commands_r)

        self.diff_ik_controller_l.reset()
        self.ik_commands_l[:] = left_goal_ee_pose
        self.ik_commands_l[:, :3] -= self.left_robot_base_offset
        self.diff_ik_controller_l.set_command(self.ik_commands_l)

        success = False
        while not success:
            # object pose
            self.object_root_state_r[:, :7] = right_object_pose
            self.object_root_state_l[:, :7] = left_object_pose 
            
            joint_pos_des_r, reached_r = self.calculate_right_qpos_ik()
            joint_pos_des_l, reached_l = self.calculate_left_qpos_ik()
            self.joint_pos[:, self.right_arm_joint_ids] = joint_pos_des_r
            self.joint_pos[:, self.left_arm_joint_ids] = joint_pos_des_l
            if reached_r and reached_l:
                init_robot_qpos_r = joint_pos_des_r[0, :].cpu().numpy()
                init_robot_qpos_l = joint_pos_des_l[0, :].cpu().numpy()
                init_robot_qpos_dict = {}
                init_robot_qpos_dict.update(
                    {k:v for (k, v) in zip(self.right_arm_joint_names, init_robot_qpos_r)}
                )
                init_robot_qpos_dict.update(
                    {k:v for (k, v) in zip(self.left_arm_joint_names, init_robot_qpos_l)}
                )


                init_robot_qpos = self.robot.data.default_joint_pos.clone()[0, :]
                init_robot_qpos[self.right_arm_joint_ids] = joint_pos_des_r[0, :]
                init_robot_qpos[self.left_arm_joint_ids] = joint_pos_des_l[0, :]
                success = True
            else:
                self.robot.set_joint_position_target(
                    joint_pos_des_r, 
                    joint_ids=self.right_arm_joint_ids
                )
                self.robot.set_joint_position_target(
                    joint_pos_des_l, 
                    joint_ids=self.left_arm_joint_ids
                )
                self.object_r.write_root_pose_to_sim(self.object_root_state_r[:, :7])
                self.object_l.write_root_pose_to_sim(self.object_root_state_l[:, :7])
                
                self.scene.write_data_to_sim()
                # perform step
                self.sim.step()
                
                # update buffers
                self.scene.update(self.sim_dt)
            
        
        return init_robot_qpos_dict, init_robot_qpos
                

    def move_to_next_waypoint(
        self, 
        right_goal_ee_pose, left_goal_ee_pose,
        right_object_pose, left_object_pose,
        right_hand_ref_qpos, left_hand_ref_qpos,
        inactive_side='right'
    ):
        self.diff_ik_controller_r.reset()
        self.ik_commands_r[:] = right_goal_ee_pose
        self.ik_commands_r[:, :3] -= self.right_robot_base_offset
        self.diff_ik_controller_r.set_command(self.ik_commands_r)

        self.diff_ik_controller_l.reset()
        self.ik_commands_l[:] = left_goal_ee_pose
        self.ik_commands_l[:, :3] -= self.left_robot_base_offset

        self.diff_ik_controller_l.set_command(self.ik_commands_l)

        right_hand_qpos = self.joint_pos[:, self.right_hand_joint_ids]
        left_hand_qpos = self.joint_pos[:, self.left_hand_joint_ids]

        right_hand_qpos[:] = right_hand_ref_qpos * 0.95
        left_hand_qpos[:] = left_hand_ref_qpos * 0.95

        success = False
        actions = []
        count = 0
        pre_grasp_flag = None
        pre_grasp_rot = None
        while not success:
            # calculate canonical pose of the object
            if not self.object_pre_grasp_reached:
                if inactive_side == 'right':
                    if 'left' in self.kpts_approx:
                        obj_pose = self.calculate_kpts_approx_pose(
                            left_object_pose.float(), self.left_object_kpts,
                            offset=self.left_robot_base_offset
                        )
                    else:
                        obj_pose = self.calculate_local_pose(left_object_pose.float(), self.left_robot_base_offset)
                    
                    ee_pose = torch.zeros_like(obj_pose)
                    ee_pose[:, :3] = self.scene.sensors["left_ee_frame"].data.target_pos_source[..., 0, :] - self.left_robot_base_offset
                    ee_pose[:, 3:] = self.scene.sensors["left_ee_frame"].data.target_quat_source[..., 0, :]

                    # object pose
                    self.object_root_state_r[:, :7] = right_object_pose
                    self.object_root_state_l[:, :3] = obj_pose[:, :3] + self.scene.env_origins + self.left_robot_base_offset
                    self.object_root_state_l[:, 3:7] = obj_pose[:, 3:]

                else:
                    if 'right' in self.kpts_approx:
                        obj_pose = self.calculate_kpts_approx_pose(
                            right_object_pose.float(), 
                            self.right_object_kpts,
                            offset=self.right_robot_base_offset
                        )
                    else:
                        obj_pose = self.calculate_local_pose(right_object_pose, self.right_robot_base_offset)

                    ee_pose = torch.zeros_like(obj_pose)
                    ee_pose[:, :3] = self.scene.sensors["right_ee_frame"].data.target_pos_source[..., 0, :] - self.right_robot_base_offset
                    ee_pose[:, 3:] = self.scene.sensors["right_ee_frame"].data.target_quat_source[..., 0, :]

                    self.object_root_state_l[:, :7] = left_object_pose
                    self.object_root_state_r[:, :3] = obj_pose[:, :3] + self.scene.env_origins + self.right_robot_base_offset
                    self.object_root_state_r[:, 3:7] = obj_pose[:, 3:]

                _pre_grasp_flag, ee2o_dist, _pre_grasp_rot = self.pre_grasp_check(
                    ee_pose, obj_pose, threshold=0.08
                )
                
                if _pre_grasp_flag:
                    pre_grasp_flag = _pre_grasp_flag
                    pre_grasp_rot = _pre_grasp_rot
                    self.object_pre_grasp_reached = True
            else:
                # object pose
                self.object_root_state_r[:, :7] = right_object_pose
                self.object_root_state_l[:, :7] = left_object_pose 
            
            joint_pos_des_r, reached_r = self.calculate_right_qpos_ik()
            joint_pos_des_l, reached_l = self.calculate_left_qpos_ik()
            self.joint_pos[:, self.right_arm_joint_ids] = joint_pos_des_r
            self.joint_pos[:, self.left_arm_joint_ids] = joint_pos_des_l
            self.joint_pos[:, self.right_hand_joint_ids] = right_hand_ref_qpos
            self.joint_pos[:, self.left_hand_joint_ids] = left_hand_ref_qpos
            
            reached = reached_l if inactive_side == 'right' else reached_r
            if reached:
                actions.append(self.robot.data.joint_pos[0, :].cpu().numpy())
                success = True
            else:
                self.robot.set_joint_position_target(
                    joint_pos_des_r, 
                    joint_ids=self.right_arm_joint_ids
                )
                self.robot.set_joint_position_target(
                right_hand_qpos,
                joint_ids=self.right_hand_joint_ids
                )
                    
                self.robot.set_joint_position_target(
                    joint_pos_des_l, 
                    joint_ids=self.left_arm_joint_ids
                )
                
                self.robot.set_joint_position_target(
                    left_hand_qpos,
                    joint_ids=self.left_hand_joint_ids
                )

                self.object_r.write_root_pose_to_sim(self.object_root_state_r[:, :7])
                self.object_l.write_root_pose_to_sim(self.object_root_state_l[:, :7])
                if count % self.save_interval == 0:
                    actions.append(self.robot.data.joint_pos[0, :].cpu().numpy())
                
                self.scene.write_data_to_sim()
                # perform step
                self.sim.step()
                
                # update buffers
                self.scene.update(self.sim_dt)

                count += 1
            
        
        return actions, pre_grasp_flag, pre_grasp_rot
        
    
    def update_chunk(self, buffer, chunk_id, right_goal_object_pose, left_goal_object_pose, waypoint_idx):
        def create_new_chunk(chunk_id, right_obj_pose, left_obj_pose):
            buffer['action_chunks'][f'chunk_{chunk_id}'] = {
                        'goal_object_pose.right': right_obj_pose,
                        'goal_object_pose.left': left_obj_pose,
                        'qpos': []
                    }
        
        def merge_current_chunk(chunk_id, right_obj_pose, left_obj_pose):
            buffer['action_chunks'][f'chunk_{chunk_id}']['goal_object_pose.right'] = right_obj_pose
            buffer['action_chunks'][f'chunk_{chunk_id}']['goal_object_pose.left'] = left_obj_pose


        if waypoint_idx == getattr(self, f'{self.order[0]}_keyframe_indices')[0]: # create the first action chunk for the init arm
            create_new_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
        else:
            if not self.object_lifted: 
                init_right_object_pose = buffer['init_object_pose.right']
                init_left_object_pose = buffer['init_object_pose.left']
                diff_r = np.abs(init_right_object_pose[2] - right_goal_object_pose[2])
                diff_l = np.abs(init_left_object_pose[2] - left_goal_object_pose[2])
                if not self.separate:
                    lifted = diff_r > self.move_thres_r and diff_l > self.move_thres_l
                else:
                    lifted = diff_r > self.move_thres_r or diff_l > self.move_thres_l
                if lifted:
                    if not self.lift_chunk_end: # a little hack to make sure the lift-up chunk covers all lift up action
                        merge_current_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
                        self.lift_chunk_end = True
                    else:
                        chunk_id += 1
                        self.object_lifted = True
                        print(f"create {chunk_id} chunk, object lifted")
                        create_new_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
                else:
                    merge_current_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
            else:
                if not self.use_selected_keyframes:
                    if len(buffer['action_chunks'][f'chunk_{chunk_id}']['qpos']) > self.min_chunk_steps: # current chunk contains enough steps, create a new action chunk
                        chunk_id += 1
                        create_new_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
                    else: # current chunk does not have enough steps, merge with the next chunk
                        merge_current_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
                else: # use pre-selected keyframes
                    if (waypoint_idx-1) in self.keyframe_indices and (waypoint_idx-1) != getattr(self, f'{self.order[-1]}_keyframe_indices')[0]:
                        chunk_id += 1
                        print(f"create {chunk_id} chunk, keyframe reached")
                        create_new_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)
                    else: 
                        merge_current_chunk(chunk_id=chunk_id, right_obj_pose=right_goal_object_pose, left_obj_pose=left_goal_object_pose)

            
        return chunk_id
    
    
    def replay(self):
        # Simulation loop
        start_idx = 0
        self.lift_chunk_end = False
        while simulation_app.is_running():
            chunk_id = 0
            trajectorys = {
                    "init_robot_qpos": None,
                    "init_object_pose.right": None,
                    "init_object_pose.left": None,
                    "object_offset.right": self.right_offset,
                    "object_offset.left": self.left_offset,
                    "pre_grasp_pose.right": None,
                    "pre_grasp_pose.left": None,
                    "action_chunks": {}
                }
            print("Move left robots to the initial pose")
            cur_goal_object_pose_rw, cur_goal_object_pose_rb = self.transform_object_pose_cam_to_world(
                self.retarget_data['right']['ref_obj_pose'][start_idx, :],
                offset=self.right_offset
            )
            cur_goal_object_pose_lw, cur_goal_object_pose_lb = self.transform_object_pose_cam_to_world(
                self.retarget_data['left']['ref_obj_pose'][start_idx, :],
                offset=self.left_offset
            )

            # here we use the ee poses from first right & left keyframe to calculate the initial robot qpos
            init_robot_qpos_dict, init_robot_qpos = self.move_to_initial_waypoint(
                right_goal_ee_pose=self.retarget_data['right']['ref_ee_pose'][self.right_keyframe_indices[0], :], 
                left_goal_ee_pose=self.retarget_data['left']['ref_ee_pose'][self.left_keyframe_indices[0], :],
                right_object_pose=cur_goal_object_pose_rw, 
                left_object_pose=cur_goal_object_pose_lw,
            )

            trajectorys['init_robot_qpos'] = init_robot_qpos_dict
            trajectorys['init_object_pose.right'] = cur_goal_object_pose_rb
            trajectorys['init_object_pose.left'] = cur_goal_object_pose_lb

            print("Start replaying...")
            
            for waypoint_idx in range(start_idx, self.num_waypoints):
                if waypoint_idx > getattr(self, f'{self.order[0]}_keyframe_indices')[-1] + 1 and \
                    waypoint_idx < getattr(self, f'{self.order[1]}_keyframe_indices')[0]:
                    print('skip: ', waypoint_idx, self.switch_indices, getattr(self, f"{self.order[0]}_keyframe_indices")[-1], getattr(self, f"{self.order[1]}_keyframe_indices")[0], self.keyframe_indices)
                    continue
                else:
                    print('running: ', waypoint_idx, self.switch_indices, getattr(self, f"{self.order[0]}_keyframe_indices")[-1], getattr(self, f"{self.order[1]}_keyframe_indices")[0], self.keyframe_indices)
                
                cur_goal_object_pose_rw, cur_goal_object_pose_rb = self.transform_object_pose_cam_to_world(
                    self.retarget_data['right']['ref_obj_pose'][waypoint_idx, :],
                    offset=self.right_offset
                )
                cur_goal_object_pose_lw, cur_goal_object_pose_lb = self.transform_object_pose_cam_to_world(
                    self.retarget_data['left']['ref_obj_pose'][waypoint_idx, :],
                    offset=self.left_offset
                )
                
                chunk_id = self.update_chunk(
                    buffer=trajectorys, 
                    chunk_id=chunk_id, 
                    right_goal_object_pose=cur_goal_object_pose_rb, 
                    left_goal_object_pose=cur_goal_object_pose_lb,
                    waypoint_idx=waypoint_idx
                )

                inactive_side = self.order[1] if waypoint_idx <= self.switch_indices+1 else self.order[0]
                active_side = self.order[0] if waypoint_idx <= self.switch_indices+1 else self.order[1]
                if inactive_side == 'right':
                    right_goal_ee_pose=self.retarget_data['right']['ref_ee_pose'][self.right_keyframe_indices[0], :]
                    right_object_pose=cur_goal_object_pose_rw
                    right_hand_ref_qpos=self.retarget_data['right']['ref_hand_qpos'][self.right_keyframe_indices[0], :]

                    left_goal_ee_pose=self.retarget_data['left']['ref_ee_pose'][waypoint_idx, :]
                    left_object_pose=cur_goal_object_pose_lw
                    left_hand_ref_qpos=self.retarget_data['left']['ref_hand_qpos'][waypoint_idx, :]
                
                else:
                    right_goal_ee_pose=self.retarget_data['right']['ref_ee_pose'][waypoint_idx, :]
                    right_object_pose=cur_goal_object_pose_rw
                    right_hand_ref_qpos=self.retarget_data['right']['ref_hand_qpos'][waypoint_idx, :]

                    left_goal_ee_pose=self.retarget_data['left']['ref_ee_pose'][self.left_keyframe_indices[0], :]
                    left_object_pose=cur_goal_object_pose_lw
                    left_hand_ref_qpos=self.retarget_data['left']['ref_hand_qpos'][self.left_keyframe_indices[0], :]
                

                actions, pre_grasp_flag, pre_grasp_rot = self.move_to_next_waypoint(
                    right_goal_ee_pose=right_goal_ee_pose, 
                    right_object_pose=right_object_pose, 
                    right_hand_ref_qpos=right_hand_ref_qpos, 
                    left_goal_ee_pose=left_goal_ee_pose,
                    left_object_pose=left_object_pose,
                    left_hand_ref_qpos=left_hand_ref_qpos, 
                    inactive_side=self.order[1] if waypoint_idx < self.switch_indices else self.order[0]
                )
            
                trajectorys['action_chunks'][f'chunk_{chunk_id}']['qpos'].extend(actions)

                if pre_grasp_flag and trajectorys[f'pre_grasp_pose.{active_side}'] is None:
                    trajectorys[f'pre_grasp_pose.{active_side}'] = pre_grasp_rot
                    print(f"{active_side} side pre-grasp pose: {pre_grasp_rot}")
                

                if waypoint_idx == (self.switch_indices+1) and self.object_lifted == True:
                    print("refresh object lift state")
                    self.object_lifted = False # fresh state for another side
                    self.lift_chunk_end = False
                    self.object_pre_grasp_reached = False
                    trajectorys['switch_idx'] = chunk_id
                    print("switch_idx: ", chunk_id)
                print()
            
            
            
            if not self.repeat:
                for key in trajectorys["action_chunks"].keys():
                    print(key, len(trajectorys["action_chunks"][key]['qpos']))
                    trajectorys["action_chunks"][key]['qpos'] = np.stack(trajectorys["action_chunks"][key]['qpos'], axis=0)
                with open(f"{args_cli.data_dir}/{args_cli.seq_name}/{args_cli.trajectory_output_f}", 'wb') as f:
                    pickle.dump(trajectorys, f)
                break
        
        self.object_lifted = False
        self.object_pre_grasp_reached = False

def main():
    """Main function."""
    scene_cfg = ReplaySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)

    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    entry = DemoReplay(scene_cfg=scene_cfg, args_cli=args_cli)
    entry.replay()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()



# @TODO
# 1. change the replay to reach -> grasp way, similar to the evn
# 2. change the lift condition from object mass center to keypoints