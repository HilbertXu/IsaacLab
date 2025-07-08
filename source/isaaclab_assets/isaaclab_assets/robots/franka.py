# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/Robots/FrankaPanda/franka_panda.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=12, solver_velocity_iteration_count=1
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=2e3,
            damping=1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_ALLEGRO_RIGHT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/Robots/FrankaAllegro/franka_allegro_v2.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=12, 
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint_1": 0.0,
            "panda_joint_2": -0.569,
            "panda_joint_3": 0.0,
            "panda_joint_4": -2.,
            "panda_joint_5": 0.0,
            "panda_joint_6": 3.037,
            "panda_joint_7": 0.741,
            "index_joint_0": 0.0,
            "index_joint_1": 0.0,
            "index_joint_2": 0.0,
            "index_joint_3": 0.0,
            "middle_joint_0": 0.0,
            "middle_joint_1": 0.0,
            "middle_joint_2": 0.0,
            "middle_joint_3": 0.0,
            "ring_joint_0": 0.0,
            "ring_joint_1": 0.0,
            "ring_joint_2": 0.0,
            "ring_joint_3": 0.0,
            "thumb_joint_0": 0.0,
            "thumb_joint_1": 0.0,
            "thumb_joint_2": 0.0,
            "thumb_joint_3": 0.0,
        },
        # pos=[0.0, -0.5, 0.0], 
        # rot=[1, 0, 0, 0]
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint_[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint_[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "allegro_index_finger": ImplicitActuatorCfg(
            joint_names_expr=["index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "allegro_middle_finger": ImplicitActuatorCfg(
            joint_names_expr=["middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "allegro_ring_finger": ImplicitActuatorCfg(
            joint_names_expr=["ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "allegro_thumb_finger": ImplicitActuatorCfg(
            joint_names_expr=["thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


FRANKA_ALLEGRO_LEFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/Robots/FrankaAllegroLeft/franka_allegro_left.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=12, 
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint_1": 0.0,
            "panda_joint_2": -0.569,
            "panda_joint_3": 0.0,
            "panda_joint_4": -2.,
            "panda_joint_5": 0.0,
            "panda_joint_6": 3.037,
            "panda_joint_7": 0.741,
            "index_joint_0": 0.0,
            "index_joint_1": 0.0,
            "index_joint_2": 0.0,
            "index_joint_3": 0.0,
            "middle_joint_0": 0.0,
            "middle_joint_1": 0.0,
            "middle_joint_2": 0.0,
            "middle_joint_3": 0.0,
            "ring_joint_0": 0.0,
            "ring_joint_1": 0.0,
            "ring_joint_2": 0.0,
            "ring_joint_3": 0.0,
            "thumb_joint_0": 0.4,
            "thumb_joint_1": 0.0,
            "thumb_joint_2": 0.0,
            "thumb_joint_3": 0.0,
        },
        # pos=[0.0, 0.5, 0.0], 
        # rot=[1, 0, 0, 0]
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint_[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint_[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "allegro_index_finger": ImplicitActuatorCfg(
            joint_names_expr=["index_joint_0", "index_joint_1", "index_joint_2", "index_joint_3"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "allegro_middle_finger": ImplicitActuatorCfg(
            joint_names_expr=["middle_joint_0", "middle_joint_1", "middle_joint_2", "middle_joint_3"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "allegro_ring_finger": ImplicitActuatorCfg(
            joint_names_expr=["ring_joint_0", "ring_joint_1", "ring_joint_2", "ring_joint_3"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "allegro_thumb_finger": ImplicitActuatorCfg(
            joint_names_expr=["thumb_joint_0", "thumb_joint_1", "thumb_joint_2", "thumb_joint_3"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0


FRANKA_ALLEGRO_RIGHT_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG = FRANKA_ALLEGRO_RIGHT_CFG.copy()
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["allegro_index_finger"].stiffness = 30.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["allegro_index_finger"].damping = 1.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["allegro_middle_finger"].stiffness = 30.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["allegro_middle_finger"].damping = 1.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["allegro_ring_finger"].stiffness = 30.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["allegro_ring_finger"].damping = 1.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["allegro_thumb_finger"].stiffness = 30.0
FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.actuators["allegro_thumb_finger"].damping = 1.0

FRANKA_ALLEGRO_CFG = FRANKA_ALLEGRO_RIGHT_CFG.copy()
FRANKA_ALLEGRO_HIGH_PD_CFG = FRANKA_ALLEGRO_RIGHT_HIGH_PD_CFG.copy()

FRANKA_ALLEGRO_LEFT_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG = FRANKA_ALLEGRO_LEFT_CFG.copy()
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["allegro_index_finger"].stiffness = 30.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["allegro_index_finger"].damping = 1.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["allegro_middle_finger"].stiffness = 30.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["allegro_middle_finger"].damping = 1.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["allegro_ring_finger"].stiffness = 30.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["allegro_ring_finger"].damping = 1.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["allegro_thumb_finger"].stiffness = 30.0
FRANKA_ALLEGRO_LEFT_HIGH_PD_CFG.actuators["allegro_thumb_finger"].damping = 1.0

"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""



FRANKA_ALLEGRO_BIMANUAL_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/workspace/isaaclab/source/isaaclab_assets/data/Robots/FrankaAllegroBimanual_v2/franka_allegro.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, 
            solver_position_iteration_count=12, 
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "left_panda_joint_1": 0.0,
            "left_panda_joint_2": -0.569,
            "left_panda_joint_3": 0.0,
            "left_panda_joint_4": -2.,
            "left_panda_joint_5": 0.0,
            "left_panda_joint_6": 3.037,
            "left_panda_joint_7": 0.741,
            "left_index_joint_0": 0.0,
            "left_index_joint_1": 0.0,
            "left_index_joint_2": 0.0,
            "left_index_joint_3": 0.0,
            "left_middle_joint_0": 0.0,
            "left_middle_joint_1": 0.0,
            "left_middle_joint_2": 0.0,
            "left_middle_joint_3": 0.0,
            "left_ring_joint_0": 0.0,
            "left_ring_joint_1": 0.0,
            "left_ring_joint_2": 0.0,
            "left_ring_joint_3": 0.0,
            "left_thumb_joint_0": 0.4,
            "left_thumb_joint_1": 0.0,
            "left_thumb_joint_2": 0.0,
            "left_thumb_joint_3": 0.0,
            
            "right_panda_joint_1": 0.0,
            "right_panda_joint_2": -0.569,
            "right_panda_joint_3": 0.0,
            "right_panda_joint_4": -2.,
            "right_panda_joint_5": 0.0,
            "right_panda_joint_6": 3.037,
            "right_panda_joint_7": 0.741,
            "right_index_joint_0": 0.0,
            "right_index_joint_1": 0.0,
            "right_index_joint_2": 0.0,
            "right_index_joint_3": 0.0,
            "right_middle_joint_0": 0.0,
            "right_middle_joint_1": 0.0,
            "right_middle_joint_2": 0.0,
            "right_middle_joint_3": 0.0,
            "right_ring_joint_0": 0.0,
            "right_ring_joint_1": 0.0,
            "right_ring_joint_2": 0.0,
            "right_ring_joint_3": 0.0,
            "right_thumb_joint_0": 0.4,
            "right_thumb_joint_1": 0.0,
            "right_thumb_joint_2": 0.0,
            "right_thumb_joint_3": 0.0,

        },
        # pos=[0.0, -0.5, 0.0], 
        # rot=[1, 0, 0, 0]
    ),
    actuators={
        "left_panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["left_panda_joint_[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "left_panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["left_panda_joint_[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "left_allegro_index_finger": ImplicitActuatorCfg(
            joint_names_expr=["left_index_joint_[0-3]"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "left_allegro_middle_finger": ImplicitActuatorCfg(
            joint_names_expr=["left_middle_joint_[0-3]"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "left_allegro_ring_finger": ImplicitActuatorCfg(
            joint_names_expr=["left_ring_joint_[0-3]"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "left_allegro_thumb_finger": ImplicitActuatorCfg(
            joint_names_expr=["left_thumb_joint_[0-3]"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),


        "right_panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["right_panda_joint_[1-4]"],
            effort_limit=87.0,
            velocity_limit=2.175,
            stiffness=80.0,
            damping=4.0,
        ),
        "right_panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["right_panda_joint_[5-7]"],
            effort_limit=12.0,
            velocity_limit=2.61,
            stiffness=80.0,
            damping=4.0,
        ),
        "right_allegro_index_finger": ImplicitActuatorCfg(
            joint_names_expr=["right_index_joint_[0-3]"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "right_allegro_middle_finger": ImplicitActuatorCfg(
            joint_names_expr=["right_middle_joint_[0-3]"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "right_allegro_ring_finger": ImplicitActuatorCfg(
            joint_names_expr=["right_ring_joint_[0-3]"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
        "right_allegro_thumb_finger": ImplicitActuatorCfg(
            joint_names_expr=["right_thumb_joint_[0-3]"],
            effort_limit=0.5,
            velocity_limit=100.0,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
FRANKA_ALLEGRO_BIMANUAL_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG = FRANKA_ALLEGRO_BIMANUAL_CFG.copy()
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_panda_shoulder"].stiffness = 400.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_panda_shoulder"].damping = 80.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_panda_forearm"].stiffness = 400.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_panda_forearm"].damping = 80.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_allegro_index_finger"].stiffness = 30.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_allegro_index_finger"].damping = 1.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_allegro_middle_finger"].stiffness = 30.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_allegro_middle_finger"].damping = 1.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_allegro_ring_finger"].stiffness = 30.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_allegro_ring_finger"].damping = 1.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_allegro_thumb_finger"].stiffness = 30.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["left_allegro_thumb_finger"].damping = 1.0

FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_panda_shoulder"].stiffness = 400.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_panda_shoulder"].damping = 80.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_panda_forearm"].stiffness = 400.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_panda_forearm"].damping = 80.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_allegro_index_finger"].stiffness = 30.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_allegro_index_finger"].damping = 1.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_allegro_middle_finger"].stiffness = 30.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_allegro_middle_finger"].damping = 1.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_allegro_ring_finger"].stiffness = 30.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_allegro_ring_finger"].damping = 1.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_allegro_thumb_finger"].stiffness = 30.0
FRANKA_ALLEGRO_BIMANUAL_HIGH_PD_CFG.actuators["right_allegro_thumb_finger"].damping = 1.0
