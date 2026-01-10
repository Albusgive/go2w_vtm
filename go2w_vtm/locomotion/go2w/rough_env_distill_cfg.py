from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import go2w_vtm.locomotion.mdp as mdp

from ..velocity_env_distill_cfg import ActionsCfg, LocomotionVelocityRoughEnvCfg

from .asset.unitree import UNITREE_GO2W_CFG
##
# Pre-defined configs
##


from isaaclab.utils.noise import UniformNoiseCfg

@configclass
class UnitreeGo2WActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class UnitreeGo2WRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: UnitreeGo2WActionsCfg = UnitreeGo2WActionsCfg()

    base_link_name = "base"
    foot_link_name = ".*_foot"
    wheel_joint_name = ".*_foot_joint"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    contact_link_names = [
        "base",
        "FR_foot", "FL_foot", "RR_foot", "RL_foot",
    ]
    
    min_camera_distance = 0.3
    max_camera_distance = 1.0

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.ray_caster_camera.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_down.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_up.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.debug_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # self.scene.debug_scanner2.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names
        )
        self.observations.teacher.joint_pos.func = mdp.joint_pos_rel
        self.observations.teacher.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names
        )
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        
        self.observations.teacher.base_lin_vel.scale = 2.0
        self.observations.teacher.base_ang_vel.scale = 0.25
        self.observations.teacher.joint_pos.scale = 1.0
        self.observations.teacher.joint_vel.scale = 0.05
        
        self.observations.policy.ray_caster_camera.params["min"] = self.min_camera_distance
        self.observations.policy.ray_caster_camera.params["max"] = self.max_camera_distance
        # self.observations.policy.ray_caster_camera.params["noise_level"] = "low"
        self.observations.policy.ray_caster_camera.params["zero_p"] = 0.005
        self.observations.policy.ray_caster_camera.params["noise_range"] = 0.01
        self.observations.policy.ray_caster_camera.params["max_gradient"] = 16.5
        self.observations.policy.ray_caster_camera.params["min_gradient"] = 2.0
        self.observations.policy.ray_caster_camera.params["high_probability"] = 0.2
        self.observations.policy.ray_caster_camera.params["low_probability"] = 0.6

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names[:-4]
        self.actions.joint_vel.joint_names = self.joint_names[-4:]

        # ------------------------------Events------------------------------
        # self.events.randomize_reset_base.params = {
        #     "pose_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (0.0, 0.2),
        #         "roll": (-3.14, 3.14),
        #         "pitch": (-3.14, 3.14),
        #         "yaw": (-3.14, 3.14),
        #     },
        #     "velocity_range": {
        #         "x": (-0.5, 0.5),
        #         "y": (-0.5, 0.5),
        #         "z": (-0.5, 0.5),
        #         "roll": (-0.5, 0.5),
        #         "pitch": (-0.5, 0.5),
        #         "yaw": (-0.5, 0.5),
        #     },
        # }
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.0, 2.0)
