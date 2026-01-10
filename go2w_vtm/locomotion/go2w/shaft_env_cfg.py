from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import go2w_vtm.locomotion.mdp as mdp

from ..velocity_env_cfg import ActionsCfg, LocomotionVelocityRoughEnvCfg
from ..velocity_env_cfg import RewardsCfg


from .asset.unitree import UNITREE_GO2W_CFG
from go2w_vtm.terrains.config.rough import GO2W_SHAFT_TERRAINS_CFG
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
class UnitreeGo2WRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class UnitreeGo2WShaftEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: UnitreeGo2WActionsCfg = UnitreeGo2WActionsCfg()
    rewards: UnitreeGo2WRewardsCfg = UnitreeGo2WRewardsCfg()

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
    foot_link_names = [
        "FR_foot", "FL_foot", "RR_foot", "RL_foot",
    ]
    
    min_camera_distance = 0.3
    max_camera_distance = 1.0

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # self.scene.height_scanner_down.prim_path = None
        # self.scene.height_scanner_up.prim_path = None
        # self.scene.height_scanner.prim_path = None
        # ------------------------------Terrain------------------------------
        self.scene.terrain.terrain_generator = GO2W_SHAFT_TERRAINS_CFG
        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel
        self.observations.critic.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names
        )
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        
        
        self.observations.policy.velocity_commands.params["asset_cfg"].body_names = self.foot_link_names
        self.observations.critic.velocity_commands.params["asset_cfg"].body_names = self.foot_link_names
        
        # self.observations.policy.height_scanner = None
        # self.observations.policy.height_scanner_down = None
        # self.observations.critic.height_scanner_up = None
        # self.observations.critic.height_scanner_down = None

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_vel.scale = 5.0
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names[:-4]
        self.actions.joint_vel.joint_names = self.joint_names[-4:]

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0.0

        self.rewards.base_height_l2_ref.weight = 0.0
        self.rewards.base_height_l2_ref.params["target_height"] = 0.4
        self.rewards.base_height_l2_ref.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.base_height_l2_ref.params["command_threshold"] = 0.3
        
        self.rewards.feet_pos_in_pit_l2.weight = 0.0
        self.rewards.feet_pos_in_pit_l2.params["asset_cfg"].body_names = self.foot_link_names
        self.rewards.base_pos_in_pit_l2.weight = 0.0
        self.rewards.base_pos_in_pit_l2.params["asset_cfg"].body_names = [self.base_link_name]
        
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penaltie
        self.rewards.joint_torques_l2.weight = -2e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.joint_torques_wheel_l2.weight = 0
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = [self.wheel_joint_name]
        self.rewards.joint_vel_l2.weight = -5e-5
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.joint_vel_wheel_l2.weight = -1e-4
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = [self.wheel_joint_name]
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = [self.wheel_joint_name]
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = [self.wheel_joint_name]
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.stand_still_without_cmd.weight = -2.0
        self.rewards.stand_still_without_cmd.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.joint_pos_penalty.weight = -0.5 
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = [f"^(?!{self.wheel_joint_name}).*"]
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = [self.wheel_joint_name]
        self.rewards.joint_mirror.weight = -0.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -0.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 8.0
        self.rewards.track_ang_vel_z_exp.weight = 4.0
        self.rewards.track_lin_vel_z_exp.weight = 10.0
        self.rewards.track_lin_vel_z_exp.params["asset_cfg"].body_names = self.foot_link_names

        # Others
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = 0
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.0
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.0
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 0.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2WShaftEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        # self.terminations.bad_orientation_and_pos.params["asset_cfg"].body_names = [self.base_link_name]
        self.terminations.bad_orientation_and_pos.params["asset_cfg"].body_names = []


        # ------------------------------Commands------------------------------
        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.1)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.1, 0.1)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_z = (0.0, 0.5)
        self.commands.base_velocity.is_lin_vel_z_zero = False
