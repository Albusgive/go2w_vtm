from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import go2w_vtm.locomotion.mdp as mdp

from ..multi_motion_env_cfg import ActionsCfg, MultiMotionEnvCfg, MySceneCfg
from ..multi_motion_env_cfg import RewardsCfg

import go2w_vtm
from go2w_vtm.Robot.go2w import UNITREE_GO2W_CFG,UNITREE_GO2W_NO_MOTOR_LIMIT_CFG,UNITREE_GO2W_GHOST_CFG
from go2w_vtm.terrains.config.rough import MIMIC_GYM_TERRAIN_CFG,CONFIRM_TERRAIN_CFG,CONFIRM_TERRAIN_CFG2

from isaaclab.utils.noise import UniformNoiseCfg
import os

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
class UnitreeGo2WMultiMotionEnvCfg(MultiMotionEnvCfg):
    actions: UnitreeGo2WActionsCfg = UnitreeGo2WActionsCfg()
    # rewards: UnitreeGo2WRewardsCfg = UnitreeGo2WRewardsCfg()

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
    collection_leg_names = ["base", "FR_foot", "FL_foot", "RR_foot", "RL_foot"]
    body_names = [
            'base', 'FL_hip', 'FR_hip', 'RL_hip', 'RR_hip', 'FL_thigh', 'FR_thigh', 'RL_thigh', 'RR_thigh', 
            'FL_calf', 'FR_calf', 'RL_calf', 'RR_calf', 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot'
        ]
    
    low_joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": 0.0,
            "F.*_thigh_joint": 1.5,
            "R.*_thigh_joint": 1.5,
            ".*_calf_joint": -2.7,
            ".*_foot_joint": 0.0,
        }
    
    min_camera_distance = 0.3
    max_camera_distance = 1.0

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2W_NO_MOTOR_LIMIT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.ghost_robot = UNITREE_GO2W_GHOST_CFG.replace(prim_path="{ENV_REGEX_NS}/GhostRobot")
        # ------------------------------Terrain------------------------------
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = CONFIRM_TERRAIN_CFG2
        # ------------------------------commands------------------------------
        high_platform_path = os.path.join(go2w_vtm.GO2W_MJCF_DIR, "multi_motion_platform_terrain_k.npz")
        self.commands.motion.anchor_body_name = "base"
        self.commands.motion.body_names = self.body_names
        self.commands.motion.joint_names = self.leg_joint_names
        self.commands.motion.terrain_and_checkpoint_file = {"high_platform": high_platform_path} #TODO
        self.commands.motion.terrain_and_cmd_vel = {"high_platform": [(0.8,1.0),(0.0,0.0),(0.0,0.0)]}
        self.commands.motion.ik_cfg.legs_config={
            "FL_foot": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
            "FR_foot": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"],
            "RL_foot": ["RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"],
            "RR_foot": ["RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"],
        }
        self.commands.motion.ik_cfg.debug_vis = True
        # ------------------------------Observations------------------------------
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel
        self.observations.critic.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.leg_joint_names
        )
        self.observations.policy.base_lin_vel = None
        # self.observations.policy.base_ang_vel.scale = 0.25
        # self.observations.policy.joint_pos.scale = 1.0
        # self.observations.policy.joint_vel.scale = 0.05

        # self.observations.policy.base_lin_vel = None
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
        # ------------------------------Rewards------------------------------
        self.rewards.action_rate_l2.weight = -1e-2
        self.rewards.motion_body_pos.weight = 1.0
        self.rewards.motion_body_lin_vel.weight = 1.0
        self.rewards.motion_global_anchor_pos.weight = 1.0

        # 取消轮子body的姿态和角速度rew
        self.rewards.motion_body_pos.params["body_names"] = self.body_names
        self.rewards.motion_body_ori.params["body_names"] = self.body_names[:-4]
        self.rewards.motion_body_lin_vel.params["body_names"] = self.body_names
        self.rewards.motion_body_ang_vel.params["body_names"] = self.body_names[:-4]
        
        # ------------------------------Termination------------------------------
        # self.terminations.anchor_pos = None
        # self.terminations.anchor_ori = None

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2WMultiMotionEnvCfg":
            self.disable_zero_weight_rewards()
    
    