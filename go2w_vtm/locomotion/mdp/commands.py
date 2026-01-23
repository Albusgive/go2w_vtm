from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
import omni.log
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import RayCaster
import go2w_vtm.locomotion.mdp as mdp
import go2w_vtm.track as track

from dataclasses import MISSING
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG,RED_ARROW_X_MARKER_CFG
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique
import isaaclab.utils.string as string_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
   

class UniformThresholdVelocity3DCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold."""

    cfg: mdp.UniformThresholdVelocity3DCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformThresholdVelocity3DCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, z_vel(heading) , z_acc
        self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)
        
    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        # logs data
        self.metrics["error_vel"] += (
            torch.norm(self.vel_command_b[:, [0,1,3]] - self.robot.data.root_lin_vel_b, dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2]) / max_command_step
        )
        
    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > self.cfg.threshold).unsqueeze(1)
        if not self.cfg.is_lin_vel_z_zero:
            self.vel_command_b[env_ids, 3] = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ranges.lin_vel_z)


    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat
    
        
@configclass
class UniformThresholdVelocity3DCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformThresholdVelocity3DCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    threshold: float = 0.1
    
    is_lin_vel_z_zero: bool = True
    
    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """
        
        lin_vel_z: tuple[float, float] = MISSING
        """Range for the linear-z velocity command (in m/s)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    
    

class JumpTrackCommand(CommandTerm):
    
    cfg: JumpTrackCommandCfg
    jump_track_calculator: track.JumpCurveDynamicCalculatorW
    # mdp.UniformPoseCommand
    def __init__(self, cfg: JumpTrackCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self.height_scanner: RayCaster = env.scene[self.cfg.height_scanner_name]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz,bool) in root frame
        self.pose_command_w = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_w[:,3] = 1.0
        self.vel_track = torch.zeros(self.num_envs, 3, device=self.device)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        
        self.jump_track_calculator = track.JumpCurveDynamicCalculatorW(
            max_envs = self.num_envs,
            device=self.device,
            h1=cfg.h1,
            h2=cfg.h2,
            t1=cfg.t1,
            a2=cfg.a2,
            g=cfg.g,
            default_end_height=cfg.default_end_height,
            is_track_gravity = cfg.is_track_gravity
        )
        # 轨迹时间
        self.time_points = torch.zeros((self.num_envs,), device=self.device)
        self.terrain_z = torch.zeros((self.num_envs,), device=self.device)
        self.start_z = torch.zeros((self.num_envs,), device=self.device)
        self.jump_mask = torch.zeros((self.num_envs,), dtype=torch.bool ,device=self.device)
        
        # 设置一下更新时间
        self.cfg.resampling_time_range = (self._env.step_dt,self._env.step_dt)
        
        self._env: ManagerBasedRLEnv = env

    
    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_w
        
    def _resample_command(self, env_ids: Sequence[int]):
        if not self.cfg.enable_track:
            return
        cmd_linv_xy = self._env.command_manager.get_command(self.cfg.vel_command_name)[:,:2].clone()
        cmd_a = self._env.command_manager.get_command(self.cfg.jump_command_name)[:,0]
        
        # 如果有大于0的就要启动跳跃
        jump_flags = cmd_a > 0.0
        # 还不是跳跃阶段的环境
        jumped_flags = self.time_points == 0
        # 设置参数的环境
        set_parameters_idx = jump_flags & jumped_flags
        # 设置跳跃参数 并更跳跃开始位置
        if torch.any(set_parameters_idx):
            self.terrain_z[set_parameters_idx] = self.height_scanner.data.ray_hits_w[set_parameters_idx,..., 2].mean(dim=1)
            self.start_z[set_parameters_idx] = \
                self.robot.data.body_link_pose_w[set_parameters_idx, self.body_idx,2] - self.terrain_z[set_parameters_idx]
            # start_heights = body z - terrain z
            self.jump_track_calculator.set_parameters(set_parameters_idx, cmd_a, self.start_z)
            self.pose_command_w[set_parameters_idx,:2] = \
            self.robot.data.body_link_pose_w[set_parameters_idx, self.body_idx,:2].clone()
            
            # self.terrain_z[set_parameters_idx] = self.robot.data.body_link_pose_w[set_parameters_idx, self.body_idx,2].clone()
            
        # 计算高度
        if torch.any(jump_flags):
            target_heigh = self.jump_track_calculator.compute_height(jump_flags, self.time_points)
            target_vel = self.jump_track_calculator.compute_velocity(jump_flags, self.time_points)
            # 判断有效的高度
            target_heigh_inf_idx = torch.isinf(target_heigh)
            if torch.any(target_heigh_inf_idx):
                # 把inf的命令和time置0
                cmd_a[target_heigh_inf_idx] = 0.0
                self.time_points[target_heigh_inf_idx] = 0.0
            # 更新轨迹时间 和轨迹标志位
            jump_flags = cmd_a > 0.0
            if torch.any(jump_flags):
                self.jump_mask = jump_flags
                self.time_points[jump_flags] += self._env.step_dt
                # cmd_从body转到world
                cmd_linv = torch.cat([cmd_linv_xy[jump_flags], torch.zeros((cmd_linv_xy[jump_flags].shape[0], 1), device=self.device)], dim=-1)
                world_velocity = math_utils.quat_apply(self.robot.data.body_link_quat_w[jump_flags, self.body_idx], cmd_linv)
                self.pose_command_w[jump_flags, :2] += world_velocity[:, :2] * self._env.step_dt
                self.pose_command_w[jump_flags, 2] = self.terrain_z[jump_flags] + target_heigh[jump_flags]
                self.vel_track[jump_flags,2] = target_vel[jump_flags]
                no_jump = ~jump_flags
                self.pose_command_w[no_jump,:3] = self.robot.data.body_link_pose_w[no_jump, self.body_idx,:3].clone()
        
    def termination(self, envs: torch.Tensor):
        self.jump_track_calculator.stile[envs] = 0.0
        self.jump_mask[envs] = False
        self._env.command_manager.get_command(self.cfg.jump_command_name)[envs,0] = 0.0
        
    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:7])
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])

    def _update_command(self):
        pass
    
    def _update_metrics(self):
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:7],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)


@configclass
class JumpTrackCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = JumpTrackCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """
    
    resampling_time_range = (0.0, 0.0)
    ''' 无需指定,会自动按照step_dt生成 '''

    vel_command_name: str = MISSING,
    ''' 从该cmd_term获取跳跃指令 '''
    
    jump_command_name: str = MISSING,
    ''' 从该cmd_term获取跳跃指令 '''
    
    height_scanner_name: str = MISSING,
    ''' 测量当前位置地面的z '''
    
    is_track_gravity: bool = True
    
    enable_track: bool = True
    ''' 是否启用跳跃轨迹跟踪 '''
        
    h1: float = 0.2,
    ''' z (m) '''
    h2: float = 0.5,
    ''' z (m) '''
    t1: float = 0.4,
    ''' base heigh 2 low heigh time (s) '''
    a2: float = -1.0,
    ''' the force(acc) rate '''
    g: float = -9.8,
    ''' world G '''
    default_end_height: float = 0.45,
    ''' track termitate (m) '''

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    
    
class JumpAccCommand(CommandTerm):

    cfg: JumpAccCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: JumpAccCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        self._env: ManagerBasedRLEnv = env
        self.vel_command_b = torch.zeros((self.num_envs,1), device=self.device)
        self.acc_z_probability = torch.full((self.num_envs,),self.cfg.acc_z_probability, device=self.device)
        self.jump_mask = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # 从给定cmd之后达到最低点（time）
        self.lowing = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # 从最低点到离地前
        self.uping = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        self.cmd_time = torch.zeros((self.num_envs,), device=self.device)
        # 从给定cmd之后离地到落地
        self.flying = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # 向上飞行
        self.flying_up = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # 向下飞行
        self.flying_down = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        # 下蹲默认关节角度
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.low_joint_pos = torch.zeros_like(self.robot.data.default_joint_pos)
        indices_list, _, values_list = string_utils.resolve_matching_names_values(
            self.cfg.joint_pos, self.robot.joint_names
        )
        self.low_joint_pos[:,indices_list] = torch.tensor(values_list, device=self.device)
        

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "JumpAccCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        return msg

    
    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        times = (self._env.episode_length_buf[env_ids] * self._env.step_dt) > self.cfg.start_cmd
        _ids = env_ids[times]
        is_produce = torch.rand(_ids.shape, device=self.device)
        idx = is_produce < self.acc_z_probability[_ids]
        self.vel_command_b[_ids] = 0.0
        self.cmd_time[env_ids] = 0.0
        self.flying[env_ids] = False
        self.flying_down[env_ids] = False
        self.lowing[env_ids] = False
        self.uping[env_ids] = False
        if torch.any(idx):
            rand_values = torch.empty(len(_ids[idx]),1, device=self.device)
            rand_values.uniform_(*self.cfg.ranges.lin_acc_z)
            self.vel_command_b[_ids[idx]] = rand_values
            

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class JumpAccCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""
    class_type: type = JumpAccCommand

    @configclass
    class Ranges:
        lin_acc_z: tuple[float, float] = MISSING
        """Range for the linear-z velocity command (in m^2/s)."""

    ranges: Ranges = MISSING
    
    acc_z_probability: float = 1.0
    
    start_cmd: float = 1.0
    ''' 每个环境从第n秒之后才开始进行命令生成 '''
    
    asset_name: str = MISSING
    
    joint_pos: dict[str, float] = {".*": 0.0}
    ''' 下蹲时的关节角度 '''
    
    
    
    
'''     montion tarcking    '''
import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
    quat_apply_inverse
)
from torch.nn.utils.rnn import pad_sequence
from go2w_vtm.terrains import ConfirmTerrainImporter


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MultiMotionLoader:
    def __init__(
        self,
        motion_files: dict[str, str],
        body_indexes: Sequence[int],
        device: str = "cpu"
    ):
        """
        Load multiple motion files and stack them into a unified dataset.
        
        Args:
            motion_files (dict): Mapping from motion name to file path.
            body_indexes (Sequence[int]): Indices of bodies to extract from each motion's full body list.
            device (str): Device to load tensors onto.
        """
        self.motion_names = list(motion_files.keys())
        self.motion_name_to_idx = {name: i for i, name in enumerate(self.motion_names)}
        self._body_indexes = list(body_indexes)  # ensure it's a list for indexing
        self.device = device

        # Load all MotionLoaders
        loaders = []
        for name in self.motion_names:
            file_path = motion_files[name]
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"Motion file not found: {file_path}")
            loader = MotionLoader(file_path, body_indexes=[], device=device)  # ← 注意：这里传空 body_indexes！
            loaders.append(loader)

        # Validate consistency using the first motion as reference
        ref_loader = loaders[0]
        ref_joint_dim = ref_loader.joint_pos.shape[1]
        ref_body_num = ref_loader._body_pos_w.shape[1]  # full body count
        self.fps = float(ref_loader.fps)

        for i, (name, loader) in enumerate(zip(self.motion_names, loaders)):
            if loader.joint_pos.shape[1] != ref_joint_dim:
                raise ValueError(f"Joint dimension mismatch in '{name}': "
                                 f"expected {ref_joint_dim}, got {loader.joint_pos.shape[1]}")
            if loader._body_pos_w.shape[1] != ref_body_num:
                raise ValueError(f"Body count mismatch in '{name}': "
                                 f"expected {ref_body_num}, got {loader._body_pos_w.shape[1]}")
            if abs(loader.fps - self.fps) > 1e-3:
                print(f"Warning: FPS differs in '{name}' ({loader.fps} vs {self.fps})")

        # Stack RAW tensors (before body selection)
        max_T = max(l.time_step_total for l in loaders)

        # Pad and stack
        self.joint_pos = self._pad_sequence([l.joint_pos for l in loaders], max_T)
        self.joint_vel = self._pad_sequence([l.joint_vel for l in loaders], max_T)
        self._body_pos_w = self._pad_sequence([l._body_pos_w for l in loaders], max_T)
        self._body_quat_w = self._pad_sequence([l._body_quat_w for l in loaders], max_T)
        self._body_lin_vel_w = self._pad_sequence([l._body_lin_vel_w for l in loaders], max_T)
        self._body_ang_vel_w = self._pad_sequence([l._body_ang_vel_w for l in loaders], max_T)
        
        # Store time step totals per motion
        self.time_step_totals = torch.tensor(
            [l.time_step_total for l in loaders], dtype=torch.long, device=device
        )  # [M]
        self.num_motions = len(loaders)

    # Helper: pad a list of [T_i, ...] tensors to [max_T, ...]
    def _pad_sequence(self,tensor_list, max_len, padding_value=0.0):
        # pad_sequence expects [T, ...] in a list, returns [max_T, ...]
        # But it pads at the beginning by default; we want end-padding
        padded = []
        for t in tensor_list:
            pad_len = max_len - t.size(0)
            if pad_len > 0:
                last_dim = t.size()[1:]
                padding = torch.full((pad_len,) + last_dim, padding_value, device=t.device, dtype=t.dtype)
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded, dim=0)  # [M, max_T, ...]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, :, self._body_indexes]  # [M, T, B_selected, 3]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, :, self._body_indexes]  # [M, T, B_selected, 4]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, :, self._body_indexes]  # [M, T, B_selected, 3]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, :, self._body_indexes]  # [M, T, B_selected, 3]

    def get_max_duration(self) -> int:
        return self.time_step_totals.max().item()

    def __len__(self) -> int:
        return self.num_motions

    

class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MultiMotionLoader(self.cfg.motion_files, self.body_indexes, device=self.device)
        self.motion_ids = torch.full((self.num_envs,), 0, dtype=torch.long, device=self.device) #[env,motion_id]
        self.enable = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)  #对于不进行mimic的环境给False
        
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.max_time_steps = self.motion.get_max_duration()  # ← from MultiMotionLoader
        self.bin_count = int(self.max_time_steps // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.motion.num_motions, self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.motion.num_motions, self.bin_count, dtype=torch.float, device=self.device)

        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        
        if self.cfg.joint_names is not None:
            self.joint_indexes = self.robot.find_joints(self.cfg.joint_names, preserve_order=True)[0]
            self.joint_indexes = torch.tensor(self.joint_indexes, dtype=torch.long, device=self.device)
        
        self._velocity_command = torch.zeros(self.num_envs, 3, device=self.device)
            
        self.terrain = self._env.scene.terrain
        if isinstance(self.terrain, ConfirmTerrainImporter):
            self.terrain: ConfirmTerrainImporter = self._env.scene.terrain
        
        # 构建地形 -> motion_id 的映射表 TODO 并行framekey
        if self.cfg.terrain_motion_map is not None:
            sub_terrain_type_names = self.terrain.sub_terrain_type_names
            if sub_terrain_type_names is None:
                raise ValueError("terrain is not a ConfirmTerrainImporter or terrain_type is not generator \
                                 or curriculum is not enabled.")
            self.terrain_to_idx = {name: i for i, name in enumerate(sub_terrain_type_names)}
            
            # Step 1: 构建 terrain_id -> list[motion_id]
            num_terrains = len(self.terrain_to_idx)
            terrain_id_to_motion_ids: list[list[int]] = [[] for _ in range(num_terrains)]

            for terrain_name, motion_names in self.cfg.terrain_motion_map.items():
                if terrain_name not in self.terrain_to_idx:
                    print(f"[Warning] Terrain '{terrain_name}' in terrain_motion_map but not in environment terrains. Ignored.")
                    continue
                tid = self.terrain_to_idx[terrain_name]
                motion_ids = []
                for mn in motion_names:
                    if mn not in self.motion.motion_name_to_idx:
                        raise KeyError(f"Motion name '{mn}' not found in motion library.")
                    motion_ids.append(self.motion.motion_name_to_idx[mn])
                terrain_id_to_motion_ids[tid] = motion_ids
                # 注意：如果 motion_names 为空列表，这里会赋值 []

            # Step 2: 计算 max_len，但要排除全空的情况（否则 max() 报错）
            non_empty_lengths = [len(m) for m in terrain_id_to_motion_ids if len(m) > 0]
            if non_empty_lengths:
                max_len = max(non_empty_lengths)
            else:
                # 所有地形都无 motion，设 max_len=1（至少能建表）
                max_len = 1

            # Step 3: 构建向量化采样张量表 (T, max_len)，未配置的地形用 -1 填充
            table = torch.full((num_terrains, max_len), -1, dtype=torch.long, device=self.device)
            env_ids = torch.ones(self.num_envs, dtype=torch.long, device=self.device)
            env_terrain_ids = self.terrain.get_sub_terrain_type(env_ids)

            for tid, motions in enumerate(terrain_id_to_motion_ids):
                if len(motions) == 0:
                    # 保持 -1（已由 torch.full 初始化） 修改enable
                    self.enable[env_terrain_ids == tid] = False
                    continue
                L = len(motions)
                motions_t = torch.tensor(motions, dtype=torch.long, device=self.device)
                repeats = (max_len + L - 1) // L  # 向上取整
                padded = motions_t.repeat(repeats)[:max_len]
                table[tid] = padded

            self.terrain_motion_table = table  # shape: (num_terrains, max_len)
        else:
            # 如果没有配置
            self.terrain_motion_table = None
    
    @property
    def command(self) -> torch.Tensor:
        if self.cfg.joint_names is not None:
            raw = torch.cat([self.joint_pos[:,self.joint_indexes], self.joint_vel[:,self.joint_indexes]], dim=1)
        else:
            raw = torch.cat([self.joint_pos, self.joint_vel], dim=1)
        return torch.where(
            self.enable.unsqueeze(-1),
            raw,
            torch.zeros_like(raw))
        
    @property
    def velocity_command(self) -> torch.Tensor:
        ''' 根据anchor 的速度和角速度计算cmd (vx, vy, ωz)'''
        return torch.where(
            self.enable.unsqueeze(-1),
            self._velocity_command,
            torch.zeros_like(self._velocity_command))
        
    @property
    def command_with_vel(self) -> torch.Tensor:
        return torch.cat([self.command, self.velocity_command], dim=1)
    
    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.motion_ids, self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.motion_ids, self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_ids, self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_ids, self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_ids, self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_ids, self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.motion_ids, self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    def _update_velocity_command(self):
        """
        从 motion 数据中提取 anchor 的 (vx, vy, ωz) 命令，并更新 self._velocity_command。
        
        - vx, vy: anchor 局部坐标系下的线速度（x/y 分量）
        - ωz: 绕世界 Z 轴的角速度（等价于局部 z，若无 roll/pitch）
        """
        # 1. 获取 world 系下的线速度和角速度
        lin_vel_w = self.anchor_lin_vel_w      # [E, 3]
        ang_vel_w = self.anchor_ang_vel_w      # [E, 3]
        # 2. 获取 anchor 的朝向（用于将线速度转到局部系）
        quat_w = self.anchor_quat_w          # [E, 4]
        # 3. 将线速度转到 anchor 局部坐标系
        lin_vel_local = quat_apply_inverse(quat_w, lin_vel_w)  # [E, 3]
        # 4. 提取 vx, vy（局部 x, y）
        vx = lin_vel_local[:, 0]
        vy = lin_vel_local[:, 1]
        # 5. 提取 ωz（世界系 z 分量，通常更稳定；也可用局部 z）
        wz = ang_vel_w[:, 2]  # 绕世界 Z 轴的角速度
        # 6. 更新缓存
        self._velocity_command[:, 0] = vx
        self._velocity_command[:, 1] = vy
        self._velocity_command[:, 2] = wz

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        # 1. 获取当前环境的状态
        episode_failed = self._env.termination_manager.terminated[env_ids]
        current_motion_ids = self.motion_ids[env_ids]  # [E]

        # 2. 统计失败 (Update Failures)
        if torch.any(episode_failed):
            # 获取每个 motion 实际的总时长，避免 padding 区域影响计算
            motion_totals = self.motion.time_step_totals[current_motion_ids].float()
            
            # 计算归一化时间并映射到 Bin
            current_time_norm = self.time_steps[env_ids].float() / motion_totals
            current_bin_index = (current_time_norm * self.bin_count).long().clamp(0, self.bin_count - 1)

            # 筛选出失败的环境
            fail_mask = episode_failed
            fail_motion_ids = current_motion_ids[fail_mask] # [F]
            fail_bins = current_bin_index[fail_mask]       # [F]

            # 只有当确实有失败发生时才更新
            if fail_motion_ids.numel() > 0:
                self._current_bin_failed.view(-1).index_add_(
                    0,
                    fail_motion_ids * self.bin_count + fail_bins,
                    torch.ones_like(fail_bins, dtype=torch.float)
                )

        # 3. 采样新时间步 (Sample New Time Steps)
        new_time_steps = torch.zeros_like(self.time_steps[env_ids]) # 默认为0
        
        # [优化点] 只遍历当前 envs 涉及到的 motion_id，极大提升稀疏情况下的性能
        unique_mids = torch.unique(current_motion_ids)

        for mid_tensor in unique_mids:
            mid = mid_tensor.item()
            mask = (current_motion_ids == mid) # 当前批次中属于该动作的环境掩码
            
            # --- 概率计算 ---
            # 取出该动作的失败分布行
            raw_probs = self.bin_failed_count[mid] + self.cfg.adaptive_uniform_ratio / self.bin_count
            
            # 卷积平滑 (Conv1d 需要 [1, 1, L] 格式)
            padded_probs = torch.nn.functional.pad(
                raw_probs.view(1, 1, -1),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate"
            )
            smoothed_probs = torch.nn.functional.conv1d(
                padded_probs, self.kernel.view(1, 1, -1)
            ).view(-1)
            
            # 归一化
            prob_sum = smoothed_probs.sum()
            if prob_sum > 1e-6:
                sampling_probs = smoothed_probs / prob_sum
            else:
                # 极端情况兜底：均匀分布
                sampling_probs = torch.ones_like(smoothed_probs) / self.bin_count

            # --- 采样 ---
            n_sample = mask.sum().item()
            sampled_bins = torch.multinomial(sampling_probs, n_sample, replacement=True)

            # --- 映射回时间步 ---
            max_t = self.motion.time_step_totals[mid].item()
            
            # 连续化 + 缩放 + 截断
            # 注意：这里使用 random 偏移让采样更平滑，不只停留在 bin 的左边界
            t_float = (sampled_bins.float() + torch.rand(n_sample, device=self.device)) / self.bin_count * (max_t - 1)
            
            new_time_steps[mask] = t_float.long().clamp(0, max_t - 1)

        self.time_steps[env_ids] = new_time_steps

        # Update metrics (optional: use average or per-motion)
        avg_sampling_probs = self.bin_failed_count.mean(dim=0) + self.cfg.adaptive_uniform_ratio / self.bin_count
        avg_sampling_probs = torch.nn.functional.pad(
            avg_sampling_probs.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate"
        )
        avg_sampling_probs = torch.nn.functional.conv1d(avg_sampling_probs, self.kernel.view(1, 1, -1)).view(-1)
        avg_sampling_probs = avg_sampling_probs / avg_sampling_probs.sum()

        H = -(avg_sampling_probs * (avg_sampling_probs + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = avg_sampling_probs.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        mimic_env_ids = env_ids[self.enable[env_ids]]
        if len(mimic_env_ids) == 0:
            return
        if not self.cfg.is_play_mode:
            self._adaptive_sampling(mimic_env_ids)
        else:
            self.time_steps[mimic_env_ids] = 0.0
        
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(mimic_env_ids), 6), device=self.device)
        root_pos[mimic_env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[mimic_env_ids] = quat_mul(orientations_delta, root_ori[mimic_env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(mimic_env_ids), 6), device=self.device)
        root_lin_vel[mimic_env_ids] += rand_samples[:, :3]
        root_ang_vel[mimic_env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[mimic_env_ids]
        joint_pos[mimic_env_ids] = torch.clip(
            joint_pos[mimic_env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[mimic_env_ids], joint_vel[mimic_env_ids], env_ids=mimic_env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[mimic_env_ids], root_ori[mimic_env_ids], root_lin_vel[mimic_env_ids], root_ang_vel[mimic_env_ids]], dim=-1),
            env_ids=mimic_env_ids,
        )
        
        # 抽取motion 在这里会出现卡死bug，使用event就不会
        # self.reset_motion_by_terrain(mimic_env_ids)

    def _update_command(self):
        self.time_steps[self.enable] += 1

        # Check if any env exceeds its motion's length
        max_steps = self.motion.time_step_totals[self.motion_ids]  # [N]
        env_ids = torch.where(self.time_steps >= max_steps)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )

        self._update_velocity_command()
        
        # Update bin_failed_count with exponential moving average
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed +
            (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()
    
    
    def get_env_terrains(self):
        '''
        获取环境的地形位置 需要地形type=generator TerrainGeneratorCfg开启curriculum 
        见isaaclab/terrains/terrain_generator.py -> _generate_curriculum_terrains
        col_names 为TerrainGeneratorCfg.sub_terrains的key
        '''
        from go2w_vtm.terrains import ConfirmTerrainImporter
        if self._env.scene.terrain is None:
            raise ValueError("Terrain is not initialized")
        if not self._env.scene.terrain.cfg.terrain_generator.curriculum:
            raise ValueError("TerrainGeneratorCfg.curriculum is not True")

        terrain:ConfirmTerrainImporter = self._env.scene.terrain
        return terrain.sub_terrain_type_names
    
    
    # def reset_motion_by_terrain(self, env_ids: Sequence[int]):
    #     if self.terrain_motion_table is None:
    #         return
    #     terrain_ids = self.env_terrain_ids[env_ids]
    #     max_len = self.terrain_motion_table.shape[1]
    #     rand_idx = torch.randint(0, max_len, (len(env_ids),), device=self.device)
    #     self.motion_ids[env_ids] = self.terrain_motion_table[terrain_ids, rand_idx]


    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_body_visualizers"):
                self.current_ref_body_visualizers = []
                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )
                    self.current_ref_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.ref_body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)
                self.current_ref_body_visualizers[i].set_visibility(True)
            
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)

        else:
            if hasattr(self, "current_body_visualizers"):
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)
                    self.current_ref_body_visualizers[i].set_visibility(False)
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])
            self.current_ref_body_visualizers[i].visualize(self.body_pos_w[:, i], self.body_quat_w[:, i])
        
        anchor_pos_w = self.robot_anchor_pos_w.clone()
        anchor_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.velocity_command[:, :2])
        robot_anchor_lin_vel_b = quat_apply_inverse(self.robot_anchor_quat_w, self.robot_anchor_lin_vel_w)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(robot_anchor_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(anchor_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(anchor_pos_w, vel_arrow_quat, vel_arrow_scale)

    
    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_files: dict[str, str] = MISSING
    
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    
    joint_names: list[str] = None # cmd观测的motion的joint名称
    
    is_play_mode :bool = False

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}
    
    ''' 地形名:对应的motion_files的key即motion_name  没写或者空motion的地形会在command中 enable=False'''
    terrain_motion_map: dict[str, list[str]] = None   

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    ref_body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    ref_body_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    


""" IK """
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from typing import Literal
from isaaclab.utils.math import subtract_frame_transforms,quat_apply_inverse, matrix_from_quat, quat_inv

class IKCommand(CommandTerm):
    cfg: IKCommandCfg

    def __init__(self, cfg: IKCommandCfg, env: ManagerBasedRLEnv):
        # 1. 初始化基础属性
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_names = list(cfg.legs_config.keys())
        self.num_legs = len(self.body_names)
        self.is_pose_mode = cfg.use_pose_mode
        
        # 2. 调用基类
        super().__init__(cfg, env)

        # 3. 解析索引
        self.cmd_type = "pose" if self.is_pose_mode else "position"
        self.joint_ids_list = []
        self.body_ids_list = []
        self.jacobi_joint_ids_list = []
        
        for body_name in self.body_names:
            joint_names = cfg.legs_config[body_name]
            leg_entity_cfg = SceneEntityCfg(cfg.asset_name, joint_names=joint_names, body_names=[body_name])
            leg_entity_cfg.resolve(env.scene)
            self.joint_ids_list.append(leg_entity_cfg.joint_ids)
            self.body_ids_list.append(leg_entity_cfg.body_ids[0])
            
            if self.robot.is_fixed_base:
                self.jacobi_joint_ids_list.append(leg_entity_cfg.joint_ids)
            else:
                self.jacobi_joint_ids_list.append([idx + 6 for idx in leg_entity_cfg.joint_ids])

        # 4. 控制器配置
        total_ik_envs = self.num_envs * self.num_legs
        diff_ik_cfg = DifferentialIKControllerCfg(
            command_type=self.cmd_type,
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 0.03}
        )
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=total_ik_envs, device=self.device)
        
        # 5. 【新增】状态缓存：用于存储参考机器人的完整 FK 结果
        self.num_joints = self.robot.num_joints
        self.num_bodies = self.robot.num_bodies
        
        # 关节空间
        self.joint_pos = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self.num_joints), device=self.device)
        
        # 笛卡尔空间 (Body)
        self.body_pos_w = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_quat_w = torch.zeros((self.num_envs, self.num_bodies, 4), device=self.device)
        self.body_lin_vel_w = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)
        self.body_ang_vel_w = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device)

        # 历史记录 (用于差分计算)
        self._prev_joint_pos = torch.zeros_like(self.joint_pos)
        self._prev_body_pos_w = torch.zeros_like(self.body_pos_w)
        self._prev_body_quat_w = torch.zeros_like(self.body_quat_w)

        self.command_len = 7 if self.is_pose_mode else 3
        
        self.ik_command = torch.zeros((self.num_envs, self.num_legs, self.command_len), device=self.device)

    def get_jacobians_b(self, leg_idx: int) -> torch.Tensor:
        jacobi_body_idx = self.body_ids_list[leg_idx]
        if self.robot.is_fixed_base:
            jacobi_body_idx -= 1
        
        jac_w = self.robot.root_physx_view.get_jacobians()[:, jacobi_body_idx, :, self.jacobi_joint_ids_list[leg_idx]]
        base_rot_matrix = matrix_from_quat(quat_inv(self.robot.data.root_quat_w))
        
        jac_b = torch.zeros_like(jac_w)
        jac_b[:, :3, :] = torch.bmm(base_rot_matrix, jac_w[:, :3, :])
        jac_b[:, 3:, :] = torch.bmm(base_rot_matrix, jac_w[:, 3:, :])
        return jac_b

    def compute_ik(self, ik_commands: torch.Tensor) -> torch.Tensor:
        """纯 IK 解算逻辑"""
        self.ik_command = ik_commands[:, :, :self.command_len]
        
        all_ee_pos_b, all_ee_quat_b, all_jac_b, all_q = [], [], [], []

        for i in range(self.num_legs):
            all_q.append(self.robot.data.joint_pos[:, self.joint_ids_list[i]])
            ee_p_b, ee_q_b = subtract_frame_transforms(
                self.robot.data.root_pos_w, self.robot.data.root_quat_w, 
                self.robot.data.body_pos_w[:, self.body_ids_list[i]], 
                self.robot.data.body_quat_w[:, self.body_ids_list[i]]
            )
            all_ee_pos_b.append(ee_p_b)
            all_ee_quat_b.append(ee_q_b)
            
            jac_b = self.get_jacobians_b(i)
            row_end = 6 if self.is_pose_mode else 3
            all_jac_b.append(jac_b[:, :row_end, :])

        flat_ee_pos_b = torch.cat(all_ee_pos_b, dim=0)
        flat_ee_quat_b = torch.cat(all_ee_quat_b, dim=0)
        flat_jac_b = torch.cat(all_jac_b, dim=0)
        flat_q = torch.cat(all_q, dim=0)
        flat_cmd = self.ik_command.transpose(0, 1).reshape(-1, self.ik_command.shape[-1])

        self.diff_ik_controller.set_command(flat_cmd, ee_quat=flat_ee_quat_b)
        flat_q_des = self.diff_ik_controller.compute(flat_ee_pos_b, flat_ee_quat_b, flat_jac_b, flat_q)

        target_q = self.robot.data.default_joint_pos.clone()
        split_q_des = flat_q_des.reshape(self.num_legs, self.num_envs, -1)
        for i in range(self.num_legs):
            target_q[:, self.joint_ids_list[i]] = split_q_des[i]
        return target_q

    def compute_ik_and_fk(self, root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, ik_commands: torch.Tensor, dt: float):
        """
        基于当前影子机器人的状态解算 IK 得到关节位置
        将新的 Root Pose 和解出的关节位置同时写入物理引擎
        执行 Scene.update(0.0) 触发正向动力学 (FK)
        计算差分速度
        """
        # 1. 解算 IK (基于影子机器人当前那一瞬间的状态)
        # 此时 self.robot.data 还是上一帧更新后的结果
        target_q = self.compute_ik(ik_commands) # 内部使用当前的 root_pos_w/quat_w 和 joint_pos
        
        # 2. 统一写入目标位姿（Root + Joints）
        root_states = torch.zeros((self.num_envs, 13), device=self.device)
        root_states[:, 0:3] = root_pos_w
        root_states[:, 3:7] = root_quat_w
        # root_states[:, 7:13] = 0.0 # 影子机器人通常设为 0 速度
        
        self.robot.write_root_state_to_sim(root_states)
        self.robot.write_joint_state_to_sim(target_q, torch.zeros_like(target_q))
        
        # 3. 核心：执行物理引擎更新，此时会根据新写入的 Root 和 Joint 计算出所有 Body 的 World Pose (FK)
        self._env.scene.update(0.0)

        # 4. 从更新后的 robot.data 中提取 FK 结果
        self.joint_pos[:] = self.robot.data.joint_pos
        self.body_pos_w[:] = self.robot.data.body_pos_w
        self.body_quat_w[:] = self.robot.data.body_quat_w

        # 5. 差分计算速度 (基于本次 FK 结果和上一帧缓存的 FK 结果)
        if dt > 0:
            self.joint_vel[:] = (self.joint_pos - self._prev_joint_pos) / dt
            self.body_lin_vel_w[:] = (self.body_pos_w - self._prev_body_pos_w) / dt
            # 角速度差分：omega = 2 * (q_curr * q_prev_inv).xyz / dt
            dq = quat_mul(self.body_quat_w, quat_inv(self._prev_body_quat_w))
            # 修正：处理四元数双倍覆盖（确保 w 分量为正）
            sign = torch.sign(dq[..., 0:1])
            self.body_ang_vel_w[:] = 2.0 * sign * dq[..., 1:4] / dt
        
        # 6. 更新历史缓存
        self._prev_joint_pos[:] = self.joint_pos
        self._prev_body_pos_w[:] = self.body_pos_w
        self._prev_body_quat_w[:] = self.body_quat_w

        return target_q

    def sync_history_to_current(self, env_ids: Sequence[int]):
        """重置时同步历史记录，防止速度差分爆炸"""
        self._prev_joint_pos[env_ids] = self.joint_pos[env_ids]
        self._prev_body_pos_w[env_ids] = self.body_pos_w[env_ids]
        self._prev_body_quat_w[env_ids] = self.body_quat_w[env_ids]

    def reset_ghost_robot(self, env_ids: Sequence[int], root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, joint_pos: torch.Tensor):
        """专门用于重置环境时，强制设置影子机器人的状态，不涉及 IK 解算"""
        # 1. 写入物理引擎
        root_states = self.robot.data.root_state_w[env_ids].clone()
        root_states[:, 0:3] = root_pos_w
        root_states[:, 3:7] = root_quat_w
        root_states[:, 7:13] = 0.0 # 重置时速度归零
        
        self.robot.write_root_state_to_sim(root_states, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
        
        # 2. 强制该环境的 FK 缓存立即更新 (这一步可选，因为稍后全局 update 也会跑)
        # 但为了保证 prev_pos 准确，我们需要更新内存中的缓存
        self.joint_pos[env_ids] = joint_pos
        self.body_pos_w[env_ids] = root_pos_w.unsqueeze(1) # 这是一个近似，真实的需等全局 update 的 FK
        self.body_quat_w[env_ids] = root_quat_w.unsqueeze(1)
        
        # 3. 同步历史，确保第一帧 dt 计算出的速度为 0
        self._prev_joint_pos[env_ids] = self.joint_pos[env_ids]
        self._prev_body_pos_w[env_ids] = self.body_pos_w[env_ids]
        self._prev_body_quat_w[env_ids] = self.body_quat_w[env_ids]

    @property
    def command(self) -> torch.Tensor:
        return self.ik_command.clone()
    
    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass
    
    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "markers"):
                self.markers = []
                for i, name in enumerate(self.body_names):
                    m_cfg = FRAME_MARKER_CFG.copy()
                    m_cfg.markers["frame"].scale = (self.cfg.vis_scale,)*3
                    self.markers.append({
                        "cur": VisualizationMarkers(m_cfg.replace(prim_path=f"/Visuals/IK/{name}_cur")),
                        "goal": VisualizationMarkers(m_cfg.replace(prim_path=f"/Visuals/IK/{name}_goal"))
                    })
        for m in self.markers:
            m["cur"].set_visibility(debug_vis)
            m["goal"].set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        """可视化直接读取 compute_ik 记录的位姿数据。"""
        for i in range(self.num_legs):
            # 1. 计算目标点（Goal）的世界坐标
            goal_pos_b = self.ik_command[:, i, 0:3]
            goal_quat_b = self.ik_command[:, i, 3:7] if self.is_pose_mode else torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
            
            g_p_w, g_q_w = combine_frame_transforms(self.robot.data.root_pos_w, self.robot.data.root_quat_w, goal_pos_b, goal_quat_b)
            
            # 2. 获取当前末端（Current）的世界位姿
            ee_p_w = self.robot.data.body_state_w[:, self.body_ids_list[i], 0:3]
            ee_q_w = self.robot.data.body_state_w[:, self.body_ids_list[i], 3:7]
            
            # 3. 更新可视化
            self.markers[i]["cur"].visualize(ee_p_w, ee_q_w)
            self.markers[i]["goal"].visualize(g_p_w, g_q_w)

@configclass
class IKCommandCfg(CommandTermCfg):
    
    class_type: type = IKCommand
    
    asset_name: str = MISSING
    legs_config: dict[str, list[str]] = MISSING 
    use_pose_mode: bool = False
    vis_scale: float = 0.1
    

"""" 动作生成
获取地形的check_points ✅
加载npz的MocapInterpolator_map ✅
motion数据结构张量 ✅
储存到一个key_data中(envs,keys,root_pose)[envs, max_frames, 7] (envs,keys,N,targets_pose_b)[envs, max_frames,N,7] ✅
插帧计算函数 env_ids-> 按地形分类-> 插帧-> 返回插帧数据 ,每次 _resample_command 时判断是否需要重新插帧（超过motion_max_episode） ✅
IK更新函数 每次update_command时判断是否需要达到最大time_step_totals ，没有的选下一帧key然后进行IK计算 ✅
在num_epochs内进行_adaptive_sampling ✅
body_linv和body_angv可以通过差分计算 ✅
"""


from go2w_vtm.utils.MocapInterpolator import MocapInterpolator
class MotionGenerator(CommandTerm):
    
    cfg: MotionGeneratorCfg

    def __init__(self, cfg: MotionGeneratorCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.robot: Articulation = env.scene[cfg.asset_name]

        if not isinstance(self._env.scene.terrain, ConfirmTerrainImporter):
            raise ValueError("Terrain must be ConfirmTerrainImporter")
        self.terrain: ConfirmTerrainImporter = self._env.scene.terrain
        self.preprocess_checkpoint_data()
        
        self.terrain_names = self.cfg.terrain_and_checkpoint_file.keys()
        self.terrain_types = [self.terrain.terrain_name2type[name] for name in self.terrain_names]
        self.interpolator_map = {} # terrain_type -> MocapInterpolator
        for terrain_type in self.terrain_types:
            file_path = self.cfg.terrain_and_checkpoint_file[self.terrain.terrain_type2name[terrain_type]]
            self.interpolator_map[terrain_type] = MocapInterpolator(file_path,device=self.device)
            
       # 计算motion轨迹长度 每个地形的最慢速度计算轨迹长度
        max_length = 0
        num_targets = 0
        for terrain_type in self.interpolator_map.keys():
            interpolator: MocapInterpolator = self.interpolator_map[terrain_type]
            env_ids, terrain_key_pos = self.get_data_by_terrain_type(terrain_type)
            if len(env_ids) == 0:
                continue
            # 目前都是vx方向
            cmd_vel = torch.zeros(len(env_ids), 3, device=self.device)
            # 取该地形配置的速度范围中的最小值（最慢速度对应最长轨迹）
            v_min = min(self.cfg.terrain_and_cmd_vel[self.terrain.terrain_type2name[terrain_type]][0][0],
            self.cfg.terrain_and_cmd_vel[self.terrain.terrain_type2name[terrain_type]][0][1])
            cmd_vel[:, 0] = v_min
            total_frames = interpolator.get_total_frames_per_env(terrain_key_pos, cmd_vel, self.cfg.fps)
            max_length = max(max_length, int(torch.max(total_frames).item()))
            num_targets = interpolator.num_targets
            
            
        self.interpolator_dt = 1.0 / self.cfg.fps
        self.time_step_totals = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 插帧数据，root和关键点
        self.interpolator_root_pose_w = torch.zeros(self.num_envs, max_length, 7, device=self.device)
        self.interpolator_tergets_pose_b = torch.zeros(self.num_envs, max_length, num_targets, 7, device=self.device)
        
        self.ik_cmd = IKCommand(self.cfg.ik_cfg, self._env)
        
        # motion数据
        self.joint_indices, _ = self.robot.find_joints(self.cfg.joint_names)

        # track
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        
        # --- 补充自适应采样相关的变量 ---
        control_dt = env.cfg.decimation * env.cfg.sim.dt
        # 这样可以确保统计的分辨率与你的控制频率相匹配
        self.adaptive_bins = int(max_length // (1 / control_dt)) + 1
        self.bin_failed_count = torch.zeros(self.adaptive_bins, device=self.device)
        self._current_bin_failed = torch.zeros(self.adaptive_bins, device=self.device)
        # 卷积核用于平滑采样概率
        self.kernel = torch.ones(self.cfg.adaptive_kernel_size, device=self.device) / self.cfg.adaptive_kernel_size


        # 初始化命令速度缓存 [num_envs, 3] (vx, vy, wz)
        self._cmd_vel = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        
        
    @property
    def command(self) -> torch.Tensor:
        # 拼接：[目标速度(3), 目标关节位置(N), 目标关节速度(N)]
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)
    
    @property
    def velocity_command(self) -> torch.Tensor:
        """返回当前每个环境对应的目标参考速度 [num_envs, 3]"""
        return self._cmd_vel

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.ik_cmd.joint_pos[:, self.joint_indices]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.ik_cmd.joint_vel[:, self.joint_indices]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.ik_cmd.body_pos_w[:, self.body_indexes]+ self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.ik_cmd.body_quat_w[:, self.body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.ik_cmd.body_lin_vel_w[:, self.body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.ik_cmd.body_ang_vel_w[:, self.body_indexes]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.ik_cmd.body_pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.ik_cmd.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.ik_cmd.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.ik_cmd.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        # 只返回 joint_names 指定的关节
        return self.robot.data.joint_pos[:, self.joint_indices]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        # 只返回 joint_names 指定的关节
        return self.robot.data.joint_vel[:, self.joint_indices]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    
            
    def _compute_interpolation(self, env_ids: torch.Tensor):
        """并行计算并存储插值后的 root_pose 和 target_poses"""
        if len(env_ids) == 0:
            return
        current_rows = self.terrain.terrain_levels[env_ids]
        current_cols = self.terrain.terrain_types[env_ids]
        env_terrain_types = self.sub_terrain_type_tensor[current_rows, current_cols]
        
        unique_types = torch.unique(env_terrain_types)
        
        for t_type in unique_types:
            t_type_item = t_type.item()
            if t_type_item not in self.interpolator_map:
                continue
                
            # 找到当前类型匹配的子 env_ids
            type_mask = (env_terrain_types == t_type)
            type_env_ids = env_ids[type_mask]
            
            # 获取 Checkpoints
            m_rows = current_rows[type_mask]
            m_cols = current_cols[type_mask]
            checkpoints = self.checkpoint_buffer[m_rows, m_cols]
            real_n = self.key_counts[m_rows[0], m_cols[0]].item()
            compact_checkpoints = checkpoints[:, :real_n, :]
            
            # 生成随机速度 (从配置的范围采样)
            # 明确提取每个轴的 min 和 max
            vel_cfg = self.cfg.terrain_and_cmd_vel[self.terrain.terrain_type2name[t_type_item]]
            vx_min, vx_max = vel_cfg[0][0], vel_cfg[0][1]
            vy_min, vy_max = vel_cfg[1][0], vel_cfg[1][1]
            wz_min, wz_max = vel_cfg[2][0], vel_cfg[2][1]
            # 构造正确的 low 和 high Tensor
            low = torch.tensor([vx_min, vy_min, wz_min], device=self.device)
            high = torch.tensor([vx_max, vy_max, wz_max], device=self.device)
            cmd_vel = sample_uniform(low, high, (len(type_env_ids), 3), device=self.device)
            self._cmd_vel[type_env_ids] = cmd_vel
            
            # 调用插值器
            interpolator:MocapInterpolator = self.interpolator_map[t_type_item]
            root_pose, target_pose_b = interpolator.interpolate(compact_checkpoints, cmd_vel, self.cfg.fps)
            
            # 存入缓冲区
            num_f = root_pose.shape[1]
            self.interpolator_root_pose_w[type_env_ids, :num_f] = root_pose
            self.interpolator_tergets_pose_b[type_env_ids, :num_f] = target_pose_b
            self.time_step_totals[type_env_ids] = num_f
        
        
    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)


    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            # 使用 self.adaptive_bins (即参考代码的 bin_count) 进行映射
            # 核心逻辑：(当前步 * 总桶数) // 总步数
            current_bin_index = torch.clamp(
                (self.time_steps[env_ids] * self.adaptive_bins) // 
                torch.clamp(self.time_step_totals[env_ids], min=1), 
                0, self.adaptive_bins - 1
            )
            fail_bins = current_bin_index[episode_failed]
            # 记录失败
            self._current_bin_failed.index_add_(0, fail_bins, torch.ones_like(fail_bins, dtype=torch.float))

        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.adaptive_bins)
        
        # 卷积平滑 (让失败点附近的 Bin 也有更高概率被采样)
        if self.cfg.adaptive_kernel_size > 1:
            sampling_probabilities = torch.nn.functional.pad(
                sampling_probabilities.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            sampling_probabilities = torch.nn.functional.conv1d(
                sampling_probabilities, self.kernel.view(1, 1, -1)
            ).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        # 将采样到的 Bin 转化为具体的 time_steps
        # 在该 Bin 范围内随机偏移，避免每次都从 Bin 的开头开始
        bin_offsets = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        self.time_steps[env_ids] = (
            (sampled_bins.float() + bin_offsets) / self.adaptive_bins * (self.time_step_totals[env_ids].float() - 1)
        ).long()

        # 5. 更新 Metrics (用于 Tensorboard 观察采样熵)
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        self.metrics["sampling_entropy"][:] = H / math.log(self.adaptive_bins)
        
    def _resample_command(self, env_ids: torch.Tensor):
        # 1. 轨迹重新插值与自适应采样 (保持原有逻辑)
        self._compute_interpolation(env_ids)
        self._adaptive_sampling(env_ids)
        # 2. 获取采样时间步对应的目标参考数据
        ee_targets_b = self.interpolator_tergets_pose_b[torch.arange(self.num_envs, device=self.device), self.time_steps]
        # 接下来进行 IK 解算
        with torch.no_grad():
            target_q = self.ik_cmd.compute_ik(ee_targets_b)
        # 写入真实机器人
        root_pose_ref = self.interpolator_root_pose_w[env_ids, self.time_steps[env_ids]]
        root_states = torch.zeros((len(env_ids), 13), device=self.device)
        root_states[:, 0:3] = root_pose_ref[:, :3]
        root_states[:, 3:7] = root_pose_ref[:, 3:7]
        root_states[:, 7:13] = 0.0 # 重置时速度清零
        self.robot.write_root_state_to_sim(root_states, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(target_q[env_ids], torch.zeros_like(target_q[env_ids]), env_ids=env_ids)
        # 同步影子机器人历史，防止下一帧速度爆炸
        self.ik_cmd.reset_ghost_robot(
            env_ids, 
            root_pose_ref[:, :3], 
            root_pose_ref[:, 3:7], 
            target_q[env_ids]
        )

    def _update_command(self):
        # 1. 推进时间步
        self.time_steps += 1
        
        # 2. 检查是否需要重置环境轨迹
        reset_mask = self.time_steps >= (self.time_step_totals - 1)
        reset_env_ids = torch.where(reset_mask)[0]
        if len(reset_env_ids) > 0:
            self._resample_command(reset_env_ids)

        # 3. 获取当前步目标位姿
        all_ids = torch.arange(self.num_envs, device=self.device)
        root_pose_ref = self.interpolator_root_pose_w[all_ids, self.time_steps]
        ee_targets_b = self.interpolator_tergets_pose_b[all_ids, self.time_steps]

        # 4. 调用迁移后的 IK+FK 一体化函数
        # 这一步会自动更新 ik_cmd 内部的 joint_pos, body_pos_w, body_vel 等全量数据
        self.ik_cmd.compute_ik_and_fk(
            root_pose_ref[:, :3], 
            root_pose_ref[:, 3:], 
            ee_targets_b, 
            self.interpolator_dt
        )

        # 5. 更新统计量与指标
        self._update_relative_poses()
        self._update_metrics()
        
    def _update_relative_poses(self):
        """将参考 FK 状态投影到机器人当前的水平航向坐标系下"""
        num_bodies = self.body_pos_relative_w.shape[1] # 获取 body 数量 (如 17 或 4)
        # 1. 获取参考轨迹锚点 (Motion Reference Anchor)
        ref_anchor_pos = self.anchor_pos_w # 参考轨迹的当前位置
        ref_anchor_quat = self.anchor_quat_w # 参考轨迹的当前姿态
        # 2. 获取机器人真实锚点 (Robot Real Anchor)
        robot_anchor_pos = self.robot_anchor_pos_w # 机器人当前实时的位置
        robot_anchor_quat = self.robot_anchor_quat_w # 机器人当前实时的姿态
        # 3. 计算目标变换中心：取机器人当前的水平坐标 (XY)，但保留参考轨迹的垂直高度 (Z)
        target_pos_w = robot_anchor_pos.clone()
        target_pos_w[:, 2] = ref_anchor_pos[:, 2] # 确保参考动作的高度与设计一致
        # 4. 计算航向对齐量 (Yaw Alignment)：计算机器人 Yaw 与参考 Yaw 的差值
        # 使用 yaw_quat 提取纯偏航角的四元数
        target_yaw_quat = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))
        # 5. 显式扩展对齐量以适配 TorchScript 的严格形状检查
        # 将 [num_envs, 4] 扩展为 [num_envs, num_bodies, 4]
        target_yaw_quat_expanded = target_yaw_quat[:, None, :].expand(-1, num_bodies, -1)
        # 6. 批量应用变换
        # A. 姿态对齐：所有 body 旋转到机器人当前航向
        self.body_quat_relative_w[:] = quat_mul(target_yaw_quat_expanded, self.body_quat_w)
        # B. 位置对齐：目标中心坐标 + 旋转后的相对偏移量
        # 首先计算参考 body 相对于参考锚点的偏移
        rel_pos = self.body_pos_w - ref_anchor_pos[:, None, :]
        # 应用旋转并加回目标中心
        self.body_pos_relative_w[:] = target_pos_w[:, None, :] + quat_apply(target_yaw_quat_expanded, rel_pos)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
            

    def preprocess_checkpoint_data(self):
        """处理变长数据：对齐到最大长度并生成长度记录表"""
        # 1. 找到全局最大 Key 数量
        max_keys = max([data.shape[0] for data in self.terrain.terrains_checkpoint_data.values()])
        num_rows, num_cols = self.terrain.sub_terrain_type.shape
        # 2. 创建缓冲区和长度记录表
        # checkpoint_buffer: [rows, cols, max_keys, 3]
        self.checkpoint_buffer = torch.zeros((num_rows, num_cols, max_keys, 3), device=self.device)
        # key_counts: 记录每个格子的真实 key 数量 [rows, cols]
        self.key_counts = torch.zeros((num_rows, num_cols), device=self.device, dtype=torch.long)
        # 3. 填充数据并进行 Padding
        for (r, c), data in self.terrain.terrains_checkpoint_data.items():
            n = data.shape[0]
            data_tensor = torch.from_numpy(data).to(self.device).float()
            # 填充真实数据
            self.checkpoint_buffer[r, c, :n] = data_tensor
            # 关键：Padding 剩余部分。使用最后一个点填充，这样插值到后面时会停留在终点
            if n < max_keys:
                self.checkpoint_buffer[r, c, n:] = data_tensor[-1]
            self.key_counts[r, c] = n
        self.sub_terrain_type_tensor = self.terrain.sub_terrain_type
        
        
    def get_data_by_terrain_type(self, target_type):
        """
        给定地形类型，返回在该地形上的 env_ids 和紧凑的(无冗余Padding) checkpoints。
        """
        current_rows = self.terrain.terrain_levels
        current_cols = self.terrain.terrain_types
        # 1. 查找匹配的环境
        env_terrain_types = self.sub_terrain_type_tensor[current_rows, current_cols]
        matching_env_ids = (env_terrain_types == target_type).nonzero(as_tuple=True)[0]
        if matching_env_ids.shape[0] == 0:
            return matching_env_ids, torch.empty(0)
        # 2. 提取数据
        m_rows = current_rows[matching_env_ids]
        m_cols = current_cols[matching_env_ids]
        raw_checkpoints = self.checkpoint_buffer[m_rows, m_cols]
        # 3. 既然“同类型长度相同”，我们只需要看第一个匹配项的真实长度
        real_n = self.key_counts[m_rows[0], m_cols[0]].item()
        # 4. 切掉多余的 Padding，返回紧凑的 [num_envs, real_n, 3]
        compact_checkpoints = raw_checkpoints[:, :real_n, :]
        return matching_env_ids, compact_checkpoints
    

@configclass 
class MotionGeneratorCfg(CommandTermCfg):
    
    class_type: type = MotionGenerator
    
    ik_cfg: IKCommandCfg = MISSING
    
    asset_name: str = MISSING # robot
    
    fps: int = MISSING #重要
    
    terrain_and_checkpoint_file: dict[str, list[str]] = MISSING # [地形名, 检查点npz文件] 
    terrain_and_cmd_vel: dict[str, list[tuple[float, float]]] = MISSING # [地形名, 速度范围] (vx, vy, wz)

    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    joint_names: list[str] = MISSING
    
    motion_max_episode: int = 10

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
