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
    quat_rotate_inverse
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
        lin_vel_local = quat_rotate_inverse(quat_w, lin_vel_w)  # [E, 3]
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

        episode_failed = self._env.termination_manager.terminated[env_ids]
        current_motion_ids = self.motion_ids[env_ids]  # [E]

        if torch.any(episode_failed):
            # Compute normalized time (0～1) for each env
            current_time_norm = self.time_steps[env_ids].float() / self.motion.time_step_totals[current_motion_ids].float()
            # Map to bin index using global bin_count
            current_bin_index = (current_time_norm * self.bin_count).long().clamp(0, self.bin_count - 1)

            # Accumulate failures per motion
            fail_mask = episode_failed
            fail_motion_ids = current_motion_ids[fail_mask]      # [F]
            fail_bins = current_bin_index[fail_mask]            # [F]

            # Use scatter_add to accumulate per motion per bin
            self._current_bin_failed.zero_()
            self._current_bin_failed.view(-1).index_add_(
                0,
                fail_motion_ids * self.bin_count + fail_bins,
                torch.ones_like(fail_bins, dtype=torch.float)
            )

        # Now sample per env based on its motion's failure stats
        new_time_steps = torch.empty_like(self.time_steps[env_ids])

        # Group envs by motion_id
        for mid in range(self.motion.num_motions):
            mask = (current_motion_ids == mid)
            if not mask.any():
                continue

            # Get failure stats for this motion
            sampling_probs = self.bin_failed_count[mid] + self.cfg.adaptive_uniform_ratio / self.bin_count
            sampling_probs = torch.nn.functional.pad(
                sampling_probs.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate"
            )
            sampling_probs = torch.nn.functional.conv1d(sampling_probs, self.kernel.view(1, 1, -1)).view(-1)
            sampling_probs = sampling_probs / sampling_probs.sum()

            n_sample = mask.sum().item()
            sampled_bins = torch.multinomial(sampling_probs, n_sample, replacement=True)

            # Convert bin → time step (clamped to this motion's actual length)
            max_t = self.motion.time_step_totals[mid].item()
            new_time_steps[mask] = (
                (sampled_bins.float() + torch.rand(n_sample, device=self.device))
                / self.bin_count
                * (max_t - 1)
            ).long().clamp(0, max_t - 1)

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

        self.anchor_lin_vel_local()
        
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
            if not hasattr(self, "current_anchor_visualizer"):
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

        else:
            if hasattr(self, "current_anchor_visualizer"):
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)
                    self.current_ref_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])
            self.current_ref_body_visualizers[i].visualize(self.body_pos_w[:, i], self.body_quat_w[:, i])
            


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
