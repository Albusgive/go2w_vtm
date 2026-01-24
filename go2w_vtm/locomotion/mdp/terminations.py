from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from dataclasses import MISSING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm
    
import go2w_vtm.locomotion.mdp as mdp

def drop_groove(
    env: ManagerBasedRLEnv, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    角度大于一定程度并且z小于0
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) > 0:
        termination = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle
        termination = termination & (asset.data.body_pos_w[:, asset_cfg.body_ids[0], 2] < 0)
    else:
        termination = torch.full((env.scene.num_envs,),False,dtype=torch.bool,device=env.scene.device)
    return termination


def long_time_no_move(
    env: ManagerBasedRLEnv, limit_time: float, limit_distance: float, command_name: str,
    limit_cmd: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    长时间不移动
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # 初始化
    if env.cfg.long_time_no_move_flag:
        env.long_time_step = torch.zeros(env.num_envs,device=env.device)
        env.root_last_pos_w = asset.data.root_pos_w.clone()
        env.last_long_time_termination = torch.full((env.num_envs,),True,device=env.device,dtype=torch.bool)
        env.cfg.long_time_no_move_flag = False
    
    # 判断命令
    cmd = env.command_manager.get_command(command_name)
    # cmd大并且自身速度更不上cmd的一半
    cmd_vel = torch.norm(cmd[:, :2], dim=1)
    cmd_move = cmd_vel > limit_cmd
    robot_move = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1) <  (cmd_vel * 0.5)
    no_long_time = cmd_move & robot_move
    no_long_time = ~no_long_time
    env.long_time_step[~no_long_time] += 1
    env.long_time_step[no_long_time] = 0
    # 到时间判断位置
    long_time = (env.long_time_step * env.cfg.decimation * env.cfg.sim.dt) > limit_time
    termination = torch.norm((asset.data.root_pos_w - env.root_last_pos_w), dim=1) < limit_distance
    termination = termination & long_time
    # 重置清0
    env.long_time_step[termination] = 0
    # 更新位置 对于不记录的和上次重置的
    env.root_last_pos_w[env.last_long_time_termination|no_long_time] = \
    asset.data.root_pos_w[env.last_long_time_termination|no_long_time]
    env.last_long_time_termination = termination
    return termination
    

def jump_timely(
    env: ManagerBasedRLEnv, limit_distance: float, command_name: str = None,
) -> torch.Tensor:
    """
    跳跃距离跟踪点太远
    """
    if command_name is None:
        return torch.zeros(env.num_envs,dtype=torch.bool, device=env.device)
    # extract the used quantities (to enable type-hinting)
    cmd: mdp.JumpTrackCommand = env.command_manager.get_term(command_name)
    cmd_pos = cmd.command[:,:3]
    termination = torch.zeros(env.num_envs,dtype=torch.bool, device=env.device)
    # 获取跳转环境的索引
    jump_indices = torch.where(cmd.jump_mask)[0]
    # 计算错误掩码
    err_mask = torch.norm(
        cmd.robot.data.body_link_pose_w[cmd.jump_mask, cmd.body_idx, :3] - 
        cmd_pos[cmd.jump_mask], 
        dim=1
    ) > limit_distance
    termination[jump_indices[err_mask]] = True
    cmd.termination(jump_indices[err_mask])
    return termination


def jump_state(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    lowing_time: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    仅置0跳跃cmd,不对环境进行终止
    如果从跳跃命令生成开始，出现腾空到速度向下就结束跳跃命令
    """
    cmd: mdp.JumpAccCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    
    # in_air = torch.all(contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids],dim=1)
    
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norms = torch.norm(contacts, dim=2)
    in_air = torch.min(force_norms,dim=1).values > 1.0
    
    landed = ~in_air
    
    cmd.jump_mask = cmd.command[:,0] > 0
    
    # flying并且速度向下
    linv_z = asset.data.root_lin_vel_w[:, 2]
    cmd.flying_down[:] = False
    # 在空中,上一帧的flying_up为True
    cmd.flying_down[in_air & (linv_z < 0.0) & cmd.flying_up] = True
    cmd.command[cmd.flying_down] = 0.0
    
    cmd.flying_up[:] = False
    cmd.flying_up[in_air & (linv_z >= 0.0) & cmd.jump_mask] = True
    
    # 有cmd或者max_heigh并且离开地面
    cmd.flying[:] = cmd.flying_up | cmd.flying_down
    
    # 从给定命令的一段时间内为下蹲状态
    cmd.lowing[:] = False
    cmd.lowing[(cmd.cmd_time < lowing_time)&cmd.jump_mask&landed] = True
    # 有命令并且过了下蹲时间
    cmd.uping[:] = False
    cmd.uping[(cmd.cmd_time >= lowing_time)&cmd.jump_mask&landed] = True
    cmd.cmd_time += env.step_dt
    

    return torch.zeros(env.num_envs,dtype=torch.bool, device=env.device)


''' mimic '''
import isaaclab.utils.math as math_utils
from go2w_vtm.locomotion.mdp.commands import MotionCommand
from go2w_vtm.locomotion.mdp.rewards import _get_body_indexes


def mul_bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def mul_bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def mul_bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def mul_bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def mul_bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


''' mul motion '''
from go2w_vtm.locomotion.mdp.rewards import motion_generator_get_body_indexes
from go2w_vtm.locomotion.mdp.commands import MotionGenerator

def mul_bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionGenerator = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def mul_bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionGenerator = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def mul_bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionGenerator = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def mul_bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionGenerator = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def mul_bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionGenerator = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def test_termination_all(env: ManagerBasedRLEnv) -> torch.Tensor:
    
    return torch.ones(env.num_envs,dtype=torch.bool, device=env.device)