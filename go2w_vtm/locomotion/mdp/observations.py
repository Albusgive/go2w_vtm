# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
import isaaclab.utils.math as math_utils
from go2w_vtm.locomotion.mdp.commands import MotionCommand


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


def lidar(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # sensor._set_debug_vis_impl(True)
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset


import torch.nn.functional as F
def image_max(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    min: float = 0.25,
    max: float = 2.0,
    zero_p: float = 0.05,
    noise_range: float = 0.1,
    noise_level: str = "high",
    # 新增参数用于基于梯度的噪声
    max_gradient: float = 0.5,      # 最大梯度阈值
    min_gradient: float = 0.1,      # 最小梯度阈值
    high_probability: float = 0.8,  # 高梯度区域的置零概率
    low_probability: float = 0.1,   # 低梯度区域的置零概率
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor."""
    # 提取传感器数据
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]

    # 深度图像转换
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # RGB/深度/法线图像归一化
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = max
        elif "normals" in data_type:
            images = (images + 1.0) * 0.5
    
    # 保存原始图像用于梯度计算
    original_images = images.clone()
    
    # 添加均匀分布的噪声
    images = images + (torch.rand_like(images) * noise_range * 2 - noise_range)
    images = images.clamp(min=min, max=max)
    dif = max - min
    images = (images - min) / dif  # 归一化到[0, 1]

    # 确保输入形状为 [n_envs, w, h, 1]
    # 移除最后一个维度（通道维度），因为它是1
    original_images = original_images.squeeze(-1)  # 形状变为 [n_envs, w, h]
    
    # 添加通道维度，使其变为 [n_envs, 1, w, h] 适合卷积
    original_4d = original_images.unsqueeze(1)  # 形状变为 [n_envs, 1, w, h]
    
    # 计算梯度 (模拟Scharr算子)
    scharr_x_kernel = torch.tensor([[-3, 0, 3], 
                                    [-10, 0, 10], 
                                    [-3, 0, 3]], dtype=torch.float32, device=images.device)
    scharr_y_kernel = torch.tensor([[-3, -10, -3], 
                                    [0, 0, 0], 
                                    [3, 10, 3]], dtype=torch.float32, device=images.device)
    
    # 扩展卷积核维度 [out_channels=1, in_channels=1, height=3, width=3]
    scharr_x_kernel = scharr_x_kernel.view(1, 1, 3, 3)
    scharr_y_kernel = scharr_y_kernel.view(1, 1, 3, 3)
    
    # 计算x和y方向的梯度
    scharr_x = F.conv2d(original_4d, scharr_x_kernel, padding=1)
    scharr_y = F.conv2d(original_4d, scharr_y_kernel, padding=1)
    
    # 计算梯度幅值
    scharr_grad = torch.sqrt(scharr_x**2 + scharr_y**2)
    
    # 移除通道维度，形状变为 [n_envs, w, h]
    scharr_grad = scharr_grad.squeeze(1)
    
    # 计算梯度差异
    _gradient_dif = max_gradient - min_gradient
    
    # 确保所有张量形状一致
    # 首先，确保 images 的形状也是 [n_envs, w, h]
    if images.dim() == 4 and images.shape[-1] == 1:
        images = images.squeeze(-1)
    
    # 创建基于梯度的置零掩码，形状为 [n_envs, w, h]
    gradient_mask = torch.zeros_like(images, dtype=torch.bool)
    
    # 高梯度区域
    high_grad_mask = scharr_grad > max_gradient
    # 确保 high_prob_mask 的形状与 high_grad_mask 相同
    high_prob_mask = torch.rand_like(scharr_grad) < high_probability
    gradient_mask |= (high_grad_mask & high_prob_mask)
    
    # 中等梯度区域
    mid_grad_mask = (scharr_grad > min_gradient) & (scharr_grad <= max_gradient)
    if mid_grad_mask.any():
        # 计算每个像素的置零概率
        p_values = ((scharr_grad[mid_grad_mask] - min_gradient) / _gradient_dif) * \
                   (high_probability - low_probability) + low_probability
        # 生成随机数并与概率比较
        mid_prob_mask = torch.rand_like(p_values) < p_values
        # 应用置零
        gradient_mask[mid_grad_mask] = mid_prob_mask
    
    # 应用基于梯度的置零
    images[gradient_mask] = 0
    
    # 应用原有的随机置零
    # 确保 mask 的形状与 images 相同
    mask = torch.rand_like(images) > zero_p
    images = images * mask
    
    # 如果需要，恢复原始形状 [n_envs, w, h, 1]
    if env.scene.sensors[sensor_cfg.name].data.output[data_type].shape[-1] == 1:
        images = images.unsqueeze(-1)
                
    return images


def generated_commands_low_then_pos_z(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    low_z: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
    ) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name).clone()
    min_z = torch.min(asset.data.body_pos_w[:, asset_cfg.body_ids, 2],dim=1).values
    heigh_idx = min_z > low_z
    cmd[heigh_idx,3] = 0.0
    return cmd


def joint_pos_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel

''' mimic '''
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms
from go2w_vtm.locomotion.mdp.commands import MotionCommand


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)



def motion_commands_and_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """The motion command from command term in the command manager with the given name."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.command_with_vel


def motion_velocity_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """The motion command from command term in the command manager with the given name."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.velocity_command