# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import numpy as np
import scipy.spatial.transform as tf
import torch
import trimesh
from typing import TYPE_CHECKING
from isaaclab.terrains.trimesh.utils import make_plane

if TYPE_CHECKING:
    from . import mesh_terrains_cfg


def floating_boxes_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshFloatingBoxesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """
    生成一个带有随机悬浮方块的地形。

    这个地形包含一个平坦的地面和一个中心平台，以及多个随机放置的、
    尺寸和悬浮高度可变的悬浮方块。
    """
    # 初始化网格列表
    meshes_list = []

    # -- 1. 解析课程参数 --
    p_start = cfg.box_params_start
    p_end = cfg.box_params_end

    # 使用 difficulty (0.0 -> 1.0) 在起始和结束参数之间进行线性插值
    num_boxes = int(p_start.num_boxes + difficulty * (p_end.num_boxes - p_start.num_boxes))
    
    size_min = np.array(p_start.size_min) + difficulty * (np.array(p_end.size_min) - np.array(p_start.size_min))
    size_max = np.array(p_start.size_max) + difficulty * (np.array(p_end.size_max) - np.array(p_start.size_max))

    height_min = p_start.floating_height_min + difficulty * (p_end.floating_height_min - p_start.floating_height_min)
    height_max = p_start.floating_height_max + difficulty * (p_end.floating_height_max - p_start.floating_height_max)

    # -- 2. 创建基础地面 --
    ground_plane = make_plane(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)

    # -- 3. 创建中心安全平台 (可选，但推荐) --
    # 平台稍高于地面，以防万一
    platform_height = 0.01
    platform_pos = (0.5 * cfg.size[0], 0.5 * cfg.size[1], platform_height / 2)
    platform_dims = (cfg.platform_width, cfg.platform_width, platform_height)
    platform = trimesh.creation.box(platform_dims, trimesh.transformations.translation_matrix(platform_pos))
    meshes_list.append(platform)
    
    # -- 4. 生成悬浮方块 --
    # 计算平台安全区边界，防止障碍物生成在平台上
    platform_radius = cfg.platform_width * 0.707 # 使用外接圆半径作为安全区
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1]])

    for _ in range(num_boxes):
        # 随机化方块尺寸
        box_size = np.random.uniform(size_min, size_max)

        # 随机化悬浮高度
        floating_height = np.random.uniform(height_min, height_max)

        # 随机化位置，直到找到一个不在平台上的位置
        while True:
            pos_x = np.random.uniform(0, cfg.size[0])
            pos_y = np.random.uniform(0, cfg.size[1])
            if np.linalg.norm(np.array([pos_x, pos_y]) - terrain_center) > platform_radius:
                break
        
        # 计算方块中心点的3D坐标
        # Z = 离地高度 + 方块自身高度的一半
        box_center_pos = [pos_x, pos_y, floating_height + box_size[2] / 2]

        # 使用 trimesh 创建方块
        box_mesh = trimesh.creation.box(box_size, trimesh.transformations.translation_matrix(box_center_pos))
        meshes_list.append(box_mesh)

    # -- 5. 定义机器人出生点 --
    # 通常是平台的中心
    origin = np.array([terrain_center[0], terrain_center[1], 0.05]) # 稍高于平台

    return meshes_list, origin