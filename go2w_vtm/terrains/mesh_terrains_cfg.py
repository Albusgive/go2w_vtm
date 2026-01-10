# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings
from dataclasses import MISSING
from typing import Literal

import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.utils as mesh_utils_terrains
from isaaclab.utils import configclass

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from go2w_vtm.terrains import mesh_terrains

@configclass
class MeshFloatingBoxesTerrainCfg(SubTerrainBaseCfg):
    """为带有多个可配置的悬浮方块的地形进行配置。"""

    function = mesh_terrains.floating_boxes_terrain

    @configclass
    class BoxObjectCfg:
        num_boxes: int = MISSING
        size_min: tuple[float, float, float] = MISSING
        size_max: tuple[float, float, float] = MISSING
        floating_height_min: float = MISSING
        floating_height_max: float = MISSING

    # 为课程学习定义开始和结束参数
    box_params_start: BoxObjectCfg = MISSING
    box_params_end: BoxObjectCfg = MISSING

    platform_width: float = 1.0
    """中心安全平台的宽度 (m)。"""