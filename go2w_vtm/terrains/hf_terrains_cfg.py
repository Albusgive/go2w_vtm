# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from go2w_vtm.terrains import hf_terrains

@configclass
class RectangularTerrainCfgX(HfTerrainBaseCfg):
    """
    按照x方向阵列,适合跑酷向前地形
    """

    function = hf_terrains.rectangular_terrain

    obstacle_height_mode: str = "convex"
    '''  convex or concave 凸台和凹陷'''
    
    obstacle_width_range: tuple[float, float] = MISSING
    ''' x m'''
    
    obstacle_length_range: tuple[float, float] = MISSING
    ''' y m'''

    obstacle_height_range: tuple[float, float] = MISSING
    ''' z  m'''

    num_obstacles: int = MISSING

    platform_width: float = 1.0
    
    
@configclass
class RectangularTerrainCfgAround(HfTerrainBaseCfg):

    function = hf_terrains.rectangular_terrain_around

    obstacle_height_mode: str = "convex"
    ''' 凸台或凹陷模式: "convex" 或 "concave" '''
    
    ring_width_range: tuple[float, float] = MISSING
    ''' 环宽度范围 (最小值, 最大值) '''
    
    ring_spacing_range: tuple[float, float] = MISSING
    ''' 环间距范围 (最小值, 最大值) '''

    obstacle_height_range: tuple[float, float] = MISSING
    ''' 高度/深度范围 (最小值, 最大值) '''

    platform_width: float = 1.0
    ''' 中心平台宽度 '''
    
    difficulty_width: float = 0.1
    '''  概率范围 '''
    
    
@configclass
class ShaftTerrainCfgAround(HfTerrainBaseCfg):

    function = hf_terrains.shaft_terrain_around

    obstacle_height_range: tuple[float, float] = MISSING
    ''' 井高度范围 '''
    
    bevel_angle: float = 45.0
    ''' 边缘倒角 '''

    platform_width: float = 1.0
    ''' 井中尺寸 '''
    
    difficulty_width: float = 0.1
    '''  概率范围 '''
