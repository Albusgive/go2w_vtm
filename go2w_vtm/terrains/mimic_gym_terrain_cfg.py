# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from go2w_vtm.terrains import mimic_gym_terrain

@configclass
class MimicTrenchTerrainCfg(HfTerrainBaseCfg):
    
    function = mimic_gym_terrain.trench_terrain
    trench_start_x: float = 2.0                     # 沟中心在 X 轴的位置（米，从地形中心起算）
    trench_width: float = 0.4                       # 沟宽（米）
    trench_depth: float = 0.3                       # 沟深（米）
    
    save_to_mjcf: bool = False
    png_path: str = None
    mjcf_path: str = None
    save_name: str = "trench_terrain"
    

@configclass
class MimicFixBoxTerrainCfg(SubTerrainBaseCfg):
    
    function = mimic_gym_terrain.fix_box_terrain
    
    ''' 下列参数均是基于机器人坐标偏移 '''
    high_platform_x: list[float] = MISSING                 
    high_platform_z: list[float] = MISSING                     
    high_platform_half_width: list[float] = MISSING                       
    high_platform_half_height: list[float] = MISSING  
    
    robot_origin_x: float = 1.5
    
    save_to_mjcf: bool = False
    mesh_path: str = None
    mjcf_path: str = None
    save_name: str = "trench_box_terrain"
    
    
@configclass
class MimicHighPlatformTerrainCfg(SubTerrainBaseCfg):
    
    function = mimic_gym_terrain.high_platform_terrain
    high_platform_start_x: list[float] = MISSING                     
    high_platform_width: list[float] = MISSING                       
    high_platform_height: list[float] = MISSING  
    
    robot_origin_x: float = 1.5
    
    save_to_mjcf: bool = False
    mesh_path: str = None
    mjcf_path: str = None
    save_name: str = "high_platform"
    
    

    