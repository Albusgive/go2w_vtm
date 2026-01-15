# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.terrains.height_field import HfTerrainBaseCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from go2w_vtm.terrains import mimic_gym_terrain
from go2w_vtm.terrains import terrain_2_mjcf
from collections.abc import Callable
import trimesh
import numpy as np
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from go2w_vtm.utils import make_terrain_check_point


@configclass
class SaveTerrainCfg(SubTerrainBaseCfg):

    save_to_mjcf: bool = False
    mesh_path: str = None
    mjcf_path: str = None
    terrain_name: str = MISSING
    
    def make_check_points(self, difficulty: float):
        return None
    

@configclass
class MimicFixBoxTerrainCfg(SaveTerrainCfg):
    
    function = mimic_gym_terrain.fix_box_terrain
    ''' 下列参数均是基于机器人坐标偏移 '''
    high_platform_x: list[float] = MISSING                 
    high_platform_z: list[float] = MISSING                     
    high_platform_half_width: list[float] = MISSING                       
    high_platform_half_height: list[float] = MISSING  
    
    robot_origin_x: float = 1.5
    
    terrain_name: str = "trench_box_terrain"



@configclass
class MimicHighPlatformTerrainCfg(SaveTerrainCfg):
    
    function = mimic_gym_terrain.high_platform_terrain
    high_platform_start_x: list[float] = MISSING                     
    high_platform_width: list[float] = MISSING                       
    high_platform_height: list[float] = MISSING  
    
    robot_origin_x: float = 1.5
    
    terrain_name: str = "high_platform"
    
    

''' 全新地形 ''' 
@configclass
class BoxTrenchTerrainCfg(SaveTerrainCfg):
    ''' 两块box组成的沟壑 '''
    function = mimic_gym_terrain.box_trench_terrain
    
    ''' 在机器人面前 trench_x位置为沟壑边缘 宽度为trench_width'''
    trench_x: float = MISSING                                   
    trench_width: tuple[float, float] = MISSING  #宽度范围
    trench_depth: float = MISSING
    
    
    ''' 机器人x坐标 '''
    robot_origin_x: float = 1.5
    
    terrain_name: str = "trench_box_terrain"
    
    ''' 自动计算中间key_pos 然后k帧数据保存key_name和相对于key的pos 
    返回的为 terrain_key_pos_list 为 terrain 坐标系,即原点在右下'''
    def make_check_points(self, difficulty: float) ->list[np.ndarray]:
        from . import make_terrain_check_point
        return make_terrain_check_point.makeBoxTrenchTerrainCheckPoint(
            difficulty=difficulty,
            size=self.size,
            trench_x=self.trench_x,
            trench_width=self.trench_width,
            robot_origin_x=self.robot_origin_x,
            init_key_pos_distance=self.init_key_pos_distance,
            end_key_pos_distance=self.end_key_pos_distance)
    
    init_key_pos_distance: float = 0.1 # 距离跳跃前沟壑边缘的距离
    end_key_pos_distance: float = 0.1 # 距离跳跃后沟壑边缘的距离
    
    

