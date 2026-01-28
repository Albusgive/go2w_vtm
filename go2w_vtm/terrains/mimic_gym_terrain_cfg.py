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
from typing import List, Tuple
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
    
    ''' 机器人x坐标 '''
    terrain_x: float = MISSING  #地形距离机器人x坐标的距离
    robot_origin_x: float = 1.5
    
    def make_check_points(self, difficulty: float)->Tuple[List[np.ndarray], List[str]]:
        return None, None
    

@configclass
class MimicFixBoxTerrainCfg(SaveTerrainCfg):
    
    function = mimic_gym_terrain.fix_box_terrain
    ''' 下列参数均是基于机器人坐标偏移 '''
    high_platform_x: list[float] = MISSING                 
    high_platform_z: list[float] = MISSING                     
    high_platform_half_width: list[float] = MISSING                       
    high_platform_half_height: list[float] = MISSING  
    
    robot_origin_x: float = 1.5
    
    terrain_x: float = 0.0
    
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
    
    ''' 在机器人面前 trench_x位置为沟壑边缘(距离机器人) 宽度为trench_width'''                                 
    trench_width: tuple[float, float] = MISSING  #宽度范围
    trench_depth: float = MISSING
    
    terrain_name: str = "box_trench_terrain"
    
    ''' 自动计算中间key_pos 然后k帧数据保存key_name和相对于key的pos 
    返回的为 terrain_key_pos_list 为 terrain 坐标系,即原点在右下
    '''
    def make_check_points(self, difficulty: float) ->Tuple[List[np.ndarray], List[str]]:
        from . import make_terrain_check_point
        return make_terrain_check_point.makeBoxTrenchTerrainCheckPoint(
            difficulty=difficulty,
            size=self.size,
            terrain_x=self.terrain_x,
            robot_origin_x=self.robot_origin_x,
            trench_width=self.trench_width)
    
    
@configclass
class BoxHighPlatformTerrainCfg(SaveTerrainCfg):
    ''' 两块box组成的地面和平台 '''
    function = mimic_gym_terrain.box_platform_terrain
    
    ''' 在机器人面前 terrain_x 位置为平台边缘(距离机器人) 宽度为platform_width'''                                 
    platform_width: float = MISSING  #宽度
    platform_height: tuple[float, float] = MISSING #高度范围
    
    
    terrain_name: str = "box_platform_terrain"
    
    ''' 自动计算中间key_pos 然后k帧数据保存key_name和相对于key的pos 
    返回的为 terrain_key_pos_list 为 terrain 坐标系,即原点在右下'''
    def make_check_points(self, difficulty: float) ->Tuple[List[np.ndarray], List[str]]:
        from . import make_terrain_check_point
        return make_terrain_check_point.makeBoxHighPlatformTerrainCheckPoint(
            difficulty=difficulty,
            size=self.size,
            terrain_x=self.terrain_x,
            robot_origin_x=self.robot_origin_x,
            platform_width=self.platform_width,
            platform_height=self.platform_height,)

    

@configclass
class BoxRockFissureTerrainCfg(SaveTerrainCfg):
    ''' 三块box组成的地面和垂直狭缝 rock_fissure'''
    function = mimic_gym_terrain.box_rock_fissure_terrain
    
    ''' 在机器人面前 terrain_x 位置为裂缝边缘(距离机器人) 宽度为platform_width'''
         
    rock_fissure_long: float = MISSING                     
    rock_fissure_width: tuple[float, float] = MISSING  # 宽度范围
    rock_fissure_height: float = MISSING #高度
    
    terrain_name: str = "box_rock_fissure_terrain"
    
    ''' 自动计算中间key_pos 然后k帧数据保存key_name和相对于key的pos 
    返回的为 terrain_key_pos_list 为 terrain 坐标系,即原点在右下'''
    def make_check_points(self, difficulty: float) ->Tuple[List[np.ndarray], List[str]]:
        from . import make_terrain_check_point
        return make_terrain_check_point.makeBoxRockFissureTerrainCheckPoint(
            difficulty=difficulty,
            size=self.size,
            terrain_x=self.terrain_x,
            robot_origin_x=self.robot_origin_x,
            rock_fissure_long=self.rock_fissure_long,
            rock_fissure_width=self.rock_fissure_width,
            rock_fissure_height=self.rock_fissure_height,)

    
    
@configclass
class BoxFloatBoxTerrainCfg(SaveTerrainCfg):
    ''' 两块box组成的地面和悬空box '''
    function = mimic_gym_terrain.box_float_box_terrain

    ''' 在机器人面前 terrain_x 位置为float box边缘(距离机器人) 宽度为platform_width''' 
    float_box_long: float = MISSING   # 地形长度
    float_box_float_height: tuple[float, float] = MISSING #悬浮高度 
    float_box_height: tuple[float, float] = MISSING #高度 随机
    
    
    terrain_name: str = "box_float_box_terrain"
    
    ''' 自动计算中间key_pos 然后k帧数据保存key_name和相对于key的pos 
    返回的为 terrain_key_pos_list 为 terrain 坐标系,即原点在右下'''
    def make_check_points(self, difficulty: float) ->Tuple[List[np.ndarray], List[str]]:
        from . import make_terrain_check_point
        return make_terrain_check_point.makeBoxFloatBoxTerrainCheckPoint(
            difficulty=difficulty,
            size=self.size,
            terrain_x=self.terrain_x,
            robot_origin_x=self.robot_origin_x,
            float_box_long=self.float_box_long,
            float_box_float_height=self.float_box_float_height,
            float_box_height=self.float_box_height,)