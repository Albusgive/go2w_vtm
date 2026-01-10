from __future__ import annotations

import numpy as np
import math
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING
from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg

@height_field_to_mesh
def rectangular_terrain(difficulty: float, cfg: hf_terrains_cfg.RectangularTerrainCfgX) -> np.ndarray:
    """
    生成一个带有随机生成的矩形障碍物的地形。

    这个地形在中心有一个平坦的平台，周围散布着随机生成的长方体障碍物。
    这些障碍物的宽度、长度和高度都在指定的范围内随机生成。

    Args:
        difficulty: 地形难度，介于 0 和 1 之间。
        cfg: 地形的配置参数 (HfRectangularObstaclesTerrainCfg)。

    Returns:
        地形的高度场，为一个2D numpy数组。
    """
    # 解析地形配置
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

    # 将参数转换为离散单位
    # -- 地形
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- 障碍物
    obs_height = int(obs_height / cfg.vertical_scale)
    # 关键改动：分别处理宽度和长度
    obs_width_min = round(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = round(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    obs_length_min = round(cfg.obstacle_length_range[0] / cfg.horizontal_scale)
    obs_length_max = round(cfg.obstacle_length_range[1] / cfg.horizontal_scale)
    
    # -- 地形中心
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # 为障碍物创建离散的范围
    # 关键改动：为宽度和长度创建独立的范围
    obs_width_range = np.arange(obs_width_min, obs_width_max)
    obs_length_range = np.arange(obs_length_min, obs_length_max)
    
    # -- 位置
    dobs_x = width_pixels / cfg.num_obstacles
    obs_x_range = np.arange(0, width_pixels, dobs_x)

    # 创建一个中心有平坦平台的地形
    hf_raw = np.zeros((width_pixels, length_pixels))
    # 生成障碍物
    for i in range(cfg.num_obstacles):
        # 采样高度
        if cfg.obstacle_height_mode == "convex":
            height = obs_height
        elif cfg.obstacle_height_mode == "concave":
            height = -obs_height
        else:
            raise ValueError(f"未知地形模式 '{cfg.obstacle_height_mode}'。必须是 'convex' 或 'concave'。")
        
        # 关键改动：从各自的范围中采样宽度和长度
        width = int(np.random.choice(obs_width_range))
        length = int(np.random.choice(obs_length_range))
        
        x_start = int(obs_x_range[i])
        y_start = 0
            
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height
        
    # 裁剪地形以形成中心平台
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    
    # 将高度四舍五入到最近的垂直步长
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def rectangular_terrain_around(difficulty: float, cfg: hf_terrains_cfg.RectangularTerrainCfgAround) -> np.ndarray:
    # 将参数转换为离散单位
    # -- 地形
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    # -- 地形中心平台
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    
    # 创建一个中心有平坦平台的地形
    hf_raw = np.zeros((width_pixels, length_pixels))
    
    # 计算中心位置
    center_x = width_pixels // 2
    center_y = length_pixels // 2
    
    # 计算平台边界
    platform_half = platform_width // 2
    
    difficulty_min = difficulty - cfg.difficulty_width
    if difficulty_min < 0:
        difficulty_min = 0.0
    
    # 根据难度调整采样范围
    if cfg.obstacle_height_mode == "convex":
        # 凸起模式：高度和宽度范围乘以难度
        height_range = (
            cfg.obstacle_height_range[0] + (cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]) * difficulty_min,
            cfg.obstacle_height_range[0] + (cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]) * difficulty
        )
        ring_width_range = (
            cfg.ring_width_range[0] + (cfg.ring_width_range[1] - cfg.ring_width_range[0]) * difficulty_min,
            cfg.ring_width_range[0] + (cfg.ring_width_range[1] - cfg.ring_width_range[0]) * difficulty
        )
    elif cfg.obstacle_height_mode == "concave":
        # 凹陷模式：环宽度范围乘以难度
        height_range = cfg.obstacle_height_range  # 高度范围不变
        ring_width_range = (
            cfg.ring_width_range[0] + (cfg.ring_width_range[1] - cfg.ring_width_range[0]) * difficulty_min,
            cfg.ring_width_range[0] + (cfg.ring_width_range[1] - cfg.ring_width_range[0]) * difficulty
        )
    else:
        raise ValueError(f"未知地形模式 '{cfg.obstacle_height_mode}'。必须是 'convex' 或 'concave'。")
    
    # 生成回型地形环，直到达到边界
    ring_idx = 0
    current_offset = platform_half
    
    while True:
        # 从范围中采样环宽度、间距和高度
        ring_width = np.random.uniform(ring_width_range[0], ring_width_range[1])
        
        # 对于第一个环，不使用间距（紧挨着平台边缘）
        if ring_idx == 0:
            ring_spacing = 0
        else:
            ring_spacing = np.random.uniform(cfg.ring_spacing_range[0], cfg.ring_spacing_range[1])
        
        # 根据模式确定高度
        if cfg.obstacle_height_mode == "convex":
            ring_height = np.random.uniform(height_range[0], height_range[1])
        else:  # concave
            ring_height = -np.random.uniform(height_range[0], height_range[1])
        
        # 转换为像素单位
        ring_width_pixels = int(ring_width / cfg.horizontal_scale)
        ring_spacing_pixels = int(ring_spacing / cfg.horizontal_scale)
        ring_height_pixels = int(ring_height / cfg.vertical_scale)
        
        # 计算当前环的内外边界
        inner_offset = current_offset + ring_spacing_pixels
        outer_offset = inner_offset + ring_width_pixels
        
        # 确保不超出地形边界
        if outer_offset >= min(center_x, center_y):
            break
            
        # 创建当前环的四个边
        # 上边
        hf_raw[center_x - outer_offset:center_x + outer_offset, 
               center_y - outer_offset:center_y - inner_offset] = ring_height_pixels
        
        # 下边
        hf_raw[center_x - outer_offset:center_x + outer_offset, 
               center_y + inner_offset:center_y + outer_offset] = ring_height_pixels
        
        # 左边
        hf_raw[center_x - outer_offset:center_x - inner_offset, 
               center_y - inner_offset:center_y + inner_offset] = ring_height_pixels
        
        # 右边
        hf_raw[center_x + inner_offset:center_x + outer_offset, 
               center_y - inner_offset:center_y + inner_offset] = ring_height_pixels
        
        # 更新当前偏移量，为下一个环做准备
        current_offset = outer_offset
        ring_idx += 1
    
    # 确保中心平台平坦
    platform_x1 = center_x - platform_half
    platform_x2 = center_x + platform_half
    platform_y1 = center_y - platform_half
    platform_y2 = center_y + platform_half
    hf_raw[platform_x1:platform_x2, platform_y1:platform_y2] = 0
    
    # 将高度四舍五入到最近的垂直步长
    return np.rint(hf_raw).astype(np.int16)



@height_field_to_mesh
def shaft_terrain_around(difficulty: float, cfg: hf_terrains_cfg.ShaftTerrainCfgAround) -> np.ndarray:
    # -- 地形
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    
    # -- 地形中心平台
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    
    # 井深度
    difficulty_min = difficulty - cfg.difficulty_width
    if difficulty_min < 0:
        difficulty_min = 0.0
    height_range = (
            cfg.obstacle_height_range[0] + (cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]) * difficulty_min,
            cfg.obstacle_height_range[0] + (cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]) * difficulty
        )
    ring_height = np.random.uniform(height_range[0], height_range[1]) / cfg.vertical_scale
    
    # 计算中心位置
    center_x = width_pixels // 2
    center_y = length_pixels // 2
    
    # 计算平台边界
    platform_half = platform_width // 2
    
    offset = (width_pixels-platform_width)/2
    
    # 计算地形边缘高度
    edge_heigh = math.tan(cfg.bevel_angle/180.0*math.pi) * (offset * cfg.horizontal_scale / cfg.vertical_scale)
    
    # 创建一个中心有平坦平台的地形
    hf_raw = np.zeros((width_pixels, length_pixels))
    
    heigh_step = edge_heigh / offset 
    offset = int(offset)
    while True:
        if offset < 0:
            break
        hf_raw[center_x - offset:center_x + offset, 
               center_y - offset:center_y + offset] -= heigh_step
        offset -= 1
    
    # 确保中心平台平坦
    platform_x1 = center_x - platform_half
    platform_x2 = center_x + platform_half
    platform_y1 = center_y - platform_half
    platform_y2 = center_y + platform_half
    hf_raw[platform_x1:platform_x2, platform_y1:platform_y2] = -ring_height - edge_heigh
    
    return np.rint(hf_raw).astype(np.int16)