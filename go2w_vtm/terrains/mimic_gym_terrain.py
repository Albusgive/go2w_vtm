from __future__ import annotations

import numpy as np
import math
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING
from isaaclab.terrains.height_field.utils import height_field_to_mesh
import trimesh
from isaaclab.terrains.trimesh.utils import make_plane
from go2w_vtm.utils import *

if TYPE_CHECKING:
    from . import mimic_gym_terrain_cfg


import trimesh
import numpy as np

def make_plane_box(
    size: tuple[float, float],
    height: float,
    thickness: float = 0.02,
    center_zero: bool = True
) -> trimesh.Trimesh:
    """
    Generate a *thick* plane mesh (a thin box) suitable for MuJoCo collision.

    Args:
        size: (length_x, width_y) in meters.
        height: Z position of the *top surface* of the plane.
        thickness: Thickness of the plane (default 2 cm). Must be > 0.
        center_zero: If True, center of top surface is at (0, 0, height).
                     If False, bottom-left corner is at (0, 0, height - thickness).

    Returns:
        A closed, watertight trimesh.Trimesh representing a thin box.
    """
    if thickness <= 0:
        raise ValueError("Thickness must be positive for MuJoCo compatibility.")

    # Box half-extents
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, thickness / 2.0

    # Create a box centered at origin with given half-sizes
    box = trimesh.creation.box(extents=[size[0], size[1], thickness])

    # Move so that TOP surface is at z = height
    # Default box is centered at (0,0,0), so top is at z = +hz
    # We want top at z = height → translate by (0, 0, height - hz)
    box.apply_translation([0.0, 0.0, height - hz])

    if not center_zero:
        # Shift so that bottom-left corner is at (0, 0, ...)
        box.apply_translation([hx, hy, 0.0])

    return box

  
@height_field_to_mesh
def trench_terrain(difficulty: float, cfg: mimic_gym_terrain_cfg.MimicTrenchTerrainCfg) -> np.ndarray:
    """
    在 X 方向指定位置生成一条沿 Y 轴局部范围的沟（trench）。
    """
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)   # X
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)  # Y
    trench_depth_px = cfg.trench_depth / cfg.vertical_scale # Z

    hf_raw = np.zeros((width_pixels, length_pixels), dtype=np.float32)

    x_center_px = (cfg.trench_start_x + cfg.size[0] / 2.0) / cfg.horizontal_scale
    half_width_px = (cfg.trench_width / 2.0) / cfg.horizontal_scale
    x_low = int(np.clip(x_center_px - half_width_px, 0, width_pixels - 1))
    x_high = int(np.clip(x_center_px + half_width_px, 0, width_pixels - 1))

    hf_raw[x_low:x_high, 0:length_pixels-1] = -trench_depth_px
    if cfg.save_to_mjcf:
        save_heightfield_as_mjcf(hf_raw,cfg.horizontal_scale,
                    cfg.vertical_scale,cfg.mjcf_path,cfg.save_name,cfg.png_path)
        
    return np.rint(hf_raw).astype(np.int16)


def create_mujoco_box_mesh(
    size: list | tuple | np.ndarray,
    pos: list | tuple | np.ndarray,
    offset: list | tuple | np.ndarray = (0.0, 0.0, 0.0),
) -> trimesh.Trimesh:
    """
    创建一个与 MuJoCo box 几何完全一致的 trimesh 网格。

    参数:
    - size: MuJoCo 风格的半长，例如 [0.1, 0.2, 0.3]
            → 盒子总尺寸为 (0.2, 0.4, 0.6)
    - pos: MuJoCo 中的 pos，即盒子中心位置，例如 [1.0, 2.0, 3.0]
    - offset: 在 pos 基础上额外添加的偏移量，例如 [0, 0, 0.1]

    返回:
    - trimesh.Trimesh: 已正确缩放和平移的 box 网格
    """
    size = np.asarray(size, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64)

    # MuJoCo 的 size 是半长 → 转换为 trimesh.box 的 extents（全长）
    extents = 2.0 * size

    # 计算最终中心位置
    center = pos + offset

    # 创建 box：trimesh.creation.box 默认中心在原点
    box = trimesh.creation.box(extents=extents)

    # 平移到目标位置
    box.apply_translation(center)

    return box


def trench_box_terrain(difficulty: float, cfg: mimic_gym_terrain_cfg.MimicTrenchBoxTerrainCfg) -> np.ndarray:
    """
    高台只能使用box,如果修改高度场就会出现初始z为最高位置
    """
    if len(cfg.high_platform_x) != len(cfg.high_platform_half_width) or len(cfg.high_platform_x) != len(cfg.high_platform_half_width):
        raise ValueError("high_platform_x, high_platform_width, high_platform_height must have the same length.")
    
    # 初始化网格列表
    meshes_list = []

    # --  创建基础地面 --
    # ground_plane = make_plane_box(cfg.size, height=0.0, center_zero=False)
    # meshes_list.append(ground_plane)
    
    # -- 定义机器人出生点 --
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1]])
    origin = np.array([cfg.robot_origin_x, terrain_center[1], 0.05]) # 稍高于平台

    for i in range(len(cfg.high_platform_x)):
        box_size=[cfg.high_platform_half_width[i],cfg.size[1]/2,cfg.high_platform_half_height[i]]
        box_center_pos=[cfg.high_platform_x[i],0,cfg.high_platform_z[i]]
        box_mesh = create_mujoco_box_mesh(box_size, box_center_pos, origin)
        meshes_list.append(box_mesh)
    
    if cfg.save_to_mjcf:
        save_terrain_as_mjcf_with_stl(meshes_list=meshes_list,
                            origin=origin,output_path=cfg.mjcf_path,
                            filename=cfg.save_name,mesh_output_dir=cfg.mesh_path)
        

    return meshes_list, origin


def high_platform_terrain(difficulty: float, cfg: mimic_gym_terrain_cfg.MimicHighPlatformTerrainCfg) -> np.ndarray:
    """
    高台只能使用box,如果修改高度场就会出现初始z为最高位置
    """
    if len(cfg.high_platform_start_x) != len(cfg.high_platform_width) or len(cfg.high_platform_start_x) != len(cfg.high_platform_height):
        raise ValueError("high_platform_start_x, high_platform_width, high_platform_height must have the same length.")
    
    # 初始化网格列表
    meshes_list = []

    # --  创建基础地面 --
    ground_plane = make_plane_box(cfg.size, height=0.0, center_zero=False)
    meshes_list.append(ground_plane)
    
    # -- 定义机器人出生点 --
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1]])
    origin = np.array([cfg.robot_origin_x, terrain_center[1], 0.05]) # 稍高于平台

    for i in range(len(cfg.high_platform_start_x)):
        box_size=[cfg.high_platform_width[i]/2,cfg.size[1]/2,cfg.high_platform_height[i]/2]
        box_center_pos=[cfg.high_platform_start_x[i],0,cfg.high_platform_height[i]/2]
        box_mesh = create_mujoco_box_mesh(box_size, box_center_pos, origin)
        meshes_list.append(box_mesh)
    
    #再x=0处加一个档条
    extents = [0.05, cfg.size[1], 1]
    box = trimesh.creation.box(extents=extents)
    # 平移到目标位置
    box_pos = [0,cfg.size[1]/2,0.5]
    box.apply_translation(box_pos)
    meshes_list.append(box)
    
    if cfg.save_to_mjcf:
        save_terrain_as_mjcf_with_stl(meshes_list=meshes_list,
                            origin=origin,output_path=cfg.mjcf_path,
                            filename=cfg.save_name,mesh_output_dir=cfg.mesh_path)
        

    return meshes_list, origin