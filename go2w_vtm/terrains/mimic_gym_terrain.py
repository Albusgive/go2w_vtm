from __future__ import annotations

import numpy as np
import math
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING
from isaaclab.terrains.height_field.utils import height_field_to_mesh
import trimesh
from xml.dom import minidom
from isaaclab.terrains.trimesh.utils import make_plane
import xml.etree.ElementTree as ET
import os


if TYPE_CHECKING:
    from . import mimic_gym_terrain_cfg

def save_terrain_as_mjcf_with_stl(
    cfg: mimic_gym_terrain_cfg.SaveTerrainCfg,
    meshes_list: list,
    origin: np.ndarray,
    difficulty: float = 0.0,
    terrain_key_pos_list: list[np.ndarray] = None,
    descriptions: list[str] = None,
    rgba: str = "0.8 0.6 0.4 1",
) -> None:
    """
    将 terrain 保存为 MJCF + 多个 STL 文件。
    
    每个 mesh 保存为独立 STL，MJCF 通过 <mesh file="..."/> 引用，并显式命名。
    
    参数:
    - cfg: SaveTerrainCfg
    - meshes_list: List[trimesh.Trimesh]
    - origin: np.ndarray (3,) —— 地形在 MuJoCo 中的期望中心位置（XY 对齐）
    - difficulty: float —— 难度系数
    - check_point_key: 两个检查点 init_key_pos  end_key_pos
    - rgba: str —— geom 颜色
    """
    if not cfg.save_to_mjcf:
        return
    if not meshes_list:
        raise ValueError("meshes_list is empty.")

    stl_dir = cfg.mesh_path if cfg.mesh_path is not None else cfg.mjcf_path
    os.makedirs(cfg.mjcf_path, exist_ok=True)
    os.makedirs(stl_dir, exist_ok=True)

    # 计算整体平移
    combined = trimesh.util.concatenate(meshes_list)
    bounds = combined.bounds
    if bounds is None:
        raise ValueError("All meshes are empty.")

    mujoco = ET.Element("mujoco", model=cfg.terrain_name)
    asset = ET.SubElement(mujoco, "asset")
    worldbody = ET.SubElement(mujoco, "worldbody")

    for i, mesh in enumerate(meshes_list):
        if not mesh.vertices.size or not mesh.faces.size:
            continue

        mesh_trans = mesh.copy()

        # ✅ STL 文件名（不含路径）
        stl_name = f"{cfg.terrain_name}_mesh{i}.stl"
        stl_full_path = os.path.join(stl_dir, stl_name)
        mesh_trans.export(stl_full_path)

        # ✅ Mesh 的逻辑名称（用于 <geom mesh="..."/>）
        mesh_id = f"{cfg.terrain_name}_mesh{i}"

        # 相对路径（用于 file=""）
        if os.path.abspath(stl_dir) == os.path.abspath(cfg.mjcf_path):
            file_rel = stl_name
        else:
            file_rel = os.path.relpath(stl_full_path, cfg.mjcf_path).replace("\\", "/")

        # ✅ 显式设置 name，避免依赖隐式命名
        mesh_elem = ET.SubElement(asset, "mesh")
        mesh_elem.set("name", mesh_id)      # ←←← 关键：显式命名
        mesh_elem.set("file", file_rel)

        # ✅ geom 引用这个 name
        geom = ET.SubElement(worldbody, "geom")
        geom.set("type", "mesh")
        geom.set("mesh", mesh_id)           # ←←← 必须匹配上面的 name
        geom.set("rgba", rgba)
        geom.set("pos", str(-origin[0]) + " " + str(-origin[1]) + " " + str(-origin[2]))
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("group", "1")
    
    # 添加 <custom><text ... /></custom>
    custom = ET.SubElement(mujoco, "custom")
    text_elem = ET.SubElement(custom, "text")
    text_elem.set("name", f"terrain:{cfg.terrain_name}")        
    text_elem.set("data", f"difficulty:{difficulty}")  # 示例：写入难度信息
    # 关键点
    if terrain_key_pos_list is not None:
        for i,pos in enumerate(terrain_key_pos_list):
            _pos = pos - origin
            text_elem = ET.SubElement(custom, "text")
            text_elem.set("name", f"terrain_key_pos{i}:{descriptions[i]}")
            text_elem.set("data", f"{_pos[0]} {_pos[1]} {_pos[2]}")
    

    # 格式化 XML
    rough_string = ET.tostring(mujoco, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    lines = pretty_xml.splitlines()
    final_xml = "\n".join(lines[1:]) if lines and lines[0].startswith('<?xml') else pretty_xml

    mjcf_filepath = os.path.join(cfg.mjcf_path, f"{cfg.terrain_name}.xml")
    with open(mjcf_filepath, "w") as f:
        f.write(final_xml)

    print(f"[INFO] MJCF saved to: {mjcf_filepath}")
    print(f"[INFO] STL files saved to: {stl_dir}/")
    

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


def fix_box_terrain(difficulty: float, cfg: mimic_gym_terrain_cfg.FixBoxTerrainCfg) -> np.ndarray:
    meshes_list = []

    
    # -- 定义机器人出生点 --
    origin = np.array([0, 0, 0.0]) 
    # 生成box
    for i,pos in enumerate(cfg.box_pos):
        box_size=cfg.box_half_size[i]
        box_center_pos=pos
        box_mesh = create_mujoco_box_mesh(box_size, box_center_pos, origin)
        meshes_list.append(box_mesh)
    meshes_list.append(make_plane(cfg.size, 0.0,False))
    return meshes_list, origin


''' -------------------------全新地形------------------------- '''
def box_trench_terrain(difficulty: float, cfg: mimic_gym_terrain_cfg.BoxTrenchTerrainCfg) -> np.ndarray:
    """
    生成两块box的沟壑
    """
    #计算沟壑宽度 trench_width range*difficulty
    trench_width = cfg.trench_width[0] + (cfg.trench_width[1] - cfg.trench_width[0]) * difficulty
    box1_x_width = cfg.terrain_x+cfg.robot_origin_x
    terrain_x_max = box1_x_width + trench_width
    if cfg.size[0] < terrain_x_max:
        raise ValueError("terrain_x + trench_width must be less than size[0].")
    
    # 初始化网格列表
    meshes_list = []
    
    # -- 定义机器人出生点 --
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1]])
    origin = np.array([cfg.robot_origin_x, terrain_center[1], 0.05]) # 稍高于平台

    box1_extents = [box1_x_width, cfg.size[1], cfg.trench_depth]
    box1_center_pos = [box1_extents[0]/2, cfg.size[1]/2, -box1_extents[2]/2]
    box1 = trimesh.creation.box(extents=box1_extents)
    box1.apply_translation(box1_center_pos)
    meshes_list.append(box1)
    
    
    box2_extents = [cfg.size[0]-terrain_x_max, cfg.size[1], cfg.trench_depth]
    box2_center_pos = [box2_extents[0]/2+terrain_x_max, cfg.size[1]/2, -box2_extents[2]/2]
    box2 = trimesh.creation.box(extents=box2_extents)
    box2.apply_translation(box2_center_pos)
    meshes_list.append(box2)

    terrain_check_point_list,descriptions = cfg.make_check_points(difficulty)
    save_terrain_as_mjcf_with_stl(cfg=cfg, meshes_list=meshes_list, origin=origin, 
                                  difficulty=difficulty,
                                  terrain_key_pos_list=terrain_check_point_list,
                                  descriptions=descriptions)

    return meshes_list, origin


def box_platform_terrain(difficulty: float, cfg: mimic_gym_terrain_cfg.BoxHighPlatformTerrainCfg) -> np.ndarray:
    """
    生成两块box的地面和高台
    """
    terrain_x_min = cfg.terrain_x+cfg.robot_origin_x
    if cfg.size[0] < terrain_x_min:
        raise ValueError("size[0] must be greater than terrain_x + robot_origin_x.")
    
    # 初始化网格列表
    meshes_list = []
    
    # -- 定义机器人出生点 --
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1]])
    origin = np.array([cfg.robot_origin_x, terrain_center[1], 0.05]) # 稍高于平台

    # 地面
    box1_extents = [cfg.size[0], cfg.size[1], 0.2]
    box1_center_pos = [cfg.size[0]/2, cfg.size[1]/2, -box1_extents[2]/2]
    box1 = trimesh.creation.box(extents=box1_extents)
    box1.apply_translation(box1_center_pos)
    meshes_list.append(box1)
    
    # 高台
    platform_height = cfg.platform_height[0] + (cfg.platform_height[1] - cfg.platform_height[0]) * difficulty
    box2_extents = [cfg.platform_width, cfg.size[1], platform_height]
    box2_center_pos = [box2_extents[0]/2+terrain_x_min, cfg.size[1]/2, box2_extents[2]/2]
    box2 = trimesh.creation.box(extents=box2_extents)
    box2.apply_translation(box2_center_pos)
    meshes_list.append(box2)

    terrain_check_point_list,descriptions = cfg.make_check_points(difficulty)
    save_terrain_as_mjcf_with_stl(cfg=cfg, meshes_list=meshes_list, origin=origin, 
                                  difficulty=difficulty,
                                  terrain_key_pos_list=terrain_check_point_list,
                                  descriptions=descriptions)


    return meshes_list, origin


def box_rock_fissure_terrain(difficulty: float, cfg: mimic_gym_terrain_cfg.BoxRockFissureTerrainCfg) -> np.ndarray:
    """
    三块box组成的地面和垂直狭缝
    """
    #计算沟壑宽度 platform_width range*difficulty
    terrain_x_min = cfg.terrain_x + cfg.robot_origin_x
    if cfg.size[0] < terrain_x_min:
        raise ValueError("size[0] must be greater than terrain_x + robot_origin_x.")
    
    # 初始化网格列表
    meshes_list = []

    # -- 定义机器人出生点 --
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1]])
    origin = np.array([cfg.robot_origin_x, terrain_center[1], 0.05]) # 稍高于平台

    # 地面
    box1_extents = [cfg.size[0], cfg.size[1], 0.2]
    box1_center_pos = [cfg.size[0]/2, cfg.size[1]/2, -box1_extents[2]/2]
    box1 = trimesh.creation.box(extents=box1_extents)
    box1.apply_translation(box1_center_pos)
    meshes_list.append(box1)

    # 左侧box
    rock_fissure_width = cfg.rock_fissure_width[0] + (cfg.rock_fissure_width[1] - cfg.rock_fissure_width[0]) * difficulty
    box_y_size = (cfg.size[1] - rock_fissure_width)/2
    box_x_size = cfg.rock_fissure_long
    
    left_box_extents = [box_x_size, box_y_size, cfg.rock_fissure_height]
    left_box_center_pos = [terrain_x_min+box_x_size/2, box_y_size*3/2 + rock_fissure_width, left_box_extents[2]/2]
    left_box = trimesh.creation.box(extents=left_box_extents)
    left_box.apply_translation(left_box_center_pos)
    meshes_list.append(left_box)
    
    right_box_extents = [box_x_size, box_y_size, cfg.rock_fissure_height]
    right_box_center_pos = [terrain_x_min+box_x_size/2, box_y_size/2 , right_box_extents[2]/2]
    right_box = trimesh.creation.box(extents=right_box_extents)
    right_box.apply_translation(right_box_center_pos)
    meshes_list.append(right_box)

    terrain_check_point_list,descriptions = cfg.make_check_points(difficulty)
    save_terrain_as_mjcf_with_stl(cfg=cfg, meshes_list=meshes_list, origin=origin, 
                                  difficulty=difficulty,
                                  terrain_key_pos_list=terrain_check_point_list,
                                  descriptions=descriptions)


    return meshes_list, origin


def box_float_box_terrain(difficulty: float, cfg: mimic_gym_terrain_cfg.BoxFloatBoxTerrainCfg) -> np.ndarray:
    """
    两块box组成的地面和悬空box
    """
    #计算沟壑宽度 platform_width range*difficulty
    terrain_x_min = cfg.terrain_x + cfg.robot_origin_x
    if cfg.size[0] < terrain_x_min:
        raise ValueError("size[0] must be greater than terrain_x + robot_origin_x.")
    
    # 初始化网格列表
    meshes_list = []

    # -- 定义机器人出生点 --
    terrain_center = np.array([0.5 * cfg.size[0], 0.5 * cfg.size[1]])
    origin = np.array([cfg.robot_origin_x, terrain_center[1], 0.05]) # 稍高于平台

    # 地面
    box1_extents = [cfg.size[0], cfg.size[1], 0.2]
    box1_center_pos = [cfg.size[0]/2, cfg.size[1]/2, -box1_extents[2]/2]
    box1 = trimesh.creation.box(extents=box1_extents)
    box1.apply_translation(box1_center_pos)
    meshes_list.append(box1)

    # float box
    float_box_float_height = cfg.float_box_float_height[0] + (cfg.float_box_float_height[1] - cfg.float_box_float_height[0]) * difficulty
    # 随机float_box_height
    float_box_height = np.random.uniform(low=cfg.float_box_height[0], high=cfg.float_box_height[1])
    
    float_box_extents = [cfg.float_box_long, cfg.size[1], float_box_height]
    float_box_center_pos = [terrain_x_min+float_box_extents[0]/2,cfg.size[1]/2, float_box_float_height+float_box_extents[2]/2]
    float_box = trimesh.creation.box(extents=float_box_extents)
    float_box.apply_translation(float_box_center_pos)
    meshes_list.append(float_box)
    

    terrain_check_point_list,descriptions = cfg.make_check_points(difficulty)
    save_terrain_as_mjcf_with_stl(cfg=cfg, meshes_list=meshes_list, origin=origin, 
                                  difficulty=difficulty,
                                  terrain_key_pos_list=terrain_check_point_list,
                                  descriptions=descriptions)


    return meshes_list, origin