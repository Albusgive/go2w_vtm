import os
import trimesh
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

def save_heightfield_as_mjcf(
    hf_raw: np.ndarray,
    horizontal_scale: float = 0.1,
    vertical_scale: float = 0.005,
    output_path: str = ".",
    filename: str = "heightfield",
    png_path: str= None
) -> None:
    """
    Save a heightfield array as a PNG image and generate a corresponding MJCF file
    that loads it as a MuJoCo hfield (height field).

    The generated MJCF uses <hfield> in <asset> and places a <geom type="hfield"> at correct z-position.

    Args:
        hf_raw: 2D numpy array of height values (shape: [H, W])
        horizontal_scale: meters per pixel along x and y (default: 0.1 m/pixel)
        vertical_scale: meters per unit of height (default: 0.005 m/unit)
        output_path: directory to save .png and .xml files
        filename: base name (without extension) for both .png and .xml

    Output:
        {output_path}/{filename}.png
        {output_path}/{filename}.xml
    """
    import cv2  
    os.makedirs(output_path, exist_ok=True)

    # --- 1. Process and save PNG ---
    # Note: MuJoCo expects image origin at bottom-left.
    # OpenCV saves with top-left origin, so we flip vertically.
    img = hf_raw[:,::-1].T  # flip vertically (so that PNG bottom = array[0])
    img_min = np.min(img)
    img_max = np.max(img)
    img_normalized = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)

    png_path1 = os.path.join(png_path, f"{filename}.png")
    png_path2 = os.path.join(output_path, f"{filename}.png")
    cv2.imwrite(png_path1, img_normalized)
    cv2.imwrite(png_path2, img_normalized)

    # --- 2. Compute MuJoCo hfield parameters ---
    H, W = hf_raw.shape  # rows = y, cols = x

    # Physical size: [x_size, y_size, z_range, nrow?] → MuJoCo format: "x y z ncol"
    # According to MuJoCo docs: size = "x y z ncol", where:
    #   x = total size in x (meters) = (W - 1) * horizontal_scale
    #   y = total size in y (meters) = (H - 1) * horizontal_scale
    #   z = max elevation range = (img_max - img_min) * vertical_scale
    #   ncol = number of columns = W (optional, but recommended)
    x_size = (W - 1) * horizontal_scale
    y_size = (H - 1) * horizontal_scale
    z_range = (img_max - img_min) * vertical_scale
    ncol = W

    # Position: MuJoCo places hfield so that its *center* is at (pos_x, pos_y),
    # and the *bottom* of the heightfield is at pos_z.
    # We usually want the lowest point to sit at world z = img_min * vertical_scale.
    # But MuJoCo's hfield geom has its reference point at (center_x, center_y, bottom_z).
    # So: pos_z = img_min * vertical_scale
    pos_z = img_min * vertical_scale

    # --- 3. Build MJCF ---
    mujoco = ET.Element("mujoco", model=filename)
    asset = ET.SubElement(mujoco, "asset")
    worldbody = ET.SubElement(mujoco, "worldbody")

    # Add hfield asset
    hfield_elem = ET.SubElement(asset, "hfield")
    hfield_elem.set("name", filename)
    hfield_elem.set("file", f"{filename}.png")
    hfield_elem.set("size", f"{x_size/2} {y_size/2} {z_range} {0.5}")

    # Add geom
    geom = ET.SubElement(worldbody, "geom")
    geom.set("type", "hfield")
    geom.set("hfield", filename)
    geom.set("pos", f"0 0 {pos_z}")
    geom.set("rgba", "0.5 0.7 1.0 1")  # optional: default color

    # --- 4. Save MJCF ---
    rough_string = ET.tostring(mujoco, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    # Remove XML declaration
    lines = pretty_xml.splitlines()
    if lines and lines[0].startswith('<?xml'):
        final_xml = "\n".join(lines[1:])
    else:
        final_xml = pretty_xml

    xml_path = os.path.join(output_path, f"{filename}.xml")
    with open(xml_path, "w") as f:
        f.write(final_xml)

    print(f"[INFO] Heightfield PNG saved to: {png_path}")
    print(f"[INFO] MJCF saved to: {xml_path}")

    
def save_terrain_as_mjcf_with_stl(
    meshes_list: list,
    origin: np.ndarray,
    output_path: str,
    filename: str,
    mesh_output_dir: str = None,
    rgba: str = "0.8 0.6 0.4 1",
    difficulty: float = 0.0
) -> None:
    """
    将 terrain 保存为 MJCF + 多个 STL 文件。
    
    每个 mesh 保存为独立 STL，MJCF 通过 <mesh file="..."/> 引用，并显式命名。
    
    参数:
    - meshes_list: List[trimesh.Trimesh]
    - origin: np.ndarray (3,) —— 地形在 MuJoCo 中的期望中心位置（XY 对齐）
    - output_path: str —— MJCF 文件保存的**目录**
    - filename: str —— MJCF 文件名（不含 .xml，如 "high_platform"）
    - mesh_output_dir: str —— STL 文件保存目录（若 None，则与 MJCF 同目录）
    - rgba: str —— geom 颜色
    """
    if not meshes_list:
        raise ValueError("meshes_list is empty.")

    stl_dir = mesh_output_dir if mesh_output_dir is not None else output_path
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(stl_dir, exist_ok=True)

    # 计算整体平移
    combined = trimesh.util.concatenate(meshes_list)
    bounds = combined.bounds
    if bounds is None:
        raise ValueError("All meshes are empty.")

    mujoco = ET.Element("mujoco", model=filename)
    asset = ET.SubElement(mujoco, "asset")
    worldbody = ET.SubElement(mujoco, "worldbody")

    for i, mesh in enumerate(meshes_list):
        if not mesh.vertices.size or not mesh.faces.size:
            continue

        mesh_trans = mesh.copy()

        # ✅ STL 文件名（不含路径）
        stl_name = f"{filename}_mesh{i}.stl"
        stl_full_path = os.path.join(stl_dir, stl_name)
        mesh_trans.export(stl_full_path)

        # ✅ Mesh 的逻辑名称（用于 <geom mesh="..."/>）
        mesh_id = f"{filename}_mesh{i}"

        # 相对路径（用于 file=""）
        if os.path.abspath(stl_dir) == os.path.abspath(output_path):
            file_rel = stl_name
        else:
            file_rel = os.path.relpath(stl_full_path, output_path).replace("\\", "/")

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
    text_elem.set("name", f"terrain:{filename}")          # 可自定义名称
    text_elem.set("data", f"difficulty:{difficulty}")  # 示例：写入难度信息

    # 格式化 XML
    rough_string = ET.tostring(mujoco, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    lines = pretty_xml.splitlines()
    final_xml = "\n".join(lines[1:]) if lines and lines[0].startswith('<?xml') else pretty_xml

    mjcf_filepath = os.path.join(output_path, f"{filename}.xml")
    with open(mjcf_filepath, "w") as f:
        f.write(final_xml)

    print(f"[INFO] MJCF saved to: {mjcf_filepath}")
    print(f"[INFO] STL files saved to: {stl_dir}/")