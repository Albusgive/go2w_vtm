import go2w_vtm
import mujoco_viewer
import mujoco.viewer
import glfw

import mujoco
from go2w_vtm.utils.mjcf_editor import MJCFEditor


# plugin_path = go2w_vtm.GO2W_MJCF_DIR + "/mj_plugin"
# mujoco.mj_loadAllPluginLibraries(plugin_path)


file_path = go2w_vtm.GO2W_MJCF_DIR + "/go2w_mocap.xml"
# test_box_float_box_terrain  test_box_platform_terrain test_box_trench_terrain test_high_platform_box_terrain test_rock_fissure_box_terrain
terrain_path = go2w_vtm.GO2W_MJCF_DIR + "/test_rock_fissure_box_terrain.xml"
temp_k_path = go2w_vtm.GO2W_MJCF_DIR + "/temp_k.xml"

mjcf = MJCFEditor(file_path)
mjcf.add_sub_element("worldbody", "light", attrib={"pos": "0 0 1.5","dir": "0 0 -1","directional":"true",})
mjcf.add_sub_element("worldbody", "light", attrib={"pos": "-1.5 0 1.5","dir": "1 0 -1","directional":"true",})
mjcf.add_sub_element("mujoco", "include", attrib={"file": terrain_path})

mjcf.add_sub_element("mujoco", "custom")
mjcf.add_sub_element("custom", "text", attrib={"name": "custom", "data": "aabb"})
mjcf.add_sub_element("custom", "text", attrib={"name": "custom2", "data": "bbcc"})

mjcf.save(temp_k_path)

model = mujoco.MjModel.from_xml_path(temp_k_path)
data = mujoco.MjData(model)

mujoco.viewer.launch(model, data)

        