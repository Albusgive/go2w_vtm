import time
import math

import mujoco
import mujoco.viewer
import numpy as np
from KeyFrame import KeyFrame
import mj_func
import glfw
import go2w_vtm 
# plugin_path = go2w_vtm.GO2W_MJCF_DIR + "/mj_plugin"
# mujoco.mj_loadAllPluginLibraries(plugin_path)
mujoco.mj_loadPluginLibrary('/home/albusgive2/go2w_vtm/go2w_vtm/Robot/go2w_description/mjcf/mj_plugin/libsensor_ray.so')

file_path = go2w_vtm.GO2W_MJCF_DIR + "/mocap_scene_terrain.xml"
m = mujoco.MjModel.from_xml_path(file_path)
d = mujoco.MjData(m)

# if m.nkey > 1:
#     dt = m.key_time[1]-m.key_time[0]
#     print("dt = ",dt,"  FPS = ",1/dt)
mujoco.viewer.launch(m, d,show_left_ui=False, show_right_ui=False)