import time
import math

import mujoco
import mujoco.viewer
import numpy as np
from KeyFrame import KeyFrame
import glfw
import go2w_vtm 
import os

file_path = go2w_vtm.GO2W_MJCF_DIR + "/clean_scene.xml"
m = mujoco.MjModel.from_xml_path(file_path)
d = mujoco.MjData(m)

dt = m.key_time[1]-m.key_time[0]
print("dt = ",dt,"  FPS = ",1/dt)

with mujoco.viewer.launch_passive(m, d) as viewer:
    k = 0
    step_start = time.time()
    while viewer.is_running():
        mujoco.mj_resetDataKeyframe(m, d, k)
        # mujoco.mj_setKeyframe(m, d, k)
        mujoco.mj_forward(m, d)

        viewer.sync()
        k+=1
        if k >= m.nkey:
            k = 0
            step_start = time.time()
        else:
            wait_time = d.time - (time.time() - step_start)
            if wait_time > 0:
                time.sleep(wait_time*2)