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

is_run = True
def key_callback(key:int):
    global is_run
    if key == glfw.KEY_SPACE:
        is_run = not is_run
        
with mujoco.viewer.launch_passive(m, d,key_callback=key_callback) as viewer:
    k = 0
    while viewer.is_running():
        if is_run:
            mujoco.mj_resetDataKeyframe(m, d, k)
            mujoco.mj_forward(m, d)
            k+=1
            if k >= m.nkey:
                k = 0

        time.sleep(dt)
        viewer.sync()
        