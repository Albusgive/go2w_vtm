import time
import math

import mujoco
import mujoco.viewer
import numpy as np
from KeyFrame import KeyFrame
import glfw
import go2w_vtm

file_path = go2w_vtm.GO2W_MJCF_DIR + "/mocap_foot_point_scene.xml"
m = mujoco.MjModel.from_xml_path(file_path)
d = mujoco.MjData(m)

# 记录keyframe
key_frame = KeyFrame()
key_frame.setRecordBody(m,'base_link')
key_frame.set_record_fps(30)
key_frame.setSaveFields(act=False,ctrl=False)


actuators=["FL_wheel_z","FR_wheel_z","FL_wheel_x","FR_wheel_x",
           "RL_wheel_z","RR_wheel_z","RL_wheel_x","RR_wheel_x",]

motions = [
    {"duration": 0.2,   "ctrl": [0.07,0.07,0.0,0.0, 0.07,0.07,0.0,0.0]},
    {"duration": 0.2,   "ctrl": [0.15,0.15,0.0,0.0, 0.15,0.15,0.0,0.0]},
    {"duration": 0.2,   "ctrl": [0.25,0.25,0.0,0.0, 0.25,0.25,0.0,0.0]},
    
    {"duration": 0.2,   "ctrl": [-0.08,-0.08,0.0,0.0, -0.15,-0.15,0.0,0.0]},
    
    {"duration": 0.2,   "ctrl": [0.2,0.2,0.0,0.0, 0.2,0.2,0.0,0.0]},
    
    {"duration": 0.2,   "ctrl": [0.0,0.0,0.0,0.0, 0.0,0.0,0.0,0.0]},

]

class MotionCtrl:
    def __init__(self,actuators:list,motion:list[dict]):
        self.actuator_ids = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,act) for act in actuators]
        self.duration = [m["duration"] for m in motion]
        self.ctrl_ = [m["ctrl"] for m in motion]
        self.start_time = 0.0 #记录动作开始时间
        self.apply_idx = 0
        self.is_applying = False
        self.motion_time = 0.0

    def apply_ctrl(self,d:mujoco.MjData):
        if not self.is_applying:
            return

        elapsed = d.time - self.start_time
        if elapsed > self.motion_time:
            self.apply_idx += 1
            if self.apply_idx >= len(self.duration):
                self.apply_idx = 0
                self.is_applying = False
                for act_id in self.actuator_ids:
                    d.ctrl[act_id] = 0.0
                return
            self.motion_time += self.duration[self.apply_idx]

        
        for i,act_id in enumerate(self.actuator_ids):
            d.ctrl[act_id] = self.ctrl_[self.apply_idx][i]

    def start(self,d:mujoco.MjData):
        self.start_time = d.time
        self.apply_idx = 0
        self.is_applying = True
        self.motion_time = self.duration[0]
    
    def stop(self,d:mujoco.MjData):
        self.is_applying = False
        for act_id in self.actuator_ids:
            d.ctrl[act_id] = 0.0
        
            
    
motion_ctrl = MotionCtrl(
    actuators=actuators,
    motion=motions
)

is_sim = True
def key_callback(key:int):
    global motion_ctrl,is_sim
    if key == glfw.KEY_LEFT_ALT:
        motion_ctrl.start(d)
    elif key == glfw.KEY_BACKSPACE:
        motion_ctrl.stop(d)
    if key == glfw.KEY_SPACE:
        is_sim = not is_sim

with mujoco.viewer.launch_passive(m, d,key_callback=key_callback) as viewer:
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    while viewer.is_running():
        step_start = time.time()

        if is_sim:
            mujoco.mj_step(m,d)
        else:
            mujoco.mj_forward(m,d)
        
        key_frame.record_with_dt(m,d)
        
        motion_ctrl.apply_ctrl(d)

        viewer.sync()
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    save_path = go2w_vtm.GO2W_MJCF_DIR + "/go2w_jump.xml"
    key_frame.save_as_xml(save_path)