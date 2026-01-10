import time
import math

import mujoco
import mujoco.viewer
import numpy as np
from KeyFrame import KeyFrame
import mj_func
import glfw
import go2w_vtm

file_path = go2w_vtm.GO2W_MJCF_DIR + "/mocap_foot_point_scene.xml"
m = mujoco.MjModel.from_xml_path(file_path)
d = mujoco.MjData(m)


# TODO 寻找关键点 增加resite
joint_tf_tree_chain = mj_func.find_joint_chains_mujoco_py(m,"base_link")

key_frame = KeyFrame()
key_frame.setRecordBody(m,'base_link')
key_frame.set_record_fps(30)
key_frame.setSaveFields(act=False,ctrl=False)


start_time = 0
time_point = 0
wheel = 0
is_star = False
max_heigh = 0
times = [[0.5, 0.0],
         [0.1, 0.07], [0.1, 0.13], [0.1, 0.19], [0.1, 0.25],
         [0.15, -0.15],
         [0.08, 0.2], [0.06, 0.26], [0.2, 0.3],
         ]
base_id = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"base_link")

def key_callback(key:int):
    global is_star, start_time
    if key == glfw.KEY_SPACE:
        is_star = True
        start_time = d.time
        # print("space",start_time,is_star)

with mujoco.viewer.launch_passive(m, d,key_callback=key_callback) as viewer:
    def draw_arrow(start, end, width, rgba):
        viewer.user_scn.ngeom += 1
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
        size = [0.0, 0.0, 0.0] 
        pos = [0, 0, 0]           
        mat = [0, 0, 0, 0, 0, 0, 0, 0, 0]  
        vec = end - start
        end = start + vec * 2
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, width, start, end)
    
    def draw_sphere(pos, r, rgba):
        viewer.user_scn.ngeom += 1
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]
        size = [r, r, r]          
        mat = [1, 0, 0, 0, 1, 0, 0, 0, 1]  
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE, size, pos, mat, rgba)
    
    ngeom = viewer.user_scn.ngeom
    rgba = [0, 1, 0, 0.5]
    while viewer.is_running():
        step_start = time.time()
        viewer.user_scn.ngeom = ngeom

        mujoco.mj_step(m,d)
    
        if is_star:
            dt = d.time - start_time
            if max_heigh < d.xpos[base_id][2]:
                max_heigh = d.xpos[base_id][2]
            
            d.ctrl[2] = times[time_point][1]
            d.ctrl[5] = times[time_point][1]
            d.ctrl[8] = times[time_point][1]
            d.ctrl[11] = times[time_point][1]
            d.ctrl[12:16] = wheel
            
            # is_land TODO
            is_land = False
            is_land = is_land | (d.sensor("touch0").data > 0)
            is_land = is_land | (d.sensor("touch1").data > 0)
            is_land = is_land | (d.sensor("touch2").data > 0)
            is_land = is_land | (d.sensor("touch3").data > 0)
            
            if dt > times[time_point][0]:
                start_time = d.time
                time_point +=1
                if time_point == len(times):
                    print("max_heigh: ",max_heigh)
                    is_star = False
                    time_point = 0
                    max_heigh = 0.0
                    d.ctrl[[2,5,8,11,12,13,14,15]] = 0.0
        
        key_frame.record(m,d)
        pos = [d.qpos[0],d.qpos[1],d.qpos[2]]
        draw_sphere(pos,0.1,rgba)
        
        
        for i in range(len(joint_tf_tree_chain)):
            for j in range(len(joint_tf_tree_chain[i])-1):
                draw_arrow(d.xanchor[joint_tf_tree_chain[i][j]],
                d.xanchor[joint_tf_tree_chain[i][j+1]], 0.01, rgba)
        
        viewer.sync()
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
    save_path = go2w_vtm.GO2W_MJCF_DIR + "/go2w_key.xml"
    key_frame.save_as_xml(save_path)