import go2w_vtm.produce_motion.IK_and_savekey as IK_and_savekey
import go2w_vtm
import mujoco_viewer
import mujoco.viewer
import glfw

import mujoco
plugin_path = go2w_vtm.GO2W_MJCF_DIR + "/mj_plugin"
mujoco.mj_loadAllPluginLibraries(plugin_path)


file_path = go2w_vtm.GO2W_MJCF_DIR + "/mocap_scene_terrain.xml"
file_path2 = go2w_vtm.GO2W_MJCF_DIR + "/mocap_pd_scene.xml"

# save_npz_path = go2w_vtm.GO2W_PRODUCE_MOTION_K_DIR + "/human_k.npz"
save_xml_path = go2w_vtm.GO2W_MJCF_DIR + "/human_k2.xml"

anchor = ["FL_foot_joint", "FR_foot_joint", "RR_foot_joint", "RL_foot_joint"]
anchor_ref = [ref+"_ref" for ref in anchor]

cfg = IK_and_savekey.mink_cfg("base_link",anchor,anchor_ref)
cfg.orientation_cost = 0.6
cfg2 = IK_and_savekey.mujoco_position_cfg("base_link",anchor,anchor_ref)
cfg3 = IK_and_savekey.mujoco_pid_cfg("base_link",anchor,anchor_ref)
cfg3.kp = 1000.0
cfg3.kd = 10.0
cfg3.ki = 1.0
cfg3.i_max = 1000.0
cfg3.force_limit = 1000

plk = IK_and_savekey.PlanningKeyframe(file_path,cfg,True) # mink
# plk = planning_keyframe.PlanningKeyframe(file_path2,cfg2) # mujoco pd
# plk = planning_keyframe.PlanningKeyframe(file_path,cfg3) # self pid

frame_time = 0.0
def key_callback(key:int):
    global plk,frame_time
    if key == glfw.KEY_LEFT_ALT:
        plk.record(frame_time)
        frame_time += 0.0303
    if key == glfw.KEY_SPACE:
        # plk.save_qpos(save_npz_path)
        plk.save_keyframe(save_xml_path)
    if key == glfw.KEY_BACKSPACE:
        plk.reset_world()


with mujoco.viewer.launch_passive(plk.model, plk.data,key_callback=key_callback) as mj_viewer:
    while mj_viewer.is_running():
        plk.update()
        mujoco.mj_forward(plk.model, plk.data)
        mj_viewer.sync()
        
# # # 创建渲染器
# viewer = mujoco_viewer.MujocoViewer(plk.model, plk.data)
# # 模拟循环
# while viewer.is_alive:
#     plk.update()
#     mujoco.mj_forward(plk.model, plk.data)
#     viewer.render()