import go2w_vtm.produce_motion.IK_and_savekey as IK_and_savekey
import go2w_vtm
import mujoco_viewer
import mujoco.viewer
import glfw

import mujoco
from go2w_vtm.utils.mjcf_editor import MJCFEditor

# plugin_path = go2w_vtm.GO2W_MJCF_DIR + "/mj_plugin"
# mujoco.mj_loadAllPluginLibraries(plugin_path)


file_path = go2w_vtm.GO2W_MJCF_DIR + "/go2w_mocap.xml"
terrain_path = go2w_vtm.GO2W_MJCF_DIR + "/high_platform.xml"
save_xml_path = go2w_vtm.GO2W_MJCF_DIR + "/human_k2.xml"
temp_k_path = go2w_vtm.GO2W_MJCF_DIR + "/temp_k.xml"

mjcf = MJCFEditor(file_path)
mjcf.add_sub_element("worldbody", "light", attrib={"pos": "0 0 1.5","dir": "0 0 -1","directional":"true",})
mjcf.add_sub_element("worldbody", "light", attrib={"pos": "-1.5 0 1.5","dir": "1 0 -1","directional":"true",})
mjcf.add_sub_element("mujoco", "include", attrib={"file": terrain_path})

mjcf.add_sub_element("mujoco", "custom")
mjcf.add_sub_element("custom", "text", attrib={"name": "custom", "data": "aabb"})
mjcf.add_sub_element("custom", "text", attrib={"name": "custom2", "data": "bbcc"})

mjcf.save(temp_k_path)
mjcf.add_sub_element("mujoco", "include", attrib={"file": save_xml_path})

anchor = ["FL_foot_joint", "FR_foot_joint", "RR_foot_joint", "RL_foot_joint"]
anchor_ref = [ref+"_ref" for ref in anchor]

cfg = IK_and_savekey.mink_cfg("base_link",anchor,anchor_ref)
cfg.orientation_cost = 0.6

plk = IK_and_savekey.PlanningKeyframe(temp_k_path,cfg,True) # mink

# 自定义内容
id = mujoco.mj_name2id(plk.model,mujoco.mjtObj.mjOBJ_TEXT,"custom2")
custom = plk.model.text_data[plk.model.text_adr[id]:plk.model.text_adr[id]+plk.model.text_size[id]-1]


frame_time = 0.0
def key_callback(key:int):
    global plk,frame_time
    if key == glfw.KEY_LEFT_ALT:
        plk.record(frame_time)
        frame_time += 0.0303
    if key == glfw.KEY_SPACE:
        # plk.save_qpos(save_npz_path)
        plk.save_keyframe(save_xml_path)
        mjcf.save(temp_k_path)
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