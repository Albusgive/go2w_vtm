import go2w_vtm.produce_motion.IK_and_savekey as IK_and_savekey
import go2w_vtm
import mujoco_viewer
import mujoco.viewer
import glfw

import mujoco
from go2w_vtm.utils.mjcf_editor import MJCFEditor
from go2w_vtm.produce_motion.decode_terrain import DecodeTerrain

# plugin_path = go2w_vtm.GO2W_MJCF_DIR + "/mj_plugin"
# mujoco.mj_loadAllPluginLibraries(plugin_path)


file_path = go2w_vtm.GO2W_MJCF_DIR + "/go2w_mocap.xml"
# test_box_float_box_terrain  test_box_platform_terrain test_box_rock_fissure_terrain test_box_trench_terrain 
terrain_name = "test_box_trench_terrain"
terrain_path = go2w_vtm.GO2W_MJCF_DIR + "/" + terrain_name + ".xml"
terrain_k_path = go2w_vtm.GO2W_MJCF_DIR + "/" + terrain_name + "_k.xml"
temp_path = go2w_vtm.GO2W_MJCF_DIR + "/temp.xml"

mjcf = MJCFEditor(file_path)
mjcf.add_sub_element("worldbody", "light", attrib={"pos": "0 0 1.5","dir": "0 0 -1","directional":"true",})
mjcf.add_sub_element("worldbody", "light", attrib={"pos": "-1.5 0 1.5","dir": "1 0 -1","directional":"true",})
mjcf.add_sub_element("mujoco", "include", attrib={"file": terrain_path})

# mjcf.add_sub_element("mujoco", "include", attrib={"file": terrain_k_path})
mjcf.save(temp_path)

anchor = ["FL_foot_joint", "FR_foot_joint", "RR_foot_joint", "RL_foot_joint"]
anchor_ref = [ref+"_ref" for ref in anchor]

cfg = IK_and_savekey.mink_cfg("base_link",anchor,anchor_ref)
cfg.orientation_cost = 0.6

plk = IK_and_savekey.PlanningKeyframe(temp_path,cfg,save_key_path=go2w_vtm.GO2W_MJCF_DIR,save_key_name=terrain_name+"_k") # mink
plk.load_relative_npz(go2w_vtm.GO2W_MJCF_DIR + "/" + terrain_name + "_k.npz")
# plk.sync_from_model_keys()

# TODO 调整key 添加key过程状态描述,保存为mjcf并通过name描述key过程状态, key2npz功能增加,check_point_key模式的npz文件 加载给IK_motion_loader
# 单线程mujoco测试:插值计算(并行通用计算),motion trace randmonize
with mujoco.viewer.launch_passive(plk.model, plk.data,key_callback=plk.key_callback,
                                  show_left_ui=False,show_right_ui=False) as viewer:
    plk.draw_terrain_key_pos(viewer)
    while viewer.is_running():
        plk.update()
        mujoco.mj_forward(plk.model, plk.data)

        viewer.sync()
        