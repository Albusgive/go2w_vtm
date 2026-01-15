import go2w_vtm.produce_motion.IK_and_savekey as IK_and_savekey
import go2w_vtm
import mujoco_viewer
import mujoco.viewer
import glfw

import mujoco
from go2w_vtm.utils.mjcf_editor import MJCFEditor
from go2w_vtm.produce_motion.decode_terrain import DecodeTerrain
import time

# plugin_path = go2w_vtm.GO2W_MJCF_DIR + "/mj_plugin"
# mujoco.mj_loadAllPluginLibraries(plugin_path)


file_path = go2w_vtm.GO2W_MJCF_DIR + "/go2w_mocap.xml"
terrain_path = go2w_vtm.GO2W_MJCF_DIR + "/test_trench_box_terrain.xml"
load_key_path = go2w_vtm.GO2W_MJCF_DIR + "/terrain_k.xml"
temp_k_path = go2w_vtm.GO2W_MJCF_DIR + "/temp_k.xml"

mjcf = MJCFEditor(file_path)
mjcf.add_sub_element("worldbody", "light", attrib={"pos": "0 0 1.5","dir": "0 0 -1","directional":"true",})
mjcf.add_sub_element("worldbody", "light", attrib={"pos": "-1.5 0 1.5","dir": "1 0 -1","directional":"true",})
mjcf.add_sub_element("mujoco", "include", attrib={"file": terrain_path})

mjcf.add_sub_element("mujoco", "custom")
mjcf.add_sub_element("custom", "text", attrib={"name": "custom", "data": "aabb"})
mjcf.add_sub_element("custom", "text", attrib={"name": "custom2", "data": "bbcc"})

mjcf.add_sub_element("mujoco", "include", attrib={"file": load_key_path})
mjcf.save(temp_k_path)

anchor = ["FL_foot_joint", "FR_foot_joint", "RR_foot_joint", "RL_foot_joint"]
anchor_ref = [ref+"_ref" for ref in anchor]

cfg = IK_and_savekey.mink_cfg("base_link",anchor,anchor_ref)
cfg.orientation_cost = 0.6

plk = IK_and_savekey.PlanningKeyframe(temp_k_path,cfg,save_key_path=go2w_vtm.GO2W_MJCF_DIR,save_key_name="terrain_k") # mink
hz = 50
plk.run_interpolation_and_store(go2w_vtm.GO2W_MJCF_DIR + "/terrain_k.npz",(1.0,0.0,0.0),hz)
# plk.is_normal_mode = True
with mujoco.viewer.launch_passive(plk.model, plk.data,key_callback=plk.key_callback,
                                  show_left_ui=False,show_right_ui=False) as viewer:
    plk.draw_terrain_key_pos(viewer)
    replay_k = 0
    while viewer.is_running():
        start_time = time.time()
        plk.reset_key(replay_k)
        plk.update()
        replay_k += 1
        if replay_k >= plk.nkey:
            replay_k = 0
        end_time = time.time()
        time.sleep(max(0,1.0/hz - (end_time - start_time)))
        viewer.sync()
        