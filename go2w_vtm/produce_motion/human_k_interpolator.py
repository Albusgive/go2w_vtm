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

# test_box_float_box_terrain  test_box_platform_terrain test_box_rock_fissure_terrain test_box_trench_terrain 
terrain_name = "multi_motion_platform_terrain"
temp_path = go2w_vtm.GO2W_MJCF_DIR + "/temp.xml"


anchor = ["FL_foot_joint", "FR_foot_joint", "RR_foot_joint", "RL_foot_joint"]
anchor_ref = [ref+"_ref" for ref in anchor]

cfg = IK_and_savekey.mink_cfg("base_link",anchor,anchor_ref)
cfg.orientation_cost = 0.6

plk = IK_and_savekey.PlanningKeyframe(temp_path,cfg) # mink
hz = 50
plk.load_interpolator_config(go2w_vtm.GO2W_MJCF_DIR + "/" + terrain_name + "_k.npz")
plk.compute_and_store_interpolated_frames((1.0,0.0,0.0),hz)
with mujoco.viewer.launch_passive(plk.model, plk.data,show_left_ui=False,show_right_ui=False) as viewer:
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

# plk.save_keyframe(save_path = go2w_vtm.GO2W_MJCF_DIR + "/terrain_k_interpolated.xml")
        