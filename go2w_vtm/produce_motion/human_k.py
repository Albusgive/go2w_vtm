import go2w_vtm.produce_motion.IK_and_savekey as IK_and_savekey
import go2w_vtm
import mujoco_viewer
import mujoco.viewer
import glfw
import time

import mujoco
from go2w_vtm.utils.mjcf_editor import MJCFEditor
from go2w_vtm.produce_motion.decode_terrain import DecodeTerrain

# plugin_path = go2w_vtm.GO2W_MJCF_DIR + "/mj_plugin"
# mujoco.mj_loadAllPluginLibraries(plugin_path)
is_edit_mode = True

file_path = go2w_vtm.GO2W_MJCF_DIR + "/go2w_mocap.xml"
# multi_motion_trench_terrain multi_motion_platform_terrain
terrain_name = "multi_motion_platform_terrain"
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
if is_edit_mode:
    plk.set_root_targets("base_link",["FL_foot_joint_ref","FR_foot_joint_ref","RL_foot_joint_ref","RR_foot_joint_ref"])
    plk.load_relative_npz(go2w_vtm.GO2W_MJCF_DIR + "/" + terrain_name + "_k.npz")
    # plk.sync_from_model_keys()
else:
    hz = 50
    plk.load_interpolator_config(go2w_vtm.GO2W_MJCF_DIR + "/" + terrain_name + "_k.npz")
    plk.compute_and_store_interpolated_frames((1.0,0.0,0.0),hz)

with mujoco.viewer.launch_passive(plk.model, plk.data,key_callback=plk.key_callback,
                                  show_left_ui=False,show_right_ui=False) as viewer:
    plk.draw_terrain_key_pos(viewer)
    replay_k = 0
    if is_edit_mode:
        plk.create_static_grid(viewer,0.2,1.5,rgba=[0.5,0.5,0.5,0.2],line_thickness=0.005,hide=True)
    while viewer.is_running():
        start_time = time.time()
        if not is_edit_mode:
            plk.reset_key(replay_k)
        plk.update()
        if not is_edit_mode:
            replay_k += 1
            if replay_k >= plk.nkey:
                replay_k = 0
            end_time = time.time()
            time.sleep(max(0,1.0/hz - (end_time - start_time)))
        viewer.sync()
        