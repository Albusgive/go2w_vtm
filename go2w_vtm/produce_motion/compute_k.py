import go2w_vtm.produce_motion.IK_and_savekey as IK_and_savekey
import go2w_vtm
import mujoco_viewer
import mujoco.viewer
import glfw

import mujoco
from go2w_vtm.utils.mjcf_editor import MJCFEditor

plugin_path = go2w_vtm.GO2W_MJCF_DIR + "/mj_plugin"
mujoco.mj_loadAllPluginLibraries(plugin_path)

file_path = go2w_vtm.GO2W_MJCF_DIR + "/mocap_scene_terrain_K.xml"

# save_npz_path = go2w_vtm.GO2W_PRODUCE_MOTION_K_DIR + "/human_k.npz"
save_xml_path = go2w_vtm.GO2W_MJCF_DIR + "/compute_k.xml"

anchor = ["FL_foot_joint", "FR_foot_joint", "RR_foot_joint", "RL_foot_joint"]
anchor_ref = [ref+"_ref" for ref in anchor]

cfg = IK_and_savekey.mink_cfg("base_link",anchor,anchor_ref)
cfg.orientation_cost = 0.6

plk = IK_and_savekey.PlanningKeyframe(file_path,cfg,False) # mink
plk.interpolate_and_record(fps=60.0)
plk.save_keyframe(save_xml_path)

