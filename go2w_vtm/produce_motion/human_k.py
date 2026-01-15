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
terrain_path = go2w_vtm.GO2W_MJCF_DIR + "/test_trench_box_terrain.xml"
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

terrain = DecodeTerrain(plk.model)


frame_time = 0.0
key_id = 0
last_key_id = 0
def key_callback(key:int):
    global plk,frame_time,key_id,last_key_id,terrain
    if key == glfw.KEY_LEFT_ALT:
        plk.record(frame_time)
        frame_time += 0.0303
    if key == glfw.KEY_SPACE:
        # plk.save_qpos(save_npz_path)
        plk.save_keyframe(save_xml_path)
        mjcf.save(temp_k_path)
    if key == glfw.KEY_BACKSPACE:
        plk.reset_world()
    if key == glfw.KEY_LEFT:
        last_key_id = key_id
        key_id -= 1
        if key_id < 0:
            key_id = terrain.n_points-1
    if key == glfw.KEY_RIGHT:
        last_key_id = key_id
        key_id += 1
        if key_id >= terrain.n_points:
            key_id = 0

# TODO 调整key 添加key过程状态描述,保存为mjcf并通过name描述key过程状态, key2npz功能增加,check_point_key模式的npz文件 加载给IK_motion_loader
# 单线程mujoco测试:插值计算(并行通用计算),motion trace randmonize
with mujoco.viewer.launch_passive(plk.model, plk.data,key_callback=key_callback) as viewer:
    def draw_geom(type, size, pos, mat, rgba):
        viewer.user_scn.ngeom += 1
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]   
        mujoco.mjv_initGeom(geom, type, size, pos, mat, rgba)
    ngeom = viewer.user_scn.ngeom    
    rgba = [0.0, 1.0, 0.0, 0.5]     
    for pos in terrain.terrain_key_pos:
        draw_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0.0, 0.0], pos, [1, 0, 0, 0, 1, 0, 0, 0, 1], [1.0, 0.0, 0.0, 0.5])
    
    while viewer.is_running():
        plk.update()
        mujoco.mj_forward(plk.model, plk.data)

        viewer.user_scn.geoms[ngeom + last_key_id].rgba = [1.0, 0.0, 0.0, 0.5]
        viewer.user_scn.geoms[ngeom + key_id].rgba = [0.0, 1.0, 0.0, 1.0]
        viewer.sync()
        