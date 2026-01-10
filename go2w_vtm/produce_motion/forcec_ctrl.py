import mujoco_viewer
import mujoco.viewer
import mujoco
from actuator import PDForceCtrl,StateCtrl
import go2w_vtm
from go2w_vtm.utils.gamepad import Gamepad
import glfw
from ray_caster_data import RayCasterData
import cv2
import numpy as np  

plugin_path = go2w_vtm.GO2W_MJCF_DIR + "/mj_plugin"
mujoco.mj_loadAllPluginLibraries(plugin_path)

file_path = go2w_vtm.GO2W_MJCF_DIR + "/go2w_point_force.xml"
model = mujoco.MjModel.from_xml_path(file_path)
data = mujoco.MjData(model)

# gamepad = Gamepad()

sc = StateCtrl(model,data)

ray = RayCasterData(model,data,"raycaster")

def joystick_to_wheel_speeds(
    left_y: float,
    right_x: float,
    max_linear_speed: float = 1.0,
    max_angular_speed: float = 1.57,
    wheel_track: float = 0.4
) -> tuple[float, float, float, float]:
    """
    将手柄左右摇杆输入转换为四轮差速机器人的四个轮速（m/s）

    参数:
        left_y (float): 左摇杆 Y 轴，范围 [-1, 1]
            -1 = 向上推（前进），+1 = 向下拉（后退）
        right_x (float): 右摇杆 X 轴，范围 [-1, 1]
            -1 = 向左推，+1 = 向右推（顺时针转向）
        max_linear_speed (float): 最大前进速度（m/s），默认 1.0
        max_angular_speed (float): 最大角速度（rad/s），默认 1.57（≈90°/s）
        wheel_track (float): 左右轮中心距（米），默认 0.4m

    返回:
        (v_fl, v_fr, v_rl, v_rr) —— 四个轮子的线速度（m/s）
        FL=前左, FR=前右, RL=后左, RR=后右
    """
    # 映射手柄到物理命令
    v_x = -left_y * max_linear_speed      # 上推 left_y=-1 → v_x = +max
    omega = right_x * max_angular_speed   # 右推 right_x=+1 → omega = +max

    # 计算左右侧轮速（基于CoM的差速模型）
    R = wheel_track / 2.0
    v_left = v_x - omega * R
    v_right = v_x + omega * R

    # 四轮分配：左右同侧前后轮速度相同
    return v_left, v_right, v_left, v_right

def step():
    # gamepad.update()
    mujoco.mj_step(model, data)
    z_data = ray.get_pos_w(0)
    z_data = z_data[:,2]
    sc.leap(z_data)
    #NAN 转换为0.0
    z_data = np.nan_to_num(z_data, nan=0.0)
    zero_mask = z_data < -0.2
    z_data[zero_mask] = 0.0
    z_data[~zero_mask] = 255.0
    z_data = z_data.reshape(ray.v_ray_num,ray.h_ray_num)
    z_data = z_data.astype(np.uint8)
    cv2.imshow("z_data",z_data)
    cv2.waitKey(1)
    # v_fl, v_fr, v_rl, v_rr = joystick_to_wheel_speeds(gamepad.axis_left_y,gamepad.axis_right_x)
    # sc.ctrl_wheel([v_fl,v_fr,v_rl,v_rr])
    sc.ctrl_wheel([10.0,10.0,10.0,10.0])

def key_callback(key:int):
    global ff
    if key == glfw.KEY_LEFT_ALT:
        ff = not ff
with mujoco.viewer.launch_passive(model, data,key_callback=key_callback) as mj_viewer:
    while mj_viewer.is_running():
        step()
        mj_viewer.sync()


        
# 创建渲染器
viewer = mujoco_viewer.MujocoViewer(model, data, width=1920, height=1080)
marker = {
    "type": mujoco.mjtGeom.mjGEOM_SPHERE,   # 几何体类型
    "size": [0.05,0.05, 0.05],                         # 半径（球只需要1个值）
    "pos": [0.0, 0.0, 0.5],                 # 世界坐标位置
    "rgba": [1, 0, 0, 1],                   # 红色，不透明
} 


# import cv2
# import numpy as np  
# while viewer.is_alive:
#     # marker["pos"] = data.sensor("subtreecom").data
#     # viewer.add_marker(**marker)
    
#     step()
#     viewer.render()