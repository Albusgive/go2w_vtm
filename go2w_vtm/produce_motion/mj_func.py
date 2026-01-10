import mujoco

def find_joint_chains_mujoco_py(model, body_name: str) -> list[list[int]]:
    chains = []
    
    # 获取body ID
    body_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,body_name)
    if body_id == -1:
        print(f"body: {body_name} can't find")
        return chains

    tree_id = model.body_treeid[body_id]
    if tree_id == -1:
        print(f"body: {body_name} can't find tree_id")
        return chains
    
    # 使用字典存储父子关系
    children_map = {}
    
    for i in range(model.nv):
        if model.dof_parentid[i] != -1 and model.dof_treeid[i] == tree_id:
            parent_jnt = model.dof_jntid[model.dof_parentid[i]]
            child_jnt = model.dof_jntid[i]
            
            # 只有当父关节和子关节不同时才添加关系
            if parent_jnt != child_jnt:
                if parent_jnt not in children_map:
                    children_map[parent_jnt] = []
                children_map[parent_jnt].append(child_jnt)
    
    # 输出父子关系
    for parent, children in children_map.items():
        print(f"parent {parent}")
        print(f"sub: {children}")
    
    # 构建完整的链
    
    # 找到根节点（没有父节点的节点）
    all_children = set()
    for children in children_map.values():
        all_children.update(children)
    
    roots = []
    for parent in children_map.keys():
        if parent not in all_children:
            roots.append(parent)
    
    # 从每个根节点开始DFS构建链
    for root in roots:
        stack = [(root, [root])]
        
        while stack:
            current, current_chain = stack.pop()
            
            # 如果当前节点没有子节点，则这是一个完整的链
            if current not in children_map or not children_map[current]:
                chains.append(current_chain)
            else:
                # 将子节点加入栈
                for child in children_map[current]:
                    new_chain = current_chain.copy()
                    new_chain.append(child)
                    stack.append((child, new_chain))
    
    # 输出链信息
    print(f"Found {len(chains)} chains:")
    for i, chain in enumerate(chains):
        chain_str = " -> ".join(str(joint) for joint in chain)
        print(f"Chain {i}: {chain_str}")
    
    return chains


import mujoco
import numpy as np

class PDRoatAttitudeController:
    def __init__(self):
        # 存储上一次的状态（用于微分）
        self.roll_prev = 0.0
        self.pitch_prev = 0.0
        self.prev_time = 0.0

        # 控制参数
        self.kp = 2000.0
        self.kd = 10.0

        # 目标姿态（可外部修改）
        self.roll_ref = 0.0   # radians
        self.pitch_ref = 0.0  # radians

        # 执行器 ID（初始化时未设置）
        self.act_rx = -1
        self.act_ry = -1

    def find_actuators(self, model):
        """查找执行器 ID"""
        self.act_rx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_rx")
        self.act_ry = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "base_ry")

        if self.act_rx == -1 or self.act_ry == -1:
            raise ValueError("Error: Cannot find actuators 'base_rx' or 'base_ry'")

    def set_reference(self, roll: float = 0.0, pitch: float = 0.0):
        """设置期望姿态（弧度）"""
        self.roll_ref = roll
        self.pitch_ref = pitch

    def step(self, model, data):
        """执行一步 PD 姿态控制"""
        # 第一次调用时查找执行器
        if self.act_rx == -1 or self.act_ry == -1:
            self.find_actuators(model)

        # 当前时间
        curr_time = data.time

        # 获取当前 roll 和 pitch（通过 actuator_length，假设 transmission 映射为角度）
        roll_curr = data.actuator_length[self.act_rx]
        pitch_curr = data.actuator_length[self.act_ry]

        # 时间步长
        dt = curr_time - self.prev_time
        if dt <= 0:
            dt = model.opt.timestep  # 使用模拟器 timestep

        # 数值微分：计算当前角速度
        droll_curr = (roll_curr - self.roll_prev) / dt
        dpitch_curr = (pitch_curr - self.pitch_prev) / dt

        # 误差的导数（期望角速度为 0）
        droll_err = 0.0 - droll_curr
        dpitch_err = 0.0 - dpitch_curr

        # PD 控制律
        torque_roll = self.kp * (self.roll_ref - roll_curr) + self.kd * droll_err
        torque_pitch = self.kp * (self.pitch_ref - pitch_curr) + self.kd * dpitch_err

        # 输出到控制输入
        data.ctrl[self.act_rx] = torque_roll
        data.ctrl[self.act_ry] = torque_pitch

        # 更新历史值
        self.roll_prev = roll_curr
        self.pitch_prev = pitch_curr
        self.prev_time = curr_time

        # 可选：打印调试信息
        # print(f"Roll: {np.degrees(roll_curr):.2f}°, Pitch: {np.degrees(pitch_curr):.2f}°, "
        #       f"Torque: ({torque_roll:.2f}, {torque_pitch:.2f})")