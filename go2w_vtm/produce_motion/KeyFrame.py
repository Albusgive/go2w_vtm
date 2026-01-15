# KeyFrame.py
import xml.etree.ElementTree as ET
from typing import List, Optional
import numpy as np
import mujoco

class KeyData:
    """内存中存储的关键帧数据，全部使用 numpy 数组以匹配 MuJoCo 格式"""
    def __init__(self):
        self.name: str = ""
        self.time: float = 0.0
        self.qpos: Optional[np.ndarray] = None
        self.qvel: Optional[np.ndarray] = None
        self.act: Optional[np.ndarray] = None
        self.ctrl: Optional[np.ndarray] = None
        self.mocap_pos: Optional[np.ndarray] = None  # Shape: (nmocap, 3)
        self.mocap_quat: Optional[np.ndarray] = None # Shape: (nmocap, 4)


class KeyFrame:
    def __init__(self):
        # Body filtering / Mocap binding
        self.record_single_body: bool = False
        self.body_id: int = -1
        self.qpos_indices: List[int] = []
        self.qvel_indices: List[int] = []
        
        self.is_mocap_body: bool = False
        self.mocap_body_id: int = -1

        # Field selection flags
        self.save_qpos: bool = True
        self.save_qvel: bool = True
        self.save_act: bool = True
        self.save_ctrl: bool = True
        self.save_mocap_pos: bool = True
        self.save_mocap_quat: bool = True

        self.record_dt: Optional[float] = None
        self.last_record_time: float = 0.0     
        self.keys_in_memory: List[KeyData] = []
    
    def setRecordMocapBody2Body(self, model, body_name: str):
        """记录时保持独立，但在保存 XML 时将此 body 的 mocap 数据合入 qpos 头部"""
        body_id = model.body(name=body_name).id
        self.mocap_body_id = model.body_mocapid[body_id]
        if self.mocap_body_id != -1:
            self.is_mocap_body = True
        else:
            raise ValueError(f"Body '{body_name}' is not a mocap body.")
        print(f"✅ Bound mocap body '{body_name}' to qpos header for export.")

    def setRecordBody(self, model, body_name: str):
        """设置只记录某个 body 及其子树的 qpos/qvel"""
        try:
            body_id = model.body(name=body_name).id
        except KeyError:
            raise ValueError(f"Body '{body_name}' not found in model.")
        

        tree_id = model.body_treeid[body_id]
        self.qpos_indices.clear()
        self.qvel_indices.clear()

        addr = 0
        for i in range(model.njnt):
            jnt_body = model.jnt_bodyid[i]
            jnt_type = model.jnt_type[i]
            jnt_size = 1
            if jnt_type == 0:  # free joint
                jnt_size = 7
            elif jnt_type == 1:  # ball joint
                jnt_size = 4
            elif jnt_type in (2, 3):  # slide or hinge
                jnt_size = 1
            else:
                continue

            if model.body_treeid[jnt_body] == tree_id:
                for j in range(jnt_size):
                    self.qpos_indices.append(addr + j)
            addr += jnt_size

        for i in range(model.nv):
            dof_body = model.dof_bodyid[i]
            if model.body_treeid[dof_body] == tree_id:
                self.qvel_indices.append(i)

        self.body_id = body_id
        self.record_single_body = True
        print(f"✅ Now recording body subtree '{body_name}' (id={body_id}), "
              f"qpos: {len(self.qpos_indices)}, qvel: {len(self.qvel_indices)}")

    def clearRecordBody(self):
        """取消 body 过滤，记录完整状态"""
        self.record_single_body = False
        self.qpos_indices.clear()
        self.qvel_indices.clear()
        print("✅ Cleared body filter. Will record full state.")

    def setSaveFields(self, qpos: Optional[bool] = False, qvel: Optional[bool] = False,
                      act: Optional[bool] = False, ctrl: Optional[bool] = False,mocap_pos: Optional[bool] = False,
                      mocap_quat: Optional[bool] = False):
        """
        设置哪些字段需要保存到 XML。
        time 始终保存，不可关闭。
        """
        self.save_qpos = qpos
        self.save_qvel = qvel
        self.save_act = act
        self.save_ctrl = ctrl
        self.save_mocap_pos = mocap_pos
        self.save_mocap_quat = mocap_quat

        print(f"✅ Save fields updated: qpos={self.save_qpos}, qvel={self.save_qvel}, "
              f"act={self.save_act}, ctrl={self.save_ctrl}, mocap_pos={self.save_mocap_pos}, mocap_quat={self.save_mocap_quat}")

    def set_record_dt(self, dt: float):
        """设置记录的时间间隔（秒），例如 dt=0.1 表示每 0.1 秒记录一帧"""
        if dt <= 0:
            raise ValueError("dt must be positive.")
        self.record_dt = dt
        print(f"✅ Set recording interval: dt = {dt}s")

    def set_record_fps(self, fps: float):
        """设置记录的帧率（帧/秒），例如 fps=10 表示每秒 10 帧"""
        if fps <= 0:
            raise ValueError("fps must be positive.")
        self.set_record_dt(1.0 / fps)
        
    def get_now_key(self, model, data, name: str = "", time: float = None) -> KeyData:
        key = KeyData()
        key.name = name
        key.time = time if time is not None else data.time
        
        # 使用 np.array 或 copy 确保获取的是数据快照而非引用
        if self.save_qpos:
            if self.record_single_body and self.qpos_indices:
                key.qpos = np.array(data.qpos[self.qpos_indices], dtype=np.float64)
            else:
                key.qpos = np.array(data.qpos, dtype=np.float64)

        if self.save_qvel:
            if self.record_single_body and self.qvel_indices:
                key.qvel = np.array(data.qvel[self.qvel_indices], dtype=np.float64)
            else:
                key.qvel = np.array(data.qvel, dtype=np.float64)

        if self.save_act and model.na > 0:
            key.act = np.array(data.act, dtype=np.float64)
            
        if self.save_ctrl and model.nu > 0:
            key.ctrl = np.array(data.ctrl, dtype=np.float64)
        
        if self.save_mocap_pos and model.nmocap > 0:
            key.mocap_pos = np.array(data.mocap_pos, dtype=np.float64) # (nmocap, 3)
            
        if self.save_mocap_quat and model.nmocap > 0:
            key.mocap_quat = np.array(data.mocap_quat, dtype=np.float64) # (nmocap, 4)
            
        return key
    
    def record(self, model, data, name: str = "",time:float=None):
        key = self.get_now_key(model, data, name,time)
        self.keys_in_memory.append(key)
        
        
    def record_with_dt(self, model, data, name: str = "",time:float=None):
        """
        记录当前状态到内存（如果满足时间间隔要求）
        """
        current_time = data.time
        # 判断是否该记录
        should_record = False
        if self.record_dt is None:
            # 无频率限制，每次都记录
            should_record = True
        else:
            if current_time - self.last_record_time > self.record_dt:
                should_record = True
                self.last_record_time = current_time 
                
        if not should_record:
            return  # skip recording
        self.record(model, data, name,time=time)

    def remove_key(self,n:int):
        """Remove the nth keyframe from memory"""
        self.keys_in_memory.pop(n)
        
    def cover_key(self,n:int, model, data, name: str = "",time:float=None):
        """Replace the nth keyframe with the given keyframe"""
        self.keys_in_memory[n] = self.get_now_key(model, data, name,time)

    def save_as_xml(self, file_path: str):
        if not self.keys_in_memory:
            print("No keys to save.")
            return

        root = ET.Element("mujoco")
        keyframe_elem = ET.SubElement(root, "keyframe")

        for key_data in self.keys_in_memory:
            key_elem = ET.SubElement(keyframe_elem, "key")
            if key_data.name: key_elem.set("name", key_data.name)
            key_elem.set("time", str(key_data.time))

            # --- QPOS 逻辑: 处理 Mocap 拼接 ---
            if self.save_qpos and key_data.qpos is not None:
                if self.is_mocap_body and key_data.mocap_pos is not None:
                    # 提取特定 body 的 3位 pos 和 4位 quat
                    m_pos = key_data.mocap_pos[self.mocap_body_id]
                    m_quat = key_data.mocap_quat[self.mocap_body_id]
                    # 拼接为 [pos, quat, qpos...]
                    combined_qpos = np.concatenate([m_pos, m_quat, key_data.qpos])
                    key_elem.set("qpos", " ".join(map(str, combined_qpos)))
                else:
                    key_elem.set("qpos", " ".join(map(str, key_data.qpos)))

            # --- 其他标准字段 ---
            if self.save_qvel and key_data.qvel is not None:
                key_elem.set("qvel", " ".join(map(str, key_data.qvel)))
            if self.save_act and key_data.act is not None:
                key_elem.set("act", " ".join(map(str, key_data.act)))
            if self.save_ctrl and key_data.ctrl is not None:
                key_elem.set("ctrl", " ".join(map(str, key_data.ctrl)))
            
            # --- 原生 Mocap 字段 (拉平存储) ---
            if self.save_mocap_pos and key_data.mocap_pos is not None:
                key_elem.set("mpos", " ".join(map(str, key_data.mocap_pos.flatten())))
            if self.save_mocap_quat and key_data.mocap_quat is not None:
                key_elem.set("mquat", " ".join(map(str, key_data.mocap_quat.flatten())))

        indent(root)
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)
        print(f"✅ Exported {len(self.keys_in_memory)} keys to XML (Mocap-to-Qpos: {self.is_mocap_body})")
        
    def save_as_npz(self, file_path: str):
        if not self.keys_in_memory: return
        # 将内存中的 List[np.ndarray] 堆叠成高维数组方便直接加载
        save_dict = {"time": np.array([k.time for k in self.keys_in_memory], dtype=np.float32)}
        if self.save_qpos: save_dict["qpos"] = np.stack([k.qpos for k in self.keys_in_memory])
        if self.save_qvel: save_dict["qvel"] = np.stack([k.qvel for k in self.keys_in_memory])
        if self.save_act: save_dict["act"] = np.stack([k.act for k in self.keys_in_memory])
        if self.save_ctrl: save_dict["ctrl"] = np.stack([k.ctrl for k in self.keys_in_memory])
        if self.save_mocap_pos: save_dict["mocap_pos"] = np.stack([k.mocap_pos for k in self.keys_in_memory])
        if self.save_mocap_quat: save_dict["mocap_quat"] = np.stack([k.mocap_quat for k in self.keys_in_memory])
        np.savez_compressed(file_path, **save_dict)
        print(f"✅ Saved to NPZ: {file_path}")

    def clear(self):
        """清空内存中所有记录"""
        self.keys_in_memory.clear()
        print("✅ Cleared all recorded keys.")


# ===========================
# 工具函数：美化 XML 缩进
# ===========================
def indent(elem, level=0):
    """为 ElementTree 添加缩进，使 XML 更美观"""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i