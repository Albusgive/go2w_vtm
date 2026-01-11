# KeyFrame.py
import xml.etree.ElementTree as ET
from typing import List, Optional
import numpy as np
import mujoco

class KeyData:
    """内存中存储的关键帧数据"""
    def __init__(self):
        self.name: str = ""
        self.time: float = 0.0
        self.qpos: List[float] = []
        self.qvel: List[float] = []
        self.mocap_pos: List[float] = []
        self.mocap_quat: List[float] = []
        self.act: List[float] = []
        self.ctrl: List[float] = []
        
        
class KeyFrame:
    def __init__(self):
        # Body filtering
        self.record_single_body: bool = False
        self.body_id: int = -1
        self.qpos_indices: List[int] = []
        self.qvel_indices: List[int] = []
        self.is_mocap_body: bool = False
        self.mocap_body_id: int = -1

        # Field selection flags (time is always saved)
        self.save_qpos: bool = True
        self.save_qvel: bool = True
        self.save_act: bool = True
        self.save_ctrl: bool = True
        self.save_mocap_pos: bool = True
        self.save_mocap_quat: bool = True

        # Recording frequency control
        self.record_dt: Optional[float] = None   # desired time interval between keys
        self.last_record_time: float = 0.0     

        self.keys_in_memory: List[KeyData] = []
    
    def setRecordMocapBody2Body(self, model, body_name: str):
        """ 如果是mocap body会记录第一个mocap_pos mocap_quat 并拼接到qpos中"""
        body_id = model.body(name=body_name).id
        self.mocap_body_id = model.body_mocapid[body_id]
        if self.mocap_body_id != -1:
            self.is_mocap_body = True
        else:
            raise ValueError(f"Body '{body_name}' is not a mocap body.")
        print(f"✅ Now recording mocap body '{body_name}' ,nqpos={model.nq},nqvel={model.nv}")

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
        
    def get_now_key(self, model, data, name: str = "",time:float=None):
        key = KeyData()
        key.name = name
        if time is not None:
            key.time = time
        else:
            key.time = data.time
        # qpos
        if self.save_qpos:
            if self.is_mocap_body:
                key.qpos = (
                data.mocap_pos[self.mocap_body_id].tolist() +
                data.mocap_quat[self.mocap_body_id].tolist() +
                data.qpos.tolist())
            else:
                if self.record_single_body and self.qpos_indices:
                    key.qpos = [float(data.qpos[i]) for i in self.qpos_indices]
                elif model.nq > 0:
                    key.qpos = data.qpos.tolist()
        # qvel
        if self.save_qvel:
            if self.is_mocap_body:
                key.qvel = data.qvel.tolist()
            else:
                if self.record_single_body and self.qvel_indices:
                    key.qvel = [float(data.qvel[i]) for i in self.qvel_indices]
                elif model.nv > 0:
                    key.qvel = data.qvel.tolist()
        # act
        if self.save_act and model.na > 0:
            key.act = data.act.tolist()
        # ctrl
        if self.save_ctrl and model.nu > 0:
            key.ctrl = data.ctrl.tolist()
        if self.save_mocap_pos and model.nmocap > 0:
            key.mocap_pos = data.mocap_pos.flatten().tolist()
        if self.save_mocap_quat and model.nmocap > 0:
            key.mocap_quat = data.mocap_quat.flatten().tolist()
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
        """
        保存为 MuJoCo keyframe XML 格式，自动覆盖
        """
        if not self.keys_in_memory:
            print("No keys to save.")
            return

        root = ET.Element("mujoco")
        keyframe_elem = ET.SubElement(root, "keyframe")

        for key_data in self.keys_in_memory:
            key_elem = ET.SubElement(keyframe_elem, "key")
            if key_data.name:
                key_elem.set("name", key_data.name)
            key_elem.set("time", str(key_data.time))

            if self.save_qpos and key_data.qpos:
                key_elem.set("qpos", " ".join(map(str, key_data.qpos)))
            if self.save_qvel and key_data.qvel:
                key_elem.set("qvel", " ".join(map(str, key_data.qvel)))
            if self.save_act and key_data.act:
                key_elem.set("act", " ".join(map(str, key_data.act)))
            if self.save_ctrl and key_data.ctrl:
                key_elem.set("ctrl", " ".join(map(str, key_data.ctrl)))
            if self.save_mocap_pos and key_data.mocap_pos:
                key_elem.set("mpos", " ".join(map(str, key_data.mocap_pos)))
            if self.save_mocap_quat and key_data.mocap_quat:
                key_elem.set("mquat", " ".join(map(str, key_data.mocap_quat)))

        indent(root)
        tree = ET.ElementTree(root)
        tree.write(file_path, encoding="utf-8", xml_declaration=True)
        print(f"✅ Saved {len(self.keys_in_memory)} keys to XML: {file_path}")
        
    def save_as_npz(self, file_path: str):
        """
        保存为 NumPy NPZ 格式，自动覆盖
        """
        if not self.keys_in_memory:
            print("No keys to save.")
            return

        times = np.array([key.time for key in self.keys_in_memory], dtype=np.float32)

        qpos = None
        if self.save_qpos:
            qpos = np.array([key.qpos for key in self.keys_in_memory], dtype=np.float32)
        qvel = None
        if self.save_qvel:
            qvel = np.array([key.qvel for key in self.keys_in_memory], dtype=np.float32)
        act = None
        if self.save_act:
            act = np.array([key.act for key in self.keys_in_memory], dtype=np.float32)
        ctrl = None
        if self.save_ctrl:
            ctrl = np.array([key.ctrl for key in self.keys_in_memory], dtype=np.float32)
        if self.save_mocap_pos:
            mocap_pos = np.array([key.mocap_pos for key in self.keys_in_memory], dtype=np.float32)
        mocap_pos = None
        if self.save_mocap_quat:
            mocap_quat = np.array([key.mocap_quat for key in self.keys_in_memory], dtype=np.float32)
        mocap_quat = None
        np.savez_compressed(
            file_path,
            time=times,
            qpos=qpos,
            qvel=qvel,
            act=act,
            ctrl=ctrl,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat
        )
        print(f"✅ Saved {len(self.keys_in_memory)} keys to NPZ: {file_path}")

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