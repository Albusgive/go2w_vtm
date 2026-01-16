from pathlib import Path

import mujoco
import numpy as np
import glfw
import os
import torch

import mink
import KeyFrame
from go2w_vtm.produce_motion.decode_terrain import DecodeTerrain
from go2w_vtm.utils.keyframe_ui import ProportionalKeyframeDialog
from go2w_vtm.utils.MocapInterpolator import MocapInterpolator

class ik_cfg:
    def __init__(self,mocap_robot_name:str,anchor:list,anchor_ref:list):
        self.mocap_robot_name = mocap_robot_name
        self.anchor = anchor
        self.anchor_ref = anchor_ref
        self.dt = 0.002
        self.mode:str = None

class mujoco_position_cfg(ik_cfg):
    ''' 很纯粹的大力出奇迹 '''
    def __init__(self,mocap_robot_name:str,anchor:list,anchor_ref:list):
        super().__init__(mocap_robot_name,anchor,anchor_ref)
        self.mode = "mujoco_position"
        self.num_step = 10
        self.pos_error_threshold = 1e-3 # m

class mink_cfg(ik_cfg):
    ''' 好用的mink ik'''
    def __init__(self,mocap_robot_name:str,anchor:list,anchor_ref:list):
        super().__init__(mocap_robot_name,anchor,anchor_ref)
        self.mode = "mink"
        self.position_cost = 1.0
        self.orientation_cost = 0.0
        self.posture_task_cost = 1e-5   
        self.damping = 1e-5
        self.solver = "daqp"
        
        
class PlanningKeyframe:
    ''' 
    is_normal_mode: 是否是普通模式
        True: normal模式 手动k然后保存mocap ,一帧帧进行IK测试
        False: terrain模式 根据地形的checkpoint 编辑mocap 然后保存keyframe,
        会额外保存一个同名的npz表达每帧相对于terrain_key_pos的offset

        robot要求为mocap ,IK通过其他的mocap跟踪,保存keyframe为mocap和qpos的信息
        插帧根据mocap运动插值计算(并行),测试根据没帧的mocap解IK
    '''
    def __init__(self,mjcf:str,cfg:ik_cfg,save_key_path:str=None,save_key_name:str=None):
        if mjcf[-4:] != ".xml":
            self.model = mujoco.MjModel.from_xml_string(mjcf)
        else:
            self.model = mujoco.MjModel.from_xml_path(mjcf)
        self.anchor_ref_body_mids = [self.model.body(site).mocapid[0] for site in cfg.anchor_ref]
        self.anchor_body_ids = [self.model.site_bodyid[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,anchor)] for anchor in cfg.anchor]
        self.anchor_ref_site_ids = [mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,anchor_ref) for anchor_ref in cfg.anchor_ref]
        self.anchor_site_ids = [mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_SITE,anchor) for anchor in cfg.anchor]
        self.mocap_robot_id = self.model.body(cfg.mocap_robot_name).mocapid[0]
        self.error = np.zeros((len(cfg.anchor),3))
        
        self.avg_pos_error = np.zeros(len(cfg.anchor))
        if cfg.mode=="mink":
            self.cfg : mink_cfg = cfg
            self.configuration = mink.Configuration(self.model)
            self.data = self.configuration.data
            self.mink_site_tasks = []
            for i in range(len(self.cfg.anchor)):
                task = mink.FrameTask(
                    frame_name=self.cfg.anchor[i],
                    frame_type="site",
                    position_cost=self.cfg.position_cost,
                    orientation_cost=self.cfg.orientation_cost,
                )
                self.mink_site_tasks.append(task)
            self.posture_task = mink.PostureTask(self.model, cost=self.cfg.posture_task_cost)
            self.tasks = [self.posture_task, *self.mink_site_tasks]
        elif cfg.mode=="mujoco_position":
            self.cfg : mujoco_position_cfg = cfg
            self.data = mujoco.MjData(self.model)
            
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        if cfg.mode=="mink":
            self.configuration.update()
            self.posture_task.set_target_from_configuration(self.configuration)
        
        
        # 数据保存
        self.keyframe_data = KeyFrame.KeyFrame()
        self.keyframe_data.setSaveFields(qpos=True, mocap_pos=True,mocap_quat=True)

        # 总帧数
        self.nkey = 0
        self.is_normal_mode = True 
        self.change_key = False
        self.preset_labels = ["smooth", "jump", "jump_pre","start","end"]
        
        # 解析地形
        self.terrain = DecodeTerrain(self.model)
        self.save_key_path = save_key_path
        self.save_key_name = save_key_name
        self.save_mjcf_key_path = os.path.join(self.save_key_path,self.save_key_name+".xml")
        self.save_npz_key_path = os.path.join(self.save_key_path,self.save_key_name+".npz")
        if self.terrain.n_points > 0:
            self.key_id = 0
            self.key_id_last = 0
            self.is_normal_mode = False
            self.nkey = self.terrain.n_points
            # 记录初始的第一个 Key
            self.keyframe_data.record(self.model, self.data, name="0_initial", time=0.0)
            # 状态追踪：记录每个 index 是否已经“手动保存(Record)”过
            # 只有第 0 位初始为 True
            self.key_edited_status = [False] * self.nkey
            self.key_edited_status[0] = True
                
        # 更新渲染
        self.viewer = None
        self.ngeom = 0
        self.select_key_rgba = [0.0, 1.0, 0.0, 0.5]
        self.terrain_key_pos_rgba = [1.0, 0.0, 0.0, 0.5]
        
    
    def key_callback(self,key:int):
        if key == glfw.KEY_LEFT_ALT:
            self.record()
        if key == glfw.KEY_SPACE:
            self.save_keyframe(self.save_mjcf_key_path)
            self.save_relative_npz(self.save_npz_key_path)
        if key == glfw.KEY_RIGHT:
            # 拦截逻辑：如果当前 key 还没手动 Record 过，不允许去下一个
            if not self.is_normal_mode and not self.key_edited_status[self.key_id]:
                print(f"⚠️  Cannot move to next. Please Record (Alt) current key {self.key_id} first!")
                return
            self.key_id_last = self.key_id
            self.key_id += 1
            if self.key_id >= self.nkey:
                self.key_id = 0
            self.change_key = True
        if key == glfw.KEY_LEFT:
            # 向左切通常允许，方便回溯修改
            self.key_id_last = self.key_id
            self.key_id -= 1
            if self.key_id < 0:
                self.key_id = self.nkey - 1
            self.change_key = True
    
    def draw_terrain_key_pos(self,viewer):
        def draw_geom(type, size, pos, mat, rgba):
            viewer.user_scn.ngeom += 1
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]   
            mujoco.mjv_initGeom(geom, type, size, pos, mat, rgba)
        self.viewer = viewer
        self.ngeom = viewer.user_scn.ngeom       
        for pos in self.terrain.terrain_key_pos:
            draw_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0.0, 0.0], pos, [1, 0, 0, 0, 1, 0, 0, 0, 1], self.terrain_key_pos_rgba)
    
            
    def update(self):
        if self.change_key:
            self.reset_key(self.key_id)
            self.change_key = False
            
        if self.cfg.mode == "mink":
            for i, task in enumerate(self.mink_site_tasks):
                task.set_target(mink.SE3.from_mocap_id(self.data, self.anchor_ref_body_mids[i]))
            vel = mink.solve_ik(self.configuration, self.tasks, self.cfg.dt, self.cfg.solver, self.cfg.damping)
            self.configuration.integrate_inplace(vel, self.cfg.dt)
            mujoco.mj_camlight(self.model, self.data)

        elif self.cfg.mode == "mujoco_position":
            mujoco.mj_step(self.model, self.data)
            for _ in range(self.cfg.num_step - 1):  # 已经 step 一次，再补 num_step-1 次
                mujoco.mj_step(self.model, self.data)
                self.error = np.linalg.norm(self.data.site_xpos[self.anchor_ref_site_ids] - self.data.site_xpos[self.anchor_site_ids], axis=1)
                max_err = np.max(self.error)
                if max_err < self.cfg.pos_error_threshold:
                    break

        # self.avg_pos_error = np.mean(self.error)
        # print("Avg Pos Error:", self.avg_pos_error)
        
        # update terrain key viewer
        if self.terrain.n_points > 0 and self.viewer is not None:
            self.viewer.user_scn.geoms[self.ngeom + self.key_id_last].rgba = self.terrain_key_pos_rgba
            self.viewer.user_scn.geoms[self.ngeom + self.key_id].rgba = self.select_key_rgba

    
    '''    记录，保存，回放，插帧   '''
    def record(self):
        default_time = self.data.time
        # 弹出 UI
        dialog = ProportionalKeyframeDialog(
            default_time=default_time,
            preset_labels=self.preset_labels,
            window_title="Record Keyframe for IK",
            scale_factor = 4.0,
        )
        user_input = dialog.show()
        # 处理结果
        if user_input is None: return
        time_val, label = user_input
        if self.is_normal_mode:
            self.keyframe_data.record(self.model, self.data, "", time=time_val)
            self.nkey += 1
        else:
            name = f"{self.key_id}_{label}"
            # 动态增长内存列表：如果当前 key_id 还没录入过，就一直 append 
            while len(self.keyframe_data.keys_in_memory) <= self.key_id:
                # 先放一个空的 KeyData 占位，或者直接 record 当前状态
                self.keyframe_data.keys_in_memory.append(KeyFrame.KeyData())

            # 覆盖/更新当前 key_id 的数据
            self.cover_key(self.key_id, name=name, time=time_val)
            
            self.key_edited_status[self.key_id] = True
            print(f"✅ Key {self.key_id} captured from current scene state.")
        
    
    def save_keyframe(self,save_path:str):
        self.keyframe_data.save_as_xml(save_path)
    
    def remove_key(self,n:int):
        self.keyframe_data.remove_key(n)
        self.nkey -= 1
        print(f"Removed key frame {n}")
    
    def cover_key(self,n:int, name: str = "",time:float=None):
        self.keyframe_data.cover_key(n,self.model,self.data,name,time)
        print(f"Covered key frame {n}")
        
    def reset_key(self, n: int):
        # 如果是地形模式
        if not self.is_normal_mode:
            # 情况 A: 这个 Key 已经 Record 过了，我们需要“回放”它的数据
            if n < len(self.keyframe_data.keys_in_memory):
                target_key = self.keyframe_data.keys_in_memory[n]
                self.data.mocap_pos[:] = target_key.mocap_pos
                self.data.mocap_quat[:] = target_key.mocap_quat
                if target_key.qpos is not None:
                    self.data.qpos[:] = target_key.qpos
                if hasattr(self, 'configuration'):
                    self.configuration.update()
                print(f"⏪ Back to recorded Key {n}")
            
            # 情况 B: 这个 Key 还没 Record 过
            else:
                # 重点：什么都不做。MjData 保持上一帧结束时的状态。
                # 这样用户可以基于当前的姿态继续调整。
                print(f"🆕 Entering new area for Key {n}. Current pose maintained.")
        else:
            if n < len(self.keyframe_data.keys_in_memory):
                target_key = self.keyframe_data.keys_in_memory[n]
                self.data.mocap_pos[:] = target_key.mocap_pos
                self.data.mocap_quat[:] = target_key.mocap_quat
                if target_key.qpos is not None:
                    self.data.qpos[:] = target_key.qpos
                
                
    def save_relative_npz(self, file_path: str):
            """
            保存所有 mocap body 的相对位姿
            数据形状: [nkey, nmocap, 3] 和 [nkey, nmocap, 4]
            """
            if not self.keyframe_data.keys_in_memory:
                print("No keys to save.")
                return

            names = []
            all_pos_offsets = []
            all_quats = []

            for i, key in enumerate(self.keyframe_data.keys_in_memory):
                names.append(key.name)
                
                # 1. 获取该帧所有的 mocap_pos (nmocap, 3)
                # 2. 获取对应的地形点 (3,) -> 广播减法
                terrain_pos = self.terrain.terrain_key_pos[i]
                relative_pos = key.mocap_pos - terrain_pos # (nmocap, 3) - (3,)
                
                all_pos_offsets.append(relative_pos)
                all_quats.append(key.mocap_quat)

            # 转换为 numpy 大数组
            np.savez_compressed(
                file_path,
                names=np.array(names),
                mocap_pos_offsets=np.array(all_pos_offsets), # [nkey, nmocap, 3]
                mocap_quats=np.array(all_quats),             # [nkey, nmocap, 4]
                # 也可以顺便存下 qpos 以防需要完全恢复状态
                all_qpos=np.array([k.qpos for k in self.keyframe_data.keys_in_memory]) 
            )
            print(f"✅ Full relative mocap data saved to {file_path}")

    def load_relative_npz(self, file_path: str):
        """
        加载并根据当前地形点还原所有 mocap body 的位置
        """
        if not Path(file_path).exists():
            print(f"File {file_path} not found.")
            return
        with np.load(file_path, allow_pickle=True) as data:
            names = data['names']
            offsets = data['mocap_pos_offsets'] # [n_saved, nmocap, 3]
            quats = data['mocap_quats']         # [n_saved, nmocap, 4]
            all_qpos = data['all_qpos'] if 'all_qpos' in data else None

        self.keyframe_data.clear()
        # 注意：如果加载的文件点数和当前地形点数不一致，以较小的为准
        load_count = min(len(names), self.nkey)
        self.key_edited_status = [False] * self.nkey

        for i in range(load_count):
            key = KeyFrame.KeyData()
            key.name = str(names[i])
            key.time = float(i)
            
            # 还原绝对坐标：当前地形点 + 偏移量
            # terrain_pos (3,) + offset (nmocap, 3) -> (nmocap, 3)
            key.mocap_pos = self.terrain.terrain_key_pos[i] + offsets[i]
            key.mocap_quat = quats[i].copy()
            
            if all_qpos is not None:
                key.qpos = all_qpos[i].copy()
            else:
                key.qpos = self.data.qpos.copy() # 退而求其次
            
            self.keyframe_data.keys_in_memory.append(key)
            self.key_edited_status[i] = True

        self.key_id = 0
        self.change_key = True
        print(f"✅ Restored {load_count} keys with full mocap hierarchy.")
        
        
    def sync_from_model_keys(self):
        """
        利用 mj_resetDataKeyframe 接口，将 MJCF 中的关键帧同步到内存。
        """
        if self.model.nkey == 0:
            print("⚠️ No keys found in model. Skipping sync.")
            return
        print(f"🔄 Syncing {self.model.nkey} keys using MuJoCo native loader...")
        # 记录同步前的原始状态，以便同步后恢复现场
        original_qpos = self.data.qpos.copy()
        original_mpos = self.data.mocap_pos.copy()
        original_mquat = self.data.mocap_quat.copy()
        # 确定需要同步的数量
        sync_count = self.model.nkey
        if not self.is_normal_mode:
            if self.model.nkey != self.nkey:
                print(f"❗ Warning: Model keys ({self.model.nkey}) != Terrain points ({self.nkey})")
            sync_count = min(self.model.nkey, self.nkey)
        for i in range(sync_count):
            mujoco.mj_resetDataKeyframe(self.model, self.data, i)
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY, i) or f"key_{i}"
            if i < len(self.keyframe_data.keys_in_memory):
                self.keyframe_data.cover_key(i, self.model, self.data, name=key_name, time=self.data.time)
            else:
                self.keyframe_data.record(self.model, self.data, name=key_name, time=self.data.time)
            if not self.is_normal_mode:
                self.key_edited_status[i] = True
        self.data.qpos[:] = original_qpos
        self.data.mocap_pos[:] = original_mpos
        self.data.mocap_quat[:] = original_mquat
        if hasattr(self, 'configuration'):
            self.configuration.update()
        print(f"✅ Successfully synced {sync_count} keys from MJCF.")
        self.change_key = True 
        
    
    def run_interpolation_and_store(self, npz_path: str, cmd_vel=(0.5, 0.0, 0.0), fps: int = 50):
        """
        利用并行插值器计算当前环境的完整轨迹，并存入主 keyframe_data。
        cmd_vel: 支持 (3,) 的 list/tuple 或 (1, 3) 的 ndarray/tensor
        """
        # 1. 准备并行输入 (Batch Size = 1)
        # terrain_key_pos: (Keys, 3) -> (1, Keys, 3)
        terrain_tensor = torch.from_numpy(self.terrain.terrain_key_pos).float().to("cuda").unsqueeze(0)
        
        # cmd_vel 处理: 确保转换成 (1, 3) 的 tensor
        if not isinstance(cmd_vel, torch.Tensor):
            cmd_tensor = torch.tensor(cmd_vel, device="cuda").float()
        else:
            cmd_tensor = cmd_vel.to("cuda").float()
        
        if cmd_tensor.ndim == 1:
            cmd_tensor = cmd_tensor.unsqueeze(0) # (3,) -> (1, 3)

        # 2. 调用并行插值器
        interpolator = MocapInterpolator(npz_path, device="cuda")
        
        # 得到并行计算结果 trajs: (1, Total_Frames, Num_Mocap, 7)
        trajs = interpolator.interpolate(terrain_tensor, cmd_vel=cmd_tensor, fps=fps)
        
        # 3. 数据回流：从 Tensor 取出并填充到 keyframe_data
        traj_np = trajs[0].detach().cpu().numpy()
        num_frames = traj_np.shape[0]

        self.keyframe_data.clear()
        print(f"🔄 Interpolating {num_frames} frames into keyframe_data at {fps} FPS...")

        # 预先获取当前的 qpos 作为初始参考
        initial_qpos = self.data.qpos.copy()

        for i in range(num_frames):
            kd = KeyFrame.KeyData()
            # 这里的 name 建议保留一些原始 key 的信息，或者简单编号
            kd.name = f"f_{i:04d}" 
            kd.time = i / fps
            
            # 赋值 Mocap 数据 (Num_Mocap, 3) 和 (Num_Mocap, 4)
            kd.mocap_pos = traj_np[i, :, :3].copy()
            kd.mocap_quat = traj_np[i, :, 3:].copy()
            
            # 重要：qpos 处理
            # 只有第一帧携带当前姿态，后续由 IK 连续跟踪
            # 如果每一帧都 copy 当前 data.qpos，在非实时回放时会导致 IK 丢失目标
            kd.qpos = initial_qpos.copy() if i == 0 else None 
            
            self.keyframe_data.keys_in_memory.append(kd)

        # 4. 更新 PlanningKeyframe 状态
        self.nkey = len(self.keyframe_data.keys_in_memory)
        self.key_id = 0
        # 标记所有生成的帧为“已录入”状态，允许自由切换预览
        self.key_edited_status = [True] * self.nkey 
        self.change_key = True
        
        print(f"✅ Generated {self.nkey} frames. Ready for playback.")
