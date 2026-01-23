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
        
        # mocap设定
        self.root_mocap_name = None
        self.target_mocap_names = None
        self.root_mocap_id = None
        self.target_mocap_ids = None

        # 总帧数
        self.nkey = 0
        self.is_normal_mode = True 
        self.change_key = False
        self.preset_labels = ["smooth", "jump", "jump_pre","start","end"]
        
        # 解析地形
        self.terrain = DecodeTerrain(self.model)
        self.save_key_path = save_key_path
        self.save_key_name = save_key_name
        self.save_mjcf_key_path = None
        self.save_npz_key_path = None
        if save_key_path is not None and save_key_name is not None:
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
        self.key_geom_start_idx = -1
        self.select_key_rgba = [0.0, 1.0, 0.0, 0.5]
        self.terrain_key_pos_rgba = [1.0, 0.0, 0.0, 0.5]
        
        self.grid_geom_start_idx = -1
        self.grid_geom_count = 0
        self.grid_rgba = [0.0, 0.0, 0.0, 0.5]
        self.close_grid = False
        self.close_grid_update = False
    
    def set_root_targets(self,root_mocap_name:str,target_mocap_names:list[str]):
        self.root_mocap_name = root_mocap_name
        self.target_mocap_names = target_mocap_names
        self.root_mocap_id = self.model.body(self.root_mocap_name).mocapid[0]
        self.target_mocap_ids = [self.model.body(name).mocapid[0] for name in self.target_mocap_names]
    
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
        if key == glfw.KEY_RIGHT_ALT:
            self.close_grid = not self.close_grid
            self.close_grid_update = True

            
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
            self.viewer.user_scn.geoms[self.key_geom_start_idx + self.key_id_last].rgba = self.terrain_key_pos_rgba
            self.viewer.user_scn.geoms[self.key_geom_start_idx + self.key_id].rgba = self.select_key_rgba
        if self.close_grid_update:
            rgba = [self.grid_rgba[0],self.grid_rgba[1],self.grid_rgba[2],self.grid_rgba[3]]
            if self.close_grid:
                rgba[3] = 0.0
            self.update_grid_color(rgba)

    
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
        if not self.keyframe_data.keys_in_memory:
            print("No keys to save.")
            return

        root_pos_offsets = []
        root_quats = []
        target_rel_pos = []
        target_rel_quats = []
        names = []

        for i, key in enumerate(self.keyframe_data.keys_in_memory):
            names.append(key.name)
            
            # --- 1. 处理 Root (相对于地形) ---
            r_pos = key.mocap_pos[self.root_mocap_id]
            r_quat = key.mocap_quat[self.root_mocap_id]
            terrain_pos = self.terrain.terrain_key_pos[i]
            
            root_pos_offsets.append(r_pos - terrain_pos)
            root_quats.append(r_quat)

            # --- 2. 处理 Targets (相对于 Root) ---
            t_pos_list = []
            t_quat_list = []
            for t_id in self.target_mocap_ids:
                rel_p, rel_q = self._get_relative_pose(r_pos, r_quat, key.mocap_pos[t_id], key.mocap_quat[t_id])
                t_pos_list.append(rel_p)
                t_quat_list.append(rel_q)
            
            target_rel_pos.append(t_pos_list)
            target_rel_quats.append(t_quat_list)

        # 保存时增加 root_name 和 target_names
        np.savez_compressed(
            file_path,
            names=np.array(names),
            root_name=np.array(self.root_mocap_name), # 保存 Root 名称
            target_names=np.array(self.target_mocap_names), # 按顺序保存 Targets 名称
            root_pos_offsets=np.array(root_pos_offsets),
            root_quats=np.array(root_quats),
            target_rel_pos=np.array(target_rel_pos),
            target_rel_quats=np.array(target_rel_quats),
            all_qpos=np.array([k.qpos for k in self.keyframe_data.keys_in_memory if k.qpos is not None])
        )
        print(f"✅ Saved hierarchical data with names to {file_path}")
        
    
    def save_relative_npz(self, file_path: str):
        if not self.keyframe_data.keys_in_memory:
            print("No keys to save.")
            return

        root_pos_offsets = []
        root_quats = []
        target_rel_pos = []
        target_rel_quats = []
        names = []

        for i, key in enumerate(self.keyframe_data.keys_in_memory):
            names.append(key.name)
            
            # --- 1. 处理 Root (相对于地形) ---
            r_pos = key.mocap_pos[self.root_mocap_id]
            r_quat = key.mocap_quat[self.root_mocap_id]
            terrain_pos = self.terrain.terrain_key_pos[i]
            
            root_pos_offsets.append(r_pos - terrain_pos)
            root_quats.append(r_quat)

            # --- 2. 处理 Targets (相对于 Root) ---
            t_pos_list = []
            t_quat_list = []
            for t_id in self.target_mocap_ids:
                rel_p, rel_q = self._get_relative_pose(r_pos, r_quat, key.mocap_pos[t_id], key.mocap_quat[t_id])
                t_pos_list.append(rel_p)
                t_quat_list.append(rel_q)
            
            target_rel_pos.append(t_pos_list)
            target_rel_quats.append(t_quat_list)

        # 保存时增加 root_name 和 target_names
        np.savez_compressed(
            file_path,
            names=np.array(names),
            root_name=np.array(self.root_mocap_name), # 保存 Root 名称
            target_names=np.array(self.target_mocap_names), # 按顺序保存 Targets 名称
            root_pos_offsets=np.array(root_pos_offsets),
            root_quats=np.array(root_quats),
            target_rel_pos=np.array(target_rel_pos),
            target_rel_quats=np.array(target_rel_quats),
            all_qpos=np.array([k.qpos for k in self.keyframe_data.keys_in_memory if k.qpos is not None])
        )
        print(f"✅ Saved hierarchical data with names to {file_path}")

    def load_relative_npz(self, file_path: str):
        if not Path(file_path).exists():
            print(f"File {file_path} not found.")
            return
        
        with np.load(file_path, allow_pickle=True) as data:
            # 读取名称并转换回字符串列表
            file_root_name = str(data['root_name'])
            file_target_names = [str(n) for n in data['target_names']]
            
            r_offsets = data['root_pos_offsets']
            r_quats = data['root_quats']
            t_rel_pos = data['target_rel_pos']
            t_rel_quats = data['target_rel_quats']
            all_qpos = data.get('all_qpos')
            names = data.get('names', [])

        # --- 核心逻辑：重新根据保存的名称寻找当前模型的 ID ---
        try:
            load_root_id = self.model.body(file_root_name).mocapid[0]
            load_target_ids = [self.model.body(name).mocapid[0] for name in file_target_names]
        except Exception as e:
            print(f"❌ Load failed: Model bodies do not match names in NPZ. {e}")
            return

        self.keyframe_data.clear()
        load_count = min(len(r_offsets), self.nkey)

        for i in range(load_count):
            key = KeyFrame.KeyData()
            key.name = str(names[i]) if i < len(names) else f"key_{i}"
            key.mocap_pos = np.zeros((self.model.nmocap, 3))
            key.mocap_quat = np.tile([1, 0, 0, 0], (self.model.nmocap, 1)).astype(np.float64)

            # 1. 还原 Root 世界位姿
            world_root_pos = self.terrain.terrain_key_pos[i] + r_offsets[i]
            world_root_quat = r_quats[i]
            key.mocap_pos[load_root_id] = world_root_pos
            key.mocap_quat[load_root_id] = world_root_quat

            # 2. 还原每个 Target 的世界位姿 (对应顺序)
            for j, t_id in enumerate(load_target_ids):
                w_p, w_q = self._get_world_pose(world_root_pos, world_root_quat, t_rel_pos[i, j], t_rel_quats[i, j])
                key.mocap_pos[t_id] = w_p
                key.mocap_quat[t_id] = w_q
            
            if all_qpos is not None and len(all_qpos) > i:
                key.qpos = all_qpos[i].copy()
            
            self.keyframe_data.keys_in_memory.append(key)
            self.key_edited_status[i] = True

        self.key_id = 0
        self.change_key = True
        print(f"✅ Successfully restored {load_count} frames using hierarchical names.")
        
        
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
        
    ''' 插帧 '''
    def load_interpolator_config(self, npz_path: str):
        """
        函数 1: 加载配置。
        从 NPZ 中读取 root 和 targets 的名称，并初始化插值器。
        """
        if not os.path.exists(npz_path):
            print(f"❌ File not found: {npz_path}")
            return
        
        # 加载数据以获取名称映射
        data = np.load(npz_path, allow_pickle=True)
        self.root_mocap_name = str(data['root_name'])
        self.target_mocap_names = [str(n) for n in data['target_names']]
        
        # 重新解析当前模型的 ID (防止模型 ID 变动)
        self.root_mocap_id = self.model.body(self.root_mocap_name).mocapid[0]
        self.target_mocap_ids = [self.model.body(name).mocapid[0] for name in self.target_mocap_names]
        
        # 初始化插值器
        self.interpolator = MocapInterpolator(npz_path, device="cuda")
        print(f"✅ Interpolator loaded with root: {self.root_mocap_name}")

    def compute_and_store_interpolated_frames(self, cmd_vel=(0.5, 0.0, 0.0), fps: int = 50):
        """
        函数 2: 计算全部帧并储存。
        """
        if not hasattr(self, 'interpolator'):
            print("❌ Please call load_interpolator_config first!")
            return

        # 1. 准备输入
        terrain_tensor = torch.from_numpy(self.terrain.terrain_key_pos).float().to("cuda").unsqueeze(0)
        cmd_tensor = torch.tensor(cmd_vel, device="cuda").float().unsqueeze(0)

        # 2. 并行计算全量帧 (解包为两个 Tensor)
        root_pose_w_seq, targets_pose_b_seq = self.interpolator.interpolate(terrain_tensor, cmd_tensor, fps=fps)
        
        # 3. 清空旧数据
        self.keyframe_data.clear()
        num_frames = root_pose_w_seq.shape[1]
        
        # 提取到 CPU 以便 MuJoCo 循环填充 (取 batch 0)
        # root_pose: [frames, 7], targets_pose: [frames, N, 7]
        r_pose_seq = root_pose_w_seq[0].cpu().numpy()
        t_pose_b_seq = targets_pose_b_seq[0].cpu().numpy()

        print(f"🔄 Storing {num_frames} interpolated frames...")

        for f in range(num_frames):
            kd = KeyFrame.KeyData()
            kd.name = f"interp_{f:04d}"
            kd.time = f / fps
            
            # 初始化本帧 mocap 数据
            m_pos = np.zeros((self.model.nmocap, 3))
            m_quat = np.tile([1, 0, 0, 0], (self.model.nmocap, 1)).astype(np.float64)

            # --- A. 还原 Root ---
            # r_pose_seq[f] 结构为 [x, y, z, qw, qx, qy, qz]
            curr_r_pos = r_pose_seq[f, 0:3]
            curr_r_quat = r_pose_seq[f, 3:7]
            m_pos[self.root_mocap_id] = curr_r_pos
            m_quat[self.root_mocap_id] = curr_r_quat

            # --- B. 还原 Targets (相对 -> 世界) ---
            for i, t_id in enumerate(self.target_mocap_ids):
                # t_pose_b_seq[f, i] 也是 7D
                rel_p = t_pose_b_seq[f, i, 0:3]
                rel_q = t_pose_b_seq[f, i, 3:7]
                
                # 利用你现有的 _get_world_pose 函数
                w_p, w_q = self._get_world_pose(curr_r_pos, curr_r_quat, rel_p, rel_q)
                
                m_pos[t_id] = w_p
                m_quat[t_id] = w_q
            
            kd.mocap_pos = m_pos
            kd.mocap_quat = m_quat
            kd.qpos = self.data.qpos.copy() if f == 0 else None
            
            self.keyframe_data.keys_in_memory.append(kd)

        # UI 状态更新
        self.nkey = len(self.keyframe_data.keys_in_memory)
        self.key_edited_status = [True] * self.nkey
        self.key_id = 0
        self.change_key = True
        print(f"✅ All {self.nkey} frames stored.")


    ''' 渲染 '''
    def draw_terrain_key_pos(self,viewer):
        def draw_geom(type, size, pos, mat, rgba):
            viewer.user_scn.ngeom += 1
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom - 1]   
            mujoco.mjv_initGeom(geom, type, size, pos, mat, rgba)
        if self.viewer is None:
            self.viewer = viewer
        self.ngeom = viewer.user_scn.ngeom       
        self.key_geom_start_idx = self.ngeom
        for pos in self.terrain.terrain_key_pos:
            draw_geom(mujoco.mjtGeom.mjGEOM_SPHERE, [0.02, 0.0, 0.0], pos, [1, 0, 0, 0, 1, 0, 0, 0, 1], self.terrain_key_pos_rgba)
        

    def create_static_grid(self, viewer, 
                           resolution=1.0, 
                           scale=1.0, 
                           rgba=[0.5, 0.5, 0.5, 0.1], 
                           line_thickness=0.01,hide=False):
        """
        在场景 AABB 范围内绘制 3D 网格。
        resolution: 网格间距 (米)
        scale: 对包围盒的缩放比例
        line_thickness: 圆柱体线宽
        """
        if self.model.ngeom == 0:
            return

        # 1. 计算原始 AABB 边界 (逻辑同前)
        global_min = np.array([np.inf, np.inf, np.inf])
        global_max = np.array([-np.inf, -np.inf, -np.inf])
        self.grid_rgba = rgba
        if self.viewer is None:
            self.viewer = viewer
        
        for i in range(self.model.ngeom):
            # 过滤掉地板，否则网格会变得无限大
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            
            local_center = self.model.geom_aabb[i, :3]
            local_half_size = self.model.geom_aabb[i, 3:]
            pos = self.model.geom_pos[i]
            quat = self.model.geom_quat[i]
            mat = np.zeros(9); mujoco.mju_quat2Mat(mat, quat); mat = mat.reshape(3, 3)
            
            offsets = np.array([[i, j, k] for i in [-1, 1] for j in [-1, 1] for k in [-1, 1]])
            local_vertices = local_center + offsets * local_half_size
            world_vertices = pos + (mat @ local_vertices.T).T
            global_min = np.minimum(global_min, np.min(world_vertices, axis=0))
            global_max = np.maximum(global_max, np.max(world_vertices, axis=0))

        # 2. 应用缩放 (Scale)
        center = (global_max + global_min) / 2.0
        half_size = ((global_max - global_min) / 2.0) * scale
        g_min, g_max = center - half_size, center + half_size

        # 3. 记录起始索引
        self.grid_geom_start_idx = viewer.user_scn.ngeom
        initial_ngeom = viewer.user_scn.ngeom

        # 4. 生成网格线坐标
        # 我们需要在 X, Y, Z 三个方向分别画线
        xs = np.arange(g_min[0], g_max[0] + resolution, resolution)
        ys = np.arange(g_min[1], g_max[1] + resolution, resolution)
        zs = np.arange(g_min[2], g_max[2] + resolution, resolution)

        def add_line(p1, p2):
            if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                return
            geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
            # 计算圆柱体的位姿：中心点，长度，旋转
            mid_point = (p1 + p2) / 2.0
            direction = p2 - p1
            length = np.linalg.norm(direction)
            
            # 默认圆柱体沿 Z 轴，需要旋转到 direction 方向
            z_axis = np.array([0, 0, 1])
            target_dir = direction / (length + 1e-6)
            
            # 计算旋转矩阵 (将 Z 轴对齐到 target_dir)
            rot_mat = np.eye(3)
            if not np.allclose(target_dir, z_axis):
                v = np.cross(z_axis, target_dir)
                c = np.dot(z_axis, target_dir)
                s = np.linalg.norm(v)
                kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rot_mat = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-6))

            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                size=np.array([line_thickness, length / 2.0, 0]),
                pos=mid_point,
                mat=rot_mat.flatten(),
                rgba=rgba
            )
            viewer.user_scn.ngeom += 1

        # 绘制沿 X 方向的线 (固定 Y, Z)
        for y in ys:
            for z in zs:
                add_line(np.array([g_min[0], y, z]), np.array([g_max[0], y, z]))

        # 绘制沿 Y 方向的线 (固定 X, Z)
        for x in xs:
            for z in zs:
                add_line(np.array([x, g_min[1], z]), np.array([x, g_max[1], z]))

        # 绘制沿 Z 方向的线 (固定 X, Y)
        for x in xs:
            for y in ys:
                add_line(np.array([x, y, g_min[2]]), np.array([x, y, g_max[2]]))
        # 5. 记录总数
        self.grid_geom_count = viewer.user_scn.ngeom - initial_ngeom
        if hide:
            rgba = [self.grid_rgba[0],self.grid_rgba[1],self.grid_rgba[2],0.0]
            self.update_grid_color(rgba)
            self.close_grid = True

    def update_grid_color(self, new_rgba):
        """
        快速修改已绘制网格的颜色
        """
        if self.grid_geom_start_idx == -1:
            return
        
        for i in range(self.grid_geom_count):
            idx = self.grid_geom_start_idx + i
            if idx < self.viewer.user_scn.maxgeom:
                self.viewer.user_scn.geoms[idx].rgba[:] = new_rgba

    '''  math  '''          
    def _get_relative_pose(self, root_pos, root_quat, target_pos, target_quat):
        """计算 target 在 root 坐标系下的位姿"""
        # 1. 计算相对位置: R_root^T * (p_target - p_root)
        rel_pos = np.zeros(3)
        inv_root_quat = np.zeros(4)
        mujoco.mju_negQuat(inv_root_quat, root_quat)
        
        diff = target_pos - root_pos
        mujoco.mju_rotVecQuat(rel_pos, diff, inv_root_quat)
        
        # 2. 计算相对旋转: q_inv_root * q_target
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, inv_root_quat, target_quat)
        return rel_pos, rel_quat

    def _get_world_pose(self, root_pos, root_quat, rel_pos, rel_quat):
        """将 local 坐标还原回世界坐标"""
        # 1. 还原位置: p_root + R_root * p_rel
        world_pos = np.zeros(3)
        mujoco.mju_rotVecQuat(world_pos, rel_pos, root_quat)
        world_pos += root_pos
        
        # 2. 还原旋转: q_root * q_rel
        world_quat = np.zeros(4)
        mujoco.mju_mulQuat(world_quat, root_quat, rel_quat)
        return world_pos, world_quat