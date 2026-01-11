from pathlib import Path

import mujoco
import numpy as np

import mink
import KeyFrame


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
    
class mujoco_pid_cfg(ik_cfg):
    ''' 朴实无华的pid '''
    def __init__(self,mocap_robot_name:str,anchor:list,anchor_ref:list):
        super().__init__(mocap_robot_name,anchor,anchor_ref)
        self.mode = "mujoco_pid"
        self.num_step = 10
        self.pos_error_threshold = 1e-3 # m
        self.kp = 100.0
        self.kd = 1.0
        self.ki = 0.0
        self.i_max = 0.0
        self.force_limit = 1e3

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
    def __init__(self,file_path:str,cfg:ik_cfg,editing:bool=False):
        self.model = mujoco.MjModel.from_xml_path(file_path)
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
        elif cfg.mode=="mujoco_pid":
            self.cfg : mujoco_pid_cfg = cfg
            self.data = mujoco.MjData(self.model)
            self.error_integral = np.zeros((len(self.cfg.anchor),3))
            self.error_last = np.zeros((len(self.cfg.anchor),3))
            
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        if cfg.mode=="mink":
            self.configuration.update()
            self.posture_task.set_target_from_configuration(self.configuration)
        
        # 编辑模式 手动k然后保存mocap 插帧模式 根据mocap生成keyframe
        self.editing = editing
        
        # 数据保存
        self.keyframe_data = KeyFrame.KeyFrame()
        if self.editing:
            self.keyframe_data.setSaveFields(mocap_pos=True,mocap_quat=True)
        else:
            self.keyframe_data.setRecordMocapBody2Body(self.model,self.cfg.mocap_robot_name)
            self.keyframe_data.setSaveFields(qpos=True,qvel=True,mocap_pos=True)
        self.nkey = 0
    
    
    @staticmethod
    def _slerp(q0, q1, t):
        """Spherical linear interpolation between two unit quaternions."""
        q0 = np.asarray(q0, dtype=np.float64)
        q1 = np.asarray(q1, dtype=np.float64)
        dot = np.dot(q0, q1)

        if dot < 0.0:
            q1 = -q1
            dot = -dot

        if dot > 0.9995:
            result = q0 + t * (q1 - q0)
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return s0 * q0 + s1 * q1
    
    def compute_pid_force(self):
        target_pos = self.data.site_xpos[self.anchor_ref_site_ids]  # (N, 3)
        anchor_pos = self.data.site_xpos[self.anchor_site_ids]      # (N, 3)
        self.error = target_pos - anchor_pos  # (N, 3)
        # 积分项
        self.error_integral += self.error * self.cfg.dt
        self.error_integral = np.clip(self.error_integral, -self.cfg.i_max, self.cfg.i_max)
        # 微分项
        error_derivative = (self.error - self.error_last) / self.cfg.dt
        self.error_last = self.error.copy()  # 重要：避免引用
        # PID 计算
        force = (
            self.cfg.kp * self.error +
            self.cfg.kd * error_derivative +
            self.cfg.ki * self.error_integral
        )  # (N, 3)
        # 力限幅（按每个轴 clip）
        force = np.clip(force, -self.cfg.force_limit, self.cfg.force_limit)
        return force  # shape: (num_force, 3)
    
            
    def update(self):
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

        elif self.cfg.mode == "mujoco_pid":
            for _ in range(self.cfg.num_step):
                self.data.qfrc_applied[:] = 0.0
                forces = self.compute_pid_force()  # shape: (num_legs, 3)
                err = np.zeros((len(self.cfg.anchor),3))
                for i, (body_id, site_id) in enumerate(zip(self.anchor_body_ids, self.anchor_site_ids)):
                    force_vec = forces[i].reshape(3, 1)          # → (3,1)
                    torque_vec = np.zeros((3, 1))                # → (3,1)
                    point_vec = self.data.site_xpos[site_id].reshape(3, 1)  # → (3,1)
                    mujoco.mj_applyFT(
                        self.model,
                        self.data,
                        force_vec,
                        torque_vec,
                        point_vec,
                        int(body_id),
                        self.data.qfrc_applied  # 累加到 qfrc_applied
                    )
                mujoco.mj_step(self.model, self.data)
                self.error = np.linalg.norm(self.data.site_xpos[self.anchor_ref_site_ids] - self.data.site_xpos[self.anchor_site_ids], axis=1)
                max_err = np.max(self.error)
                if max_err < self.cfg.pos_error_threshold:
                    break

        # self.avg_pos_error = np.mean(self.error)
        # print("Avg Pos Error:", self.avg_pos_error)
    
    def reset_world(self):
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model,self.data,0)
        else:
            mujoco.mj_resetData(self.model,self.data)
    
    '''    记录，保存，回放，插帧   '''
    def record(self,time:float=None):
        self.keyframe_data.record(self.model,self.data,"",time=time)
        print(f"Recorded key frame {self.nkey}")
        self.nkey += 1
        
    def save_qpos(self,save_path:str):
        self.keyframe_data.save_as_npz(save_path)
    
    def save_keyframe(self,save_path:str):
        self.keyframe_data.save_as_xml(save_path)
    
    def remove_key(self,n:int):
        self.keyframe_data.remove_key(n)
        self.nkey -= 1
        print(f"Removed key frame {n}")
    
    def cover_key(self,n:int, name: str = "",time:float=None):
        self.keyframe_data.cover_key(n,self.model,self.data,name,time)
        print(f"Covered key frame {n}")
        
    def reset_key(self,n:int):
        self.data.mocap_pos[:] = self.keyframe_data.keys_in_memory[n].mocap_pos
        self.data.mocap_quat[:] = self.keyframe_data.keys_in_memory[n].mocap_quat
        self.update()
        print(f"Reset to key frame {n}")
        
    ''' 计算帧 '''
    def interpolate_and_record(self, fps: float):
        """
        根据 model 中的 keyframes（含 key_time）进行 MoCap 插值，
        按目标 FPS 生成中间帧，每帧调用 update() 并 record()。
        
        Args:
            fps (float): 目标帧率（如 30.0）
        """
        if self.model.nkey == 0:
            raise ValueError("No keyframes found in the model.")
        
        # --- Step 1: 获取 key 时间和 MoCap 状态 ---
        key_times = np.array(self.model.key_time)  # (nkey,)
        nkey = len(key_times)
        sort_idx = np.argsort(key_times)
        key_times = key_times[sort_idx]

        # 找出所有 mocap bodies（从 model.body_mocapid != -1）
        mocap_body_ids = []
        mocap_body_names = []
        for i in range(self.model.nbody):
            if self.model.body_mocapid[i] != -1:
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                mocap_body_ids.append(i)
                mocap_body_names.append(name)
        
        if not mocap_body_ids:
            raise ValueError("No mocap bodies found in the model.")

        # 提取每个 key 的 mocap_pos 和 mocap_quat
        key_mocap_pos = {}   # {body_name: (nkey, 3)}
        key_mocap_quat = {}  # {body_name: (nkey, 4)}

        for name, body_id in zip(mocap_body_names, mocap_body_ids):
            mocap_id = self.model.body_mocapid[body_id]
            pos_list = []
            quat_list = []
            for k in sort_idx:
                # MuJoCo 3.0+ stores mocap poses in key_mpos / key_mquat
                start3 = 3 * mocap_id
                start4 = 4 * mocap_id
                pos = self.model.key_mpos[k][start3:start3+3].copy()
                quat = self.model.key_mquat[k][start4:start4+4].copy()
                pos_list.append(pos)
                quat_list.append(quat)
            key_mocap_pos[name] = np.stack(pos_list)      # (nkey, 3)
            key_mocap_quat[name] = np.stack(quat_list)    # (nkey, 4)

        # Normalize quaternions
        for name in mocap_body_names:
            q = key_mocap_quat[name]
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            key_mocap_quat[name] = q

        # --- Step 2: 生成插值时间序列 ---
        t_start = key_times[0]
        t_end = key_times[-1]
        total_duration = t_end - t_start
        if total_duration <= 0:
            total_duration = 1.0 / fps  # fallback

        n_interp = int(np.ceil(total_duration * fps)) + 1
        interp_times = np.linspace(t_start, t_end, n_interp)

        # --- Step 3: 插值并逐帧处理 ---
        print(f"Interpolating from {t_start:.3f}s to {t_end:.3f}s at {fps} FPS → {n_interp} frames")

        # 临时重置 data 到初始状态（避免 previous state interference）
        self.reset_world()

        for t in interp_times:
            # Set mocap bodies via interpolation
            for name in mocap_body_names:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                mocap_id = self.model.body_mocapid[body_id]

                # Position: linear interpolation
                pos_interp = np.array([
                    np.interp(t, key_times, key_mocap_pos[name][:, dim])
                    for dim in range(3)
                ])

                # Quaternion: SLERP
                if t <= key_times[0]:
                    quat_interp = key_mocap_quat[name][0]
                elif t >= key_times[-1]:
                    quat_interp = key_mocap_quat[name][-1]
                else:
                    idx = np.searchsorted(key_times, t) - 1
                    t0, t1 = key_times[idx], key_times[idx + 1]
                    alpha = (t - t0) / (t1 - t0)
                    quat_interp = self._slerp(key_mocap_quat[name][idx], key_mocap_quat[name][idx + 1], alpha)

                # Assign to data
                self.data.mocap_pos[mocap_id] = pos_interp
                self.data.mocap_quat[mocap_id] = quat_interp

            # Run your IK/control pipeline
            self.update()
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)

            # Record the resulting state (qpos, qvel, etc.)
            self.record(time=t)
        print(f"✅ Interpolation complete. Recorded {len(self.keyframe_data.keys_in_memory)} frames.")