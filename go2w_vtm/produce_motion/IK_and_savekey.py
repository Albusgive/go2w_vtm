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
    def __init__(self,file_path:str,cfg:ik_cfg):
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
        
        # 数据保存
        self.keyframe_data = KeyFrame.KeyFrame()
        self.keyframe_data.setRecordMocapBody2Body(self.model,self.cfg.mocap_robot_name)
        self.keyframe_data.setSaveFields(qvel=False,act=False,ctrl=False,mocap_pos=False,mocap_quat=False)
        self.history_times = []
        self.history_anchor_ref_pos = []
        self.history_anchor_ref_quat = []
        self.nkey = 0
        
        self.ex_keyframe_data = KeyFrame.KeyFrame()
        self.ex_keyframe_data.setRecordMocapBody2Body(self.model,self.cfg.mocap_robot_name)
        self.ex_keyframe_data.setSaveFields(qvel=False,act=False,ctrl=False,mocap_pos=False,mocap_quat=False)
        
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
        if time is not None:
            self.history_times.append(time)
        self.history_anchor_ref_pos.append(self.data.mocap_pos)
        self.history_anchor_ref_quat.append(self.data.mocap_quat)
        print(f"Recorded key frame {self.nkey}")
        self.nkey += 1
        
    def save_qpos(self,save_path:str):
        self.keyframe_data.save_as_npz(save_path)
    
    def save_keyframe(self,save_path:str):
        self.keyframe_data.save_as_xml(save_path)
        
    def save_ex_keyframe(self,save_path:str):
        self.ex_keyframe_data.save_as_xml(save_path)
        
    def clear_ex_keyframe(self):
        self.ex_keyframe_data.clear()
    
    def remove_key(self,n:int):
        self.keyframe_data.remove_key(n)
        self.history_times.pop(n)
        self.history_anchor_ref_pos.pop(n)
        self.history_anchor_ref_quat.pop(n)
        self.nkey -= 1
        print(f"Removed key frame {n}")
        
    def reset_key(self,n:int):
        self.data.mocap_pos[:] = self.history_anchor_ref_pos[n]
        self.data.mocap_quat[:] = self.history_anchor_ref_quat[n]
        self.update()
        print(f"Reset to key frame {n}")
    
    def compute_frame(self,target_fps:int=30):
        if self.history_times[0] is None:
            raise ValueError("History times are not recorded")
        #读取历史的anchor位姿 计算mocap关键帧位置，计算mocap路径，update并保存到ex_keyframe_data中
        