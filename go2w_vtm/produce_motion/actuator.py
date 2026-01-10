import mujoco 
import numpy as np

class PDForceCtrl:
    
    def __init__(self,model:mujoco.MjModel,data:mujoco.MjData,joint_names:list[str],kp:float,kd:float,
                 foot_names:list[str]):
        self.actuator_ids_map = {}
        for joint_name in joint_names:
            self.actuator_ids_map[joint_name] = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,joint_name)
        for foot_name in foot_names:
            self.actuator_ids_map[foot_name] = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,foot_name)
            
        self.foot_act_ids_map = {} #{"foot_name":[act_id_x,act_id_y,act_id_z]}
        for foot_act_name in foot_names:
            act_ids = []
            for axis in ["x","y","z"]:
                act_ids.append(mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,foot_act_name+"_"+axis))
            self.foot_act_ids_map[foot_act_name] = act_ids
        self.kp = kp
        self.kd = kd
        self.data = data
        
    def ctrl_joint_pd(self,joint_names:list[str],pos_ref:list[float]):
        for i in range(len(joint_names)):
            joint_name = joint_names[i]
            act_id = self.actuator_ids_map[joint_name]
            pos = self.data.sensor(joint_name+"_pos").data[0]
            vel = self.data.sensor(joint_name+"_vel").data[0]
            pos_error = pos_ref[i] - pos
            self.data.ctrl[act_id] = self.kp*pos_error - self.kd*vel
            

    def ctrl_foot_force(self,foot_name:list[str],force:list[list[float]]):
        ''' ref_site 的力 '''
        for i in range(len(foot_name)):
            act_ids = self.foot_act_ids_map[foot_name[i]]
            for j in range(3):
                self.data.ctrl[act_ids[j]] = force[i][j]
    
    def ctrl(self,foot_name:list[str],ctrl:list[float]):
        ''' 直接ctrl '''
        for i in range(len(foot_name)):
            act_id = self.actuator_ids_map[foot_name[i]]
            self.data.ctrl[act_id] = ctrl[i]
            
            
class StateCtrl:
    joint_names = ["FL_hip_joint","FL_thigh_joint","FL_calf_joint",
                "FR_hip_joint","FR_thigh_joint","FR_calf_joint",
                "RL_hip_joint","RL_thigh_joint","RL_calf_joint",
               "RR_hip_joint","RR_thigh_joint","RR_calf_joint",]

    foot_names = ["FL_foot_joint","FR_foot_joint","RL_foot_joint","RR_foot_joint"]
    
    legs = ["FL","FR","RL","RR"]

    # 正常站立
    stand_L = [0.00571868, 1.11, -1.92]
    stand_R = [-0.00571868, 1.11, -1.92]
    # 前后脚靠拢
    pre_joint_target = [0.00571868, 1.5, -1.78,
                     0.00571868, 1.5, -1.78,
                     -0.00571868, 0.841, -2.01,
                     -0.00571868, 0.841, -2.01,]
    # 向后蹬腿
    _leg_back_force = [0.0, 20.0, 20.0]
    _leg_back_pos_L = [0.00571868, 0.9, -2.55]
    _leg_back_pos_R = [-0.00571868, 0.9, -2.55]
    
    #向前伸腿
    _leg_front_pos_L = [0.00571868, -0.621, -1.03]
    _leg_front_pos_R = [-0.00571868, -0.621, -1.03]
    
    kp = 50.0
    kd = 3.5    
    
    def __init__(self,model:mujoco.MjModel,data:mujoco.MjData):
        self.pdf = PDForceCtrl(model,data,self.joint_names,self.kp,self.kd,self.foot_names)
        
    def leap(self,ray_data,ray_threshold:float = -0.01,
             f_leg_force_threshold:float = 0.0,
             r_leg_z_threshold:float = 0.05,
             r_leg_force_threshold:float = 1.0):
        # ray_data是npy数组，如果全大于0
        if np.all(ray_data > ray_threshold):
            for leg_name in self.legs:
                self.normal_stand(leg_name)
            return
        else:
            self.pre_stand()
            # 前腿接处力
            if self.pdf.data.sensor("FL_foot_joint_force").data[2] < f_leg_force_threshold:
                self.leg_front_pos("FL")
            if self.pdf.data.sensor("FR_foot_joint_force").data[2] < f_leg_force_threshold:
                self.leg_front_pos("FR")
            # 后腿z高度
            if self.pdf.data.sensor("RL_foot_joint_pos").data[2] < r_leg_z_threshold:
                self.leg_back_force("RL")
            if self.pdf.data.sensor("RR_foot_joint_pos").data[2] < r_leg_z_threshold:
                self.leg_back_force("RR")
            # 后腿完整接触力
            if np.linalg.norm(self.pdf.data.sensor("RL_foot_joint_force").data) < r_leg_force_threshold:
                self.leg_back_pos("RL")
            if np.linalg.norm(self.pdf.data.sensor("RR_foot_joint_force").data) < r_leg_force_threshold:
                self.leg_back_pos("RR")

            
    def normal_stand(self,leg_name:str="FL"):
        joint_names = self.leg_2_joint_names(leg_name)
        if leg_name.find("L") != -1:
            self.pdf.ctrl_joint_pd(joint_names,self.stand_L)
        elif leg_name.find("R") != -1:
            self.pdf.ctrl_joint_pd(joint_names,self.stand_R)
        
    def pre_stand(self):
        self.pdf.ctrl_joint_pd(self.joint_names,self.pre_joint_target)
        
    def leg_back_force(self,leg_name:str="FL"):
        ''' 针对后腿 '''
        joint_names = self.leg_2_joint_names(leg_name)
        self.pdf.ctrl(joint_names,self._leg_back_force)
        
    def leg_back_pos(self,leg_name:str="FL"):
        joint_names = self.leg_2_joint_names(leg_name)
        if leg_name.find("L") != -1:
            self.pdf.ctrl_joint_pd(joint_names,self._leg_back_pos_L)
        elif leg_name.find("R") != -1:
            self.pdf.ctrl_joint_pd(joint_names,self._leg_back_pos_R)
        
    def leg_front_pos(self,leg_name:str="FL"):
        joint_names = self.leg_2_joint_names(leg_name)
        if leg_name.find("L") != -1:
            self.pdf.ctrl_joint_pd(joint_names,self._leg_front_pos_L)
        elif leg_name.find("R") != -1:
            self.pdf.ctrl_joint_pd(joint_names,self._leg_front_pos_R)
    
    def ctrl_wheel(self,ctrl:list[float]):
        self.pdf.ctrl(self.foot_names,ctrl)
        
    def leg_2_joint_names(self,leg_name:str="FL"):
        ''' 将腿名转换为关节名 '''
        if leg_name == "FL":
            return ["FL_hip_joint","FL_thigh_joint","FL_calf_joint"]
        elif leg_name == "FR":
            return ["FR_hip_joint","FR_thigh_joint","FR_calf_joint"]
        elif leg_name == "RL":
            return ["RL_hip_joint","RL_thigh_joint","RL_calf_joint"]
        elif leg_name == "RR":
            return ["RR_hip_joint","RR_thigh_joint","RR_calf_joint"]
        