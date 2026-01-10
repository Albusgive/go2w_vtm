import mujoco

class RayCasterData:
    
    def __init__(self,model:mujoco.MjModel,data:mujoco.MjData,sensor_name:str="raycaster"):
        self.model = model
        self.data = data
        self.h_ray_num,self.v_ray_num,self.data_ps = self.get_ray_caster_info(model,data,sensor_name)
        self.sensor_data = data.sensor(sensor_name).data
        
        
    def get_ray_caster_info(self,model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str):
        data_ps = []
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
        if sensor_id == -1:
            print("Sensor not found")
            return 0, 0, data_ps
        sensor_plugin_id = model.sensor_plugin[sensor_id]
        state_idx = model.plugin_stateadr[sensor_plugin_id]
        state_num = model.plugin_statenum[sensor_plugin_id]
        for i in range(state_idx + 2, state_idx + state_num, 2):
            if i + 1 < len(data.plugin_state):
                data_ps.append((int(data.plugin_state[i]), int(data.plugin_state[i + 1])))
        h_ray_num = (
            int(data.plugin_state[state_idx]) if state_idx < len(data.plugin_state) else 0
        )
        v_ray_num = (
            int(data.plugin_state[state_idx + 1])
            if state_idx + 1 < len(data.plugin_state)
            else 0
        )
        return h_ray_num, v_ray_num, data_ps
    
    def get_data(self,num:int=0):
        return self.sensor_data[self.data_ps[num][0]:self.data_ps[num][0]+self.data_ps[num][1]]
    
    def get_pos_w(self,num:int=0):
        return self.sensor_data[self.data_ps[num][0]:self.data_ps[num][0]+self.data_ps[num][1]].reshape(-1, 3)
        