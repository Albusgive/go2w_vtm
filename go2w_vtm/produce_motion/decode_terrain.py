import mujoco
import numpy as np

class DecodeTerrain:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.terrain_name = None
        self.terrain_difficulty = None
        self.terrain_key_pos = []
        self.n_points = 0
        for id in range(self.model.ntext):
            name:str = mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_TEXT,id)
            data = self.model.text_data[self.model.text_adr[id]:self.model.text_adr[id]+self.model.text_size[id]-1]
            if isinstance(data, bytes):
                try:
                    data = data.decode('utf-8')
                except UnicodeDecodeError:
                    raise ValueError("Input data is not valid UTF-8 text. Are you passing binary by mistake?")
            if "terrain:" in name:
                self.terrain_name = name.split(":")[1]
                if "difficulty:" in data:
                    self.terrain_difficulty = float(data.split("difficulty:")[-1].split()[0])
            if "terrain_key_pos" in name:
                self.terrain_key_pos.append(np.fromstring(data, sep=' '))
        print(self.terrain_name,self.terrain_difficulty)
        self.n_points = len(self.terrain_key_pos)
        self.terrain_key_pos = np.array(self.terrain_key_pos)
        print(self.terrain_key_pos)
        

            
        