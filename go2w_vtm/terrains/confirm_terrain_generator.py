from isaaclab.utils import configclass
from isaaclab.terrains.terrain_generator import TerrainGenerator
import numpy as np
import torch
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from go2w_vtm.terrains.mimic_gym_terrain_cfg import SaveTerrainCfg

class ConfirmTerrainGenerator(TerrainGenerator):
    
    def __init__(self, cfg, device: str = "cpu"):
        self.difficulty_map = None
        self.terrain_type_map = None
        self.terrain_type_names = None
        # terrains_checkpoint_data 保存每个 terrains 的 checkpoint 数据(pos_w)
        self.terrains_checkpoint_data: dict[tuple[int, int], np.ndarray] = {}
        self.num_rows = cfg.num_rows
        self.num_cols = cfg.num_cols    
        super().__init__(cfg, device)
        
        
    def _generate_curriculum_terrains(self):
        proportions = np.array([sub_cfg.proportion for sub_cfg in self.cfg.sub_terrains.values()])
        proportions /= np.sum(proportions)

        sub_indices = []
        for index in range(self.cfg.num_cols):
            sub_index = np.min(np.where(index / self.cfg.num_cols + 0.001 < np.cumsum(proportions))[0])
            sub_indices.append(sub_index)
        sub_indices = np.array(sub_indices, dtype=np.int32)
        sub_terrains_cfgs = list(self.cfg.sub_terrains.values())

        lower, upper = self.cfg.difficulty_range

        # 初始化两个 map
        difficulty_map_np = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.float32)
        terrain_type_map_np = np.zeros((self.cfg.num_rows, self.cfg.num_cols), dtype=np.int32)
        self.terrain_type_names = list(self.cfg.sub_terrains.keys())

        for sub_col in range(self.cfg.num_cols):
            terrain_type_id = sub_indices[sub_col]  #  当前列的地形类型 ID
            for sub_row in range(self.cfg.num_rows):
                # --- Difficulty ---
                difficulty_norm = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                difficulty = lower + (upper - lower) * difficulty_norm
                difficulty_map_np[sub_row, sub_col] = difficulty

                # --- Terrain Type ---
                terrain_type_map_np[sub_row, sub_col] = terrain_type_id  #  所有行相同
                
                # --- Generate mesh ---
                mesh, origin = self._get_terrain_mesh(difficulty, sub_terrains_cfgs[terrain_type_id])
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_terrains_cfgs[terrain_type_id])
                if isinstance(sub_terrains_cfgs[terrain_type_id], SaveTerrainCfg):
                    sub_terrain:SaveTerrainCfg = sub_terrains_cfgs[terrain_type_id]
                    terrain_checkpoint,_ = sub_terrain.make_check_points(difficulty)
                    if terrain_checkpoint is not None:
                        offset1 = np.array([-self.cfg.size[0] * 0.5, -self.cfg.size[1] * 0.5, 0.0]) # 中心点偏移
                        offset2 = np.array([(sub_row + 0.5) * self.cfg.size[0], (sub_col + 0.5) * self.cfg.size[1], 0.0]) # sub_terrain 中心点偏移
                        offset3 = np.array([-self.cfg.size[0] * self.cfg.num_rows * 0.5, -self.cfg.size[1] * self.cfg.num_cols * 0.5, 0.0]) # 总体偏移
                        terrain_checkpoint += offset1 + offset2 + offset3
                        self.terrains_checkpoint_data[(sub_row, sub_col)] = np.array(terrain_checkpoint)

        # 转为 tensor
        self.difficulty_map = torch.from_numpy(difficulty_map_np).to(self.device)
        self.terrain_type_map = torch.from_numpy(terrain_type_map_np).to(self.device)  # 保存
                
    @property
    def difficulty(self):
        """Shape: [num_rows, num_cols], value = difficulty level."""
        return self.difficulty_map

    @property
    def terrain_types(self):
        """Shape: [num_rows, num_cols], value = type_id (int)."""
        return self.terrain_type_map
    
                
@configclass
class ConfirmTerrainGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = ConfirmTerrainGenerator