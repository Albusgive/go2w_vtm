from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from .confirm_terrain_generator import ConfirmTerrainGeneratorCfg,ConfirmTerrainGenerator
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG,SPHERE_MARKER_CFG
import torch
import numpy as np

class ConfirmTerrainImporter(TerrainImporter):
    
    def __init__(self, cfg: TerrainImporterCfg):
        """Initialize the terrain importer.

        Args:
            cfg: The configuration for the terrain importer.

        Raises:
            ValueError: If input terrain type is not supported.
            ValueError: If terrain type is 'generator' and no configuration provided for ``terrain_generator``.
            ValueError: If terrain type is 'usd' and no configuration provided for ``usd_path``.
            ValueError: If terrain type is 'usd' or 'plane' and no configuration provided for ``env_spacing``.
        """
        # check that the config is valid
        cfg.validate()
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore

        # create buffers for the terrains
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None  # assigned later when `configure_env_origins` is called
        # private variables
        self._terrain_flat_patches = dict()
        self.difficulty = None
        self.sub_terrain_type = None
        self.sub_terrain_type_names = None
        # terrains_checkpoint_data 保存每个 terrains 的 checkpoint 数据(pos_w,根据pos_w上插值便于对接command)
        self.terrains_checkpoint_data: dict[tuple[int, int], np.ndarray] = None
        self.num_rows = None
        self.num_cols = None    

        # auto-import the terrain based on the config
        if self.cfg.terrain_type == "generator":
            # check config is provided
            if self.cfg.terrain_generator is None:
                raise ValueError("Input terrain type is 'generator' but no value provided for 'terrain_generator'.")
            # generate the terrain  
            if isinstance(self.cfg.terrain_generator,ConfirmTerrainGeneratorCfg):
                terrain_generator:ConfirmTerrainGenerator = self.cfg.terrain_generator.class_type(
                cfg=self.cfg.terrain_generator, device=self.device
                )
                self.difficulty = terrain_generator.difficulty
                self.sub_terrain_type = terrain_generator.terrain_types
                self.sub_terrain_type_names = terrain_generator.terrain_type_names
                self.terrains_checkpoint_data = terrain_generator.terrains_checkpoint_data
                self.num_rows = terrain_generator.num_rows
                self.num_cols = terrain_generator.num_cols
            else:
                terrain_generator = self.cfg.terrain_generator.class_type(
                    cfg=self.cfg.terrain_generator, device=self.device
                )
            self.import_mesh("terrain", terrain_generator.terrain_mesh)
            # configure the terrain origins based on the terrain generator
            self.configure_env_origins(terrain_generator.terrain_origins)
            # refer to the flat patches
            self._terrain_flat_patches = terrain_generator.flat_patches
                
        elif self.cfg.terrain_type == "usd":
            # check if config is provided
            if self.cfg.usd_path is None:
                raise ValueError("Input terrain type is 'usd' but no value provided for 'usd_path'.")
            # import the terrain
            self.import_usd("terrain", self.cfg.usd_path)
            # configure the origins in a grid
            self.configure_env_origins()
        elif self.cfg.terrain_type == "plane":
            # load the plane
            self.import_ground_plane("terrain")
            # configure the origins in a grid
            self.configure_env_origins()
        else:
            raise ValueError(f"Terrain type '{self.cfg.terrain_type}' not available.")

        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        if isinstance(self.cfg,ConfirmTerrainImporterCfg):
            self.set_chceckpoint_debug_vis(self.cfg.checkpoint_debug_vis)
        # 均匀环境分布
        if self.cfg.evenly_distributed:
            self.evenly_distributed_env_origins()

    def evenly_distributed_env_origins(self):
        """
        Reset all environment origins to be evenly distributed across all available
        terrain levels and types.
        """
        # 检查地形是否存在
        if self.terrain_origins is None:
            return
        # 1. 获取地形网格的维度
        # self.terrain_origins 的形状通常是 (num_levels, num_types, 3)
        num_rows = self.terrain_origins.shape[0] # max_terrain_level / difficulty levels
        num_cols = self.terrain_origins.shape[1] # terrain types
        total_cells = num_rows * num_cols
        # 2. 生成均匀分布的索引
        # 我们生成一个 0 到 num_envs 的序列，然后对 total_cells 取余。
        # 这保证了每个地形格子被分配到的次数几乎完全一致（最多相差1个）。
        indices = torch.arange(self.cfg.num_envs, device=self.device) % total_cells
        # 3. 将线性索引转换回 (level, type) 矩阵坐标
        # level = index / num_cols (整除)
        # type  = index % num_cols (取余)
        levels = torch.div(indices, num_cols, rounding_mode='floor')
        types = indices % num_cols
        # 4. 随机打乱分配 (Shuffle)
        # 如果不打乱，env 0 永远会在 level 0, type 0。
        # 我们希望分布是均匀的，但具体哪个 robot 去哪应该是随机的。
        perm = torch.randperm(self.cfg.num_envs, device=self.device)
        # 5. 更新类属性
        # 注意这里使用切片[:]来原地更新tensor，保持引用不变
        self.terrain_levels[:] = levels[perm]
        self.terrain_types[:] = types[perm]
        # 6. 更新物理坐标 origins
        self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
        
        
    def get_difficulty(self,env_ids: torch.Tensor):
        if self.difficulty is None:
            return None
        return self.difficulty[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def get_sub_terrain_type(self,env_ids: torch.Tensor):
        if self.sub_terrain_type is None:
            return None
        return self.sub_terrain_type[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def get_sub_terrain_type_names(self):
        if self.sub_terrain_type_names is None:
            return None
        return self.sub_terrain_type_names
    
    
    def set_chceckpoint_debug_vis(self, debug_vis: bool) -> bool:
        if debug_vis:
            if not hasattr(self, "checkpoint_visualizer"): 
                self.checkpoint_visualizer = VisualizationMarkers(
                    cfg=SPHERE_MARKER_CFG.replace(prim_path="/Visuals/TerrainCheckpoint")
                )
                if self.terrains_checkpoint_data is not None:
                    all_points = []
                    for (row, col), checkpoint_data in self.terrains_checkpoint_data.items():
                        all_points.append(checkpoint_data)
                    if len(all_points) > 0:
                        all_points_combined = np.concatenate(all_points, axis=0)
                        self.checkpoint_visualizer.visualize(all_points_combined)
            self.checkpoint_visualizer.set_visibility(True)
        else:
            if hasattr(self, "checkpoint_visualizer"):
                self.checkpoint_visualizer.set_visibility(False)
        
        # report success
        return True
        
@configclass
class ConfirmTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = ConfirmTerrainImporter
    checkpoint_debug_vis: bool = False
    evenly_distributed: bool = True