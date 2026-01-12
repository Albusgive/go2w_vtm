from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from .confirm_terrain_generator import ConfirmTerrainGeneratorCfg,ConfirmTerrainGenerator
import torch

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
        
@configclass
class ConfirmTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = ConfirmTerrainImporter