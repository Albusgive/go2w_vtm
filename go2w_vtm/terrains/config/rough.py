import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import go2w_vtm.terrains as terrain_gen
import go2w_vtm
from go2w_vtm.terrains.confirm_terrain_generator import ConfirmTerrainGeneratorCfg

"""Configuration for custom terrains."""
GO2W_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "convex_terrain": terrain_gen.RectangularTerrainCfgAround(
            proportion=0.3, 
            obstacle_height_mode="convex",
            ring_width_range=(0.1, 0.5), 
            ring_spacing_range=(0.5, 1.3), 
            obstacle_height_range=(0.05, 0.4),   
            platform_width = 2.0,
            difficulty_width = 0.1
        ),
        "concave_terrain": terrain_gen.RectangularTerrainCfgAround(
            proportion=0.3, 
            obstacle_height_mode="concave",
            ring_width_range=(0.05, 0.5),
            ring_spacing_range=(0.5, 1.3), 
            obstacle_height_range=(0.8, 1.2), 
            platform_width = 2.0,
            difficulty_width = 0.1
        ),
        "floating_blocks_easy": terrain_gen.MeshFloatingBoxesTerrainCfg( # The value is a FULL terrain config object
            proportion=0.3,
            platform_width=2.0,
            size=(20.0, 20.0),
            
            box_params_start=terrain_gen.MeshFloatingBoxesTerrainCfg.BoxObjectCfg(
                num_boxes=10,
                size_min=(0.4, 0.4, 0.2),
                size_max=(0.8, 0.8, 0.4),
                floating_height_min=0.4,
                floating_height_max=0.5
            ),
            
            box_params_end=terrain_gen.MeshFloatingBoxesTerrainCfg.BoxObjectCfg(
                num_boxes=20,
                size_min=(0.5, 0.5, 0.3),
                size_max=(1.5, 2.5, 0.5), 
                floating_height_min=0.3,
                floating_height_max=0.4            
            )
        ),
    }
)


GO2W_SHAFT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(3.0, 3.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "convex_terrain": terrain_gen.ShaftTerrainCfgAround(
            proportion=0.3, 
            obstacle_height_range=(0.8, 1.5),   
            bevel_angle = 15.0,
            platform_width = 0.7,
        ),
    }
)


FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(3.0, 3.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "convex_terrain": terrain_gen.ShaftTerrainCfgAround(
            proportion=0.3, 
            obstacle_height_range=(0.8, 1.5),   
            bevel_angle = 15.0,
            platform_width = 0.7,
        ),
    }
)


MIMIC_GYM_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(3, 2.5),
    border_width=20.0,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    # color_scheme="random",
    sub_terrains={
        "mimic_trench": terrain_gen.MimicFixBoxTerrainCfg(
            # proportion=0.5, 
            
            high_platform_x=[0.0,2.35],  
            high_platform_z=[-0.5,-0.5],      
            high_platform_half_width=[1.0,1.0],
            high_platform_half_height=[0.5,0.5],
            
            robot_origin_x = 0.8,
                
            save_to_mjcf = True,
            mesh_path = go2w_vtm.GO2W_MJCF_DIR + "/meshs/",
            mjcf_path = go2w_vtm.GO2W_MJCF_DIR,
            save_name = "trench_box_terrain"
        ),
        # "mimic_high_platform": terrain_gen.MimicFixBoxTerrainCfg(
        #     # proportion=0.5,
            
        #     high_platform_x=[0.0,1.5],  
        #     high_platform_z=[-0.5,0.0],      
        #     high_platform_half_width=[0.75,0.75],
        #     high_platform_half_height=[0.5,0.4],
            
        #     robot_origin_x = 0.8,
            
        #     save_to_mjcf = True,
        #     mesh_path = go2w_vtm.GO2W_MJCF_DIR + "/meshs/",
        #     mjcf_path = go2w_vtm.GO2W_MJCF_DIR,
        #     save_name = "high_platform"
        # ),
    }
)



CONFIRM_TERRAIN_CFG = ConfirmTerrainGeneratorCfg(
    size=(3, 2.5),
    border_width=20.0,
    num_rows=5,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    # color_scheme="random",
    sub_terrains={
        "mimic_trench": terrain_gen.MimicFixBoxTerrainCfg(
            # proportion=0.5, 
            
            high_platform_x=[0.0,2.35],  
            high_platform_z=[-0.5,-0.5],      
            high_platform_half_width=[1.0,1.0],
            high_platform_half_height=[0.5,0.5],
            
            robot_origin_x = 0.8,
                
            save_to_mjcf = True,
            mesh_path = go2w_vtm.GO2W_MJCF_DIR + "/meshs/",
            mjcf_path = go2w_vtm.GO2W_MJCF_DIR,
            save_name = "trench_box_terrain"
        ),
        "mimic_high_platform": terrain_gen.MimicFixBoxTerrainCfg(
            # proportion=0.5,
            
            high_platform_x=[0.0,1.5],  
            high_platform_z=[-0.5,0.0],      
            high_platform_half_width=[0.75,0.75],
            high_platform_half_height=[0.5,0.4],
            
            robot_origin_x = 0.8,
            
            save_to_mjcf = True,
            mesh_path = go2w_vtm.GO2W_MJCF_DIR + "/meshs/",
            mjcf_path = go2w_vtm.GO2W_MJCF_DIR,
            save_name = "high_platform"
        ),
        # "mimic_trench": terrain_gen.BoxTrenchTerrainCfg(
        #     # proportion=0.5, 
            
        #     trench_x = 1.2,                                 
        #     trench_width = (0.2,0.5),
        #     trench_depth = 1.5,
            
        #     robot_origin_x = 0.8,
                
        #     save_to_mjcf = True,
        #     mesh_path = go2w_vtm.GO2W_MJCF_DIR + "/meshs/",
        #     mjcf_path = go2w_vtm.GO2W_MJCF_DIR,
        #     save_name = "trench_box_terrain"
        # ),
        
        # "mimic_trench": terrain_gen.MimicFixBoxTerrainCfg(
        #     # proportion=0.5, 
            
        #     high_platform_x=[0.0,2.35],  
        #     high_platform_z=[-0.5,-0.5],      
        #     high_platform_half_width=[1.0,1.0],
        #     high_platform_half_height=[0.5,0.5],
            
        #     robot_origin_x = 0.8,
                
        #     save_to_mjcf = True,
        #     mesh_path = go2w_vtm.GO2W_MJCF_DIR + "/meshs/",
        #     mjcf_path = go2w_vtm.GO2W_MJCF_DIR,
        #     save_name = "trench_box_terrain"
        # ),
    }
)