import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import patterns, RayCasterCameraCfg, RayCasterCfg
from isaaclab.assets import Articulation
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg,TerrainImporter
from isaaclab.assets import RigidObjectCfg

##
# Pre-defined configs
##
from go2w_vtm.terrains import ConfirmTerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from go2w_vtm.terrains.config.rough import BOX_TERRAIN_CFG
from go2w_vtm.terrains.mimic_gym_terrain_cfg import SaveTerrainCfg
from go2w_vtm.Robot.go2w import UNITREE_GO2W_CFG
from go2w_vtm.sensors.ray_caster_camera import MyRayCasterCameraCfg

@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        terrain_generator = BOX_TERRAIN_CFG,
        debug_vis=True,
    )
    
    robot = UNITREE_GO2W_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    
    ray_caster_camera = MyRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCameraCfg.OffsetCfg(pos=(0.35, 0.0, 0.03),
                                            rot = (math.cos(math.radians(15)/2), 0, math.sin(math.radians(15)/2), 0),
                                            convention="world"),
        ray_alignment='yaw',
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=1.0,
            horizontal_aperture=2 * math.tan(math.radians(89.51) / 2),  # fovx
            vertical_aperture=2 * math.tan(math.radians(58.29) / 2),  # fovy
            width=32,
            height=18,),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
        data_types=["distance_to_camera"],
    )


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    robot: Articulation = scene["robot"]
    terrain: TerrainImporter = scene["terrain"]
    while simulation_app.is_running():
        root_state = robot.data.default_root_state.clone()
        root_state[0,:2] = terrain.env_origins[0,:2]
        robot.write_root_state_to_sim(root_state)
        scene.update(sim.cfg.dt)
        sim.step()
        
def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = SceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
