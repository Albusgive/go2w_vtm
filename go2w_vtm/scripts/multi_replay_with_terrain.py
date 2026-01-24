# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a motion without any RL agent — pure rendering + scene update."""

import argparse
import sys

from isaaclab.app import AppLauncher
import go2w_vtm

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play motion without RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg,ManagerBasedRLEnv
from isaaclab.utils.dict import print_dict

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config
from go2w_vtm.locomotion.mdp import MotionCommand



# ✅ Fix 1: Accept agent_cfg even if unused
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):  # ← added agent_cfg
    """Play motion without RL agent — only render and update scene."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else 42
    log_dir = os.path.join("logs", "playback", args_cli.task.replace(":", "_"))
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    dt = env.unwrapped.scene.physics_dt
    obs, _ = env.reset()

    timestep = 0
    print("[INFO] Starting motion playback (no physics stepping)...")

    # Get motion command term
    rl_env: ManagerBasedRLEnv = env.unwrapped
    motion_cmd: MotionCommand = rl_env.command_manager.get_term("motion")
    env_ids = torch.ones(rl_env.num_envs, dtype=torch.int64)

    while simulation_app.is_running():
        start_time = time.time()

        # ✅ Fix 2: Correctly index motion data for all environments
        num_envs = rl_env.num_envs
        current_steps = motion_cmd.time_steps  # shape: [num_envs]

        # Clone default root state
        root_states = motion_cmd.robot.data.default_root_state.clone()

        # Set from motion data (all at once using advanced indexing)
        root_states[:, :3] = motion_cmd.anchor_pos_w
        root_states[:, 3:7] = motion_cmd.anchor_quat_w
        root_states[:, 7:10] = motion_cmd.anchor_lin_vel_w
        root_states[:, 10:] = motion_cmd.anchor_ang_vel_w
        
        # Write root state
        motion_cmd.robot.write_root_state_to_sim(root_states)

        # ✅ CORRECT WAY TO INDEX JOINT STATES:
        # joint_pos: [num_envs, num_frames, num_joints]
        # We want: [num_envs, num_joints] at current frame per env
        joint_pos = motion_cmd.joint_pos
        joint_vel = motion_cmd.joint_vel

        motion_cmd.robot.write_joint_state_to_sim(joint_pos, joint_vel)

        # Update sim buffers and render
        rl_env.scene.write_data_to_sim()
        rl_env.sim.render()
        rl_env.scene.update(dt)

        # Advance motion time (this increments time_steps internally)
        motion_cmd._update_command()

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()