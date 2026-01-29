# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import argparse
import random
from typing import TYPE_CHECKING
import os
import re

if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """Add RSL-RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
    # -- experiment arguments
    arg_group.add_argument(
        "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
    )
    arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
    # -- load arguments
    arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
    arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
    arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
    # -- logger arguments
    arg_group.add_argument(
        "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
    )
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
    parser.add_argument("--teacher_load_run", type=str, default=".*", 
                        help="Regex for teacher's run directory (e.g., 'policy experiment_name').")
    parser.add_argument("--teacher_checkpoint", type=str, default=None, 
                        help="Regex for teacher's checkpoint file (e.g., 'model_xxx.pt' or date 'yyy-mm-25_22-38-21').")

def get_teacher_checkpoint_path(
    log_path: str, 
    run_dir: str = None, 
    checkpoint_spec: str = None
) -> str:
    """智能解析教师检查点路径
    
    支持:
    1. 路径型: '2026-01-29/model_100.pt'
    2. 正则/文件名型: 'model_.*\.pt' 或 'model_10000.pt'
    3. None型: 找最新日期目录下序号最大的 .pt 文件
    """
    import re
    import os
    if run_dir is None:
        raise ValueError("teacher_load_run must be specified")
    
    # 1. 确定实验根目录 (直接拼接，不使用正则匹配目录名以确保精准)
    experiment_root = os.path.join(log_path, run_dir)
    if not os.path.isdir(experiment_root):
        raise ValueError(f"实验根目录不存在: {os.path.abspath(experiment_root)}")

    date_pattern = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"

    # 2. 确定目标目录 (base_dir) 和 文件匹配模式 (file_pattern)
    if checkpoint_spec and (os.sep in checkpoint_spec or (os.altsep and os.altsep in checkpoint_spec)):
        # 路径型：从 spec 中分离目录和文件名
        sub_dir, file_pattern = os.path.split(checkpoint_spec)
        base_dir = os.path.join(experiment_root, sub_dir)
    else:
        # 非路径型：先找最新的日期目录
        date_dirs = [
            d for d in os.listdir(experiment_root) 
            if os.path.isdir(os.path.join(experiment_root, d)) and re.match(date_pattern, d)
        ]
        if not date_dirs:
            raise ValueError(f"在 {experiment_root} 中未找到日期格式的子目录")
        
        date_dirs.sort()
        base_dir = os.path.join(experiment_root, date_dirs[-1])
        file_pattern = checkpoint_spec if checkpoint_spec else r"model_.*\.pt"
        print(f"[TEACHER] 选定最新日期目录: {date_dirs[-1]}")

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"目录不存在: {base_dir}")

    # 3. 在目标目录中根据正则查找所有匹配的文件
    # 如果 file_pattern 是普通文件名，re.match 也能匹配成功
    files = [
        f for f in os.listdir(base_dir) 
        if os.path.isfile(os.path.join(base_dir, f)) and re.match(file_pattern, f)
    ]

    if not files:
        raise FileNotFoundError(f"在目录 {base_dir} 中未找到匹配 '{file_pattern}' 的文件")

    # 4. 数字感知排序 (核心修复：确保 model_10000.pt > model_999.pt)
    def natural_sort_key(s):
        # 将字符串拆分为数字和非数字部分，并将数字部分转为 int 进行比较
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    
    files.sort(key=natural_sort_key)
    selected_file = files[-1] # 取序号最大的

    checkpoint_path = os.path.join(base_dir, selected_file)
    print(f"[TEACHER] 成功匹配检查点: {selected_file}")
    
    return os.path.abspath(checkpoint_path)

def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlBaseRunnerCfg:
    """Parse configuration for RSL-RL agent based on inputs.

    Args:
        task_name: The name of the environment.
        args_cli: The command line arguments.

    Returns:
        The parsed configuration for RSL-RL agent based on inputs.
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # load the default configuration
    rslrl_cfg: RslRlBaseRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
    rslrl_cfg = update_rsl_rl_cfg(rslrl_cfg, args_cli)
    return rslrl_cfg


def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
