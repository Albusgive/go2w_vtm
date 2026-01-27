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
    parser.add_argument("--teacher_checkpoint", type=str, default="model_.*", 
                        help="Regex for teacher's checkpoint file (e.g., 'model_xxx.pt' or date 'yyy-mm-25_22-38-21').")

def get_teacher_checkpoint_path(
    log_path: str, 
    run_dir: str, 
    checkpoint_spec: str
) -> str:
    """智能解析教师检查点路径（零外部依赖），精准处理多实验目录场景"""
    import re
    import os
    
    date_pattern = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
    
    # ========== 步骤1: 解析 checkpoint_spec ==========
    if os.sep in checkpoint_spec or (os.altsep and os.altsep in checkpoint_spec):
        head, tail = os.path.split(checkpoint_spec.strip(os.sep))
        intermediate_dirs = head.split(os.sep) if head else []
        file_pattern = tail if tail else "model_.*"
    else:
        if re.fullmatch(date_pattern, checkpoint_spec):
            intermediate_dirs = [checkpoint_spec]
            file_pattern = "model_.*"
        else:
            intermediate_dirs = None  # 标记需自动查找日期目录
            file_pattern = checkpoint_spec
    
    # ========== 步骤2: 精准查找实验目录（关键修复）==========
    if not os.path.isdir(log_path):
        raise ValueError(f"日志根目录不存在: {os.path.abspath(log_path)}")
    
    candidate_runs = []
    try:
        with os.scandir(log_path) as entries:
            for entry in entries:
                if not (entry.is_dir() and re.match(run_dir, entry.name)):
                    continue
                
                # 【核心修复】当用户指定中间路径时，验证该实验目录是否包含目标子目录
                if intermediate_dirs is not None:
                    target_subdir = os.path.join(entry.path, *intermediate_dirs)
                    if os.path.isdir(target_subdir):
                        candidate_runs.append(entry.path)
                else:
                    candidate_runs.append(entry.path)  # 模式1：保留所有匹配项，后续选最新日期
    except PermissionError as e:
        raise PermissionError(f"无权限访问 '{log_path}': {e}")
    
    # 生成精准错误提示
    if not candidate_runs:
        all_dirs = [d.name for d in os.scandir(log_path) if d.is_dir()]
        if intermediate_dirs is not None:
            hint = f"（且包含子目录: {'/'.join(intermediate_dirs)}）"
            detail = f"\n请确认:\n  • 实验目录 '{run_dir}' 下是否存在 '{'/'.join(intermediate_dirs)}' 子目录\n  • run_dir 正则是否过于宽泛（如匹配了 *_gru 目录）"
        else:
            hint = ""
            detail = ""
        raise ValueError(
            f"在 '{log_path}' 中未找到匹配 '{run_dir}'{hint} 的有效实验目录。\n"
            f"可用目录: {all_dirs}{detail}"
        )
    
    candidate_runs.sort()
    experiment_dir = candidate_runs[-1]
    print(f"[TEACHER] 选定实验目录: {os.path.basename(experiment_dir)} "
          f"(匹配正则 '{run_dir}'，共 {len(candidate_runs)} 个候选)")
    
    # ========== 步骤3: 确定中间目录（日期目录）==========
    if intermediate_dirs is None:  # 模式1：自动查找最新日期目录
        date_dirs = []
        try:
            with os.scandir(experiment_dir) as entries:
                for entry in entries:
                    if entry.is_dir() and re.fullmatch(date_pattern, entry.name):
                        date_dirs.append(entry.name)
        except PermissionError as e:
            raise PermissionError(f"无权限访问 '{experiment_dir}': {e}")
        
        if not date_dirs:
            raise ValueError(
                f"实验目录 '{os.path.basename(experiment_dir)}' 中无有效日期格式子目录。\n"
                f"可用子目录: {[d.name for d in os.scandir(experiment_dir) if d.is_dir()]}"
            )
        
        date_dirs.sort()
        latest_date = date_dirs[-1]
        intermediate_dirs = [latest_date]
        print(f"[TEACHER] 自动选择最新日期目录: {latest_date}")
    
    # 拼接基础路径并验证
    base_dir = os.path.join(experiment_dir, *intermediate_dirs)
    if not os.path.isdir(base_dir):
        raise RuntimeError(
            f"逻辑错误：已验证存在的目录 '{base_dir}' 突然消失！\n"
            f"请检查文件系统或权限问题"
        )
    print(f"[TEACHER] 检查点目录: {os.path.basename(base_dir)}")
    
    # ========== 步骤4: 查找匹配的检查点文件 ==========
    candidates = []
    try:
        with os.scandir(base_dir) as entries:
            for entry in entries:
                if entry.is_file() and re.match(file_pattern, entry.name):
                    candidates.append(entry.name)
    except PermissionError as e:
        raise PermissionError(f"无权限访问 '{base_dir}': {e}")
    
    if not candidates:
        all_files = [f.name for f in os.scandir(base_dir) if f.is_file()]
        sample = all_files[:5] if len(all_files) > 5 else all_files
        raise ValueError(
            f"在 '{base_dir}' 中未找到匹配 '{file_pattern}' 的文件。\n"
            f"目录内容示例: {', '.join(sample) if sample else '（空目录）'}\n"
            f"请检查文件名拼写或确认文件存在"
        )
    
    # 数字感知排序（确保 model_100.pt > model_99.pt）
    def natural_sort_key(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]
    
    candidates.sort(key=natural_sort_key)
    selected_file = candidates[-1]
    checkpoint_path = os.path.join(base_dir, selected_file)
    
    print(f"[TEACHER] 选定检查点: {selected_file} (匹配 {len(candidates)} 个文件)")
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
