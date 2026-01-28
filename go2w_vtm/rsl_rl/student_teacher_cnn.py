# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal
from typing import Any, NoReturn

from rsl_rl.networks import CNN, MLP, EmpiricalNormalization, HiddenState


class StudentTeacherCNN(nn.Module):
    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        student_obs_normalization: bool = False,
        teacher_obs_normalization: bool = False,
        student_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        teacher_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        student_cnn_cfg: dict[str, dict] | dict | None = None,
        teacher_cnn_cfg: dict[str, dict] | dict | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "StudentTeacherCNN.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.student_obs_metadata = {g: list(obs[g].shape[1:]) for g in obs_groups["policy"]}

        # =================================================================================
        # 1. Setup Student (Policy)
        # =================================================================================
        num_student_obs_1d = 0
        self.student_obs_groups_1d = []
        student_in_dims_2d = []
        student_in_channels_2d = []
        self.student_obs_groups_2d = []

        # Analyze Student Observation Shapes
        for obs_group in obs_groups["policy"]:
            shape = obs[obs_group].shape
            if len(shape) == 4:  # B, C, H, W -> CNN
                self.student_obs_groups_2d.append(obs_group)
                student_in_dims_2d.append(shape[2:4])
                student_in_channels_2d.append(shape[1])
            elif len(shape) == 2:  # B, Dim -> MLP
                self.student_obs_groups_1d.append(obs_group)
                num_student_obs_1d += shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for student {obs_group}: {shape}")

        # Student CNNs
        self.student_cnns = None
        student_encoding_dim = 0
        if self.student_obs_groups_2d:
            assert student_cnn_cfg is not None, "Student CNN config required for 2D observations."
            # Handle config dict matching
            if not all(isinstance(v, dict) for v in student_cnn_cfg.values()):
                student_cnn_cfg = {group: student_cnn_cfg for group in self.student_obs_groups_2d}
            
            self.student_cnns = nn.ModuleDict()
            for idx, obs_group in enumerate(self.student_obs_groups_2d):
                self.student_cnns[obs_group] = CNN(
                    input_dim=student_in_dims_2d[idx],
                    input_channels=student_in_channels_2d[idx],
                    **student_cnn_cfg[obs_group],
                )
                print(f"Student CNN for {obs_group}: {self.student_cnns[obs_group]}")
                if self.student_cnns[obs_group].output_channels is None:
                    student_encoding_dim += int(self.student_cnns[obs_group].output_dim)
                else:
                    raise ValueError("Student CNN output must be flattened.")

        # Student MLP
        self.student = MLP(num_student_obs_1d + student_encoding_dim, num_actions, student_hidden_dims, activation)
        print(f"Student MLP: {self.student}")

        # Student Normalization (only for 1D inputs)
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs_1d)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        # =================================================================================
        # 2. Setup Teacher
        # =================================================================================
        num_teacher_obs_1d = 0
        self.teacher_obs_groups_1d = []
        teacher_in_dims_2d = []
        teacher_in_channels_2d = []
        self.teacher_obs_groups_2d = []

        # Analyze Teacher Observation Shapes
        for obs_group in obs_groups["teacher"]:
            shape = obs[obs_group].shape
            if len(shape) == 4:  # B, C, H, W
                self.teacher_obs_groups_2d.append(obs_group)
                teacher_in_dims_2d.append(shape[2:4])
                teacher_in_channels_2d.append(shape[1])
            elif len(shape) == 2:  # B, Dim
                self.teacher_obs_groups_1d.append(obs_group)
                num_teacher_obs_1d += shape[-1]
            else:
                raise ValueError(f"Invalid observation shape for teacher {obs_group}: {shape}")

        # Teacher CNNs
        self.teacher_cnns = None
        teacher_encoding_dim = 0
        if self.teacher_obs_groups_2d:
            # Note: If teacher is loaded from checkpoint, this config ensures structure matches
            assert teacher_cnn_cfg is not None, "Teacher CNN config required for 2D observations."
            if not all(isinstance(v, dict) for v in teacher_cnn_cfg.values()):
                teacher_cnn_cfg = {group: teacher_cnn_cfg for group in self.teacher_obs_groups_2d}

            self.teacher_cnns = nn.ModuleDict()
            for idx, obs_group in enumerate(self.teacher_obs_groups_2d):
                self.teacher_cnns[obs_group] = CNN(
                    input_dim=teacher_in_dims_2d[idx],
                    input_channels=teacher_in_channels_2d[idx],
                    **teacher_cnn_cfg[obs_group],
                )
                print(f"Teacher CNN for {obs_group}: {self.teacher_cnns[obs_group]}")
                if self.teacher_cnns[obs_group].output_channels is None:
                    teacher_encoding_dim += int(self.teacher_cnns[obs_group].output_dim)
                else:
                    raise ValueError("Teacher CNN output must be flattened.")

        # Teacher MLP
        self.teacher = MLP(num_teacher_obs_1d + teacher_encoding_dim, num_actions, teacher_hidden_dims, activation)
        print(f"Teacher MLP: {self.teacher}")

        # Teacher Normalization (only for 1D inputs)
        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs_1d)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        # =================================================================================
        # 3. Action Noise & Dist (Standard)
        # =================================================================================
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(
        self, dones: torch.Tensor | None = None, hidden_states: tuple[HiddenState, HiddenState] = (None, None)
    ) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _update_distribution(self, mlp_obs: torch.Tensor, cnn_obs: dict[str, torch.Tensor]) -> None:
        # Student pass
        if self.student_cnns is not None:
            cnn_enc_list = [self.student_cnns[group](cnn_obs[group]) for group in self.student_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)
        
        mean = self.student(mlp_obs)

        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_student_obs(obs)
        mlp_obs = self.student_obs_normalizer(mlp_obs)
        self._update_distribution(mlp_obs, cnn_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        mlp_obs, cnn_obs = self.get_student_obs(obs)
        mlp_obs = self.student_obs_normalizer(mlp_obs)

        if self.student_cnns is not None:
            cnn_enc_list = [self.student_cnns[group](cnn_obs[group]) for group in self.student_obs_groups_2d]
            cnn_enc = torch.cat(cnn_enc_list, dim=-1)
            mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)
        
        return self.student(mlp_obs)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        # Teacher inference
        mlp_obs, cnn_obs = self.get_teacher_obs(obs)
        mlp_obs = self.teacher_obs_normalizer(mlp_obs)

        with torch.no_grad():
            if self.teacher_cnns is not None:
                cnn_enc_list = [self.teacher_cnns[group](cnn_obs[group]) for group in self.teacher_obs_groups_2d]
                cnn_enc = torch.cat(cnn_enc_list, dim=-1)
                mlp_obs = torch.cat([mlp_obs, cnn_enc], dim=-1)
            
            return self.teacher(mlp_obs)

    def get_student_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 1D obs
        if self.student_obs_groups_1d:
            obs_list_1d = [obs[obs_group] for obs_group in self.student_obs_groups_1d]
            mlp_obs = torch.cat(obs_list_1d, dim=-1)
        else:
            # Handle case where student only has 2D obs (unlikely but possible)
            mlp_obs = torch.empty((obs.batch_size[0], 0), device=obs.device)

        # 2D obs
        obs_dict_2d = {}
        for obs_group in self.student_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        
        return mlp_obs, obs_dict_2d

    def get_teacher_obs(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 1D obs
        if self.teacher_obs_groups_1d:
            obs_list_1d = [obs[obs_group] for obs_group in self.teacher_obs_groups_1d]
            mlp_obs = torch.cat(obs_list_1d, dim=-1)
        else:
            mlp_obs = torch.empty((obs.batch_size[0], 0), device=obs.device)

        # 2D obs
        obs_dict_2d = {}
        for obs_group in self.teacher_obs_groups_2d:
            obs_dict_2d[obs_group] = obs[obs_group]
        
        return mlp_obs, obs_dict_2d

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        return None, None

    def detach_hidden_states(self, dones: torch.Tensor | None = None) -> None:
        pass

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        # Ensure teacher stays in eval mode
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()
        if self.teacher_cnns is not None:
            self.teacher_cnns.eval()

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            student_obs, _ = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """
        Loads parameters. Can handle two cases:
        1. Loading a distilled checkpoint (has 'student' and 'teacher' keys).
        2. Loading an RL checkpoint (has 'actor' keys) -> Maps 'actor' to 'teacher'.
        """
        # Case 1: Load from RL training (ActorCritic / ActorCriticCNN) -> Teacher
        if any("actor" in key for key in state_dict):
            print("Loading teacher from ActorCritic checkpoint...")
            
            teacher_state_dict = {}
            teacher_norm_state_dict = {}
            teacher_cnn_state_dict = {}

            for key, value in state_dict.items():
                # Map MLP
                if "actor." in key and "actor_cnns." not in key:
                    # e.g. "actor.0.weight" -> "0.weight"
                    teacher_state_dict[key.replace("actor.", "")] = value
                
                # Map Normalizer
                if "actor_obs_normalizer." in key:
                    teacher_norm_state_dict[key.replace("actor_obs_normalizer.", "")] = value
                
                # Map CNNs
                if "actor_cnns." in key:
                    # e.g. "actor_cnns.depth.conv1.weight" -> "depth.conv1.weight"
                    # The ModuleDict keys in state_dict are prefixed by the dict name.
                    # We need to strip "actor_cnns." to match what self.teacher_cnns expects 
                    # if we were to use load_state_dict on the submodule, OR
                    # rename to "teacher_cnns." if we use top-level load.
                    # Here we collect to load into submodules or construct a new dict.
                    
                    # Strategy: constructing a dict for teacher_cnns module
                    key_suffix = key.replace("actor_cnns.", "")
                    teacher_cnn_state_dict[key_suffix] = value

            # Load MLP
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            
            # Load Normalizer
            self.teacher_obs_normalizer.load_state_dict(teacher_norm_state_dict, strict=strict)
            
            # Load CNNs (if teacher expects them)
            if self.teacher_cnns is not None:
                if not teacher_cnn_state_dict and strict:
                    raise ValueError("Teacher expects CNN weights but none found in checkpoint (actor_cnns).")
                self.teacher_cnns.load_state_dict(teacher_cnn_state_dict, strict=strict)
            
            self.loaded_teacher = True
            self.train() # Resets teacher to eval mode
            return False # Training does not resume (fresh start for student)

        # Case 2: Load from Distillation training (StudentTeacher)
        elif any("student" in key for key in state_dict):
            print("Loading StudentTeacher checkpoint...")
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.train()
            return True # Training resumes

        else:
            raise ValueError("state_dict does not contain recognized keys (actor or student).")