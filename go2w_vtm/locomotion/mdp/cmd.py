from go2w_vtm.utils.MocapInterpolator import MocapInterpolator
class MotionGenerator(CommandTerm):
    
    cfg: MotionGeneratorCfg

    def __init__(self, cfg: MotionGeneratorCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        
        self.robot: Articulation = env.scene[cfg.asset_name]

        if not isinstance(self._env.scene.terrain, ConfirmTerrainImporter):
            raise ValueError("Terrain must be ConfirmTerrainImporter")
        self.terrain: ConfirmTerrainImporter = self._env.scene.terrain
        self.preprocess_checkpoint_data()
        
        self.terrain_names = self.cfg.terrain_and_checkpoint_file.keys()
        self.terrain_types = [self.terrain.terrain_name2type[name] for name in self.terrain_names]
        self.interpolator_map = {} # terrain_type -> MocapInterpolator
        for terrain_type in self.terrain_types:
            file_path = self.cfg.terrain_and_checkpoint_file[self.terrain.terrain_type2name[terrain_type]]
            self.interpolator_map[terrain_type] = MocapInterpolator(file_path,device=self.device)
            
       # 计算motion轨迹长度 每个地形的最慢速度计算轨迹长度
        max_length = 0
        num_targets = 0
        for terrain_type in self.interpolator_map.keys():
            interpolator: MocapInterpolator = self.interpolator_map[terrain_type]
            env_ids, terrain_key_pos = self.get_data_by_terrain_type(terrain_type)
            if len(env_ids) == 0:
                continue
            # 目前都是vx方向
            cmd_vel = torch.zeros(len(env_ids), 3, device=self.device)
            # 取该地形配置的速度范围中的最小值（最慢速度对应最长轨迹）
            v_min = min(self.cfg.terrain_and_cmd_vel[self.terrain.terrain_type2name[terrain_type]][0][0],
            self.cfg.terrain_and_cmd_vel[self.terrain.terrain_type2name[terrain_type]][0][1])
            cmd_vel[:, 0] = v_min
            total_frames = interpolator.get_total_frames_per_env(terrain_key_pos, cmd_vel, self.cfg.fps)
            max_length = max(max_length, int(torch.max(total_frames).item()))
            num_targets = interpolator.num_targets
            
            
        self.interpolator_dt = 1.0 / self.cfg.fps
        self.time_step_totals = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 插帧数据，root和关键点
        self.interpolator_root_pose_w = torch.zeros(self.num_envs, max_length, 7, device=self.device)
        self.interpolator_tergets_pose_b = torch.zeros(self.num_envs, max_length, num_targets, 7, device=self.device)
        
        self.ik_cmd = IKCommand(self.cfg.ik_cfg, self._env)
        self.ik_robot = self.ik_cmd.robot
        
        # motion数据
        self.joint_indices, _ = self.robot.find_joints(self.cfg.joint_names)

        # track
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        
        # --- 补充自适应采样相关的变量 ---
        control_dt = env.cfg.decimation * env.cfg.sim.dt
        # 这样可以确保统计的分辨率与你的控制频率相匹配
        self.adaptive_bins = int(max_length // (1 / control_dt)) + 1
        self.bin_failed_count = torch.zeros(self.adaptive_bins, device=self.device)
        self._current_bin_failed = torch.zeros(self.adaptive_bins, device=self.device)
        # 卷积核用于平滑采样概率
        self.kernel = torch.ones(self.cfg.adaptive_kernel_size, device=self.device) / self.cfg.adaptive_kernel_size


        # 初始化命令速度缓存 [num_envs, 3] (vx, vy, wz)
        self._cmd_vel = torch.zeros(self.num_envs, 3, device=self.device)

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        
        
    @property
    def command(self) -> torch.Tensor:
        # 拼接：[目标速度(3), 目标关节位置(N), 目标关节速度(N)]
        return torch.cat([self._cmd_vel, self.joint_pos, self.joint_vel], dim=1)
    
    @property
    def velocity_command(self) -> torch.Tensor:
        """返回当前每个环境对应的目标参考速度 [num_envs, 3]"""
        return self._cmd_vel

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.ik_cmd.joint_pos[:, self.joint_indices]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.ik_cmd.joint_vel[:, self.joint_indices]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.ik_cmd.body_pos_w[:, self.body_indexes]+ self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.ik_cmd.body_quat_w[:, self.body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.ik_cmd.body_lin_vel_w[:, self.body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.ik_cmd.body_ang_vel_w[:, self.body_indexes]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.ik_cmd.body_pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.ik_cmd.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.ik_cmd.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.ik_cmd.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        # 只返回 joint_names 指定的关节
        return self.robot.data.joint_pos[:, self.joint_indices]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        # 只返回 joint_names 指定的关节
        return self.robot.data.joint_vel[:, self.joint_indices]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]
    
    
            
    def _compute_interpolation(self, env_ids: torch.Tensor):
        """并行计算并存储插值后的 root_pose 和 target_poses"""
        if len(env_ids) == 0:
            return
        current_rows = self.terrain.terrain_levels[env_ids]
        current_cols = self.terrain.terrain_types[env_ids]
        env_terrain_types = self.sub_terrain_type_tensor[current_rows, current_cols]
        
        unique_types = torch.unique(env_terrain_types)
        
        for t_type in unique_types:
            t_type_item = t_type.item()
            if t_type_item not in self.interpolator_map:
                continue
                
            # 找到当前类型匹配的子 env_ids
            type_mask = (env_terrain_types == t_type)
            type_env_ids = env_ids[type_mask]
            
            # 获取 Checkpoints
            m_rows = current_rows[type_mask]
            m_cols = current_cols[type_mask]
            checkpoints = self.checkpoint_buffer[m_rows, m_cols]
            real_n = self.key_counts[m_rows[0], m_cols[0]].item()
            compact_checkpoints = checkpoints[:, :real_n, :]
            
            # 生成随机速度 (从配置的范围采样)
            # 明确提取每个轴的 min 和 max
            vel_cfg = self.cfg.terrain_and_cmd_vel[self.terrain.terrain_type2name[t_type_item]]
            vx_min, vx_max = vel_cfg[0][0], vel_cfg[0][1]
            vy_min, vy_max = vel_cfg[1][0], vel_cfg[1][1]
            wz_min, wz_max = vel_cfg[2][0], vel_cfg[2][1]
            # 构造正确的 low 和 high Tensor
            low = torch.tensor([vx_min, vy_min, wz_min], device=self.device)
            high = torch.tensor([vx_max, vy_max, wz_max], device=self.device)
            cmd_vel = sample_uniform(low, high, (len(type_env_ids), 3), device=self.device)
            self._cmd_vel[type_env_ids] = cmd_vel
            
            # 调用插值器
            interpolator:MocapInterpolator = self.interpolator_map[t_type_item]
            root_pose, target_pose_b = interpolator.interpolate(compact_checkpoints, cmd_vel, self.cfg.fps)
            
            # 存入缓冲区
            num_f = root_pose.shape[1]
            self.interpolator_root_pose_w[type_env_ids, :num_f] = root_pose
            self.interpolator_tergets_pose_b[type_env_ids, :num_f] = target_pose_b
            self.time_step_totals[type_env_ids] = num_f
        
        
    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )
        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)


    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            # 使用 self.adaptive_bins (即参考代码的 bin_count) 进行映射
            # 核心逻辑：(当前步 * 总桶数) // 总步数
            current_bin_index = torch.clamp(
                (self.time_steps[env_ids] * self.adaptive_bins) // 
                torch.clamp(self.time_step_totals[env_ids], min=1), 
                0, self.adaptive_bins - 1
            )
            fail_bins = current_bin_index[episode_failed]
            # 记录失败
            self._current_bin_failed.index_add_(0, fail_bins, torch.ones_like(fail_bins, dtype=torch.float))

        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.adaptive_bins)
        
        # 卷积平滑 (让失败点附近的 Bin 也有更高概率被采样)
        if self.cfg.adaptive_kernel_size > 1:
            sampling_probabilities = torch.nn.functional.pad(
                sampling_probabilities.unsqueeze(0).unsqueeze(0),
                (0, self.cfg.adaptive_kernel_size - 1),
                mode="replicate",
            )
            sampling_probabilities = torch.nn.functional.conv1d(
                sampling_probabilities, self.kernel.view(1, 1, -1)
            ).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        # 将采样到的 Bin 转化为具体的 time_steps
        # 在该 Bin 范围内随机偏移，避免每次都从 Bin 的开头开始
        bin_offsets = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        self.time_steps[env_ids] = (
            (sampled_bins.float() + bin_offsets) / self.adaptive_bins * (self.time_step_totals[env_ids].float() - 1)
        ).long()

        # 5. 更新 Metrics (用于 Tensorboard 观察采样熵)
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        self.metrics["sampling_entropy"][:] = H / math.log(self.adaptive_bins)
        
    def _resample_command(self, env_ids: Sequence[int]):
        self._compute_interpolation(env_ids)
        self._adaptive_sampling(env_ids)
        steps = self.time_steps[env_ids]
        root_pose = self.interpolator_root_pose_w[env_ids, steps]
        initial_joint_pos = self.robot.data.default_joint_pos[env_ids]
        self.ik_cmd.reset_ghost_robot(
            env_ids, 
            root_pose[:, :3], 
            root_pose[:, 3:], 
            initial_joint_pos
        )

    def _update_command(self):
        # 1. 推进时间步
        self.time_steps += 1
        
        # 2. 检查是否需要重置环境轨迹
        reset_mask = self.time_steps >= (self.time_step_totals - 1)
        reset_env_ids = torch.where(reset_mask)[0]
        if len(reset_env_ids) > 0:
            self._resample_command(reset_env_ids)

        # 3. 获取当前步目标位姿
        root_pose_ref = self.interpolator_root_pose_w[:, self.time_steps]
        ee_targets_b = self.interpolator_tergets_pose_b[:, self.time_steps]

        # 4. 调用迁移后的 IK+FK 一体化函数
        # 这一步会自动更新 ik_cmd 内部的 joint_pos, body_pos_w, body_vel 等全量数据
        self.ik_cmd.compute_ik_and_fk(
            root_pose_ref[:, :3], 
            root_pose_ref[:, 3:], 
            ee_targets_b, 
            self.interpolator_dt
        )

        # 5. 更新统计量与指标
        self._update_relative_poses()
        self._update_metrics()
        
    def _update_relative_poses(self):
        """将参考 FK 状态投影到机器人当前的水平航向坐标系下"""
        num_bodies = self.body_pos_relative_w.shape[1] # 获取 body 数量 (如 17 或 4)
        # 1. 获取参考轨迹锚点 (Motion Reference Anchor)
        ref_anchor_pos = self.anchor_pos_w # 参考轨迹的当前位置
        ref_anchor_quat = self.anchor_quat_w # 参考轨迹的当前姿态
        # 2. 获取机器人真实锚点 (Robot Real Anchor)
        robot_anchor_pos = self.robot_anchor_pos_w # 机器人当前实时的位置
        robot_anchor_quat = self.robot_anchor_quat_w # 机器人当前实时的姿态
        # 3. 计算目标变换中心：取机器人当前的水平坐标 (XY)，但保留参考轨迹的垂直高度 (Z)
        target_pos_w = robot_anchor_pos.clone()
        target_pos_w[:, 2] = ref_anchor_pos[:, 2] # 确保参考动作的高度与设计一致
        # 4. 计算航向对齐量 (Yaw Alignment)：计算机器人 Yaw 与参考 Yaw 的差值
        # 使用 yaw_quat 提取纯偏航角的四元数
        target_yaw_quat = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(ref_anchor_quat)))
        # 5. 显式扩展对齐量以适配 TorchScript 的严格形状检查
        # 将 [num_envs, 4] 扩展为 [num_envs, num_bodies, 4]
        target_yaw_quat_expanded = target_yaw_quat[:, None, :].expand(-1, num_bodies, -1)
        # 6. 批量应用变换
        # A. 姿态对齐：所有 body 旋转到机器人当前航向
        self.body_quat_relative_w[:] = quat_mul(target_yaw_quat_expanded, self.body_quat_w)
        # B. 位置对齐：目标中心坐标 + 旋转后的相对偏移量
        # 首先计算参考 body 相对于参考锚点的偏移
        rel_pos = self.body_pos_w - ref_anchor_pos[:, None, :]
        # 应用旋转并加回目标中心
        self.body_pos_relative_w[:] = target_pos_w[:, None, :] + quat_apply(target_yaw_quat_expanded, rel_pos)

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
            

    def preprocess_checkpoint_data(self):
        """处理变长数据：对齐到最大长度并生成长度记录表"""
        # 1. 找到全局最大 Key 数量
        max_keys = max([data.shape[0] for data in self.terrain.terrains_checkpoint_data.values()])
        num_rows, num_cols = self.terrain.sub_terrain_type.shape
        # 2. 创建缓冲区和长度记录表
        # checkpoint_buffer: [rows, cols, max_keys, 3]
        self.checkpoint_buffer = torch.zeros((num_rows, num_cols, max_keys, 3), device=self.device)
        # key_counts: 记录每个格子的真实 key 数量 [rows, cols]
        self.key_counts = torch.zeros((num_rows, num_cols), device=self.device, dtype=torch.long)
        # 3. 填充数据并进行 Padding
        for (r, c), data in self.terrain.terrains_checkpoint_data.items():
            n = data.shape[0]
            data_tensor = torch.from_numpy(data).to(self.device).float()
            # 填充真实数据
            self.checkpoint_buffer[r, c, :n] = data_tensor
            # 关键：Padding 剩余部分。使用最后一个点填充，这样插值到后面时会停留在终点
            if n < max_keys:
                self.checkpoint_buffer[r, c, n:] = data_tensor[-1]
            self.key_counts[r, c] = n
        self.sub_terrain_type_tensor = self.terrain.sub_terrain_type
        
        
    def get_data_by_terrain_type(self, target_type):
        """
        给定地形类型，返回在该地形上的 env_ids 和紧凑的(无冗余Padding) checkpoints。
        """
        current_rows = self.terrain.terrain_levels
        current_cols = self.terrain.terrain_types
        # 1. 查找匹配的环境
        env_terrain_types = self.sub_terrain_type_tensor[current_rows, current_cols]
        matching_env_ids = (env_terrain_types == target_type).nonzero(as_tuple=True)[0]
        if matching_env_ids.shape[0] == 0:
            return matching_env_ids, torch.empty(0)
        # 2. 提取数据
        m_rows = current_rows[matching_env_ids]
        m_cols = current_cols[matching_env_ids]
        raw_checkpoints = self.checkpoint_buffer[m_rows, m_cols]
        # 3. 既然“同类型长度相同”，我们只需要看第一个匹配项的真实长度
        real_n = self.key_counts[m_rows[0], m_cols[0]].item()
        # 4. 切掉多余的 Padding，返回紧凑的 [num_envs, real_n, 3]
        compact_checkpoints = raw_checkpoints[:, :real_n, :]
        return matching_env_ids, compact_checkpoints
    

@configclass 
class MotionGeneratorCfg(CommandTermCfg):
    
    class_type: type = MotionGenerator
    
    ik_cfg: IKCommandCfg = MISSING
    
    asset_name: str = MISSING # robot
    
    fps: int = MISSING #重要
    
    terrain_and_checkpoint_file: dict[str, list[str]] = MISSING # [地形名, 检查点npz文件] 
    terrain_and_cmd_vel: dict[str, list[tuple[float, float]]] = MISSING # [地形名, 速度范围] (vx, vy, wz)

    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    joint_names: list[str] = MISSING
    
    motion_max_episode: int = 10

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
