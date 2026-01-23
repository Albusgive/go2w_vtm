import torch
import numpy as np

class MocapInterpolator:
    def __init__(self, npz_path: str, device: str = "cuda"):
        self.device = device
        # 允许 pickle 加载以获取名称字符串
        data = np.load(npz_path, allow_pickle=True)
        
        # 载入数据并转为张量
        self.root_pos_offsets = torch.from_numpy(data['root_pos_offsets']).to(device).float() # [K, 3]
        self.root_quats = torch.from_numpy(data['root_quats']).to(device).float()             # [K, 4]
        self.target_rel_pos = torch.from_numpy(data['target_rel_pos']).to(device).float()     # [K, N_targets, 3]
        self.target_rel_quats = torch.from_numpy(data['target_rel_quats']).to(device).float() # [K, N_targets, 4]
        self.key_names = [str(n).lower() for n in data['names']]
        
        self.num_keys = len(self.key_names)
        self.num_targets = self.target_rel_pos.shape[1]

    def _quat_rotate(self, q, v):
        """并行向量旋转: R(q) * v"""
        q_xyz = q[..., 1:]
        q_w = q[..., :1]
        t = 2.0 * torch.cross(q_xyz, v, dim=-1)
        return v + q_w * t + torch.cross(q_xyz, t, dim=-1)

    def interpolate(self, terrain_key_pos, cmd_vel, fps=50):
        """
        返回: 
            root_pose_w: [envs, total_frames, 7] (World frame)
            targets_pose_b: [envs, total_frames, N, 7] (Relative to root)
        """
        num_envs = terrain_key_pos.shape[0]
        gravity = torch.tensor([0, 0, -9.81], device=self.device)
        
        # --- 1 & 2 & 3: 位置计算与速度规划 (保持原逻辑) ---
        root_abs_pos = terrain_key_pos + self.root_pos_offsets.unsqueeze(0)
        diffs = root_abs_pos[:, 1:] - root_abs_pos[:, :-1]
        dist_xy = torch.norm(diffs[..., :2], dim=-1)
        v_xy_cmd = torch.norm(cmd_vel[..., :2], dim=-1).unsqueeze(1)
        dt = torch.clamp(dist_xy / (v_xy_cmd + 1e-6), min=0.1, max=2.0)
        
        is_jump = torch.tensor(["jump" in n and "pre" not in n for n in self.key_names[:-1]], device=self.device)
        dt = torch.where(is_jump, dt * 0.8, dt)
        
        v_k = torch.zeros((num_envs, self.num_keys, 3), device=self.device)
        v_k[:, :] = cmd_vel.unsqueeze(1)
        for i in range(self.num_keys - 1):
            if "jump" in self.key_names[i] and "pre" not in self.key_names[i]:
                p0, p1, dt_i = root_abs_pos[:, i], root_abs_pos[:, i+1], dt[:, i:i+1]
                v_launch = (p1 - p0 - 0.5 * gravity * (dt_i**2)) / dt_i
                v_k[:, i] = v_launch
                v_k[:, i+1] = v_launch + gravity * dt_i

        # --- 4: 插值生成 ---
        all_root_pose = []
        all_targets_pose = []

        for i in range(self.num_keys - 1):
            # 这里的 steps 取所有环境该段最大的 dt，以保证拼接对齐
            steps = max(int(fps * torch.max(dt[:, i]).item()), 2)
            t = torch.linspace(0, 1, steps, device=self.device).view(1, -1, 1) # [1, steps, 1]
            dt_i = dt[:, i].view(-1, 1, 1) # [envs, 1, 1]
            
            # --- Root Pos ---
            p0, p1, v0, v1 = root_abs_pos[:, i:i+1], root_abs_pos[:, i+1:i+2], v_k[:, i:i+1], v_k[:, i+1:i+2]
            if "jump" in self.key_names[i] and "pre" not in self.key_names[i]:
                pos_t = p0 + v0*(t*dt_i) + 0.5*gravity*(t*dt_i)**2
            else:
                t2, t3 = t**2, t**3
                h00, h10, h01, h11 = 2*t3-3*t2+1, (t3-2*t2+t)*dt_i, -2*t3+3*t2, (t3-t2)*dt_i
                pos_t = h00*p0 + h10*v0 + h01*p1 + h11*v1
            # 此时 pos_t 形状为 [envs, steps, 3]

            # --- Root Quat (核心修正点 1) ---
            q0, q1 = self.root_quats[i], self.root_quats[i+1]
            # 计算单环境插值：[1, steps, 4]
            quat_t_single = torch.nn.functional.normalize(q0.view(1, 1, 4) + (q1 - q0).view(1, 1, 4) * t, p=2, dim=-1)
            # 扩展到所有环境：[envs, steps, 4]
            quat_t = quat_t_single.expand(num_envs, -1, -1)
            
            # --- Targets Rel Pose (核心修正点 2) ---
            tp0, tp1 = self.target_rel_pos[i], self.target_rel_pos[i+1]
            tq0, tq1 = self.target_rel_quats[i], self.target_rel_quats[i+1]
            
            # 计算单环境目标插值: [steps, N, 3/4]
            # 这里使用了广播技巧，将 [N, 3] 和 [1, steps, 1] 结合
            t_pos_t_single = tp0.unsqueeze(0).unsqueeze(0) + (tp1 - tp0).unsqueeze(0).unsqueeze(0) * t.unsqueeze(-1)
            t_quat_t_single = torch.nn.functional.normalize(
                tq0.unsqueeze(0).unsqueeze(0) + (tq1 - tq0).unsqueeze(0).unsqueeze(0) * t.unsqueeze(-1), 
                p=2, dim=-1
            )
            t_pos_t = t_pos_t_single.expand(num_envs, -1, -1, -1)
            t_quat_t = t_quat_t_single.expand(num_envs, -1, -1, -1)

            # 拼接 7D Pose
            root_pose_t = torch.cat([pos_t, quat_t], dim=-1) # [envs, steps, 7] -> 此时维度匹配
            targets_pose_t = torch.cat([t_pos_t, t_quat_t], dim=-1) # [envs, steps, N, 7]

            cut = -1 if i < self.num_keys - 2 else None
            all_root_pose.append(root_pose_t[:, :cut])
            all_targets_pose.append(targets_pose_t[:, :cut])

        return torch.cat(all_root_pose, dim=1), torch.cat(all_targets_pose, dim=1)
        
    
    def get_total_frames_per_env(self, terrain_key_pos, cmd_vel, fps=50):
        """
        计算每个环境各自的总帧数。
        输入: terrain_key_pos [num_envs, num_keys, 3], cmd_vel [num_envs, 3]
        输出: total_frames_per_env [num_envs] (LongTensor)
        """
        num_envs = terrain_key_pos.shape[0]
        
        # 1. 计算 Root 全局目标位置
        root_abs_pos = terrain_key_pos + self.root_pos_offsets.unsqueeze(0)
        
        # 2. 计算每段的时间 dt: [num_envs, num_keys - 1]
        diffs = root_abs_pos[:, 1:] - root_abs_pos[:, :-1]
        dist_xy = torch.norm(diffs[..., :2], dim=-1)
        v_xy_cmd = torch.norm(cmd_vel[..., :2], dim=-1).unsqueeze(1)
        dt = torch.clamp(dist_xy / (v_xy_cmd + 1e-6), min=0.1, max=2.0)
        
        # 针对跳跃节点进行时间压缩
        is_jump = torch.tensor(["jump" in n and "pre" not in n for n in self.key_names[:-1]], device=self.device)
        dt = torch.where(is_jump, dt * 0.8, dt)
        
        # 3. 计算每一段的帧数: [num_envs, num_keys - 1]
        # 注意：这里去掉了 .item()，保持 Tensor 计算
        steps_per_segment = torch.clamp((fps * dt).long(), min=2)
        
        # 4. 根据插值逻辑计算总帧数
        # 除最后一段外贡献 steps - 1，最后一段贡献 steps
        # 等价于: sum(steps_per_segment) - (num_segments - 1)
        total_frames = torch.sum(steps_per_segment, dim=1) - (self.num_keys - 2)
                
        return total_frames