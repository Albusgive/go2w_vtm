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
        terrain_key_pos: [num_envs, num_keys, 3]
        cmd_vel: [num_envs, 3]
        """
        num_envs = terrain_key_pos.shape[0]
        gravity = torch.tensor([0, 0, -9.81], device=self.device)
        
        # 1. 计算 Root 的全局目标位置 (envs, keys, 3)
        # 根据需求：相对于第一个坐标系，这里直接加上 terrain 偏移
        root_abs_pos = terrain_key_pos + self.root_pos_offsets.unsqueeze(0)
        
        # 2. 时间分配 (基于 Root 移动距离)
        diffs = root_abs_pos[:, 1:] - root_abs_pos[:, :-1]
        dist_xy = torch.norm(diffs[..., :2], dim=-1)
        v_xy_cmd = torch.norm(cmd_vel[..., :2], dim=-1).unsqueeze(1)
        dt = torch.clamp(dist_xy / (v_xy_cmd + 1e-6), min=0.1, max=2.0)
        
        # 针对跳跃节点进行时间压缩 (逻辑同前)
        is_jump = torch.tensor(["jump" in n and "pre" not in n for n in self.key_names[:-1]], device=self.device)
        dt = torch.where(is_jump, dt * 0.8, dt)
        
        # 3. 边界速度规划 (Root)
        v_k = torch.zeros((num_envs, self.num_keys, 3), device=self.device)
        v_k[:, :] = cmd_vel.unsqueeze(1)
        for i in range(self.num_keys - 1):
            if "jump" in self.key_names[i] and "pre" not in self.key_names[i]:
                p0, p1, dt_i = root_abs_pos[:, i], root_abs_pos[:, i+1], dt[:, i:i+1]
                v_launch = (p1 - p0 - 0.5 * gravity * (dt_i**2)) / dt_i
                v_k[:, i] = v_launch
                v_k[:, i+1] = v_launch + gravity * dt_i

        # 4. 插值生成轨迹
        all_root_pos, all_root_quat = [], []
        all_target_pos, all_target_quat = [], []

        for i in range(self.num_keys - 1):
            steps = max(int(fps * torch.max(dt[:, i]).item()), 2)
            # t: [1, steps, 1]
            t = torch.linspace(0, 1, steps, device=self.device).view(1, -1, 1)
            dt_i = dt[:, i].view(-1, 1, 1)
            
            # --- Root Pos ---
            p0, p1, v0, v1 = root_abs_pos[:, i:i+1], root_abs_pos[:, i+1:i+2], v_k[:, i:i+1], v_k[:, i+1:i+2]
            if "jump" in self.key_names[i] and "pre" not in self.key_names[i]:
                pos_t = p0 + v0*(t*dt_i) + 0.5*gravity*(t*dt_i)**2
            else:
                t2, t3 = t**2, t**3
                h00, h10, h01, h11 = 2*t3-3*t2+1, (t3-2*t2+t)*dt_i, -2*t3+3*t2, (t3-t2)*dt_i
                pos_t = h00*p0 + h10*v0 + h01*p1 + h11*v1

            # --- Root Quat ---
            q0, q1 = self.root_quats[i], self.root_quats[i+1]
            quat_t = torch.nn.functional.normalize(q0.view(1, 1, 4) + (q1 - q0).view(1, 1, 4) * t, p=2, dim=-1)
            
            # --- Targets Rel Pose ---
            tp0, tp1 = self.target_rel_pos[i], self.target_rel_pos[i+1]
            tq0, tq1 = self.target_rel_quats[i], self.target_rel_quats[i+1]
            
            # 使用 transpose 确保返回 [1, steps, N_targets, 3/4]
            target_p_t = (tp0.unsqueeze(1) + (tp1 - tp0).unsqueeze(1) * t).transpose(0, 1).unsqueeze(0)
            target_q_t = torch.nn.functional.normalize(
                (tq0.unsqueeze(1) + (tq1 - tq0).unsqueeze(1) * t).transpose(0, 1), p=2, dim=-1
            ).unsqueeze(0)

            # 拼接段落逻辑
            cut = -1 if i < self.num_keys - 2 else None
            all_root_pos.append(pos_t[:, :cut])
            all_root_quat.append(quat_t[:, :cut])
            all_target_pos.append(target_p_t[:, :cut])
            all_target_quat.append(target_q_t[:, :cut])

        # 返回格式: Dict 包含 Root 绝对轨迹和 Targets 相对轨迹
        return {
            "root_pos": torch.cat(all_root_pos, dim=1),       # [envs, total_frames, 3]
            "root_quat": torch.cat(all_root_quat, dim=1),     # [envs, total_frames, 4]
            "target_rel_pos": torch.cat(all_target_pos, dim=1),   # [envs, total_frames, N, 3]
            "target_rel_quat": torch.cat(all_target_quat, dim=1)  # [envs, total_frames, N, 4]
        }