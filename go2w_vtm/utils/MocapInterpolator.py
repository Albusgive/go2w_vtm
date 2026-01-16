import torch
import numpy as np

class MocapInterpolator:
    def __init__(self, npz_path: str, device: str = "cuda"):
        self.device = device
        data = np.load(npz_path, allow_pickle=True)
        self.names = [str(n).lower() for n in data['names']]
        self.offsets = torch.from_numpy(data['mocap_pos_offsets']).to(device).float()
        self.quats = torch.from_numpy(data['mocap_quats']).to(device).float()
        self.num_keys = len(self.names)
        self.num_mocap = self.offsets.shape[1]

    def interpolate(self, terrain_key_pos, cmd_vel, fps=50, g=-9.81):
        num_envs = terrain_key_pos.shape[0]
        device = self.device
        abs_mpos = terrain_key_pos.unsqueeze(2) + self.offsets.unsqueeze(0)
        gravity_vec = torch.tensor([0, 0, g], device=device)
        
        all_trajs = []
        
        # --- 1. 预计算每一段的时长 dt ---
        dts = []
        for i in range(self.num_keys - 1):
            name = self.names[i]
            diff = abs_mpos[:, i+1, 0, :] - abs_mpos[:, i, 0, :]
            L = torch.norm(diff[:, :2], dim=-1)
            dz = diff[:, 2]
            
            v_xy_cmd = torch.norm(cmd_vel[:, :2], dim=-1)
            
            # 基础时长：距离 / 速度
            dt_base = torch.where(v_xy_cmd > 1e-3, L / (v_xy_cmd + 1e-6), torch.abs(dz) / 0.5)
            
            # --- 速度修正逻辑 ---
            # 如果是跳跃相关段，当 cmd_vel 慢时，强制缩短时长以增加加速度
            if "jump" in name or (i > 0 and "jump_pre" in self.names[i-1]):
                # v_ref 以下会触发加速。v_ref 设为 1.0 或更高
                v_ref = 1.0 
                # 计算压缩系数：速度越慢，压缩越厉害（最小压缩到原来的 0.5 倍时间）
                compression = torch.clamp(v_xy_cmd / v_ref, min=0.5, max=1.0)
                dt_base = dt_base * compression

            dts.append(torch.clamp(dt_base, min=2.0/fps))

        # --- 2. 轨迹生成 ---
        for i in range(self.num_keys - 1):
            name = self.names[i]
            p0, p1 = abs_mpos[:, i], abs_mpos[:, i+1]
            q0, q1 = self.quats[i], self.quats[i+1]
            dt_curr = dts[i]
            
            steps = int(fps * torch.max(dt_curr).item())
            if steps < 2: steps = 2
            
            t_norm = torch.linspace(0.0, 1.0, steps, device=device).view(1, -1, 1, 1)
            t_abs = t_norm * dt_curr.view(-1, 1, 1, 1)

            if "jump_pre" in name and (i + 1 < self.num_keys - 1):
                # 【加速段】
                p2 = abs_mpos[:, i+2]
                dt_next = dts[i+1]
                # 先根据下一段（腾空段）的时长，算出需要的起跳初速度
                v_launch = (p2 - p1 - 0.5 * gravity_vec * (dt_next.view(-1, 1, 1)**2)) / dt_next.view(-1, 1, 1)
                
                # 假设进入 pre 段时的速度是 cmd_vel
                v_start = cmd_vel.unsqueeze(1).expand(-1, self.num_mocap, -1)
                
                # 计算 a_pre：为了在 dt_curr 时间内从 p0 移动到 p1 且末速度达到 v_launch
                # 这里我们使用末速度约束公式：p1 = p0 + (v_start + v_launch)/2 * dt
                # 如果这个公式不成立（因为 dt 被我们压缩了），我们优先保证到达 p1 且末速度尽量接近 v_launch
                acc_pre = 2 * (p1 - p0 - v_start * dt_curr.view(-1, 1, 1)) / (dt_curr.view(-1, 1, 1)**2)
                
                pos_t = p0.unsqueeze(1) + v_start.unsqueeze(1) * t_abs + 0.5 * acc_pre.unsqueeze(1) * (t_abs**2)
                
            elif "jump" in name:
                # 【自由落体段】只受重力
                # 重新计算初速度，确保经过被压缩后的 dt_curr 后能准时到达 p1
                v_launch = (p1 - p0 - 0.5 * gravity_vec * (dt_curr.view(-1, 1, 1)**2)) / dt_curr.view(-1, 1, 1)
                pos_t = p0.unsqueeze(1) + v_launch.unsqueeze(1) * t_abs + 0.5 * gravity_vec * (t_abs**2)
                
            else:
                # 【平滑段】
                pos_t = p0.unsqueeze(1) + (p1 - p0).unsqueeze(1) * t_norm

            # 边界修正
            pos_t[:, -1, :, :] = p1.unsqueeze(1)
            quat_t = torch.nn.functional.normalize(q0 + (q1 - q0) * t_norm, p=2, dim=-1)
            quat_t[:, -1, :, :] = q1.unsqueeze(0).unsqueeze(0)
            
            traj_seg = torch.cat([pos_t, quat_t], dim=-1)
            if i < self.num_keys - 2:
                all_trajs.append(traj_seg[:, :-1, :, :])
            else:
                all_trajs.append(traj_seg)

        return torch.cat(all_trajs, dim=1)