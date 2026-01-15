import torch
import numpy as np

class MocapInterpolator:
    def __init__(self, npz_path: str, device: str = "cuda"):
        self.device = device
        data = np.load(npz_path, allow_pickle=True)
        self.names = [str(n) for n in data['names']]
        # (Keys, Num_Mocap, 3)
        self.offsets = torch.from_numpy(data['mocap_pos_offsets']).to(device).float()
        # (Keys, Num_Mocap, 4)
        self.quats = torch.from_numpy(data['mocap_quats']).to(device).float()
        self.num_keys = len(self.names)
        self.num_mocap = self.offsets.shape[1]

    def interpolate(self, terrain_key_pos, cmd_vel, fps=50, g=-9.81):
        """
        terrain_key_pos: (Env, Keys, 3)
        cmd_vel: (Env, 3) -> 并行参考速度 (vx, vy, wz)
        """
        num_envs = terrain_key_pos.shape[0]
        device = self.device
        
        # 绝对位置计算: (Env, Keys, Num_Mocap, 3)
        abs_mpos = terrain_key_pos.unsqueeze(2) + self.offsets.unsqueeze(0)
        
        all_trajs = []
        gravity = torch.tensor([0, 0, g], device=device)

        for i in range(self.num_keys - 1):
            name = self.names[i].lower()
            
            p0 = abs_mpos[:, i]      # (Env, Num_Mocap, 3)
            p1 = abs_mpos[:, i+1]    # (Env, Num_Mocap, 3)
            q0 = self.quats[i]       # (Num_Mocap, 4)
            q1 = self.quats[i+1]     # (Num_Mocap, 4)

            # --- 1. 并行计算每个环境的时长 dt (Env,) ---
            diff_pos = p1[:, 0, :] - p0[:, 0, :]  # 取第一个mocap作为基准 (Env, 3)
            dist_xy = torch.norm(diff_pos[:, :2], dim=-1)
            dist_z = torch.abs(diff_pos[:, 2])
            
            v_xy = torch.norm(cmd_vel[:, :2], dim=-1)
            
            # 计算 dt：如果 v_xy 有效则用 xy，否则用 z 轴分量(0.5m/s兜底)
            dt_env = torch.where(v_xy > 1e-3, dist_xy / (v_xy + 1e-6), dist_z / 0.5)
            dt_env = torch.clamp(dt_env, min=0.1) # 最小 0.1s

            # 为了并行，该段取 batch 内最长的时长作为总步数
            max_dt = torch.max(dt_env).item()
            steps = int(fps * max_dt)
            if steps < 1: steps = 1

            # --- 2. 计算每个环境的归一化时间进度 (Env, steps, 1, 1) ---
            # t_norm = 当前时间 / 自己的 dt。如果超过 1 则截断为 1 (代表动作完成)
            step_indices = torch.arange(steps, device=device).view(1, -1, 1, 1)
            t_raw = (step_indices / fps) / dt_env.view(-1, 1, 1, 1)
            t_mask = (t_raw <= 1.0).float()
            t_norm = torch.clamp(t_raw, max=1.0)
            
            # 实际物理时间 (Env, steps, 1, 1)
            t_abs = t_norm * dt_env.view(-1, 1, 1, 1)

            # --- 3. 位置插值逻辑 ---
            if "jump" in name:
                # v0 = (p1 - p0 - 0.5*g*dt^2) / dt
                # 注意这里 dt 是每个环境自己的 dt_env
                v0 = (p1 - p0 - 0.5 * gravity * (dt_env.view(-1, 1, 1)**2)) / dt_env.view(-1, 1, 1)
                pos_t = p0.unsqueeze(1) + v0.unsqueeze(1) * t_abs + 0.5 * gravity * (t_abs**2)
            elif "force" in name:
                try:
                    acc_vals = [float(x) for x in name.split('_')[1:4]]
                    acc = torch.tensor(acc_vals, device=device)
                except: acc = torch.zeros(3, device=device)
                v0 = (p1 - p0 - 0.5 * acc * (dt_env.view(-1, 1, 1)**2)) / dt_env.view(-1, 1, 1)
                pos_t = p0.unsqueeze(1) + v0.unsqueeze(1) * t_abs + 0.5 * acc * (t_abs**2)
            else:
                # Smooth 线性
                pos_t = p0.unsqueeze(1) + (p1 - p0).unsqueeze(1) * t_norm

            # --- 4. 姿态插值 ---
            quat_t = q0.unsqueeze(0).unsqueeze(0) + (q1 - q0).unsqueeze(0).unsqueeze(0) * t_norm
            quat_t = torch.nn.functional.normalize(quat_t, p=2, dim=-1)
            # 此时 quat_t 已经是 (Env, steps, Num_Mocap, 4) 因为 t_norm 带有 Env 维度

            traj_seg = torch.cat([pos_t, quat_t], dim=-1)
            all_trajs.append(traj_seg)

        return torch.cat(all_trajs, dim=1)