import torch
import numpy as np

class MocapInterpolator:
    def __init__(self, npz_path: str, device: str = "cpu"):
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
        gravity = torch.tensor([0, 0, g], device=device)

        # --- STEP 1: 全局时间分配 ---
        diffs = abs_mpos[:, 1:] - abs_mpos[:, :-1]
        dist_xy = torch.norm(diffs[:, :, 0, :2], dim=-1)
        v_xy_cmd = torch.norm(cmd_vel[:, :2], dim=-1).unsqueeze(1)
        
        dt = torch.clamp(dist_xy / (v_xy_cmd + 1e-6), min=0.15, max=2.0)
        
        is_jump = torch.tensor(["jump" in n and "pre" not in n for n in self.names[:-1]], device=device)
        is_pre = torch.tensor(["jump_pre" in n for n in self.names[:-1]], device=device)
        
        # 针对跳跃和起跳准备，压缩时间以获得爆发力
        dt = torch.where(is_jump | is_pre, dt * 0.75, dt)
        dt = torch.clamp(dt, min=2.0/fps)

        # --- STEP 2: 边界速度求解 (改进点：前瞻起跳速度) ---
        # 只针对躯干 (Index 0) 进行精密速度规划，四肢跟随
        v_k_root = torch.zeros((num_envs, self.num_keys, 3), device=device)
        v_k_root[:, :] = cmd_vel.unsqueeze(1) # 默认所有节点速度为指令速度

        # 逆向遍历或前瞻遍历，确保 jump_pre 承上启下
        for i in range(self.num_keys - 1):
            if "jump" in self.names[i] and "pre" not in self.names[i]:
                p0, p1 = abs_mpos[:, i, 0], abs_mpos[:, i+1, 0]
                dt_i = dt[:, i].view(-1, 1)
                
                # 计算抛物线初速度
                v_launch = (p1 - p0 - 0.5 * gravity * (dt_i**2)) / dt_i
                v_k_root[:, i] = v_launch
                
                # 落地速度
                v_land = v_launch + gravity * dt_i
                v_k_root[:, i+1] = v_land
                
                # 【关键修正】：如果前一帧是 jump_pre，它的末速度必须是 v_launch
                # 这样 jump_pre 段就会从 cmd_vel 一路加速到 v_launch，而不会减速
                if i > 0 and "jump_pre" in self.names[i-1]:
                    v_k_root[:, i] = v_launch

        # --- STEP 3: 分段插值 ---
        all_pos = []
        all_quat = []

        for i in range(self.num_keys - 1):
            # 提取本段起终点位置和预设速度
            p0, p1 = abs_mpos[:, i], abs_mpos[:, i+1]
            q0, q1 = self.quats[i], self.quats[i+1]
            
            # 速度流：四肢速度默认跟随躯干规划
            v0 = v_k_root[:, i].unsqueeze(1).expand(-1, self.num_mocap, -1)
            v1 = v_k_root[:, i+1].unsqueeze(1).expand(-1, self.num_mocap, -1)
            
            dt_i = dt[:, i].view(-1, 1, 1)
            steps = max(int(fps * torch.max(dt[:, i]).item()), 2)
            t = torch.linspace(0, 1, steps, device=device).view(1, -1, 1, 1)

            if "jump" in self.names[i] and "pre" not in self.names[i]:
                # 腾空段：物理抛物线
                pos_t = p0.unsqueeze(1) + v0.unsqueeze(1)*(t*dt_i) + 0.5*gravity*(t*dt_i)**2
            else:
                # 包含 jump_pre 在内的所有地面段：三次埃尔米特样条
                # 因为 v1 已经提前被设为 v_launch，这里会自动执行加速逻辑
                t2, t3 = t**2, t**3
                h00, h10, h01, h11 = 2*t3-3*t2+1, (t3-2*t2+t)*dt_i, -2*t3+3*t2, (t3-t2)*dt_i
                pos_t = h00*p0.unsqueeze(1) + h10*v0.unsqueeze(1) + h01*p1.unsqueeze(1) + h11*v1.unsqueeze(1)

            # 姿态插值
            quat_t = torch.nn.functional.normalize(q0 + (q1 - q0) * t, p=2, dim=-1)
            
            if i < self.num_keys - 2:
                all_pos.append(pos_t[:, :-1])
                all_quat.append(quat_t[:, :-1].expand(num_envs, -1, self.num_mocap, -1))
            else:
                all_pos.append(pos_t)
                all_quat.append(quat_t.expand(num_envs, -1, self.num_mocap, -1))

        return torch.cat([torch.cat(all_pos, dim=1), torch.cat(all_quat, dim=1)], dim=-1)