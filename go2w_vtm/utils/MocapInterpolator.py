import torch
import numpy as np

class MocapInterpolator:
    def __init__(self, npz_path: str, device: str = "cuda"):
        self.device = device
        data = np.load(npz_path, allow_pickle=True)
        
        # 基础数据
        self.root_pos_offsets = torch.from_numpy(data['root_pos_offsets']).to(device).float()
        self.root_quats = torch.from_numpy(data['root_quats']).to(device).float()
        self.target_rel_pos = torch.from_numpy(data['target_rel_pos']).to(device).float()
        self.target_rel_quats = torch.from_numpy(data['target_rel_quats']).to(device).float()
        self.key_names = [str(n).lower() for n in data['names']]
        self.frame_time = torch.from_numpy(data['frame_time']).to(device).float()

        self.num_keys = len(self.key_names)
        self.num_targets = self.target_rel_pos.shape[1]
        self.is_jump_mask = torch.tensor(["jump" in n and "pre" not in n for n in self.key_names[:-1]], device=device)

    def _quat_rotate(self, q, v):
        q_xyz = q[..., 1:4]
        q_w = q[..., :1]
        t = 2.0 * torch.cross(q_xyz, v, dim=-1)
        return v + q_w * t + torch.cross(q_xyz, t, dim=-1)
    
    def _safe_nlerp(self, q0, q1, t):
        dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
        q1 = torch.where(dot < 0, -q1, q1)
        return torch.nn.functional.normalize(q0 + (q1 - q0) * t, p=2, dim=-1)

    def _calculate_dt_and_steps(self, terrain_key_pos, cmd_vel, fps):
        """内部工具函数：计算每一段的 dt 和步数"""
        num_envs = terrain_key_pos.shape[0]
        root_abs_pos = terrain_key_pos + self.root_pos_offsets.unsqueeze(0)
        
        # 1. 计算原始基于速度的 dt (K-1 段)
        diffs = root_abs_pos[:, 1:] - root_abs_pos[:, :-1]
        dist_xy = torch.norm(diffs[..., :2], dim=-1)
        v_xy_cmd = torch.norm(cmd_vel[..., :2], dim=-1).unsqueeze(1)
        dt_cmd = torch.clamp(dist_xy / (v_xy_cmd + 1e-6), min=0.1, max=2.0)
        dt_cmd = torch.where(self.is_jump_mask.unsqueeze(0), dt_cmd * 0.8, dt_cmd)
        
        # 2. 融合 frame_time
        # 前 K-1 段：如果 frame_time > 0 则覆盖 dt_cmd
        dt_final = torch.where(self.frame_time[:-1].unsqueeze(0) > 0, 
                               self.frame_time[:-1].unsqueeze(0), 
                               dt_cmd)
        
        # 3. 处理末尾 Hold 段 (第 K 段)
        hold_time = self.frame_time[-1]
        if hold_time > 0:
            hold_dt = hold_time.expand(num_envs, 1)
            dt_final = torch.cat([dt_final, hold_dt], dim=1) # [num_envs, K]
            
        steps_per_seg = torch.clamp((fps * dt_final).long(), min=2)
        return dt_final, steps_per_seg, root_abs_pos

    def interpolate(self, terrain_key_pos, cmd_vel, max_buffer_len, fps=50):
        num_envs = terrain_key_pos.shape[0]
        gravity = torch.tensor([0, 0, -9.81], device=self.device)
        
        dt, steps_per_seg, root_abs_pos = self._calculate_dt_and_steps(terrain_key_pos, cmd_vel, fps)
        actual_num_segs = dt.shape[1] # 可能是 K-1 或 K
        
        num_main_segs = self.num_keys - 1
        last_key_frame_idx = torch.sum(steps_per_seg[:, :num_main_segs] - 1, dim=1) 
        
        # 计算总帧数 (所有段步数和 - 重叠点数)
        total_frames = torch.sum(steps_per_seg, dim=1) - (actual_num_segs - 1)
        
        # 速度规划
        v_k = cmd_vel.unsqueeze(1).repeat(1, self.num_keys, 1)
        jump_indices = torch.where(self.is_jump_mask)[0]
        for i in jump_indices:
            p0, p1, dt_i = root_abs_pos[:, i], root_abs_pos[:, i+1], dt[:, i:i+1]
            v_launch = (p1 - p0 - 0.5 * gravity * (dt_i**2)) / dt_i
            v_k[:, i] = v_launch
            v_k[:, i+1] = v_launch + gravity * dt_i

        res_root = torch.zeros((num_envs, max_buffer_len, 7), device=self.device)
        res_targets = torch.zeros((num_envs, max_buffer_len, self.num_targets, 7), device=self.device)
        current_offsets = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        env_ids = torch.arange(num_envs, device=self.device).unsqueeze(1)

        # 逐段插值
        for i in range(actual_num_segs):
            max_steps = steps_per_seg[:, i].max().item()
            local_step_idx = torch.arange(max_steps, device=self.device).unsqueeze(0)
            t = (local_step_idx.float() / (steps_per_seg[:, i:i+1] - 1)).unsqueeze(-1)
            t = torch.clamp(t, 0.0, 1.0)
            
            dt_i = dt[:, i:i+1].unsqueeze(-1)
            
            # 判读是否为末尾 Hold 段
            if i < self.num_keys - 1:
                # 正常段插值
                p0, p1 = root_abs_pos[:, i:i+1], root_abs_pos[:, i+1:i+2]
                v0, v1 = v_k[:, i:i+1], v_k[:, i+1:i+2]
                q0, q1 = self.root_quats[i], self.root_quats[i+1]
                tp0, tp1 = self.target_rel_pos[i], self.target_rel_pos[i+1]
                tq0, tq1 = self.target_rel_quats[i], self.target_rel_quats[i+1]
                
                if self.is_jump_mask[i]:
                    pos_t = p0 + v0 * (t * dt_i) + 0.5 * gravity * (t * dt_i)**2
                else:
                    t2, t3 = t**2, t**3
                    h = [2*t3-3*t2+1, (t3-2*t2+t)*dt_i, -2*t3+3*t2, (t3-t2)*dt_i]
                    pos_t = h[0]*p0 + h[1]*v0 + h[2]*p1 + h[3]*v1
                
                quat_t = self._safe_nlerp(q0.view(1,1,4), q1.view(1,1,4), t)
                t_pos_t = tp0 + (tp1 - tp0).view(1, 1, self.num_targets, 3) * t.unsqueeze(-1)
                t_quat_t = self._safe_nlerp(tq0.view(1,1,self.num_targets,4), tq1.view(1,1,self.num_targets,4), t.unsqueeze(-1))
            else:
                # 末尾 Hold 段：保持最后一帧位姿不动
                pos_t = root_abs_pos[:, -1:].expand(-1, max_steps, -1)
                quat_t = self.root_quats[-1:].view(1, 1, 4).expand(num_envs, max_steps, -1)
                t_pos_t = self.target_rel_pos[-1:].view(1, 1, self.num_targets, 3).expand(num_envs, max_steps, -1, -1)
                t_quat_t = self.target_rel_quats[-1:].view(1, 1, self.num_targets, 4).expand(num_envs, max_steps, -1, -1)

            # 填充
            is_last_item = (i == actual_num_segs - 1)
            copy_counts = steps_per_seg[:, i] if is_last_item else (steps_per_seg[:, i] - 1)
            target_slot_idx = current_offsets.unsqueeze(1) + local_step_idx
            mask = (local_step_idx < copy_counts.unsqueeze(1)) & (target_slot_idx < max_buffer_len)
            
            b_idx = env_ids.expand(-1, max_steps)[mask]
            t_idx = target_slot_idx[mask]
            
            res_root[b_idx, t_idx, 0:3] = pos_t[mask]
            res_root[b_idx, t_idx, 3:7] = quat_t[mask]
            res_targets[b_idx, t_idx, :, 0:3] = t_pos_t[mask]
            res_targets[b_idx, t_idx, :, 3:7] = t_quat_t[mask]
            
            current_offsets += copy_counts

        return res_root, res_targets, total_frames, last_key_frame_idx

    def get_total_frames_per_env(self, terrain_key_pos, cmd_vel, fps=50):
        # 复用逻辑确保总帧数一致
        _, steps_per_seg, _ = self._calculate_dt_and_steps(terrain_key_pos, cmd_vel, fps)
        actual_num_segs = steps_per_seg.shape[1]
        return torch.sum(steps_per_seg, dim=1) - (actual_num_segs - 1)
    
    def local_to_world(self, root_pos, root_quat, rel_pos, rel_quat):
        """
        将相对位姿转换为世界位姿 (全向量化)
        root_pos: [B, T, 3], root_quat: [B, T, 4]
        rel_pos: [B, T, N, 3], rel_quat: [B, T, N, 4]
        """
        # 旋转位置: R_root * p_rel + p_root
        # 需要对齐维度进行广播
        B, T, N, _ = rel_pos.shape
        r_quat_expand = root_quat.unsqueeze(2).expand(-1, -1, N, -1)
        
        world_pos = self._quat_rotate(r_quat_expand, rel_pos) + root_pos.unsqueeze(2)
        
        # 旋转四元数: q_root * q_rel
        # 这里使用标准的四元数乘法逻辑 (简化版)
        q1, q2 = r_quat_expand, rel_quat
        w1, x1, y1, z1 = q1[...,0], q1[...,1], q1[...,2], q1[...,3]
        w2, x2, y2, z2 = q2[...,0], q2[...,1], q2[...,2], q2[...,3]
        
        world_quat = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ], dim=-1)
        
        return world_pos, world_quat