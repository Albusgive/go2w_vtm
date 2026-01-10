import torch
from go2w_vtm.track.jump_motion_w import JumpCurveDynamicCalculator


# ---------------------- Demo: Height, Velocity, and Acceleration Curves ----------------------
if __name__ == "__main__":
    # Select device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create calculator (supports up to 4096 environments)
    calculator = JumpCurveDynamicCalculator(
        max_envs=4096,
        device=device,
        h1=0.2, h2=0.5, t1=0.4, a2=-1.0, g=-9.8, default_start_height=0.4
    )
    
    # Configure environment parameters - 使用布尔掩码选择环境
    env_mask_1 = torch.zeros(4096, dtype=torch.bool, device=device)
    env_mask_1[0] = True  # 选择第一个环境
    
    a_basemax = torch.full((4096,), 16.0, device=device)
    start_heights = torch.full((4096,), 0.4, device=device)
    end_heights = torch.full((4096,), 0.45, device=device)
    
    calculator.set_parameters(env_mask_1, a_basemax, start_heights, end_heights)
    
    # Test point calculations - 创建时间点张量，形状为 [max_envs]
    import numpy as np
    for i in np.arange(0, 1.5, 0.05):
        # 创建时间点张量，形状为 [max_envs]，只有第一个环境有实际时间值
        time_points = torch.full((4096,), i, device=device, dtype=torch.float32)
        heights = calculator.compute_height(env_mask_1, time_points)
        velocities = calculator.compute_velocity(env_mask_1, time_points)
        accelerations = calculator.compute_acceleration(env_mask_1, time_points)
        
        # 只显示第一个环境的结果
        print(f"time: {i:.4f}s  z: {heights[0].item():.6f}m  v: {velocities[0].item():.6f}m/s  a: {accelerations[0].item():.6f}m/s²")
    
    print("并行计算速度测试：")
    # 创建全为 True 的掩码选择所有环境
    env_mask_all = torch.ones(4096, dtype=torch.bool, device=device)
    
    # 设置所有环境的参数
    h0 = torch.rand((4096,), device=device) * 0.4
    calculator.set_parameters(env_mask_all, a_basemax, h0, end_heights)
    
    # 创建随机时间点 - 形状为 (4096,)
    time_pos = torch.rand(4096, device=device) * 1.2
    
    import time
    start = time.time()
    heights = calculator.compute_height(env_mask_all, time_pos)
    velocities = calculator.compute_velocity(env_mask_all, time_pos)
    accelerations = calculator.compute_acceleration(env_mask_all, time_pos)
    end = time.time()
    
    print(f'用时 {end-start:.4f} s 计算了 4096 个环境的曲线数据')
    print(f'高度形状: {heights.shape}, 速度形状: {velocities.shape}, 加速度形状: {accelerations.shape}')
    
    # 可选: 验证第一个环境的结果是否与之前一致
    print("\n验证第一个环境的结果一致性:")
    # 创建时间点张量，形状为 [max_envs]，只有第一个环境有实际时间值
    time_point = torch.full((4096,), 0.5, device=device)
    height = calculator.compute_height(env_mask_1, time_point)
    velocity = calculator.compute_velocity(env_mask_1, time_point)
    acceleration = calculator.compute_acceleration(env_mask_1, time_point)
    print(f"time: 0.5s  z: {height[0].item():.6f}m  v: {velocity[0].item():.6f}m/s  a: {acceleration[0].item():.6f}m/s²")
    
    # 生成完整曲线示例
    print("\n生成完整曲线示例:")
    x_grid, heights, velocities, accelerations = calculator.compute_full_curves(env_mask_1, num_points=100)
    print(f"时间网格形状: {x_grid.shape}")
    print(f"高度曲线形状: {heights.shape}")
    print(f"速度曲线形状: {velocities.shape}")
    print(f"加速度曲线形状: {accelerations.shape}")