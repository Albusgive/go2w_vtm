import torch
'''
desmos实现,根据第二条曲线计算
世界坐标系:
https://www.desmos.com/calculator/jcsrquy6hv?lang=zh-CN
跳跃前自身坐标系:
https://www.desmos.com/calculator/9ejt7ubwz7?lang=zh-CN
'''


class JumpCurveDynamicCalculator:
    def __init__(
        self,
        max_envs: int,
        device: torch.device = torch.device("cpu"),
        h1: float = -0.2,
        h2: float = 0.1,
        t1: float = 0.4,
        a2: float = -1.0,
        g: float = -9.8,
        default_end_height: float = 0.05,
        default_a_basemax: float = 9.8
    ):
        """
        Initialize jump curve calculator with end height condition
        
        Parameters:
            max_envs: Maximum number of environments
            device: Computation device (default: CPU)
            h1: 相较跳跃前下蹲距离 (m)
            h2: 相较跳跃前离地距离 (m)
            t1: 下蹲用时 (s)
            a2: Acceleration during second segment (m/s²)
            g: Gravity acceleration (m/s²)
            default_end_height: 跳跃结束高度,相较跳跃前 (m)
        """
        self.device = device
        self.max_envs = max_envs
        
        # Store fixed parameters (created on specified device)
        self.h1 = torch.tensor(h1, device=device)
        self.h2 = torch.tensor(h2, device=device)
        self.t1 = torch.tensor(t1, device=device)
        self.a2 = torch.tensor(a2, device=device)
        self.g = torch.tensor(g, device=device)
        self.default_end_height = torch.tensor(default_end_height, device=device)
        
        # Precompute first segment coefficients
        self.a1 = 2 * (-self.h1) / (self.t1 ** 3)
        self.b1 = 3 * self.h1 / (self.t1 ** 2)
        
        # Initialize environment-specific parameters (using NaN for unset values)
        self.a_basemax = torch.full((max_envs,), default_a_basemax, 
                                   device=device)
        self.t2 = torch.full((max_envs,), float('inf'), 
                            device=device)
        self.b2 = torch.full((max_envs,), float('inf'), 
                            device=device)
        self.V2_t2 = torch.full((max_envs,), float('inf'), 
                               device=device)
        self.t_end = torch.full((max_envs,), float('inf'),
                               device=device)
        self.end_heights = torch.full((max_envs,), default_end_height, 
                                    device=device)

    def set_parameters(self, env_mask: torch.Tensor, a_basemax: torch.Tensor, end_heights: torch.Tensor = None) -> None:
        """
        Set base acceleration and end height for specified environments using mask
        
        Parameters:
            env_mask: Boolean mask tensor, shape [max_envs]
            a_basemax: Base acceleration tensor, shape [max_envs] (m/s²)
            end_heights: End height tensor, shape [max_envs] (m). 
                         If None, use self.default_end_height.
        """
        # Validate inputs
        assert env_mask.dim() == 1, "env_mask must be 1D tensor"
        assert env_mask.shape == (self.max_envs,), f"env_mask must have shape [max_envs={self.max_envs}]"
        assert a_basemax.dim() == 1, "a_basemax must be 1D tensor"
        assert a_basemax.shape == (self.max_envs,), f"a_basemax must have shape [max_envs={self.max_envs}]"
        
        # Ensure tensors are on correct device
        env_mask = env_mask.to(self.device)
        a_basemax = a_basemax.to(self.device)
        
        # Update stored parameters only for selected environments
        self.a_basemax[env_mask] = a_basemax[env_mask].to(torch.float32)
        
        # Handle end heights
        if end_heights is None:
            # Use default end height for all specified environments
            self.end_heights[env_mask] = self.default_end_height
        else:
            assert end_heights.dim() == 1, "end_heights must be 1D tensor"
            assert end_heights.shape == (self.max_envs,), f"end_heights must have shape [max_envs={self.max_envs}]"
            self.end_heights[env_mask] = end_heights[env_mask].to(self.device)
        
        # Extract values for the selected environments
        selected_a_basemax = a_basemax[env_mask]
        selected_end_heights = self.end_heights[env_mask]
        
        # Compute and store environment-specific parameters (vectorized)
        delta_h = self.h2 - self.h1
        denominator = selected_a_basemax + self.g
        
        # Avoid division by zero (add small offset when denominator near zero)
        denominator = torch.where(
            torch.abs(denominator) < 1e-6, 
            torch.sign(denominator) * 1e-6,
            denominator
        )
        
        sqrt_term = torch.sqrt(2 * delta_h / denominator)
        correction_term = 1 - (self.a2 * delta_h) / (6 * (denominator ** 2))
        
        t2_val = self.t1 + sqrt_term * correction_term
        delta_t = t2_val - self.t1
        
        b2_val = delta_h / (delta_t ** 2) - (self.a2 * delta_t) / 6
        V2_t2_val = (self.a2 / 2) * (delta_t ** 2) + 2 * b2_val * delta_t
        
        # Calculate end time (when height reaches end_height)
        a = 0.5 * self.g
        b = V2_t2_val
        c = self.h2 - selected_end_heights  # Per-environment end heights
        
        # Calculate discriminant
        discriminant = b**2 - 4*a*c
        
        # Ensure discriminant is non-negative
        discriminant = torch.where(discriminant < 0, torch.zeros_like(discriminant), discriminant)
        
        # Calculate time difference (take positive solution)
        dt = (-b - torch.sqrt(discriminant)) / (2*a)
        
        # Calculate end time
        t_end_val = t2_val + dt
        
        # Store computed values only for selected environments
        self.t2[env_mask] = t2_val
        self.b2[env_mask] = b2_val
        self.V2_t2[env_mask] = V2_t2_val
        self.t_end[env_mask] = t_end_val

    def compute_height(self, env_mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute height at specific time points for specified environments
        
        Parameters:
            env_mask: Boolean mask tensor, shape [max_envs]
            x: Time points tensor, shape [max_envs] (s)
        
        Returns:
            Height tensor, shape [max_envs] (m)
        """
        return self._compute_segment(env_mask, x, "height")

    def compute_velocity(self, env_mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute velocity at specific time points for specified environments
        
        Parameters:
            env_mask: Boolean mask tensor, shape [max_envs]
            x: Time points tensor, shape [max_envs] (s)
        
        Returns:
            Velocity tensor, shape [max_envs] (m/s)
        """
        return self._compute_segment(env_mask, x, "velocity")

    def compute_acceleration(self, env_mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute acceleration at specific time points for specified environments
        
        Parameters:
            env_mask: Boolean mask tensor, shape [max_envs]
            x: Time points tensor, shape [max_envs] (s)
        
        Returns:
            Acceleration tensor, shape [max_envs] (m/s²)
        """
        return self._compute_segment(env_mask, x, "acceleration")

    def _compute_segment(self, env_mask: torch.Tensor, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Core computation method for specified environments
        
        Parameters:
            env_mask: Boolean mask tensor, shape [max_envs]
            x: Time points tensor, shape [max_envs] (s)
            mode: Computation mode ('height', 'velocity', or 'acceleration')
        
        Returns:
            Result tensor, shape [max_envs]
            - For unconfigured environments: NaN
            - For time points beyond t_end: INF
            - Otherwise: calculated value
        """
        # Validate inputs
        assert env_mask.dim() == 1, "env_mask must be 1D tensor"
        assert env_mask.shape == (self.max_envs,), f"env_mask must have shape [max_envs={self.max_envs}]"
        assert x.dim() == 1, "x must be 1D tensor"
        assert x.shape == (self.max_envs,), f"x must have shape [max_envs={self.max_envs}]"
        
        # Ensure tensors are on correct device
        env_mask = env_mask.to(self.device)
        x = x.to(self.device)
        
        # Initialize result tensor with NaNs
        result = torch.full_like(x, float('nan'))
        
        # For environments to compute, set initial value to INF
        result[env_mask] = torch.inf
        
        # Create masks for each segment
        mask1 = (x >= 0) & (x <= self.t1) & env_mask
        mask2 = (x > self.t1) & (x <= self.t2) & (x <= self.t_end) & env_mask
        mask3 = (x > self.t2) & (x <= self.t_end) & env_mask
        
        # Segment 1: 0 ≤ x ≤ t1
        if mode == "height":
            result = torch.where(mask1, self.a1 * x**3 + self.b1 * x**2, result)
        elif mode == "velocity":
            result = torch.where(mask1, 3 * self.a1 * x**2 + 2 * self.b1 * x, result)
        elif mode == "acceleration":
            result = torch.where(mask1, 6 * self.a1 * x + 2 * self.b1, result)
        
        # Segment 2: t1 < x ≤ t2 and x <= t_end
        x_shifted = x - self.t1
        if mode == "height":
            result = torch.where(mask2, (self.a2 / 6) * x_shifted**3 + self.b2 * x_shifted**2 + self.h1, result)
        elif mode == "velocity":
            result = torch.where(mask2, (self.a2 / 2) * x_shifted**2 + 2 * self.b2 * x_shifted, result)
        elif mode == "acceleration":
            result = torch.where(mask2, self.a2 * x_shifted + 2 * self.b2, result)
        
        # Segment 3: x > t2 and x <= t_end
        x_shifted = x - self.t2
        if mode == "height":
            result = torch.where(mask3, 0.5 * self.g * x_shifted**2 + self.V2_t2 * x_shifted + self.h2, result)
        elif mode == "velocity":
            result = torch.where(mask3, self.g * x_shifted + self.V2_t2, result)
        elif mode == "acceleration":
            result = torch.where(mask3, self.g, result)
        
        return result

    def compute_full_curves(self, env_mask: torch.Tensor, num_points: int = 200) -> tuple:
        """
        Generate full curves for height, velocity, and acceleration for specified environments
        
        Parameters:
            env_mask: Boolean mask tensor, shape [max_envs]
            num_points: Number of time points per environment
            
        Returns:
            x: Time points tensor, shape [max_envs, num_points] (s)
            heights: Height tensor, shape [max_envs, num_points] (m)
            velocities: Velocity tensor, shape [max_envs, num_points] (m/s)
            accelerations: Acceleration tensor, shape [max_envs, num_points] (m/s²)
        """
        if not torch.any(env_mask):
            # No environments to compute
            time_range = torch.linspace(0, 1.0, num_points, device=self.device)
            x_grid = time_range.unsqueeze(0).repeat(self.max_envs, 1)
            heights = torch.full((self.max_envs, num_points), float('nan'), device=self.device)
            velocities = torch.full_like(heights, float('nan'))
            accelerations = torch.full_like(heights, float('nan'))
            return x_grid, heights, velocities, accelerations
        
        max_t_end = torch.max(self.t_end[env_mask]).item()
        time_range = torch.linspace(0, max_t_end * 1.2, num_points, device=self.device)
        
        # Create time grid for each environment
        x_grid = time_range.unsqueeze(0).repeat(self.max_envs, 1)
        
        # Preallocate result tensors
        heights = torch.full((self.max_envs, num_points), float('nan'), device=self.device)
        velocities = torch.full_like(heights, float('nan'))
        accelerations = torch.full_like(heights, float('nan'))
        
        # Compute all quantities at each time point
        for i in range(num_points):
            time_points = x_grid[:, i]
            heights[:, i] = self.compute_height(env_mask, time_points)
            velocities[:, i] = self.compute_velocity(env_mask, time_points)
            accelerations[:, i] = self.compute_acceleration(env_mask, time_points)
        
        return x_grid, heights, velocities, accelerations