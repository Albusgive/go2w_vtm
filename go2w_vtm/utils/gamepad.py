import pygame
import os

class Gamepad:
    def __init__(self, dead_zone=0.1):
        """
        初始化手柄控制器
        :param dead_zone: 摇杆死区阈值（小于该值视为0）
        """
        # 设置 SDL 为无头模式（避免需要图形界面）
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()  # 必须初始化 display 才能用 joystick
        pygame.joystick.init()

        if pygame.joystick.get_count() == 0:
            raise RuntimeError("未检测到任何手柄！请连接后重试。")

        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        print(f"[Gamepad] 已连接: {self.joystick.get_name()}")

        self.dead_zone = dead_zone

        # 按键状态（布尔值）
        self.button_a = False      # A / Cross
        self.button_b = False      # B / Circle
        self.button_x = False      # X / Square
        self.button_y = False      # Y / Triangle
        self.button_lb = False     # Left Bumper
        self.button_rb = False     # Right Bumper
        self.button_back = False   # View / Share
        self.button_start = False  # Menu / Options
        self.button_ls = False     # Left Stick Press
        self.button_rs = False     # Right Stick Press

        # 轴状态（浮点数，范围 [-1.0, 1.0]）
        self.axis_left_x = 0.0
        self.axis_left_y = 0.0
        self.axis_right_x = 0.0
        self.axis_right_y = 0.0
        self.axis_trigger_left = 0.0   # LT (0.0 ～ 1.0)
        self.axis_trigger_right = 0.0  # RT (0.0 ～ 1.0)

        # 内部：记录原始轴数量（用于兼容不同手柄）
        self._num_axes = self.joystick.get_numaxes()
        self._num_buttons = self.joystick.get_numbuttons()

    def update(self):
        """
        刷新手柄状态（必须在主循环中定期调用）
        """
        # 处理事件队列（防止系统卡死）
        pygame.event.pump()  # 比 get() 更轻量，仅更新状态

        # --- 更新按键 ---
        self.button_a = bool(self.joystick.get_button(0)) if self._num_buttons > 0 else False
        self.button_b = bool(self.joystick.get_button(1)) if self._num_buttons > 1 else False
        self.button_x = bool(self.joystick.get_button(2)) if self._num_buttons > 2 else False
        self.button_y = bool(self.joystick.get_button(3)) if self._num_buttons > 3 else False
        self.button_lb = bool(self.joystick.get_button(4)) if self._num_buttons > 4 else False
        self.button_rb = bool(self.joystick.get_button(5)) if self._num_buttons > 5 else False
        self.button_back = bool(self.joystick.get_button(6)) if self._num_buttons > 6 else False
        self.button_start = bool(self.joystick.get_button(7)) if self._num_buttons > 7 else False
        self.button_ls = bool(self.joystick.get_button(8)) if self._num_buttons > 8 else False
        self.button_rs = bool(self.joystick.get_button(9)) if self._num_buttons > 9 else False

        # --- 更新轴 ---
        def _get_axis(i):
            return self.joystick.get_axis(i) if i < self._num_axes else 0.0

        lx = _get_axis(0)
        ly = _get_axis(1)
        rx = _get_axis(2)
        ry = _get_axis(3)
        lt = _get_axis(4)
        rt = _get_axis(5)

        # 应用死区
        self.axis_left_x = 0.0 if abs(lx) < self.dead_zone else lx
        self.axis_left_y = 0.0 if abs(ly) < self.dead_zone else ly
        self.axis_right_x = 0.0 if abs(rx) < self.dead_zone else rx
        self.axis_right_y = 0.0 if abs(ry) < self.dead_zone else ry

        # 扳机轴通常范围是 [-1, 1]，但有些手柄是 [0, 1]，这里统一映射为 [0, 1]
        self.axis_trigger_left = max(0.0, lt)  # 如果是 -1～1，则 idle=-1 → 转为 0
        self.axis_trigger_right = max(0.0, rt)

    def is_connected(self):
        return self.joystick.get_init()

    def __str__(self):
        return (
            f"Buttons: A={self.button_a}, B={self.button_b}, X={self.button_x}, Y={self.button_y}\n"
            f"Triggers: LT={self.axis_trigger_left:.2f}, RT={self.axis_trigger_right:.2f}\n"
            f"Left Stick: ({self.axis_left_x:.2f}, {self.axis_left_y:.2f})\n"
            f"Right Stick: ({self.axis_right_x:.2f}, {self.axis_right_y:.2f})"
        )