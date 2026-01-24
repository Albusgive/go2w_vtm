import tkinter as tk
from tkinter import ttk, messagebox, font as tkfont
from typing import Optional, Tuple
import sys

class ProportionalKeyframeDialog:
    def __init__(
        self,
        default_time: float = 0.0,
        preset_labels: list[str] | None = None,
        scale_factor: float = 1.0,
        font_scale: float = 1.5,
        window_title: str = "Record Keyframe"
    ):
        self.default_time = default_time
        self.preset_labels = preset_labels or ["stand", "walk", "trot", "jump", "vault", "custom"]
        self.user_scale = scale_factor
        self.font_scale = font_scale
        self.window_title = window_title
        self.result: Optional[Tuple[float, str]] = None

    def _get_system_scaling(self, root: tk.Tk) -> float:
        try:
            dpi = root.winfo_fpixels('1i')
            return dpi / 96.0
        except:
            return 1.0

    def show(self) -> Optional[Tuple[float, str]]:
        try:
            if sys.platform == "win32":
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)
        except: pass

        temp_root = tk.Tk()
        total_scale = self._get_system_scaling(temp_root) * self.user_scale
        temp_root.destroy()

        dialog = tk.Tk()
        dialog.title(self.window_title)
        dialog.configure(bg="#ffffff")
        
        # === 1. 字体与测量 ===
        final_font_size = int(12 * total_scale * self.font_scale)
        font_family = "Microsoft YaHei" if sys.platform == "win32" else "Helvetica"
        font_config = (font_family, final_font_size)
        
        # 强制全局字体一致
        dialog.option_add("*Font", font_config)
        dialog.option_add("*TCombobox*Listbox.font", font_config)
        
        # 测量最长标签的像素宽度
        measure_font = tkfont.Font(family=font_family, size=final_font_size)
        longest_label = max(self.preset_labels, key=len)
        label_pixel_width = measure_font.measure(longest_label)
        
        # 布局参数
        pad_x = int(25 * total_scale)
        pad_y = int(8 * total_scale)
        inner_padding = int(final_font_size * 0.4)

        # === 2. 样式配置 ===
        style = ttk.Style()
        style.theme_use('default')
        style.configure("TCombobox", padding=inner_padding, arrowsize=int(final_font_size * 1.1))

        # === 3. 界面构建 ===
        main_container = tk.Frame(dialog, bg="#ffffff", padx=pad_x, pady=pad_x)
        main_container.pack(fill="both", expand=True)

        # Time
        time_frame = tk.Frame(main_container, bg="#ffffff")
        time_frame.pack(fill="x", pady=(0, pad_y))
        tk.Label(time_frame, text="Frame Time:", bg="#ffffff", width=7, anchor="w").pack(side="left")
        
        time_var = tk.StringVar(value=str(self.default_time))
        time_entry = tk.Entry(time_frame, textvariable=time_var, borderwidth=1, relief="solid")
        time_entry.pack(side="left", fill="x", expand=True, ipady=inner_padding//2)

        # Action
        action_frame = tk.Frame(main_container, bg="#ffffff")
        action_frame.pack(fill="x", pady=pad_y)
        tk.Label(action_frame, text="KeyState:", bg="#ffffff", width=7, anchor="w").pack(side="left")
        
        label_combo = ttk.Combobox(action_frame, values=self.preset_labels, state="readonly")
        label_combo.set(self.preset_labels[0])
        # 根据最长标签设置下拉框的宽度（字符单位，由于字号大，需要预留点余量）
        label_combo.configure(width=len(longest_label) + 2)
        label_combo.pack(side="left", fill="x", expand=True)

        # Buttons
        btn_frame = tk.Frame(main_container, bg="#ffffff")
        btn_frame.pack(fill="x", pady=(int(pad_y * 2.5), 0))

        def on_ok():
            try:
                self.result = (float(time_var.get()), label_combo.get())
                dialog.destroy()
            except: messagebox.showerror("Error", "Invalid time")

        btn_w = int(8 * total_scale)
        tk.Button(btn_frame, text="OK", command=on_ok, bg="#0078d7", fg="white", width=btn_w, relief="flat").pack(side="right")
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy, bg="#eeeeee", width=btn_w, relief="flat").pack(side="right", padx=(0, 10))

        # === 4. 动态调整窗口尺寸 ===
        dialog.update_idletasks()
        # 计算理想宽度：标签宽度 + 文本宽度 + 边距 + 下拉箭头空间
        # 我们让它在计算出的需求宽度基础上，至少保证不小于 350*scale
        base_width = int(350 * total_scale)
        calc_width = dialog.winfo_reqwidth() + int(40 * total_scale)
        final_w = max(base_width, calc_width)
        final_h = dialog.winfo_reqheight()

        x = (dialog.winfo_screenwidth() // 2) - (final_w // 2)
        y = (dialog.winfo_screenheight() // 2) - (final_h // 2)
        dialog.geometry(f"{final_w}x{final_h}+{x}+{y}")
        dialog.resizable(False, False)

        # 拖拽支持
        self._drag_data = {"x": 0, "y": 0}
        def start_move(e): self._drag_data.update(x=e.x, y=e.y)
        def do_move(e): dialog.geometry(f"+{dialog.winfo_x()+(e.x-self._drag_data['x'])}+{dialog.winfo_y()+(e.y-self._drag_data['y'])}")
        dialog.bind("<Button-1>", start_move)
        dialog.bind("<B1-Motion>", do_move)

        dialog.mainloop()
        return self.result

