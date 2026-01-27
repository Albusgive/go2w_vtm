from dataclasses import MISSING
from isaaclab.utils import configclass
from typing import Literal, Optional, Dict, Any
from isaaclab_rl.rsl_rl import RslRlDistillationStudentTeacherCfg

@configclass
class RslRlDistillationStudentTeacherCNNCfg(RslRlDistillationStudentTeacherCfg):
    """Configuration for the distillation student-teacher CNN networks with mixed 1D/2D observation support."""

    class_name: str = "go2w_vtm.rsl_rl.student_teacher_cnn:StudentTeacherCNN"

    student_cnn_cfg: Optional[Dict[str, Any]] = None

    teacher_cnn_cfg: Optional[Dict[str, Any]] = None



    
