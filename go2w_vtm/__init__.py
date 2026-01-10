from .locomotion import *
import os
MONTION_DIR = os.path.join(os.path.dirname(__file__), "motion")
GO2W_USD_DIR = os.path.join(os.path.dirname(__file__), "Robot","go2w.usd")
GO2W_MJCF_DIR = os.path.join(os.path.dirname(__file__), "Robot","go2w_description","mjcf")
GO2W_PRODUCE_MOTION_K_DIR = os.path.join(os.path.dirname(__file__), "produce_motion","k")