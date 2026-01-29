from isaaclab.sensors import RayCasterCameraCfg, RayCasterCamera
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass

class MyRayCasterCamera(RayCasterCamera):

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)

        if debug_vis:
            if not hasattr(self, "frame_visualizer"):
                marker_cfg = getattr(self.cfg, "frame_visualizer_cfg", None)
                frame_scale = getattr(self.cfg, "frame_scale", (1.0, 1.0, 1.0))
                if marker_cfg is not None:
                    marker_cfg = marker_cfg.replace(
                        markers={
                            "frame": marker_cfg.markers["frame"].replace(scale=frame_scale)
                        }
                    )
                    self.frame_visualizer = VisualizationMarkers(marker_cfg)
            
            if hasattr(self, "frame_visualizer"):
                self.frame_visualizer.set_visibility(True)
        else:
            if hasattr(self, "frame_visualizer"):
                self.frame_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)

        if hasattr(self, "frame_visualizer"):
            self.frame_visualizer.visualize(self._data.pos_w, self._data.quat_w_world)

@configclass
class MyRayCasterCameraCfg(RayCasterCameraCfg):
    class_type: type = MyRayCasterCamera

    frame_visualizer_cfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Sensor/frame"
    )
    frame_scale=(0.1, 0.1, 0.1)

