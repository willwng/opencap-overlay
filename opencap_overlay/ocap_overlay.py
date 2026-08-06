import os
from typing import Optional

import opensim as osim
from opencap_overlay.utils import rm_file_or_folder

from .opensim_helper import process_motion
from .backend_pyrender import render_pyrender
from .camera import Camera
from .utils import frames_to_video


class OpenCapOverlayTool:
    def __init__(
            self,
            model_path: str,
            mot_path: str,
            camera_path: Optional[str] = None,
            custom_geometry_map: Optional[dict] = None
    ):
        self.model_path = model_path
        self.mot_path = mot_path
        self.output_dir = None
        self.custom_geometry_map = {} if custom_geometry_map is None else custom_geometry_map

        # Process the mesh motions
        model = osim.Model(self.model_path)
        self.times, self.mesh_motions = process_motion(
            model, self.mot_path, self.custom_geometry_map
        )
        self.num_frames = len(self.times)
        duration = float(self.times[-1] - self.times[0])
        self.motion_fps = (self.num_frames - 1) / duration if duration > 0 else 30.0
        print(f'Processed {self.num_frames} frames, {len(self.mesh_motions)} meshes')

        # Prepare the camera intrinsics/extrinsics
        self.camera = Camera.from_pickle(camera_path) if camera_path else None
        return

    def set_output_dir(self, output_dir: str, clear_output: bool = False):
        self.output_dir = output_dir
        if clear_output and os.path.exists(self.output_dir):
            print(f'Clearing output directory {self.output_dir}/ ...')
            for f in os.listdir(self.output_dir):
                rm_file_or_folder(os.path.join(self.output_dir, f))
        return

    def _check_output(self):
        if self.output_dir is None:
            raise ValueError("No output directory set")
        os.makedirs(self.output_dir, exist_ok=True)
        return

    def _to_video(self, frames_dir, out_path, fps=None):
        # Default to real-time playback of the motion; callers that resampled onto
        # another timeline (e.g. a video's fps) pass the fps to encode at.
        if fps is None:
            fps = self.motion_fps
        print(f'Encoding {frames_dir}/ -> {out_path} at {fps:g} fps ...')
        frames_to_video(frames_dir, out_path, fps)
        return out_path

    def render(
            self,
            geometry_dir: str,
            background_video: Optional[str] = None,
            opacity: float = 1.0
    ):
        self._check_output()

        # Render frames in-process straight from the calibrated camera, optionally
        # compositing on top of a reference video from the same camera.
        frames_out = os.path.join(self.output_dir, "frames")
        msg = f' over {background_video}' if background_video else ''
        print(f'Rendering {self.num_frames} frames with pyrender{msg} -> {frames_out}/ ...')
        frames_out, out_fps = render_pyrender(
            mesh_motions=self.mesh_motions,
            geometry_dir=geometry_dir,
            camera=self.camera,
            frames_dir=frames_out,
            num_frames=self.num_frames,
            motion_times=self.times,
            background_video=background_video,
            opacity=opacity
        )

        # Convert to video (at the timeline fps chosen by the renderer)
        out_path = os.path.join(self.output_dir, "render.mp4")
        self._to_video(frames_out, out_path, fps=out_fps)
        return out_path
