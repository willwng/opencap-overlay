import os
from typing import Optional

import numpy as np
import opensim as osim

from opencap_overlay.utils import rm_file_or_folder
from .camera import Camera, CheckerboardPlacement
from .opensim_helper import process_motion, load_trc
from .render_backend import render_pyrender
from .utils import frames_to_video


class OpenCapOverlayTool:
    def __init__(
            self,
            model_path: str,
            mot_path: str,
            camera_path: str,
            checkerboard_placement: CheckerboardPlacement,
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

        # FPS of motion
        self.num_frames = len(self.times)
        self.motion_fps = (self.num_frames - 1) / float(self.times[-1] - self.times[0])
        print(f'Processed {self.num_frames} frames, {len(self.mesh_motions)} meshes')

        # Prepare the camera intrinsics/extrinsics
        self.camera = Camera.from_pickle(
            pickle_path=camera_path,
            checkerboard_placement=checkerboard_placement
        )

        # Optional experimental markers (.trc), resampled to the motion frames
        self.markers = None
        return

    def add_markers(self, trc_path: str):
        """
        Load experimental markers from a .trc.
        Marker times are resampled (nearest) onto the motion
        """
        trc_times, markers, names = load_trc(trc_path)
        if len(trc_times) > 1:
            idx = np.clip(np.searchsorted(trc_times, self.times), 1, len(trc_times) - 1)
            left = idx - 1
            take_left = np.abs(trc_times[left] - self.times) <= np.abs(trc_times[idx] - self.times)
            nearest = np.where(take_left, left, idx)
        else:
            nearest = np.zeros(self.num_frames, dtype=int)
        self.markers = markers[nearest]  # (num_frames, N, 3)
        print(f'Loaded {markers.shape[1]} markers from {trc_path}')
        return

    def apply_camera_offset(self, offset: np.ndarray | list | tuple):
        """
        Apply a translation offset to the camera extrinsics.
        offset: (3,) array in OpenSim world coordinates (m)
        """
        if not isinstance(offset, np.ndarray):
            offset = np.asarray(offset, dtype=float)
        self.camera.translation += offset
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
        """ Convert rendered frames to a video at the specified fps """
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
    ) -> str:
        """
        Render the motion to a video with optional background video and opacity
        Returns the path to the output video file.
        """
        self._check_output()
        frames_out = os.path.join(self.output_dir, "frames")
        out_fps = render_pyrender(
            mesh_motions=self.mesh_motions,
            geometry_dir=geometry_dir,
            camera=self.camera,
            frames_dir=frames_out,
            num_frames=self.num_frames,
            motion_times=self.times,
            background_video=background_video,
            opacity=opacity,
            markers=self.markers
        )

        # Convert to video (at the timeline fps chosen by the renderer)
        out_path = os.path.join(self.output_dir, "render.mp4")
        self._to_video(frames_out, out_path, fps=out_fps)
        return out_path
