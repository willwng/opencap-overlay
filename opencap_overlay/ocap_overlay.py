import os
from abc import ABC, abstractmethod

import opensim as osim

from .backend_blender import build_blend, find_blender, render_blend
from .backend_gltf import build_glb
from .opensim_helper import process_motion
from .backend_pyrender import render_pyrender
from .utils import frames_to_video, load_camera


class OpenCapOverlayTool(ABC):
    def __init__(self, model_path, mot_path, camera_path=None):
        self.model_path = model_path
        self.mot_path = mot_path
        self.output_dir = None

        # Process the mesh motions
        model = osim.Model(self.model_path)
        self.times, self.mesh_motions = process_motion(model, self.mot_path)
        self.frames = len(self.times)
        print(f'Processed {self.frames} frames, {len(self.mesh_motions)} meshes')

        # Prepare the camera intrinsics/extrinsics
        self.camera = load_camera(camera_path) if camera_path else None
        return

    def set_output_dir(self, output_dir: str):
        self.output_dir = output_dir

    def _check_output(self):
        if self.output_dir is None:
            raise ValueError("No output directory set")
        os.makedirs(self.output_dir, exist_ok=True)
        return

    def _to_video(self, frames_dir, out_path):
        # Real-time playback: frames per second of the source motion.
        duration = float(self.times[-1] - self.times[0])
        fps = round((self.frames - 1) / duration) if duration > 0 else 30
        print(f'Encoding {frames_dir}/ -> {out_path} at {fps} fps ...')
        frames_to_video(frames_dir, out_path, fps)
        return out_path

    @abstractmethod
    def render(self, geometry_dir: str):
        pass


class GLBOverlayTool(OpenCapOverlayTool):
    def render(self, geometry_dir: str):
        self._check_output()
        glb_path = os.path.join(self.output_dir, "out.glb")
        print(f'Exporting {self.mot_path} -> {glb_path} ...')
        build_glb(self.times, self.mesh_motions, geometry_dir, glb_path)
        return glb_path


class BlenderOverlayTool(OpenCapOverlayTool):
    def _build_blend(self, geometry_dir: str):
        self._check_output()
        blend_path = os.path.join(self.output_dir, "out.blend")

        print(f'Exporting {self.mot_path} -> {blend_path} ...')
        blender = find_blender()
        build_blend(
            mesh_motions=self.mesh_motions,
            geometry_dir=geometry_dir,
            blend_path=blend_path,
            blender_exe=blender,
            camera=self.camera,
        )
        return blend_path

    def render(self, geometry_dir: str):
        # Create blender file
        blend_path = self._build_blend(geometry_dir=geometry_dir)

        # Render frames
        frames_out = os.path.join(self.output_dir, "frames")
        print(f'Rendering {self.frames} frames of {blend_path} -> {frames_out}/ ...')
        render_blend(blend_path, frames_out, find_blender())

        # Convert to video
        out_path = os.path.join(self.output_dir, "render.mp4")
        self._to_video(frames_out, out_path)
        return out_path


class PyrenderOverlayTool(OpenCapOverlayTool):
    def render(self, geometry_dir: str):
        self._check_output()

        # Render frames in-process straight from the calibrated camera.
        frames_out = os.path.join(self.output_dir, "frames")
        print(f'Rendering {self.frames} frames with pyrender -> {frames_out}/ ...')
        render_pyrender(self.mesh_motions, geometry_dir, self.camera,
                        frames_out, self.frames)

        # Convert to video
        out_path = os.path.join(self.output_dir, "render.mp4")
        self._to_video(frames_out, out_path)
        return out_path
