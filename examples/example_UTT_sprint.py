""" Example usage of OpenCapOverlay for the UTT prosthetic sprint trial.

  - model : LaiUhlrich2022 with the blade welded in + the blade_keel marker
  - motion: the IK .mot from IK_with_blade from the 100m-close-100-percent-2 trial from July 1
  - videos: the OpenCap synced clips (413 frames, same timeline as the .mot)
  - Geometry: located at sprint/Geometry, has LaiUhlrich bones + the 2
    custom meshes (blade, .stl)
"""
from opencap_overlay import OpenCapOverlayTool, CheckerboardPlacement, stack_videos_horizontal


def main():
    model_path = "sprint/LaiUhlrich2022_withblade_bladekeel.osim"
    mot_path = "sprint/sprint.mot"
    markers_path = "sprint/sprint.trc"  # optional (study markers + blade_keel)

    geometry_dir = "sprint/Geometry"
    custom_geometry_map = {}  # not needed: model bones + custom STLs all located in sprint/Geometry/

    # DISPLAY-ONLY vertical correction (does NOT touch the model / .mot / scaling).
    # The model reprojects ~7 px (~8 cm) high vs the HRNet detections consistently
    # across the well-matched joints (elbows/wrists/knees/ankle), i.e. a small
    # systematic transform residual. This per-camera offset (= an 8.4 cm world-down
    # shift, expressed in each camera's frame) zeroes that; it only moves the camera.
    camera_offsets = {
        "cam0": [-0.0022, 0.0839, -0.0088],
        "cam1": [0.0006, 0.0843, -0.0045],
    }

    cameras = ["cam0", "cam1"]
    videos_out = []
    for camera in cameras:
        camera_path = f"sprint/{camera}/cameraIntrinsicsExtrinsics.pickle"
        video_path = f"sprint/{camera}/sprint.mp4"

        overlay_tool = OpenCapOverlayTool(
            model_path=model_path,
            mot_path=mot_path,
            camera_path=camera_path,
            # KEEP GROUND: this session's calib<->OpenSim transform is non-standard
            # (r_convert was wrong here), so our exact transform is baked into the
            # extrinsics pickle with GROUND^-1 folded in -> the tool's GROUND nets it
            # back out. Switching to BACK_WALL would double-apply and misalign.
            checkerboard_placement=CheckerboardPlacement.GROUND,
            custom_geometry_map=custom_geometry_map,
        )
        overlay_tool.apply_camera_offset(camera_offsets[camera])  # display-only vertical fix
        overlay_tool.add_markers(markers_path)  # optional
        overlay_tool.set_output_dir(f"output/sprint/{camera}", clear_output=True)
        render_out = overlay_tool.render(
            geometry_dir=geometry_dir,
            background_video=video_path,
            opacity=0.7,
        )
        videos_out.append(render_out)

    stitched_video_path = "output/sprint/stitched_sprint.mp4"
    stack_videos_horizontal(videos_out, stitched_video_path)
    print(f"Stitched {len(videos_out)} cameras -> {stitched_video_path}")


if __name__ == "__main__":
    main()
