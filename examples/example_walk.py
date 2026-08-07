""" Example usage of OpenCapOverlay for a walk """
from opencap_overlay import OpenCapOverlayTool, CheckerboardPlacement, stack_videos_horizontal


def main():
    model_path = "walk/LaiUhlrich2022_scaled.osim"
    mot_path = "walk/walk.mot"
    markers_path = "walk/walk.trc"

    geometry_dir = "Geometry"
    custom_geometry_map = {}

    # Loop through all cameras and render the overlay for each
    cameras = ["cam0", "cam1"]
    videos_out = []
    for camera in cameras:
        camera_path = f"walk/{camera}/cameraIntrinsicsExtrinsics.pickle"
        video_path = f"walk/{camera}/walk.mov"

        overlay_tool = OpenCapOverlayTool(
            model_path=model_path,
            mot_path=mot_path,
            camera_path=camera_path,
            checkerboard_placement=CheckerboardPlacement.GROUND,
            custom_geometry_map=custom_geometry_map
        )
        overlay_tool.add_markers(markers_path)
        overlay_tool.set_output_dir(f"output/walk/{camera}", clear_output=True)
        render_out = overlay_tool.render(
            geometry_dir=geometry_dir,
            background_video=video_path,
            opacity=0.7,
        )
        videos_out.append(render_out)

    # Stitch the per-camera renders together side by side
    stitched_video_path = "output/walk/stitched_walk.mp4"
    stack_videos_horizontal(videos_out, stitched_video_path)
    print(f"Stitched {len(videos_out)} cameras -> {stitched_video_path}")
    return


if __name__ == "__main__":
    main()
