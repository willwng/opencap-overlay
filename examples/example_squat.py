"""
Example usage of OpenCapOverlay for a squat
"""
from opencap_overlay import OpenCapOverlayTool, CheckerboardPlacement


def main():
    model_path = "squat/LaiUhlrich2022_scaled.osim"
    mot_path = "squat/squats1.mot"

    geometry_dir = "Geometry"
    custom_geometry_map = {}

    # Loop through all cameras and render the overlay for each
    cameras = ["cam0", "cam1", "cam2", "cam3", "cam4"]
    videos_out = []
    for camera in cameras:
        camera_path = f"squat/{camera}/cameraIntrinsicsExtrinsics.pickle"
        video_path = f"squat/{camera}/squats1.mov"

        overlay_tool = OpenCapOverlayTool(
            model_path=model_path,
            mot_path=mot_path,
            camera_path=camera_path,
            checkerboard_placement=CheckerboardPlacement.BACK_WALL,
            custom_geometry_map=custom_geometry_map
        )
        overlay_tool.set_output_dir(f"output/squat/{camera}", clear_output=True)
        render_out = overlay_tool.render(
            geometry_dir=geometry_dir,
            background_video=video_path,
            opacity=0.7,
        )
        videos_out.append(render_out)
    return


if __name__ == "__main__":
    main()
