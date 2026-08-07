"""
Example usage of OpenCapOverlay for a walk
"""
from opencap_overlay import OpenCapOverlayTool, CheckerboardPlacement


def main():
    model_path = "walk/LaiUhlrich2022_scaled.osim"
    mot_path = "walk/walk.mot"

    geometry_dir = "Geometry"
    custom_geometry_map = {}

    # Loop through all cameras and render the overlay for each
    cameras = ["cam0", "cam1"]
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
        overlay_tool.set_output_dir(f"output/walk/{camera}", clear_output=True)
        overlay_tool.render(
            geometry_dir=geometry_dir,
            background_video=video_path,
            opacity=0.7,
        )
    return


if __name__ == "__main__":
    main()
