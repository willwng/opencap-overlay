"""
Example usage of OpenCapOverlay for a drop-jump
"""
from opencap_overlay import OpenCapOverlayTool


def main():
    model_path = "drop_jump/LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim"
    mot_path = "drop_jump/DJ1.mot"
    camera_path = "drop_jump/cameraIntrinsicsExtrinsics.pickle"
    video_path = "drop_jump/DJ1_syncdWithMocap.avi"

    geometry_dir = "Geometry"
    custom_geometry_map = {}

    overlay_tool = OpenCapOverlayTool(
        model_path=model_path,
        mot_path=mot_path,
        camera_path=camera_path,
        custom_geometry_map=custom_geometry_map
    )
    overlay_tool.set_output_dir("output/drop_jump", clear_output=True)
    overlay_tool.render(
        geometry_dir=geometry_dir,
        background_video=video_path,
        opacity=0.7,
    )
    return


if __name__ == "__main__":
    main()
