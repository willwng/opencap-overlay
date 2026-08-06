"""
Example usage of OpenCapOverlay for a walk
"""
from opencap_overlay import PyrenderOverlayTool


def main():
    model_path = "walk/LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim"
    mot_path = "walk/walking2.mot"
    camera_path = "walk/cameraIntrinsicsExtrinsics.pickle"
    video_path = "walk/walking2_syncdWithMocap.avi"

    geometry_dir = "Geometry"
    custom_geometry_map = {}

    OverlayTool = PyrenderOverlayTool

    overlay_tool = OverlayTool(
        model_path=model_path,
        mot_path=mot_path,
        camera_path=camera_path,
        custom_geometry_map=custom_geometry_map
    )
    overlay_tool.set_output_dir("output/walk")
    overlay_tool.render(
        geometry_dir=geometry_dir,
        background_video=video_path,
        opacity=0.7,
    )
    return


if __name__ == "__main__":
    main()
