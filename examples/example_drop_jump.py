"""
Example usage of OpenCapOverlay
"""
from opencap_overlay import BlenderOverlayTool, PyrenderOverlayTool


def main():
    model_path = "drop_jump/LaiArnoldModified2017_poly_withArms_weldHand_scaled.osim"
    mot_path = "drop_jump/DJ1.mot"
    camera_path = "drop_jump/cameraIntrinsicsExtrinsics.pickle"
    geometry_dir = "Geometry"

    custom_geometry_map = {}

    backend = "pyrender"
    OverlayTool = PyrenderOverlayTool if backend == "pyrender" else BlenderOverlayTool

    overlay_tool = OverlayTool(
        model_path=model_path,
        mot_path=mot_path,
        camera_path=camera_path,
        custom_geometry_map=custom_geometry_map
    )
    overlay_tool.set_output_dir("output/drop_jump")
    overlay_tool.render(geometry_dir=geometry_dir)
    return


if __name__ == "__main__":
    main()
