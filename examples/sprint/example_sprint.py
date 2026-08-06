from ocap_overlay import BlenderOverlayTool, PyrenderOverlayTool


def main():
    # Example usage of OpenCapOverlay
    model_path = "model.osim"
    mot_path = "motion.sto"
    camera_path = "calibration/Cam1/cameraIntrinsicsExtrinsics.pickle"

    backend = "pyrender"
    OverlayTool = PyrenderOverlayTool if backend == "pyrender" else BlenderOverlayTool

    overlay_tool = OverlayTool(model_path, mot_path, camera_path)
    overlay_tool.set_output_dir("output")
    overlay_tool.render(geometry_dir="Geometry")
    return


if __name__ == "__main__":
    main()
