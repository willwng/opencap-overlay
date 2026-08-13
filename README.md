# OpenCap Overlay

A Python library for overlaying/validating inverse kinematics results
to video-based data sources.

<div style="display: flex; justify-content: center; gap: 20px;">
  <img src="assets/walk_stitched.gif" width="300"/>
  <img src="assets/squat_stitched.gif" width="300"/>
</div>

## Install OpenCap Overlay
First, OpenSim is required to be present. We recommend using conda to ensure OpenSim is installed. 
An example of creating a new environment with OpenSim:
```bash
conda create -n ENV_NAME python=3.11
conda activate ENV_NAME
conda install opensim-org::opensim
```

Then, the `opencap_overlay` package can be installed
```bash
cd opencap-overlay
pip install -e .
```

### Additional Dependencies
The overlay tool relies on the following command line tools (these are typically already be installed or easy to install)
- ffmpeg
- ffprobe

## Example usage

We provide examples based on the paired 
IK + Video data from [OpenCap](https://simtk.org/projects/opencap) and from
[opencap-test-data](https://github.com/stanfordnmbl/opencap-test-data/tree/main)

See the [examples](examples) folder for example usage of the OpenCap Overlay library.

