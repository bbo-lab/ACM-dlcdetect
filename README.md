# ACM (Anatomically-constrained model) dlc-detect wrapper
Wrapper for DeepLabCut to take inputs from our manual-marking GUI and outputs compatible with ACM.
For productive application, please use the [latest tag release](https://github.com/bbo-lab/ACM-dlcdetect/tags).

By Arne Monsees.

## Installation

This software utilizes DeepLabCut 2.1, which dependends on tensorflow 1.15. The version that is installed by conda requires CUDA 10.0 installed. See below for installation hints. If graphics support is not present, the software should automatically default to CPU computations, which will, however, be substantially and potentially unfeasibly  slower.

1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/linux/)
2. Clone https://github.com/bbo-lab/ACM-dlcdetect.git
3. Create conda environment `conda env create -f https://raw.githubusercontent.com/bbo-lab/ACM-dlcdetect/main/environment.yml`
4. Navigate into the ACM-dlcdetect repository
5. Install using `pip install .` (Alternatively, set repository base directory as working directory when running.)

### Headless usage

Later versions of DLC 2.1 are separated into the pip packages `deeplabcut` and `deeplabcut[gui]`. The former is installed by default, thus headless usage should be possible. The module also supports the switch `--headless`, which enables headless DLClight mode for earlier versions.

### CUDA

CUDA in the correct version is included in the conda environment and should run out of the box.

## Usage

1. Create a folder corresponding to your dataset (e.g. `~/data/YYYYMMDD_exp1`). 
2. Adjust and add to this folder the config file in `examples/`. Especially, set path for manual labels and video files, and update frame ranges.
3. Enter conda environment with `conda activate bbo_acm_dlcdetect-gpu`
4. Run with `python -m ACM-dlcdetect ~/data/YYYYMMDD_exp1`

Note that the process will fail if the algorithm has already been (fully or partially) run on this folder. In this case, delete `examples/data` recursively.
