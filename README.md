# ACM (Anatomically-constrained model) dlc-detect wrapper
Wrapper for DeepLabCut to take inputs from our manual-marking GUI and outputs compatible with ACM
By Arne Monsees.

## Installation

### Linux
(Note: Windows support is planned, but currently not present.)

1. [Install Anaconda](https://docs.anaconda.com/anaconda/install/linux/)
2. Clone https://github.com/bbo-lab/ACM-dlcdetect.git
3. Create conda environment `conda env create -f https://raw.githubusercontent.com/bbo-lab/ACM-dlcdetect/main/environment.yml`
4. Navigate into the ACM-dlcdetect repository
5. Install using `pip install .`

### GPU usage
If you have an NVidia GPU and set up all drivers, activate support by

1. Changing to the conda environment `conda activate bbo_acm_dlcdetect`
2. install the `cuddn` package: `conda install -c conda-forge cudnn`
