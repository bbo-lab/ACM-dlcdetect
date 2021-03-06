import pathlib
from setuptools import find_packages, setup
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="bbo-acm-dlcdetect",
    version="0.2.0",
    description="Wrapper for DeepLabCut to take inputs from our manual-marking GUI and outputs compatible with ACM",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/bbo-lab/ACM-dlcdetect",
    author="Arne Monsees, BBO lab",
    author_email="bbo-admin@caesar.de",
    license="LGPLv2+",
    classifiers=[
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=['ACM'],
    include_package_data=True,
    install_requires=["deeplabcut==2.1","matplotlib","scikit-image","numpy","ffmpeg"],
)
#DLC used originally: 2.1.6.4, but I hope minor version is sufficient
