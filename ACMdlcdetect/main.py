#!/usr/bin/env python3

import os
import sys

import deeplabcut as dlc

from ACMdlcdetect import setup
from ACMdlcdetect.dlctrain import DLCtrain
from ACMdlcdetect import dlctrain


def main(project_name, config):

    # define path of config.yaml
    config_path = os.path.abspath(config["working_directory"] + '/config.yaml')

    # create folders
    setup.create_new_project(project_name, config["working_directory"])
    setup.create_missing_folders(config["working_directory"], config["iteration"])

    # generate training data set
    dlc_train = DLCtrain(config)
    dlc_train.save_frames()
    dlc_train.step2_wrap()
    dlc_train.step4_wrap()
    # train
    dlc.train_network(config_path,
                      shuffle=1, trainingsetindex=0,
                      max_snapshots_to_keep=2, displayiters=None, saveiters=None, maxiters=None,
                      allow_growth=False, gputouse=0, autotune=False, keepdeconvweights=True)
