#!/usr/bin/env python3

import os
import sys

#import matplotlib as mpl
#mpl.use('TkAgg')
#import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('../'))
import deeplabcut as dlc

sys.path.append(os.path.abspath(os.path.curdir))
import config as cfg
import create
import fill
import save

def main():
    	# define path of config.yaml
	config_path = os.path.abspath(cfg.working_directory+'/'+cfg.date+'-'+cfg.task+'/config.yaml')

	# create folders
	create.create_new_project()
	create.create_missing_folders()
	# generate training data set
	fill.save_frames()
	fill.step2_wrap()
	fill.step4_wrap()
    	# train
	dlc.train_network(config_path,
                      	  shuffle=1,trainingsetindex=0,
                      	  max_snapshots_to_keep=2,displayiters=None,saveiters=None,maxiters=None,
                      	  allow_growth=False,gputouse=0,autotune=False,keepdeconvweights=True)

if __name__ == '__main__': 
    main()
