"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/AlexEMG/DeepLabCut

Please see AUTHORS for contributors.
https://github.com/AlexEMG/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""

import sys

import os
from pathlib import Path
import cv2
import shutil
import sys

from deeplabcut import DEBUG

import config as cfg

def create_new_project():
    """
    Creates a new project directory, sub-directories and a basic configuration file. The configuration file is loaded with the default values. Change its parameters to your projects need.

    """
    
    project = os.path.basename(os.path.dirname(cfg.__file__))
    working_directory = cfg.working_directory
    
    from datetime import datetime as dt
    from deeplabcut.utils import auxiliaryfunctions
    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    d = str(month[0:3]+str(day))
    date = dt.today().strftime('%Y-%m-%d')
    if working_directory == None:
        working_directory = '.'
    wd = Path(working_directory).resolve()
    project_path = wd
    
    # Create project and sub-directories
    if not DEBUG and project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        return
#     video_path = project_path / 'videos'
    data_path = project_path / 'labeled-data'
    shuffles_path = project_path / 'training-datasets'
    results_path = project_path / 'dlc-models'
#     for p in [video_path, data_path, shuffles_path, results_path]:
    for p in [data_path, shuffles_path, results_path]:
        p.mkdir(parents=True, exist_ok=DEBUG)
        print('Created "{}"'.format(p))

    #        Set values to config file:
    cfg_file,ruamelFile = auxiliaryfunctions.create_config_template()
    cfg_file['Task']=project
    cfg_file['scorer']=cfg.scorer
#     cfg_file['video_sets']=video_sets
    cfg_file['project_path']=str(project_path)
    cfg_file['date']=''
    cfg_file['bodyparts']=[]
    cfg_file['cropping']=False
    cfg_file['start']=0
    cfg_file['stop']=1
    cfg_file['numframes2pick']=20
    cfg_file['TrainingFraction']=[1.00]
    cfg_file['iteration']=0
    #cfg_file['resnet']=50
    cfg_file['default_net_type']='resnet_50'
    cfg_file['default_augmenter']='default'
    cfg_file['snapshotindex']=-1
    cfg_file['x1']=0
    cfg_file['x2']=0
    cfg_file['y1']=0
    cfg_file['y2']=0
    cfg_file['batch_size']=8 #batch size during inference (video - analysis); see https://www.biorxiv.org/content/early/2018/10/30/457242
    cfg_file['corner2move2']=(0,0)
    cfg_file['move2corner']=False
#     cfg_file['skeleton']=[['bodypart1','bodypart2'],['objectA','bodypart3']]
#     cfg_file['skeleton_color']='black'
#     cfg_file['pcutoff']=0.6
#     cfg_file['dotsize']=12 #for plots size of dots
#     cfg_file['alphavalue']=0.7 #for plots transparency of markers
#     cfg_file['colormap']='jet' #for plots type of colormap

    projconfigfile=os.path.join(str(project_path),'config.yaml')
    # Write dictionary to yaml  config file
    auxiliaryfunctions.write_config(projconfigfile,cfg_file)

    print('Generated "{}"'.format(project_path / 'config.yaml'))
    return projconfigfile


# create remaining folder structure
def create_missing_folders():
    originalDirectory = os.path.abspath(os.getcwd())
    mainDirectory = os.path.abspath(cfg.working_directory)

    os.mkdir(mainDirectory+'/dlc-models'+'/iteration-{:d}'.format(cfg.iteration))
    os.mkdir(mainDirectory+'/training-datasets'+'/iteration-{:d}'.format(cfg.iteration)) # add subfolder and subfolder/train folder
    os.mkdir(mainDirectory+'/training-datasets'+'/iteration-{:d}'.format(cfg.iteration)+'/UnaugmentedDataSet') # csv, h5, pickle, mat files
    
    return
