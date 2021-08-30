#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.color
import sys

sys.path.append(os.path.abspath(os.path.curdir))
import config as cfg
import dlc_helper
import ccv

sys.path.append(os.path.abspath('../'))
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input

# can be used to get features/heatmaps from trained network
def prepare_feature_extraction():
    mainDirectory = os.path.abspath(cfg.working_directory+'/'+cfg.date+'-'+cfg.task)

    ####################################################
    # Loading data, and defining model folder
    ####################################################
    modelfolder = mainDirectory+'/dlc-models'+'/iteration-{:d}'.format(cfg.iteration)+'/'+ \
                  cfg.date+'-trainset'+str(int(cfg.TrainingFraction[0]*100))+'shuffle'+str(cfg.Shuffles[0])
    cfg_dlc = load_config(modelfolder + '/train/' + "pose_cfg.yaml")

    ##################################################
    # Load and setup CNN part detector
    ##################################################
    # Check which snap shots are available and sort them by # iterations
    Snapshots = np.array([fn.split('.')[0] for fn in os.listdir(modelfolder + '/train/') if "index" in fn])
    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = sorted(Snapshots[increasing_indices])

    # Check if data was already generated:
    snapshotindex = -1 # always use last available snapshot # ATTENTION: make sure index of training weights file is the highest in the folder
    cfg_dlc['init_weights'] = modelfolder + '/train/' + Snapshots[snapshotindex]

    sess, inputs, outputs = predict.setup_pose_prediction(cfg_dlc)

    return cfg_dlc, sess, inputs, outputs

def get_features(cfg_dlc, sess, inputs, outputs,
                 image):
    '''Adapted from DeeperCut, see pose-tensorflow folder'''
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg_dlc)
    pose = predict.argmax_pose_predict(scmap, locref, cfg_dlc.stride)
    return scmap, locref, pose

if __name__ == '__main__':
    verbose = False
    dFrame = 1

    index_frames = cfg.index_frames_save

    # calculate backgrounds
    print('Calculating background images')
    fileList = list()
    for file in np.sort(os.listdir(cfg.folderPath_ccv)):
        if (file[-4:] == '.ccv'):
            fileList.append(cfg.folderPath_ccv+'/'+file)
    fileList = sorted(fileList)
    backgrounds, backgrounds_std = dlc_helper.calc_backgrounds(fileList,
                                                               cfg.xRes, cfg.yRes,
                                                               cfg.nFrames_background)
    # initialize needed arrays
    frame_list = np.arange(index_frames[0][0], index_frames[0][1] + dFrame, dFrame,
                           dtype=np.int64)
    nFrames = np.size(frame_list, 0)
    nCameras = np.size(fileList, 0)
    nLabels = np.size(cfg.feat_list, 0)
    labels_all = np.full((nFrames, nCameras, nLabels, 3), np.nan, dtype=np.float64)
    img_crop = np.zeros((2*cfg.dxy, 2*cfg.dxy), dtype=np.uint8)
    pixel_crop = np.zeros(2, dtype=np.float64)
    # initialize figure when verbose is True
    if verbose:
        fig = plt.figure(1)
        fig.clear()
        h_img = list()
        h_labels = list()
        for i_cam in range(nCameras):
            ax = fig.add_subplot(2,2,i_cam+1) # assumes 4 cameras
            ax.clear()
            ax.set_axis_off()
            h_img_single = ax.imshow(img_crop, 'gray',
                                     vmin=0, vmax=127)
            h_labels_single = ax.plot(labels_all[0, i_cam, :, 0], labels_all[0, i_cam, :, 1],
                                      color='darkorange', linestyle='', marker='.')[0]
            h_img.append(h_img_single)
            h_labels.append(h_labels_single)
        fig.tight_layout()
        fig.canvas.draw()
        plt.pause(2**-52)
    # prepare feature extraction
    print('Preparing for feature detection')
    cfg_dlc, sess, inputs, outputs = prepare_feature_extraction()
    # start feature extraction
    print('Starting feature detection')
    nIndex = np.size(index_frames, 0)
    for index in range(nIndex):
        frame_list = np.arange(index_frames[index][0], index_frames[index][1] + dFrame, dFrame,
                               dtype=np.int64)
        nFrames = np.size(frame_list, 0)
        labels_all = np.full((nFrames, nCameras, nLabels, 3), np.nan, dtype=np.float64)
        print('Detecting features for frames\t{:06d} - {:06d}'.format(frame_list[0], frame_list[-1]))
        for i_frame in range(nFrames):
            # ATTENTION: MAKE SURE THIS IS IDENTICAL TO fill.py
            for i_cam in range(nCameras):
#     #             if (cfg.task == 'arena'):
#                 if (False):
#                     dlc_helper.crop_image(fileList[i_cam], frame_list[i_frame],
#                                           backgrounds[i_cam], backgrounds_std[i_cam], cfg.noise_threshold,
#                                           cfg.dxy, img_crop, pixel_crop)
#                 else: # used for 20200205 and 20200207
#                     dlc_helper.crop_image2(fileList[i_cam], frame_list[i_frame],
#                                            backgrounds[i_cam], backgrounds_std[i_cam], cfg.noise_threshold,
#                                            cfg.dxy, img_crop, pixel_crop)
                dlc_helper.crop_image3(fileList[i_cam], frame_list[i_frame],
                                       i_cam, cfg.mask_para, cfg.mask_para_offset,
                                       backgrounds[i_cam], backgrounds_std[i_cam], cfg.noise_threshold,
                                       cfg.dxy, img_crop, pixel_crop)
            
                scmap, locref, labels = get_features(cfg_dlc, sess, inputs, outputs,
                                                     img_crop)
                labels_use = np.copy(labels)
                labels_use[:, :2] = labels_use[:, :2] - pixel_crop
                labels_all[i_frame, i_cam] = np.copy(labels_use)
                # update figure if verbose is True
                if verbose:
                    h_img[i_cam].set_array(img_crop)
                    h_labels[i_cam].set_data(labels[:, 0], labels[:, 1])
                    fig.canvas.draw()
                    plt.pause(2**-52)
        # save labels
        file_save = '/axon/u/amonsees/DeepLabCut/wrapping/dlc_labels/'+\
                    cfg.date+'/'+cfg.task+'/labels_dlc_{:06d}_{:06d}.npy'.format(index_frames[index][0], index_frames[index][1])
        dlc_labels = dict()
        dlc_labels['file_save'] = file_save
        dlc_labels['frame_list'] = frame_list
        dlc_labels['labels_all'] = labels_all
        np.save(file_save, dlc_labels)
        print('Saved labels to {:s}'.format(file_save))
    print('Finished feature detection')
