#!/usr/bin/env python3
from ACMdlcdetect import dlc_helper
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input

import matplotlib.pyplot as plt
import numpy as np
import os
import skimage.color
import sys

import imageio
from ccvtools import rawio

sys.path.append(os.path.abspath(os.path.curdir))


# can be used to get features/heatmaps from trained network
def prepare_feature_extraction(project_name, config):
    mainDirectory = os.path.abspath(config["working_directory"])

    ####################################################
    # Loading data, and defining model folder
    ####################################################
    modelfolder = mainDirectory + '/dlc-models' + '/iteration-{:d}'.format(
        config["iteration"]) + '/' + project_name + '-trainset' + str(
        int(config["TrainingFraction"][0] * 100)) + 'shuffle' + str(config["Shuffles"][0])
    cfg_dlc = load_config(modelfolder + '/train/' + "pose_cfg.yaml")

    ##################################################
    # Load and setup CNN part detector
    ##################################################
    # Check which snap shots are available and sort them by # iterations
    Snapshots = np.array([fn.split('.')[0] for fn in os.listdir(modelfolder + '/train/') if "index" in fn])
    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = sorted(Snapshots[increasing_indices])

    # Check if data was already generated:
    snapshotindex = -1  # always use last available snapshot # ATTENTION: make sure index of training weights file is the highest in the folder
    cfg_dlc['init_weights'] = modelfolder + '/train/' + Snapshots[snapshotindex]

    sess, inputs, outputs = predict.setup_pose_prediction(cfg_dlc)

    return cfg_dlc, sess, inputs, outputs


def get_features(cfg_dlc, sess, inputs, outputs,
                 image):
    """Adapted from DeeperCut, see pose-tensorflow folder"""
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg_dlc)
    pose = predict.argmax_pose_predict(scmap, locref, cfg_dlc['stride'])
    return scmap, locref, pose


def main(project_name, config):
    verbose = False
    dFrame = 1

    index_frames = config["index_frames_save"]

    # calculate backgrounds
    print('Calculating background images')
    fileList = list()
    readers = []
    for file in np.sort(os.listdir(config["folderPath_video"])):
        try:
            # TODO: Generate readers and file lists and so on much earlier, when parsing config ...
            readers.append(imageio.get_reader(self.config["folderPath_video"] + '/' + file))
            fileList.append(self.config["folderPath_video"] + '/' + file)

            if "xRes" not in self.config or self.config["xRes"] is None or \
                    "yRes" not in self.config or self.config["yRes"] is None:
                (self.config["xRes"], self.config["yRes"]) = get_size_from_reader(readers[-1])
        except ValueError as e:
            # No backend was found for respective file, thus not considered a video
            print(f"No backend found for {file}: {e}")

    fileList = sorted(fileList)
    backgrounds, backgrounds_std = dlc_helper.calc_backgrounds(fileList,
                                                               config["xRes"], config["yRes"],
                                                               config["nFrames_background"])
    # initialize needed arrays
    frame_list = np.arange(index_frames[0][0], index_frames[0][1] + dFrame, dFrame,
                           dtype=np.int64)
    nFrames = np.size(frame_list, 0)
    nCameras = np.size(fileList, 0)
    nLabels = np.size(config["feat_list"], 0)
    labels_all = np.full((nFrames, nCameras, nLabels, 3), np.nan, dtype=np.float64)
    img_crop = np.zeros((2 * config["dxy"], 2 * config["dxy"]), dtype=np.uint8)
    pixel_crop = np.zeros(2, dtype=np.float64)
    # initialize figure when verbose is True
    if verbose:
        fig = plt.figure(1)
        fig.clear()
        h_img = list()
        h_labels = list()
        for i_cam in range(nCameras):
            ax = fig.add_subplot(2, 2, i_cam + 1)  # assumes 4 cameras
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
        plt.pause(2 ** -52)
    # prepare feature extraction
    print('Preparing for feature detection')
    cfg_dlc, sess, inputs, outputs = prepare_feature_extraction(project_name, config)
    # start feature extraction
    print('Starting feature detection')
    print(index_frames)
    nIndex = np.size(index_frames, 0)
    for index in range(nIndex):
        frame_list = np.arange(index_frames[index][0], index_frames[index][1] + dFrame, dFrame,
                               dtype=np.int64)
        nFrames = np.size(frame_list, 0)
        print(frame_list)
        labels_all = np.full((nFrames, nCameras, nLabels, 3), np.nan, dtype=np.float64)
        print('Detecting features for frames\t{:06d} - {:06d} ({:d} cams, {:d} frames)'.format(frame_list[0],
                                                                                               frame_list[-1], nCameras,
                                                                                               nFrames))
        for i_cam in range(nCameras):
            reader = readers[i_cam]
            print()
            print(f'c{i_cam}')
            for i_frame in range(nFrames):
                print('.', end='', flush=True)
                dlc_helper.crop_image(reader, frame_list[i_frame],
                                      i_cam, config["mask_para"], config["mask_para_offset"],
                                      backgrounds[i_cam], backgrounds_std[i_cam], config["noise_threshold"],
                                      config["dxy"], img_crop, pixel_crop)

                scmap, locref, labels = get_features(cfg_dlc, sess, inputs, outputs,
                                                     img_crop)  # TODO move out od loop?
                labels_use = np.copy(labels)
                labels_use[:, :2] = labels_use[:, :2] - pixel_crop
                labels_all[i_frame, i_cam] = np.copy(labels_use)
                # update figure if verbose is True
                if verbose:
                    h_img[i_cam].set_array(img_crop)
                    h_labels[i_cam].set_data(labels[:, 0], labels[:, 1])
                    fig.canvas.draw()
                    plt.pause(2 ** -52)
        # save labels
        file_save = config["working_directory"] + '/dlc_labels/labels_dlc_{:06d}_{:06d}.npy'.format(
            index_frames[index][0], index_frames[index][1])
        os.makedirs(os.path.dirname(file_save), exist_ok=True)
        dlc_labels = dict()
        dlc_labels['file_save'] = file_save
        dlc_labels['frame_list'] = frame_list
        dlc_labels['labels_all'] = labels_all
        np.save(file_save, dlc_labels)
        print('Saved labels to {:s}'.format(file_save))
    print('Finished feature detection')
