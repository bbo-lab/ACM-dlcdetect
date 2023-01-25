import os
from pathlib import Path

import deeplabcut
import imageio
import numpy as np
import pandas as pd
import pickle
import scipy.io as sio
import yaml
from deeplabcut.utils import auxfun_models
from skimage import io

from ACMdlcdetect import dlc_helper
from ACMdlcdetect.helpers import load_labels, flatten, get_size_from_reader


def split_trials(trialindex, trainFraction):
    """Split a trial index into train and test sets"""
    trainsize = int(len(trialindex) * trainFraction)
    shuffle = np.random.permutation(trialindex)
    testIndexes = shuffle[trainsize:]
    trainIndexes = shuffle[:trainsize]
    return trainIndexes, testIndexes


def box_it_into_a_cell(joints):
    """Auxiliary function for creating matfile"""
    outer = np.array([[None]], dtype=object)
    outer[0, 0] = np.array(joints, dtype='int64')
    return outer


def make_train_pose_yaml(itemstochange, saveasfile, filename='pose_self.config["yaml"]'):
    raw = open(filename).read()
    docs = []
    for raw_doc in raw.split('\n---'):
        try:
            docs.append(yaml.load(raw_doc, Loader=yaml.SafeLoader))
        except SyntaxError:
            docs.append(raw_doc)

    for key in itemstochange.keys():
        docs[0][key] = itemstochange[key]

    with open(saveasfile, "w") as f:
        yaml.dump(docs[0], f)
    return docs[0]


def make_test_pose_yaml(dictionary, keys2save, saveasfile):
    dict_test = {}
    for key in keys2save:
        dict_test[key] = dictionary[key]

    dict_test['scoremap_dir'] = 'test'
    with open(saveasfile, "w") as f:
        yaml.dump(dict_test, f)


def read_hdf(file) -> object:
    data = pd.read_hdf(file, 'df_with_missing')
    print(type(data))
    return data


class DLCtrain:
    def __init__(self, config):
        self.config = config
        self.labels = load_labels(config["filePath_labels"])

    def save_frames(self):
        print('Saving frames')

        folderPath_save = self.config["working_directory"] + '/labeled-data'
        crop_offset = np.zeros(2, dtype=np.float64)

        # Read labeled frames from self.config["filePath_labels"]
        labels = self.labels
        framesList = np.unique(np.asarray(flatten([list(labels[ln].keys()) for ln in labels])))

        # TODO we cannot start at 0 as we remove background with previous frames. clean up!
        framesList = framesList[framesList > 100]

        if len(self.config["index_frames"]) > 0:  # If set, filter by configures ranges
            framesList = [framesList[np.bitwise_and(framesList >= r[0], framesList <= r[1])] for r in
                          self.config["index_frames"]]
            framesList = sorted([item for sublist in framesList for item in sublist])

        nFrames = np.size(framesList)
        fileList = []
        readers = []
        for file in np.sort(os.listdir(self.config["folderPath_video"])):
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
        nCams = np.size(fileList)
        label_dict = dict()
        for i_feat in sorted(self.config["feat_list"]):
            label_dict[i_feat] = np.full((nCams, nFrames, 2), np.nan, dtype=np.float64)

        backgrounds, backgrounds_std = dlc_helper.calc_backgrounds(fileList,
                                                                   self.config["xRes"], self.config["yRes"],
                                                                   self.config["nFrames_background"])
        img_crop_shape = list(backgrounds.shape)[1:]
        img_crop_shape[0] = min(2 * self.config["dxy"], backgrounds.shape[1])
        img_crop_shape[1] = min(2 * self.config["dxy"], backgrounds.shape[2])
        img_crop = np.zeros(img_crop_shape, dtype=np.uint8)
        for i_cam in range(nCams):
            reader = readers[i_cam]
            background = backgrounds[i_cam]
            background_std = backgrounds_std[i_cam]
            for i_frame in range(nFrames):
                frame = framesList[i_frame]

                dlc_helper.crop_image(reader, frame,
                                      i_cam, self.config["mask_para"], self.config["mask_para_offset"],
                                      background, background_std, self.config["noise_threshold"],
                                      self.config["dxy"], img_crop, crop_offset)

                # corrected labels
                if frame in labels.keys():
                    for feat in labels[frame]:
                        if feat in sorted(self.config["feat_list"]):
                            x_pose = labels[frame][feat][i_cam, 0]
                            y_pose = labels[frame][feat][i_cam, 1]
                            x_pose_corrected = x_pose + crop_offset[0]
                            y_pose_corrected = y_pose + crop_offset[1]
                            if (not (np.isnan(x_pose_corrected)) and not (np.isnan(y_pose_corrected))
                                    and (x_pose_corrected >= 0.0) and (y_pose_corrected >= 0.0)
                                    and (x_pose_corrected <= 2.0 * self.config["dxy"]) and (
                                            y_pose_corrected <= 2.0 * self.config["dxy"])):
                                label_is_clear = True
                            #                             frames_have_valid_labels[i_cam, i] = True
                            else:
                                label_is_clear = False
                            if label_is_clear:
                                label_dict[feat][i_cam, i_frame] = np.array([x_pose_corrected, y_pose_corrected],
                                                                            dtype=np.float64)
                # save frame
                file_save = folderPath_save + '/cam{:02d}_frame{:06d}.png'.format(i_cam, frame)
                io.imsave(file_save, img_crop)
        #             # print
        #             print('Saved frame (cam: {:02d}, index: {:06d})'.format(i_cam, frame))

        file_save = folderPath_save + '/labels.npy'
        np.save(file_save, label_dict)
        return

    # wrapper function for step 2
    def step2_wrap(self):
        print('Executing wrapping function for DeepLabCut\'s step 2')

        originalDirectory = os.path.abspath(os.getcwd())
        mainDirectory = os.path.abspath(self.config["working_directory"])

        label_dict = np.load(self.config["working_directory"] + '/labeled-data/labels.npy', allow_pickle=True)[()]

        folderPath_files = self.config["working_directory"] + '/labeled-data'
        os.chdir(folderPath_files)
        # sort image file names according to how they were stacked
        files = [fn for fn in os.listdir(os.curdir) if ('cam' in fn and
                                                        'frame' in fn and
                                                        ".png" in fn and
                                                        "_labelled" not in fn)]
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        imageaddress = [f for f in files]  # FIXME

        data_single_user = None
        for bodypart in sorted(self.config["feat_list"]):
            index = pd.MultiIndex.from_product([[self.config["scorer"]], [bodypart], ['x', 'y']],
                                               names=['scorer', 'bodyparts', 'coords'])
            Xrescaled = label_dict[bodypart][:, :, 0].ravel().astype(float)
            Yrescaled = label_dict[bodypart][:, :, 1].ravel().astype(float)

            if data_single_user is None:
                data_single_user = pd.DataFrame(np.vstack([Xrescaled, Yrescaled]).T,
                                                columns=index,
                                                index=imageaddress)
            else:
                data_single_user = pd.DataFrame(np.vstack([Xrescaled, Yrescaled]).T,
                                                columns=index,
                                                index=imageaddress)
                data_single_user = pd.concat([data_single_user, data_single_user],
                                             axis=1)  # along bodyparts & scorer dimension

        folderPath_labels = os.path.abspath(
            mainDirectory + '/training-datasets' + '/iteration-{:d}'.format(
                self.config["iteration"]) + '/UnaugmentedDataSet')
        os.chdir(folderPath_labels)
        data_single_user.to_csv("CollectedData.csv")  # breaks multi-indices HDF5 tables better!
        data_single_user.to_hdf('CollectedData.h5', 'df_with_missing', format='table', mode='w')
        os.chdir(originalDirectory)

    # wrapper function for step 4
    def step4_wrap(self):
        print('Executing wrapping function for DeepLabCut\'s step 4')

        originalDirectory = os.path.abspath(os.getcwd())
        mainDirectory = os.path.abspath(self.config["working_directory"])
        imgDirectory = os.path.abspath(mainDirectory + '/labeled-data')

        bf = os.path.abspath(
            mainDirectory + '/training-datasets' + '/iteration-{:d}'.format(
                self.config["iteration"]) + '/UnaugmentedDataSet')

        # Loading scorer's data:
        os.chdir(bf)
        Data = read_hdf('CollectedData.h5')[self.config["scorer"]]
        os.chdir(originalDirectory)

        model_path, num_shuffles = auxfun_models.Check4weights(self.config["net_type"],
                                                               Path(os.path.dirname(deeplabcut.__file__)),
                                                               1)  # if the model does not exist >> throws error!

        for shuffle in self.config["Shuffles"]:
            for trainFraction in self.config["TrainingFraction"]:
                trainIndexes, testIndexes = split_trials(range(len(Data.index)), trainFraction)
                filename_matfile = "MatlabData_" + self.config["scorer"] + str(
                    int(100 * trainFraction)) + "shuffle" + str(shuffle)
                filename_pickle = "Documentation_" + 'data_' + str(int(trainFraction * 100)) + "shuffle" + str(shuffle)

                ####################################################
                # Generating data structure with labeled information & frame metadata (for DeeperCut)
                ####################################################

                # Make matlab train file!
                data = list([])
                for jj in trainIndexes:
                    H = dict()
                    # load image to get dimensions:
                    filename = Data.index[jj]
                    im = io.imread(imgDirectory + '/' + filename)
                    H['image'] = imgDirectory + '/' + filename  # basefolder+folder+filename

                    if len(np.shape(im)) > 2:
                        H['size'] = np.array([np.shape(im)[2], np.shape(im)[0], np.shape(im)[1]])
                    else:
                        # print "Grayscale!"
                        H['size'] = np.array([1, np.shape(im)[0], np.shape(im)[1]])

                    indexjoints = 0
                    joints = np.zeros((len(self.config["feat_list"]), 3)) * np.nan
                    for bpindex, bodypart in enumerate(self.config["feat_list"]):
                        # are labels in image?
                        if Data[bodypart]['x'][jj] < np.shape(im)[1] and Data[bodypart]['y'][jj] < np.shape(im)[0]:
                            joints[indexjoints, 0] = int(bpindex)
                            joints[indexjoints, 1] = Data[bodypart]['x'][jj]
                            joints[indexjoints, 2] = Data[bodypart]['y'][jj]
                            indexjoints += 1

                    joints = joints[np.where(np.prod(np.isfinite(joints), 1))[0],
                             :]  # drop NaN, i.e. lines for missing body parts

                    assert (np.prod(np.array(joints[:, 2]) < np.shape(im)[0]))  # y coordinate within!
                    assert (np.prod(np.array(joints[:, 1]) < np.shape(im)[1]))  # x coordinate within!

                    H['joints'] = np.array(joints, dtype=int)
                    if np.size(joints) > 0:  # exclude images without labels
                        data.append(H)

                if len(data) == 0:
                    raise NoManualLabelsError('No labelled data found in configured frame range')
                else:
                    print(f"Using {len(data)} labelled frames.")

                os.chdir(bf)
                with open(filename_pickle + '.pickle', 'wb') as f:
                    # Pickle the 'data' dictionary using the highest protocol available.
                    pickle.dump([data, trainIndexes, testIndexes, trainFraction], f, pickle.HIGHEST_PROTOCOL)
                os.chdir(originalDirectory)

                ################################################################################
                # Convert to idosyncratic training file for deeper cut (*.mat)
                ################################################################################

                DTYPE = [('image', 'O'), ('size', 'O'), ('joints', 'O')]
                MatlabData = np.array([(np.array([data[item]['image']], dtype='U'),
                                        np.array([data[item]['size']]),
                                        box_it_into_a_cell(data[item]['joints'])) for item in range(len(data))],
                                      dtype=DTYPE)
                os.chdir(bf)
                sio.savemat(filename_matfile + '.mat', {'dataset': MatlabData})
                os.chdir(originalDirectory)

                ################################################################################
                # Creating file structure for training &
                # Test files as well as pose_yaml files (containing training and testing information)
                #################################################################################

                experimentname = mainDirectory + '/dlc-models' + '/iteration-{:d}'.format(
                    self.config["iteration"]) + '/' + os.path.basename(
                    os.path.dirname(self.config["__file__"])) + '-trainset' + str(
                    int(trainFraction * 100)) + 'shuffle' + str(shuffle)
                try:
                    os.mkdir(experimentname)
                    os.mkdir(experimentname + '/train')
                #                 os.mkdir(experimentname+'/test')
                except FileExistsError:
                    print(f"{experimentname} already exists!")
                except PermissionError:
                    print(f"No permission to create {experimentname}!")

                mirror = True
                if mirror:
                    all_joints = []
                    for i_joint in range(len(self.config["feat_list"])):
                        joint = sorted(self.config["feat_list"])[i_joint]
                        joint_split = joint.split('_')
                        if 'left' in joint_split:
                            index = joint_split.index('left')
                            joint_split[index] = 'right'
                            joint2 = '_'.join(joint_split)
                            i_joint2 = sorted(self.config["feat_list"]).index(joint2)
                            all_joints.append([i_joint, i_joint2])
                        elif not ('left' in joint_split) and not ('right' in joint_split):
                            all_joints.append([i_joint])
                else:
                    all_joints = [[i] for i in range(len(self.config["feat_list"]))]

                intermediate_supervision = False
                if (self.config["net_type"] == 'resnet_101') or (self.config["net_type"] == 'resnet_152'):
                    intermediate_supervision = True

                items2change = {'dataset': bf + '/' + filename_matfile + '.mat',
                                'num_joints': len(self.config["feat_list"]),
                                'all_joints': all_joints,
                                'all_joints_names': sorted(self.config["feat_list"]),
                                'init_weights': model_path,
                                'metadataset': bf + '/' + filename_pickle + '.pickle',
                                'project_path': mainDirectory,
                                'mirror': mirror,
                                'net_type': self.config["net_type"],
                                'intermediate_supervision': intermediate_supervision}

                make_train_pose_yaml(items2change,
                                     os.path.abspath(experimentname + '/train/' + 'pose_self.config["yaml"]'),
                                     filename=os.path.abspath(
                                         os.path.dirname(deeplabcut.__file__) + '/pose_self.config["yaml"]')
                                     )

        os.chdir(originalDirectory)
        return


class NoManualLabelsError(Exception):
    pass
