import os
from pathlib import Path

import deeplabcut
import dlcdetectConfig as cfg
import imageio
import numpy as np
from deeplabcut.utils import auxfun_models
from skimage import io

import pandas as pd
import scipy.io as sio
import yaml, pickle

from ACMdlcdetect import dlc_helper


def save_frames():
    print('Saving frames')

    folderPath_save = cfg.working_directory + '/labeled-data'
    img_crop = np.zeros((2 * cfg.dxy, 2 * cfg.dxy), dtype=np.uint8)
    pixel_crop = np.zeros(2, dtype=np.float64)

    # Read labeled frames from cfg.filePath_labels
    labels = load_labels(cfg.filePath_labels)
    framesList = np.asarray(sorted(labels.keys()))
    framesList = framesList[
        framesList > 100]  # TODO we cannot start at 0 as we remove background with prefious frames. clean up!
    if len(cfg.index_frames) > 0:  # If set, filter by configures ranges
        framesList = [framesList[np.bitwise_and(framesList >= r[0], framesList <= r[1])] for r in cfg.index_frames]
        framesList = sorted([item for sublist in framesList for item in sublist])

    nFrames = np.size(framesList)
    fileList = list()
    for file in np.sort(os.listdir(cfg.folderPath_video)):
        try:
            reader = imageio.get_reader(cfg.folderPath_video + '/' + file)
            fileList.append(cfg.folderPath_video + '/' + file)
        except:
            print(f"No backend found for {file}")
            pass  # No backend was found for respective file, thus not considered a video

    fileList = sorted(fileList)
    nCams = np.size(fileList)
    label_dict = dict()
    for i_feat in sorted(cfg.feat_list):
        label_dict[i_feat] = np.full((nCams, nFrames, 2), np.nan, dtype=np.float64)

    backgrounds, backgrounds_std = dlc_helper.calc_backgrounds(fileList,
                                                               cfg.xRes, cfg.yRes,
                                                               cfg.nFrames_background)
    for i_cam in range(nCams):
        reader = imageio.get_reader(fileList[i_cam])
        background = backgrounds[i_cam]
        background_std = backgrounds_std[i_cam]
        for i_frame in range(nFrames):
            frame = framesList[i_frame]

            dlc_helper.crop_image(reader, frame,
                                  i_cam, cfg.mask_para, cfg.mask_para_offset,
                                  background, background_std, cfg.noise_threshold,
                                  cfg.dxy, img_crop, pixel_crop)

            # corrected labels
            if (frame in labels.keys()):
                for feat in labels[frame]:
                    if (feat in sorted(cfg.feat_list)):
                        x_pose = labels[frame][feat][i_cam, 0]
                        y_pose = labels[frame][feat][i_cam, 1]
                        x_pose_corrected = x_pose + pixel_crop[0]
                        y_pose_corrected = y_pose + pixel_crop[1]
                        if (not (np.isnan(x_pose_corrected)) and not (np.isnan(y_pose_corrected))
                                and (x_pose_corrected >= 0.0) and (y_pose_corrected >= 0.0)
                                and (x_pose_corrected <= 2.0 * cfg.dxy) and (y_pose_corrected <= 2.0 * cfg.dxy)):
                            label_is_clear = True
                        #                             frames_have_valid_labels[i_cam, i] = True
                        else:
                            label_is_clear = False
                        if (label_is_clear):
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


def load_labels(labels_file) -> dict:
    labels = np.load(labels_file, allow_pickle=True)['arr_0'][()]

    if "version" in labels:  # New style labels file
        labels = labels["labels"]

    return labels


# wrapper function for step 2
def step2_wrap():
    print('Executing wrapping function for DeepLabCut\'s step 2')

    originalDirectory = os.path.abspath(os.getcwd())
    mainDirectory = os.path.abspath(cfg.working_directory)

    label_dict = np.load(cfg.working_directory + '/labeled-data/labels.npy', allow_pickle=True).item()

    folderPath_files = cfg.working_directory + '/labeled-data'
    os.chdir(folderPath_files)
    # sort image file names according to how they were stacked
    files = [fn for fn in os.listdir(os.curdir) if ('cam' in fn and
                                                    'frame' in fn and
                                                    ".png" in fn and
                                                    "_labelled" not in fn)]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    imageaddress = [f for f in files]  # FIXME

    frame = None
    Frame = None
    for bodypart in sorted(cfg.feat_list):
        index = pd.MultiIndex.from_product([[cfg.scorer], [bodypart], ['x', 'y']],
                                           names=['scorer', 'bodyparts', 'coords'])
        Xrescaled = label_dict[bodypart][:, :, 0].ravel().astype(float)
        Yrescaled = label_dict[bodypart][:, :, 1].ravel().astype(float)

        if Frame is None:
            frame = pd.DataFrame(np.vstack([Xrescaled, Yrescaled]).T,
                                 columns=index,
                                 index=imageaddress)
            Frame = frame
        else:
            frame = pd.DataFrame(np.vstack([Xrescaled, Yrescaled]).T,
                                 columns=index,
                                 index=imageaddress)
            Frame = pd.concat([Frame, frame], axis=1)  # along bodyparts & scorer dimension

        DataSingleUser = Frame

    folderPath_labels = os.path.abspath(
        mainDirectory + '/training-datasets' + '/iteration-{:d}'.format(cfg.iteration) + '/UnaugmentedDataSet')
    os.chdir(folderPath_labels)
    DataSingleUser.to_csv("CollectedData.csv")  # breaks multi-indices HDF5 tables better!
    DataSingleUser.to_hdf('CollectedData.h5', 'df_with_missing', format='table', mode='w')
    os.chdir(originalDirectory)
    return


# wrapper function for step 4
def step4_wrap():
    print('Executing wrapping function for DeepLabCut\'s step 4')

    originalDirectory = os.path.abspath(os.getcwd())
    mainDirectory = os.path.abspath(cfg.working_directory)
    imgDirectory = os.path.abspath(mainDirectory + '/labeled-data')

    def SplitTrials(trialindex, trainFraction):
        """Split a trial index into train and test sets"""
        trainsize = int(len(trialindex) * trainFraction)
        shuffle = np.random.permutation(trialindex)
        testIndexes = shuffle[trainsize:]
        trainIndexes = shuffle[:trainsize]
        return (trainIndexes, testIndexes)

    def boxitintoacell(joints):
        '''Auxiliary function for creating matfile'''
        outer = np.array([[None]], dtype=object)
        outer[0, 0] = np.array(joints, dtype='int64')
        return outer

    def MakeTrain_pose_yaml(itemstochange, saveasfile, filename='pose_cfg.yaml'):
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

    def MakeTest_pose_yaml(dictionary, keys2save, saveasfile):
        dict_test = {}
        for key in keys2save:
            dict_test[key] = dictionary[key]

        dict_test['scoremap_dir'] = 'test'
        with open(saveasfile, "w") as f:
            yaml.dump(dict_test, f)

    bf = os.path.abspath(
        mainDirectory + '/training-datasets' + '/iteration-{:d}'.format(cfg.iteration) + '/UnaugmentedDataSet')

    # Loading scorer's data:
    os.chdir(bf)
    Data = pd.read_hdf('CollectedData.h5', 'df_with_missing')[cfg.scorer]
    os.chdir(originalDirectory)

    model_path, num_shuffles = auxfun_models.Check4weights(cfg.net_type, Path(os.path.dirname(deeplabcut.__file__)),
                                                           1)  # if the model does not exist >> throws error!

    for shuffle in cfg.Shuffles:
        for trainFraction in cfg.TrainingFraction:
            trainIndexes, testIndexes = SplitTrials(range(len(Data.index)), trainFraction)
            filename_matfile = "MatlabData_" + cfg.scorer + str(int(100 * trainFraction)) + "shuffle" + str(shuffle)
            filename_pickle = "Documentation_" + 'data_' + str(int(trainFraction * 100)) + "shuffle" + str(shuffle)

            ####################################################
            ######## Generating data structure with labeled information & frame metadata (for DeeperCut)
            ####################################################

            # Make matlab train file!
            data = list([])
            for jj in trainIndexes:
                H = dict()
                # load image to get dimensions:
                filename = Data.index[jj]
                im = io.imread(imgDirectory + '/' + filename)
                H['image'] = imgDirectory + '/' + filename  # basefolder+folder+filename

                try:
                    H['size'] = np.array([np.shape(im)[2], np.shape(im)[0], np.shape(im)[1]])
                except:
                    # print "Grayscale!"
                    H['size'] = np.array([1, np.shape(im)[0], np.shape(im)[1]])

                indexjoints = 0
                joints = np.zeros((len(cfg.feat_list), 3)) * np.nan
                for bpindex, bodypart in enumerate(cfg.feat_list):
                    if Data[bodypart]['x'][jj] < np.shape(im)[1] and Data[bodypart]['y'][jj] < np.shape(im)[
                        0]:  # are labels in image?
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
            ######### Convert to idosyncratic training file for deeper cut (*.mat)
            ################################################################################

            DTYPE = [('image', 'O'), ('size', 'O'), ('joints', 'O')]
            MatlabData = np.array([(np.array([data[item]['image']], dtype='U'),
                                    np.array([data[item]['size']]),
                                    boxitintoacell(data[item]['joints'])) for item in range(len(data))], dtype=DTYPE)
            os.chdir(bf)
            sio.savemat(filename_matfile + '.mat', {'dataset': MatlabData})
            os.chdir(originalDirectory)

            ################################################################################
            ######### Creating file structure for training & 
            ######### Test files as well as pose_yaml files (containing training and testing information)
            #################################################################################

            experimentname = mainDirectory + '/dlc-models' + '/iteration-{:d}'.format(
                cfg.iteration) + '/' + os.path.basename(os.path.dirname(cfg.__file__)) + '-trainset' + str(
                int(trainFraction * 100)) + 'shuffle' + str(shuffle)
            try:
                os.mkdir(experimentname)
                os.mkdir(experimentname + '/train')
            #                 os.mkdir(experimentname+'/test')
            except:
                print("Apparently ", experimentname, "already exists!")

            mirror = True
            if (mirror):
                all_joints = []
                for i_joint in range(len(cfg.feat_list)):
                    joint = sorted(cfg.feat_list)[i_joint]
                    joint_split = joint.split('_')
                    if ('left' in joint_split):
                        index = joint_split.index('left')
                        joint_split[index] = 'right'
                        joint2 = '_'.join(joint_split)
                        i_joint2 = sorted(cfg.feat_list).index(joint2)
                        all_joints.append([i_joint, i_joint2])
                    elif (not ('left' in joint_split) and not ('right' in joint_split)):
                        all_joints.append([i_joint])
            else:
                all_joints = [[i] for i in range(len(cfg.feat_list))]

            intermediate_supervision = False
            if ((cfg.net_type == 'resnet_101') or (cfg.net_type == 'resnet_152')):
                intermediate_supervision = True

            items2change = {'dataset': bf + '/' + filename_matfile + '.mat',
                            'num_joints': len(cfg.feat_list),
                            'all_joints': all_joints,
                            'all_joints_names': sorted(cfg.feat_list),
                            'init_weights': model_path,
                            'metadataset': bf + '/' + filename_pickle + '.pickle',
                            'project_path': mainDirectory,
                            'mirror': mirror,
                            'net_type': cfg.net_type,
                            'intermediate_supervision': intermediate_supervision}

            trainingdata = MakeTrain_pose_yaml(items2change,
                                               os.path.abspath(experimentname + '/train/' + 'pose_cfg.yaml'),
                                               filename=os.path.abspath(
                                                   os.path.dirname(deeplabcut.__file__) + '/pose_cfg.yaml'))

    os.chdir(originalDirectory)
    return


class NoManualLabelsError(Exception):
    pass
