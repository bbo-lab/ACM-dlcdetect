import os

# Directory where output is placed 
# Default: ./data relative to this config
working_directory = f'{os.path.dirname(os.path.abspath(__file__))}/data'

# Folder in which the video files need to be placed
folderPath_video = f'/media/nfs/bbo3102/storage/bulk/pose_B.EG.1.09/experiments/20210511_table_4/' 
# Path of the file with manual labels for training
filePath_labels = f'/users/voit/Dropbox/Dropbox (NIG)/public_share/ACM/datasets_figures/required_files/20210511/table_4/labels.npz'

# Video parameters (different parameters per video currently unsupported)
xRes = 1280
yRes = 1024
frame_rate = 200

# Training frame ranges (List of start and end frames, e.g. [[1 1000],[2001 3000]]). 
# Default: [], all frames in filePath_labels are used.
index_frames = []

# Frame ranges to save labels for.
# Example: [[1 1000],[2001 3000]])
index_frames_save = list([[93600, 94800],
                            [102200, 103400],
                            [122400, 123400],
                            [127800, 128400],
                            [135500, 137200,],])

# Processing area: +/-dxy pixels around center of nonmasked area are processed
dxy = 300

# Masks to apply to an image. Each entry consists of two image coordinates x1/y1/x2/y2 and 'up'/'down' keywords.
# From these, a line through the two points is constructed and the area above ('up') or below ('down') is masked.
# The outer list corresponds to the video files from folderPath_video in alphabetical order. The inner lists are 
# lists of lines.
# Example: mask_para = [ [], [[0, 924, 100, 1024, 'down'],[1180, 1024, 1280, 924, 'down']], [], [] ]
#                        V1  V2                                                             V3  V4
#                             L1                          L2
#    masks the bottom corners in the 2nd of 4 videos
mask_para = list([
                    [[15, 553, 772, 0, 'up'],
                    [809, 0, 1249, 626, 'up'],
                    [11, 565, 579, 978, 'down'],
                    [580, 979, 1249, 626, 'down']],
                    [[9, 491, 604, 0, 'up'],
                    [676, 12, 1240, 603, 'up'],
                    [4, 522, 609, 912, 'down'],
                    [608, 913, 1088, 700, 'down']],
                    [[122, 501, 932, 111, 'up'], 
                    [945, 119, 1254, 714, 'up'],
                    [116, 514, 582, 942, 'down'],
                    [585, 943, 1252, 730, 'down']],
                    [[155, 700, 387, 127, 'up'], 
                    [428, 135, 1250, 439, 'up'], 
                    [156, 712, 826, 880, 'down'],
                    [834, 884, 1259, 455, 'down']],
                    ]) # needs to be set manually (x1, y1, x2, y2, 'up'/'down')
mask_para_offset = 0

# DLC network type
# Currently resnet_50, resnet_101, resnet_152, mobilenet_v2_1.0, mobilenet_v2_0.75, mobilenet_v2_0.5, and mobilenet_v2_0.35 are supported.
# Default 'resnet_152' 
net_type = 'resnet_152' 

# Number of frames from the beginning of each video to calculate a background and background_std from
nFrames_background = 100

# Faktor multiplied by the background std to calc threshold for checking if interframe pixel difference is above noise level
noise_threshold = 5.0 


# =========== Do not change anything below ===========

feat_list = list(['spot_ankle_left',
                  'spot_ankle_right',
                  'spot_elbow_left',
                  'spot_elbow_right',
                  'spot_finger_left_001',
                  'spot_finger_left_002',
                  'spot_finger_left_003',
                  'spot_finger_right_001',
                  'spot_finger_right_002',
                  'spot_finger_right_003',
                  'spot_head_001',
                  'spot_head_002',
                  'spot_head_003',
                  'spot_hip_left',
                  'spot_hip_right',
                  'spot_knee_left',
                  'spot_knee_right',
                  'spot_paw_front_left',
                  'spot_paw_front_right',
                  'spot_paw_hind_left',
                  'spot_paw_hind_right',
                  'spot_shoulder_left',
                  'spot_shoulder_right',
                  'spot_side_left',
                  'spot_side_right',
                  'spot_spine_001',
                  'spot_spine_002',
                  'spot_spine_003', 
                  'spot_spine_004',
                  'spot_spine_005',
                  'spot_spine_006',
                  'spot_tail_001',
                  'spot_tail_002',
                  'spot_tail_003',
                  'spot_tail_004', 
                  'spot_tail_005',
                  'spot_tail_006',
                  'spot_toe_left_001',
                  'spot_toe_left_002',
                  'spot_toe_left_003',
                  'spot_toe_right_001',
                  'spot_toe_right_002',
                  'spot_toe_right_003']) # hard coded - why are these not read from the manual labels file?

scorer = 'monsees' 
Shuffles = [1] # do not change
TrainingFraction = [1.00] # do not change
iteration = 0 # do not change
