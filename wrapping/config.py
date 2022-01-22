working_directory = 'data'
date = '20210511'
task = 'table_4'
#folderPath_ccv = '/ccv_files/'+date+'_'+task # folder in which the ccv-files need to be placed
#filePath_labels = '/manual_labels/'+date+'/'+task+'/labels.npz'
folderPath_ccv = f'/media/nfs/bbo3102/storage/bulk/pose_B.EG.1.09/experiments/{date}_{task}/' # folder in which the ccv-files need to be placed
filePath_labels = f'/users/voit/Dropbox/Dropbox (NIG)/public_share/ACM/datasets_figures/required_files/{date}/{task}/labels.npz'

mask_para = list() # will be overwritten 
mask_para_offset = 0 # will be overwritten 
                
if (date == '20210511'):
    frame_rate = 200
    dFrame = 200
    if (task == 'table_1'):
        index_frames = list([[6900, 44300],])
        index_frames_save = list([[6900, 184000]])
    elif (task == 'table_2'):
        index_frames = list([[5600, 10600],
                             [12600, 62200],])
        #index_frames_save = list([[5600, 183000]])
        index_frames_save = list([[73000, 75000],
                                  [78000, 79700],
                                  [81600, 82400],
                                  [112400, 113600],
                                  [114800, 115700],
                                  [117900, 119500],])
    elif (task == 'table_3'):
        index_frames = list([[4000, 38000],
                             [50800, 66000],])
        index_frames_save = list([[76900, 78800],
                                  [88900, 89600],
                                  [93900, 94700],
                                  [98100, 99100],
                                  [106000, 106600],
                                  [128700, 129800],])
    elif (task == 'table_4'):
        index_frames = list([[6400, 17000],
                             [18000, 23000],
                             [26000, 61800],
                             [62900, 67100]])
        #index_frames_save = list([[6400, 195000]])
        index_frames_save = list([[93600, 94800],
                                  [102200, 103400],
                                  [122400, 123400],
                                  [127800, 128400],
                                  [135500, 137200,],])
    #
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

net_type = 'resnet_152' # Type of networks. Currently resnet_50, resnet_101, resnet_152, mobilenet_v2_1.0, mobilenet_v2_0.75, mobilenet_v2_0.5, and mobilenet_v2_0.35 are supported.
xRes = 1280 # no need to change
yRes = 1024 # no need to change
dxy = 300
nFrames_background = 100
noise_threshold = 5.0 # is mutliplied by the background std to check if pixel is above noise level
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
                  'spot_toe_right_003']) # hard coded

scorer = 'monsees' # do not change
Shuffles = [1] # do not change
TrainingFraction = [1.00] # do not change
iteration = 0 # do not change
