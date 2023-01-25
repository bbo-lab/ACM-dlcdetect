import numpy as np
import os
import sys

import imageio
from ccvtools import rawio

import dlcdetectConfig as cfg


# used within generate_training_data.py to generate a training data set
def calc_background(i_reader,
                    x_res, y_res,
                    n_frames_background):
    mean_x = np.zeros(i_reader.get_data(0).shape)
    mean_x2 = np.zeros(i_reader.get_data(0).shape)

    for i_frame in range(1, n_frames_background):
        img = i_reader.get_data(i_frame)
        img = img.astype(np.float64)
        mean_x += img
        mean_x2 += img ** 2
    background = mean_x / n_frames_background
    background_var = mean_x2 / n_frames_background - background ** 2
    mask = (background_var < 0.0)
    if np.any(abs(background_var[mask]) >= 2 ** -23):
        print('ERROR: background_var has inconsistent values')
        raise
    background_var[mask] = 0.0
    background_std = np.sqrt(background_var)
    return background, background_std


def calc_backgrounds(file_list,
                     x_res, y_res,
                     n_frames_background):
    n_cams = np.size(file_list)

    reader = imageio.get_reader(file_list[0])
    img = reader.get_data(0)

    backgrounds = np.zeros((n_cams, )+img.shape,
                           dtype=np.float64)
    backgrounds_std = np.zeros((n_cams, )+img.shape,
                               dtype=np.float64)
    for i_cam in range(n_cams):
        reader = imageio.get_reader(file_list[i_cam])
        backgrounds[i_cam], backgrounds_std[i_cam] = calc_background(reader,
                                                                     x_res, y_res,
                                                                     n_frames_background)
    return backgrounds, backgrounds_std


def crop_image(i_reader, i_frame,
               i_cam, mask_para, mask_para_offset,
               i_background, i_background_std, noise_threshold,
               dxy, img_crop, pixel_crop):
    img_crop_shape = np.shape(img_crop)

    img0 = i_reader.get_data(i_frame - round(0.125 * cfg.frame_rate))  # 125ms delay

    img = i_reader.get_data(i_frame)
    img_bg = img.astype(np.float64) - img0.astype(np.float64)
    y_res, x_res = np.shape(img_bg)[0:2]

    mask_bg = (np.abs(img_bg) <= (noise_threshold * i_background_std))
    img_mask = np.copy(img)
    img_mask[mask_bg] = 0

    #
    if len(mask_para) > 0:
        mask = np.zeros_like(img_mask, dtype=bool)
        index_x = np.arange(x_res, dtype=np.int64)
        index_y = np.arange(y_res, dtype=np.int64)
        X, Y = np.meshgrid(index_x, index_y)
        n_mask = len(mask_para[i_cam])
        for i_mask in range(n_mask):
            x1 = mask_para[i_cam][i_mask][0]
            y1 = mask_para[i_cam][i_mask][1]
            x2 = mask_para[i_cam][i_mask][2]
            y2 = mask_para[i_cam][i_mask][3]
            up_down = mask_para[i_cam][i_mask][4]
            #
            m = (y2 - y1) / (x2 - x1)
            n = y1 - m * x1
            val = m * X + n
            if up_down == 'up':
                mask = (Y <= val - mask_para_offset)
            elif up_down == 'down':
                mask = (Y >= val + mask_para_offset)
            else:
                print('ERROR: provide correct masking parameters')
                raise
            img_mask[mask] = 0.0
    #

    #     # in case I use the background substracted image
    #     const_bg = np.max(i_background_std)
    #     img_bg = img_bg + const_bg # add constant to not lose information for values below 0
    #     img_bg = img_bg.astype(np.uint8)

    index = np.where(img_mask > 0)
    n = np.shape(index)[1]
    index = np.concatenate([index[1], index[0]]).reshape(2, n).T
    center_xy = np.median(index, 0).astype(np.int64)
    #     center_xy = np.mean(index, 0).astype(np.int64)

    xlim_min = center_xy[0] - dxy
    xlim_max = center_xy[0] + dxy
    ylim_min = center_xy[1] - dxy
    ylim_max = center_xy[1] + dxy
    xlim_min_use = np.max([xlim_min, 0])
    xlim_max_use = np.min([xlim_max, x_res])
    ylim_min_use = np.max([ylim_min, 0])
    ylim_max_use = np.min([ylim_max, y_res])
    dx = np.int64(xlim_max_use - xlim_min_use)
    dy = np.int64(ylim_max_use - ylim_min_use)

    center_xy_use = center_xy - np.array([xlim_min_use, ylim_min_use])
    center_xy_use = dxy - center_xy_use
    dx_add = center_xy_use[0]
    dy_add = center_xy_use[1]
    dx_add_int = np.int64(dx_add)
    dy_add_int = np.int64(dy_add)

    # using the plain image
    img_crop.fill(0)
    img_crop[dy_add_int:dy + dy_add_int, dx_add_int:dx + dx_add_int] = \
        img[ylim_min_use:ylim_max_use, xlim_min_use:xlim_max_use]
    #     img_crop[dy_add_int:dy+dy_add_int, dx_add_int:dx+dx_add_int] = \
    #         img_bg[ylim_min_use:ylim_max_use, xlim_min_use:xlim_max_use]
    # FIXME: this should save the float values of dx_add / dy_add! (is this the case though?)
    pixel_crop[0] = -xlim_min_use + dx_add_int
    pixel_crop[1] = -ylim_min_use + dy_add_int
    return
