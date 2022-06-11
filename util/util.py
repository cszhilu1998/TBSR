"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
# from PIL import Image
import os
import time
from functools import wraps
import torch
import random
import numpy as np
import cv2
import torch
# import colour_demosaicing
import glob

# 修饰函数，重新尝试600次，每次间隔1秒钟
# 能对func本身处理，缺点在于无法查看func本身的提示
def loop_until_success(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        for i in range(600):
            try:
                ret = func(*args, **kwargs)
                break
            except OSError:
                time.sleep(1)
        return ret
    return wrapper

# 修改后的print函数及torch.save函数示例
@loop_until_success
def loop_print(*args, **kwargs):
    print(*args, **kwargs)

@loop_until_success
def torch_save(*args, **kwargs):
    torch.save(*args, **kwargs)

def calc_psnr(sr, hr, range=255.):
    # shave = 2
    with torch.no_grad():
        diff = (sr - hr) / range
        # diff = diff[:, :, shave:-shave, shave:-shave]
        mse = torch.pow(diff, 2).mean()
        return (-10 * torch.log10(mse)).item()

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)

def print_numpy(x, val=True, shp=True):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, mid = %3.3f, std=%3.3f'
              % (np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def prompt(s, width=66):
    print('='*(width+4))
    ss = s.split('\n')
    if len(ss) == 1 and len(s) <= width:
        print('= ' + s.center(width) + ' =')
    else:
        for s in ss:
            for i in split_str(s, width):
                print('= ' + i.ljust(width) + ' =')
    print('='*(width+4))

def split_str(s, width):
    ss = []
    while len(s) > width:
        idx = s.rfind(' ', 0, width+1)
        if idx > width >> 1:
            ss.append(s[:idx])
            s = s[idx+1:]
        else:
            ss.append(s[:width])
            s = s[width:]
    if s.strip() != '':
        ss.append(s)
    return ss

def augment_func(img, hflip, vflip, rot90):  # CxHxW
    if hflip:   img = img[:, :, ::-1]
    if vflip:   img = img[:, ::-1, :]
    if rot90:   img = img.transpose(0, 2, 1)
    return np.ascontiguousarray(img)

def augment(*imgs):  # CxHxW
    hflip = random.random() < 0.5
    vflip = random.random() < 0.5
    rot90 = random.random() < 0.5
    return (augment_func(img, hflip, vflip, rot90) for img in imgs)

def remove_black_level(img, black_lv=63, white_lv=4*255):
    img = np.maximum(img.astype(np.float32)-black_lv, 0) / (white_lv-black_lv)
    return img

def gamma_correction(img, r=1/2.2):
    img = np.maximum(img, 0)
    img = np.power(img, r)
    return img

def extract_bayer_channels(raw):  # HxW
    ch_R  = raw[0::2, 0::2]
    ch_Gb = raw[0::2, 1::2]
    ch_Gr = raw[1::2, 0::2]
    ch_B  = raw[1::2, 1::2]
    raw_combined = np.dstack((ch_B, ch_Gb, ch_R, ch_Gr))
    raw_combined = np.ascontiguousarray(raw_combined.transpose((2, 0, 1)))
    return raw_combined  # 4xHxW

# def get_raw_demosaic(raw, pattern='RGGB'):  # HxW
#     raw_demosaic = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(raw, pattern=pattern)
#     raw_demosaic = np.ascontiguousarray(raw_demosaic.astype(np.float32).transpose((2, 0, 1)))
#     return raw_demosaic  # 3xHxW

def get_coord(H, W, x=448/3968, y=448/2976):
    x_coord = np.linspace(-x + (x / W), x - (x / W), W)
    x_coord = np.expand_dims(x_coord, axis=0)
    x_coord = np.tile(x_coord, (H, 1))
    x_coord = np.expand_dims(x_coord, axis=0)

    y_coord = np.linspace(-y + (y / H), y - (y / H), H)
    y_coord = np.expand_dims(y_coord, axis=1)
    y_coord = np.tile(y_coord, (1, W))
    y_coord = np.expand_dims(y_coord, axis=0)

    coord = np.ascontiguousarray(np.concatenate([x_coord, y_coord]))
    coord = np.float32(coord)

    return coord

def read_wb(txtfile, key):
    wb = np.zeros((1,4))
    with open(txtfile) as f:
        for l in f:
            if key in l:
                for i in range(wb.shape[0]):
                    nextline = next(f)
                    try:
                        wb[i,:] = nextline.split()
                    except:
                        print("WB error XXXXXXX")
                        print(txtfile)
    wb = wb.astype(np.float32)
    return wb

# ### CHECK
# def compute_wb(raw_path):
#     print("Computing WB for %s"%(raw_path))
#     bayer = rawpy.imread(raw_path)
#     rgb_nowb = bayer.postprocess(gamma=(1, 1),
#         no_auto_bright=True,
#         use_camera_wb=False,
#         output_bps=16)

#     rgb_wb = bayer.postprocess(gamma=(1, 1),
#         no_auto_bright=True,
#         use_camera_wb=True,
#         output_bps=16)

#     scale=[np.mean(rgb_wb[...,0])/np.mean(rgb_nowb[...,0]), 
#         np.mean(rgb_wb[...,1])/np.mean(rgb_nowb[...,1]),
#         np.mean(rgb_wb[...,1])/np.mean(rgb_nowb[...,1]),
#         np.mean(rgb_wb[...,2])/np.mean(rgb_nowb[...,2])]
#     wb = np.zeros((1,4))
#     wb[0,0] = scale[0]
#     wb[0,1] = scale[1]
#     wb[0,2] = scale[2]
#     wb[0,3] = scale[3]
#     return wb

