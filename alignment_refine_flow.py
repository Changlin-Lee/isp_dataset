import sys;
sys.path.append("/path/to/flow")
import models
import numpy as np
import cv2
import os
import argparse
from glob import glob
from scipy.stats import stats
from scipy.misc import imread
import imageio
import scipy.misc
import torch
from math import ceil
from torch.autograd import Variable

MAX_FEATURES = 10000
GOOD_MATCH_PERCENT = 0.15

parser = argparse.ArgumentParser(description="training codes")

parser.add_argument("--data_path", type=str, default=data_path,
                    help="Path to unrectified images.")

parser.add_argument("--camera", type=str, default="iPhone", choices=["iPhone", "Mi"], 
                    help="Path to unrectified images.")

parser.add_argument("--date", type=str,
                    help="Which date to align.")

args = parser.parse_args()

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))


def cal_mean_distance(flow):
    # flow: (H, W, 2)
    mean_dis = np.mean(np.sqrt(np.square(flow[:,:,0]) + np.square(flow[:,:,1])))

    return mean_dis


def flow_refine(im1, im2):
    # generate the optical flow from im1 to im2 then calculate the mean distance

    # ---- generate optical flow ----
    pwc_model_fn = './PWC-Net/PyTorch/pwc_net.pth.tar';
    im_all = [img for img in [im1, im2]]
    im_all = [im[:, :, :3] for im in im_all]

    # rescale the image size to be multiples of 64
    divisor = 64.
    H = im_all[0].shape[0]
    W = im_all[0].shape[1]

    H_ = int(ceil(H/divisor) * divisor)
    W_ = int(ceil(W/divisor) * divisor)
    for i in range(len(im_all)):
        im_all[i] = cv2.resize(im_all[i], (W_, H_))

    for _i, _inputs in enumerate(im_all):
        im_all[_i] = im_all[_i][:, :, ::-1]
        im_all[_i] = 1.0 * im_all[_i]/255.0
        
        im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
        im_all[_i] = torch.from_numpy(im_all[_i])
        im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2]) 
        im_all[_i] = im_all[_i].float()
        
    im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)

    net = models.pwc_dc_net(pwc_model_fn)
    net = net.cuda()
    net.eval()

    flo = net(im_all)
    flo = flo[0] * 20.0
    flo = flo.cpu().data.numpy()

    # scale the flow back to the input size
    flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
    u_ = cv2.resize(flo[:,:,0],(W,H))
    v_ = cv2.resize(flo[:,:,1],(W,H))
    u_ *= W/ float(W_)
    v_ *= H/ float(H_)
    flo = np.dstack((u_,v_))

    # ---- calculate mean distance ----
    mean_dis = cal_mean_distance(flo)
    return mean_dis



if __name__ == '__main__':
    rgb_folders = sorted(glob(args.data_path+"alignment_128/%s/%s/rgb/DSC*"%(args.camera,args.date)))
    raw_folders = sorted(glob(args.data_path+"alignment_128/%s/%s/raw/DSC*"%(args.camera,args.date)))

    if not os.path.isdir(args.data_path+"alignment_128_flow/%s/"%args.camera):
        os.makedirs(args.data_path+"alignment_128_flow/%s/"%args.camera)
    if not os.path.isdir(args.data_path+"alignment_128_flow/%s/%s/"%(args.camera, args.date)):
        os.makedirs(args.data_path+"alignment_128_flow/%s/%s/"%(args.camera, args.date))
    if not os.path.isdir(args.data_path+"alignment_128_flow/%s/%s/raw"%(args.camera, args.date)):
        os.makedirs(args.data_path+"alignment_128_flow/%s/%s/raw"%(args.camera, args.date))
    if not os.path.isdir(args.data_path+"alignment_128_flow/%s/%s/rgb"%(args.camera, args.date)):
        os.makedirs(args.data_path+"alignment_128_flow/%s/%s/rgb"%(args.camera, args.date))

    for i in range(len(rgb_folders)):
        input_patch_list = sorted(glob(rgb_folders[i]+"/*_input.jpg"))
        gt_patch_list = sorted(glob(rgb_folders[i]+"/*_gt.jpg"))
        raw_patch_list = sorted(glob(raw_folders[i]+"/*.npy"))

        folder_name = rgb_folders[i].split("/")[-1] # DSC_****

        if not os.path.isdir(args.data_path+"alignment_128_flow/%s/%s/rgb/"%(args.camera, args.date)+folder_name):
            os.makedirs(args.data_path+"alignment_128_flow/%s/%s/rgb/"%(args.camera, args.date)+folder_name)
        if not os.path.isdir(args.data_path+"alignment_128_flow/%s/%s/raw/"%(args.camera, args.date)+folder_name):
            os.makedirs(args.data_path+"alignment_128_flow/%s/%s/raw/"%(args.camera, args.date)+folder_name)

        for j in range(len(input_patch_list)):

            print("start " + input_patch_list[j])
            input_name = input_patch_list[j].split("/")[-1].split(".")[0]
            gt_name = gt_patch_list[j].split("/")[-1].split(".")[0]
            gt_num = gt_name.split("_")[0]
            raw_name = raw_patch_list[j].split("/")[-1].split(".")[0]

            input_patch = cv2.imread(input_patch_list[j])
            gt_patch = cv2.imread(gt_patch_list[j])
            raw_patch = np.load(raw_patch_list[j])

            retval = flow_refine(input_patch, gt_patch)

            if retval is not None:
                print(retval)
                if retval <= 1.2:
                    cv2.imwrite(args.data_path+"alignment_128_flow/%s/%s/rgb/"%(args.camera, args.date)+folder_name+"/"+input_name+".jpg", input_patch)
                    cv2.imwrite(args.data_path+"alignment_128_flow/%s/%s/rgb/"%(args.camera, args.date)+folder_name+"/"+gt_name+".jpg", gt_patch)
                    np.save(args.data_path+"alignment_128_flow/%s/%s/raw/"%(args.camera, args.date)+folder_name+"/"+raw_name+".npy", raw_patch)

                print(args.data_path+"alignment_128_flow/%s/%s/rgb/"%(args.camera, args.date)+folder_name+"/"+input_name+".jpg is done!")