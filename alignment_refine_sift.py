import numpy as np
import cv2
import os
import argparse
from glob import glob
from scipy.stats import stats
import rawpy
from scipy.misc import imread

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


def cal_mean_distance(kp1, kp2):
    # kp1, kp2: (N, 2)
    kp_dis = np.mean(np.sqrt(np.sum(np.square(kp1 - kp2), axis=1)))

    return kp_dis


def alignImages_sift(im1, im2):
    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    MIN_MATCH_COUNT = 10

    sift = cv2.xfeatures2d.SIFT_create(MAX_FEATURES)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1Gray,None)
    kp2, des2 = sift.detectAndCompute(im2Gray,None)

     # if len(kp1) <= 30 or len(kp2) <= 30:
     #     return

    # ------start to match-----

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               matchesMask = None, # draw only inliers
               flags = 2)

    if len(kp1) >=30 and len(kp2) >= 30:
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        if len(good) > 0:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])

            mean_dis = cal_mean_distance(src_pts, dst_pts)

            img3 = cv2.drawMatches(im1Gray,kp1,im2Gray,kp2,good,None,**draw_params)

            return mean_dis, img3

    # return im1Reg, img3

def sift_refine(im1, im2):

    im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    MIN_MATCH_COUNT = 10

    sift = cv2.xfeatures2d.SIFT_create(MAX_FEATURES)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1Gray, None)
    kp2, des2 = sift.detectAndCompute(im2Gray, None)
    print(max(len(kp1),len(kp2)))
    #40 for 128*128
    if len(kp1) <=40  and len(kp2) <= 40:
        return None
    else:
        return True

if __name__ == '__main__':
    rgb_folders = sorted(glob(args.data_path+"alignment_128_flow/%s/%s/rgb/DSC*"%(args.camera,args.date)))
    raw_folders = sorted(glob(args.data_path+"alignment_128_flow/%s/%s/raw/DSC*"%(args.camera,args.date)))

    if not os.path.isdir(args.data_path+"alignment_128_flow_sift_refine/%s/"%args.camera):
        os.makedirs(args.data_path+"alignment_128_flow_sift_refine/%s/"%args.camera)
    if not os.path.isdir(args.data_path+"alignment_128_flow_sift_refine/%s/%s/"%(args.camera, args.date)):
        os.makedirs(args.data_path+"alignment_128_flow_sift_refine/%s/%s/"%(args.camera, args.date))
    if not os.path.isdir(args.data_path+"alignment_128_flow_sift_refine/%s/%s/raw"%(args.camera, args.date)):
        os.makedirs(args.data_path+"alignment_128_flow_sift_refine/%s/%s/raw"%(args.camera, args.date))
    if not os.path.isdir(args.data_path+"alignment_128_flow_sift_refine/%s/%s/rgb"%(args.camera, args.date)):
        os.makedirs(args.data_path+"alignment_128_flow_sift_refine/%s/%s/rgb"%(args.camera, args.date))

    for i in range(len(rgb_folders)):
        input_patch_list = sorted(glob(rgb_folders[i]+"/*_input.jpg"))
        gt_patch_list = sorted(glob(rgb_folders[i]+"/*_gt.jpg"))
        raw_patch_list = sorted(glob(raw_folders[i]+"/*.npy"))

        folder_name = rgb_folders[i].split("/")[-1] # DSC_****

        if not os.path.isdir(args.data_path+"alignment_128_flow_sift_refine/%s/%s/rgb/"%(args.camera, args.date)+folder_name):
            os.makedirs(args.data_path+"alignment_128_flow_sift_refine/%s/%s/rgb/"%(args.camera, args.date)+folder_name)
        if not os.path.isdir(args.data_path+"alignment_128_flow_sift_refine/%s/%s/raw/"%(args.camera, args.date)+folder_name):
            os.makedirs(args.data_path+"alignment_128_flow_sift_refine/%s/%s/raw/"%(args.camera, args.date)+folder_name)
        #if not os.path.isdir(args.data_path+"alignment_128_flow_sift_refine/%s/%s/match/"%(args.camera, args.date)+folder_name):
        #    os.makedirs(args.data_path+"alignment_128_flow_sift_refine/%s/%s/match/"%(args.camera, args.date)+folder_name)

        for j in range(len(input_patch_list)):
            print("start " + input_patch_list[j])
            input_name = input_patch_list[j].split("/")[-1].split(".")[0]
            gt_name = gt_patch_list[j].split("/")[-1].split(".")[0]
            gt_num = gt_name.split("_")[0]
            raw_name = raw_patch_list[j].split("/")[-1].split(".")[0]

            input_patch = cv2.imread(input_patch_list[j])
            gt_patch = cv2.imread(gt_patch_list[j])
            raw_patch = np.load(raw_patch_list[j])

            retval = sift_refine(input_patch, gt_patch)

            if retval is not None:
                cv2.imwrite(args.data_path+"alignment_128_flow_sift_refine/%s/%s/rgb/"%(args.camera, args.date)+folder_name+"/"+input_name+".jpg", input_patch)
                cv2.imwrite(args.data_path+"alignment_128_flow_sift_refine/%s/%s/rgb/"%(args.camera, args.date)+folder_name+"/"+gt_name+".jpg", gt_patch)
                #cv2.imwrite(args.data_path+"alignment_128_flow_sift_refine/%s/%s/match/"%(args.camera, args.date)+folder_name+"/"+gt_num+"_match.jpg", retval[1])
                np.save(args.data_path+"alignment_128_flow_sift_refine/%s/%s/raw/"%(args.camera, args.date)+folder_name+"/"+raw_name+".npy", raw_patch)

            print(args.data_path+"alignment_128_flow_sift_refine/%s/%s/rgb/"%(args.camera, args.date)+folder_name+"/"+input_name+".jpg is done!")
