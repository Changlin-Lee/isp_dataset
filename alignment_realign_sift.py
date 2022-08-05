import numpy as np
import cv2
import os
import argparse
from glob import glob
from scipy.stats import stats
import rawpy
from scipy.misc import imread
from imageio import imwrite

MAX_FEATURES = 10000
GOOD_MATCH_PERCENT = 0.15


parser = argparse.ArgumentParser(description="training codes")
parser.add_argument("--data_path", type=str, default=data_path,
                    help="Path to unrectified images.")
parser.add_argument("--write_path", type=str, default= write_path, help= "Path to save cropped images")

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

    print("num of kp1: ", len(kp1))
    print("num of kp2: ", len(kp2))

    if len(kp1) < 2 or len(kp2) < 2:
        print("Not enough key points!")
        return 

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    # print(len(good))

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        # print("find homography done!")
        if M is None:
            print("No appropriate homography matrix!")
            return 

        matchesMask = mask.ravel().tolist()

        height, width, channels = im2.shape
        im1Reg = cv2.warpPerspective(im1, M, (width, height))


    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
        return

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

    img3 = cv2.drawMatches(im1Gray,kp1,im2Gray,kp2,good,None,**draw_params)

    return im1Reg, img3


if __name__ == '__main__':
    rgb_folders = sorted(glob(args.write_path+"alignment_128_flow/%s/%s/rgb/DSC*"%(args.camera,args.date)))
    raw_folders = sorted(glob(args.write_path+"alignment_128_flow/%s/%s/raw/DSC*"%(args.camera,args.date)))

    if not os.path.isdir(args.write_path+"alignment_128_flow_sift_refine/%s/"%args.camera):
        os.makedirs(args.write_path+"alignment_128_flow_sift_refine/%s/"%args.camera)
    if not os.path.isdir(args.write_path+"alignment_128_flow_sift_refine/%s/%s/"%(args.camera, args.date)):
        os.makedirs(args.write_path+"alignment_128_flow_sift_refine/%s/%s/"%(args.camera, args.date))
    if not os.path.isdir(args.write_path+"alignment_128_flow_sift_refine/%s/%s/raw"%(args.camera, args.date)):
        os.makedirs(args.write_path+"alignment_128_flow_sift_refine/%s/%s/raw"%(args.camera, args.date))
    if not os.path.isdir(args.write_path+"alignment_128_flow_sift_refine/%s/%s/rgb"%(args.camera, args.date)):
        os.makedirs(args.write_path+"alignment_128_flow_sift_refine/%s/%s/rgb"%(args.camera, args.date))
    if not os.path.isdir(args.write_path+"alignment_128_flow_sift_refine/%s/%s/match"%(args.camera, args.date)):
        os.makedirs(args.write_path+"alignment_128_flow_sift_refine/%s/%s/match"%(args.camera, args.date))

    for i in range(0, len(rgb_folders)):
        input_patch_list = sorted(glob(rgb_folders[i]+"/*_input.jpg"))
        gt_patch_list = sorted(glob(rgb_folders[i]+"/*_gt.jpg"))
        raw_patch_list = sorted(glob(raw_folders[i]+"/*.npy"))

        folder_name = rgb_folders[i].split("/")[-1]

        if not os.path.isdir(args.write_path+"alignment_128_flow_sift_refine/%s/%s/rgb/"%(args.camera, args.date)+folder_name):
            os.makedirs(args.write_path+"alignment_128_flow_sift_refine/%s/%s/rgb/"%(args.camera, args.date)+folder_name)
        if not os.path.isdir(args.write_path+"alignment_128_flow_sift_refine/%s/%s/raw/"%(args.camera, args.date)+folder_name):
            os.makedirs(args.write_path+"alignment_128_flow_sift_refine/%s/%s/raw/"%(args.camera, args.date)+folder_name)
        if not os.path.isdir(args.write_path+"alignment_128_flow_sift_refine/%s/%s/match/"%(args.camera, args.date)+folder_name):
            os.makedirs(args.write_path+"alignment_128_flow_sift_refine/%s/%s/match/"%(args.camera, args.date)+folder_name)

        for j in range(0, len(input_patch_list)):
            print("start " + input_patch_list[j])
            input_name = input_patch_list[j].split("/")[-1].split(".")[0]
            gt_name = gt_patch_list[j].split("/")[-1].split(".")[0]
            gt_num = gt_name.split("_")[0]
            raw_name = raw_patch_list[j].split("/")[-1].split(".")[0]

            input_patch = cv2.imread(input_patch_list[j])
            gt_patch = cv2.imread(gt_patch_list[j])
            raw_patch = np.load(raw_patch_list[j])

            if alignImages_sift(gt_patch, input_patch) is None:
                continue
            else:
                gt_patch_realign, match = alignImages_sift(gt_patch, input_patch)

                cv2.imwrite(args.write_path+"alignment_128_flow_sift_refine/%s/%s/rgb/"%(args.camera, args.date)+folder_name+"/"+input_name+".jpg", input_patch)
                cv2.imwrite(args.write_path+"alignment_128_flow_sift_refine/%s/%s/rgb/"%(args.camera, args.date)+folder_name+"/"+gt_name+".jpg", gt_patch_realign)
                cv2.imwrite(args.write_path+"alignment_128_flow_sift_refine/%s/%s/match/"%(args.camera, args.date)+folder_name+"/"+gt_num+"_match.jpg", match)
                np.save(args.write_path+"alignment_128_flow_sift_refine/%s/%s/raw/"%(args.camera, args.date)+folder_name+"/"+raw_name+".npy", raw_patch)
                    
                print(args.write_path+"alignment_128_flow_sift_refine/%s/%s/rgb/"%(args.camera, args.date)+folder_name+"/"+input_name+".jpg is done!")






