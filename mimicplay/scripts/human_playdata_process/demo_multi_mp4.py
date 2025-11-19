from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import json
import cv2
import torch
import shutil, copy
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from glob import glob
from IPython import embed

import torchvision.transforms as transforms
import torchvision.datasets as dset
# from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, \
    vis_detections_filtered_objects_PIL, vis_detections_filtered_objects, calculate_center  # (1) here add a function to viz
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb
import h5py
import moviepy.editor as mpy
from moviepy.editor import *
import matplotlib.pyplot as plt

try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Train a Fast R-CNN network")
    parser.add_argument("--left_video_dir",
                        help="directory of left view videos",
                        required=True, type=str)
    parser.add_argument("--right_video_dir",
                        help="directory of right view videos",
                        required=True, type=str)
    parser.add_argument("--dataset", dest="dataset",
                        help="training dataset",
                        default="pascal_voc", type=str)
    parser.add_argument("--hdf5_path",
                        help="h5py file path",
                        default="demo_hand_loc_0.hdf5", type=str)
    parser.add_argument("--target_hdf5_path",
                        help="target h5py file path",
                        default="human_play.hdf5", type=str)
    parser.add_argument("--hdf5_read_mode",
                        help="online / offline",
                        default="online", type=str)
    parser.add_argument("--cfg", dest="cfg_file",
                        help="optional config file",
                        default="cfgs/res101.yml", type=str)
    parser.add_argument("--net", dest="net",
                        help="vgg16, res50, res101, res152",
                        default="res101", type=str)
    parser.add_argument("--set", dest="set_cfgs",
                        help="set config keys", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--load_dir", dest="load_dir",
                        help="directory to load models",
                        default="models")
    parser.add_argument("--image_dir", dest="image_dir",
                        help="directory to load images for demo",
                        default="images")
    parser.add_argument("--save_dir", dest="save_dir",
                        help="directory to save results",
                        default="images_det")
    parser.add_argument("--cuda", dest="cuda",
                        help="whether use CUDA",
                        action="store_true")
    parser.add_argument("--mGPUs", dest="mGPUs",
                        help="whether use multiple GPUs",
                        action="store_true")
    parser.add_argument("--cag", dest="class_agnostic",
                        help="whether perform class_agnostic bbox regression",
                        action="store_true")
    parser.add_argument("--parallel_type", dest="parallel_type",
                        help="which part of model to parallel, 0: all, 1: model before roi pooling",
                        default=0, type=int)
    parser.add_argument("--checksession", dest="checksession",
                        help="checksession to load model",
                        default=1, type=int)
    parser.add_argument("--checkepoch", dest="checkepoch",
                        help="checkepoch to load network",
                        default=8, type=int)
    parser.add_argument("--checkpoint", dest="checkpoint",
                        help="checkpoint to load network",
                        default=132028, type=int)
    parser.add_argument("--bs", dest="batch_size",
                        help="batch_size",
                        default=1, type=int)
    parser.add_argument("--vis", dest="vis",
                        help="visualization mode",
                        default=True)
    parser.add_argument("--webcam_num", dest="webcam_num",
                        help="webcam ID number",
                        default=-1, type=int)
    parser.add_argument("--thresh_hand",
                        type=float, default=0.5,
                        required=False)
    parser.add_argument("--thresh_obj", default=0.5,
                        type=float,
                        required=False)
    parser.add_argument("--downsample_rate", default=1,
                        type=int, help="set to n: sample 1 frame every n frames")

    args = parser.parse_args()
    return args


def _get_image_blob(im):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def extractImages(pathIn):
    count = 0
    frame_list = []
    vidcap = cv2.VideoCapture(pathIn)
    success, image = vidcap.read()
    success = True
    while success:
        # vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))    # added this line
        success, image = vidcap.read()
        # print ("Read a new frame: ", success)
        if success:
            frame_list.append(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            # cv2.imwrite( pathOut + "\\frame%d.jpg" % count, image)     # save frame as JPEG file
            count = count + 1
    print ("Read {} frames.".format(len(frame_list)))
    return frame_list


if __name__ == "__main__":

    args = parse_args()

    # print("Called with args:")
    # print(args)

    # hdf5/data/demo_0/obs/[front_image_0, front_image_1, ]
    # demo.keys():  <KeysViewHDF5 ["actions", "dones", "interventions", "next_obs", "obs", "policy_acting", "rewards", "states", "user_acting"]>
    # obs.keys():  <KeysViewHDF5 ["ee_pose", "front_image_1", "front_image_2", "gripper_position", "hand_loc", "joint_positions", "joint_velocities", "wrist_image"]>

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception("There is no input directory for loading network from " + model_dir)
    load_name = os.path.join(model_dir,
                             "faster_rcnn_{}_{}_{}.pth".format(args.checksession, args.checkepoch, args.checkpoint))

    pascal_classes = np.asarray(["__background__", "targetobject", "hand"])
    args.set_cfgs = ["ANCHOR_SCALES", "[8, 16, 32, 64]", "ANCHOR_RATIOS", "[0.5, 1, 2]"]

    # initilize the network here.
    if args.net == "vgg16":
        fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == "res101":
        fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == "res50":
        fasterRCNN = resnet(pascal_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == "res152":
        fasterRCNN = resnet(pascal_classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name, map_location="cpu")
    fasterRCNN.load_state_dict(checkpoint["model"])

    if "pooling_mode" in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint["pooling_mode"]

    if args.cuda:
        cfg.CUDA = True
        fasterRCNN.cuda()

    fasterRCNN.eval()

    print("load model successfully!")

    # Find videos
    left_videos = sorted(glob(os.path.join(args.left_video_dir, "*.mp4")))
    right_videos = sorted(glob(os.path.join(args.right_video_dir, "*.mp4")))

    # check the number of video files
    num_episodes = len(left_videos)
    if not len(left_videos) == len(right_videos):
        raise ValueError("Mismatch the numbers of videos in `left_video_dir` and `right_video_dir`")

    # match the filenames
    for left_video in left_videos:
        basename = os.path.basename(left_video)
        target_right_video = os.path.join(args.right_video_dir, basename)
        if target_right_video not in right_videos:
            raise FileNotFoundError(f"{basename} in {args.left_video_dir} is not found in {args.right_video_dir}")
        
    # Initialize tensor holders
    # detection holders
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # statistics holders
    bbox_normalize_means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
    bbox_normalize_stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()
        box_info = box_info.cuda()
        bbox_normalize_means = bbox_normalize_means.cuda()
        bbox_normalize_stds = bbox_normalize_stds.cuda()

    # Create new hdf5 file
    new_hdf_path = args.target_hdf5_path[:-5] + "_new.hdf5"
    with h5py.File(new_hdf_path, "w") as hdf:
        data_group = hdf.create_group("data")
        mask_group = hdf.create_group("mask")
        num_samples = 0

        train_mask_list = []
        valid_mask_list = []

        for i, (left_video, right_video) in enumerate(zip(left_videos, right_videos)):
            demo_name = f"demo_{i}"
            demo_group = data_group.create_group(demo_name)
            obs_group = demo_group.create_group("obs")

            # make a save directory for detected images
            demo_dir = os.path.join(args.save_dir, demo_name)
            os.makedirs(demo_dir, exist_ok=True)

            for front_image_index in [1, 2]:
                if front_image_index == 1:
                    input_video_path = left_video
                elif front_image_index == 2:
                    input_video_path = right_video

                print("Processing {} front_image_index {} ...".format(demo_name, front_image_index))

                # load images
                images = extractImages(input_video_path)
                images = np.asarray(images)
                images = images[::args.downsample_rate]
                num_images = len(images)
                img_list = [f"{img_i:06d}" for img_i in range(num_images)]
                print("Loaded Photo: {} images.".format(num_images))

                # img_size = 120.
                img_height, img_width = images.shape[1:3]
                mpy_frame_list = []

                with torch.no_grad():
                    # start detection
                    start = time.perf_counter()
                    max_per_image = 100
                    # thresh_hand = args.thresh_hand
                    # thresh_obj = args.thresh_obj
                    # vis = args.vis

                    hand_det_result = []

                    for img_i in range(num_images):
                        total_tic = time.perf_counter()

                        img_in = images[img_i]
                        img_in = cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR)

                        # resized_img = cv2.resize(img_in, [int(img_size), int(img_size)])
                        mpy_frame_list.append(img_in)
                        blobs, im_scales = _get_image_blob(img_in)
                        assert len(im_scales) == 1, "Only single-image batch implemented"
                        im_blob = blobs

                        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
                        im_info_pt = torch.from_numpy(im_info_np)

                        im_data_pt = torch.from_numpy(im_blob)
                        im_data_pt = im_data_pt.permute(0, 3, 1, 2)

                        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                        gt_boxes.resize_(1, 1, 5).zero_()
                        num_boxes.resize_(1).zero_()
                        box_info.resize_(1, 1, 5).zero_()

                        detection_tic = time.perf_counter()

                        rois, cls_prob, bbox_pred, \
                        rpn_loss_cls, rpn_loss_box, \
                        RCNN_loss_cls, RCNN_loss_bbox, \
                        rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

                        scores = cls_prob.data
                        boxes = rois.data[:, :, 1:5]

                        # extact predicted params
                        contact_vector = loss_list[0][0]  # hand contact state info
                        offset_vector = loss_list[1][0].detach()  # offset vector (factored into a unit vector and a magnitude)
                        lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

                        # get hand contact
                        _, contact_indices = torch.max(contact_vector, 2)
                        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

                        # get hand side
                        lr = torch.sigmoid(lr_vector) > 0.5
                        lr = lr.squeeze(0).float()

                        if cfg.TEST.BBOX_REG:
                            # Apply bounding-box regression deltas
                            box_deltas = bbox_pred.data
                            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                                # Optionally normalize targets by a precomputed mean and stdev
                                if args.class_agnostic:
                                    box_deltas = box_deltas.view(-1, 4) * bbox_normalize_stds + bbox_normalize_means
                                    box_deltas = box_deltas.view(1, -1, 4)
                                else:
                                    box_deltas = box_deltas.view(-1, 4) * bbox_normalize_stds + bbox_normalize_means
                                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

                            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                        else:
                            # Simply repeat the boxes, once for each class
                            pred_boxes = np.tile(boxes, (1, scores.size(1)))

                        pred_boxes /= im_scales[0]

                        scores = scores.squeeze()
                        pred_boxes = pred_boxes.squeeze()
                        detect_time = time.perf_counter() - detection_tic
                        misc_tic = time.perf_counter()

                        if args.vis:
                            im2show = np.copy(img_in)

                        obj_dets, hand_dets = None, None
                        for j in xrange(1, len(pascal_classes)):
                            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                            if pascal_classes[j] == "hand":
                                inds = torch.nonzero(scores[:, j] > args.thresh_hand).view(-1)
                            elif pascal_classes[j] == "targetobject":
                                inds = torch.nonzero(scores[:, j] > args.thresh_obj).view(-1)

                            # embed()
                            # if there is det
                            if inds.numel() > 0:
                                cls_scores = scores[:, j][inds]
                                _, order = torch.sort(cls_scores, 0, True)
                                if args.class_agnostic:
                                    cls_boxes = pred_boxes[inds, :]
                                else:
                                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds],
                                                    offset_vector.squeeze(0)[inds], lr[inds]), 1)
                                cls_dets = cls_dets[order]
                                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                                cls_dets = cls_dets[keep.view(-1).long()]
                                if pascal_classes[j] == "targetobject":
                                    obj_dets = cls_dets.cpu().numpy()
                                if pascal_classes[j] == "hand":
                                    hand_dets = cls_dets.cpu().numpy()
                                    # print("hand_dets: ", hand_dets.shape)   # (1, 10)
                        if hand_dets is not None:
                            save_hand_dets = hand_dets[0:1]
                            save_hand_dets = np.concatenate((save_hand_dets, np.array(calculate_center(save_hand_dets[0, :4]))[None]), axis=1)
                            save_hand_dets[:, [0, 2, -2]] = save_hand_dets[:, [0, 2, -2]].astype(np.float32) / img_height
                            save_hand_dets[:, [1, 3, -1]] = save_hand_dets[:, [1, 3, -1]].astype(np.float32) / img_width
                            hand_det_result.append(copy.deepcopy(save_hand_dets))
                        else:
                            save_hand_dets = np.zeros((1, 10))
                            save_hand_dets = np.concatenate((save_hand_dets, np.array(calculate_center(save_hand_dets[0, :4]))[None]), axis=1)
                            save_hand_dets[:, [0, 2, -2]] = save_hand_dets[:, [0, 2, -2]].astype(np.float32) / img_height
                            save_hand_dets[:, [1, 3, -1]] = save_hand_dets[:, [1, 3, -1]].astype(np.float32) / img_width
                            hand_det_result.append(copy.deepcopy(save_hand_dets))

                        if args.vis:
                            # visualization
                            # print(num_images, "save_hand_dets", save_hand_dets)
                            im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, args.thresh_hand, args.thresh_obj)

                        nms_time = time.perf_counter() - misc_tic

                        sys.stdout.write("im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r" \
                                        .format(img_i + 1, num_images, detect_time, nms_time))
                        sys.stdout.flush()

                        if args.vis:
                            result_path = os.path.join(demo_dir, f"{img_i:06d}_{front_image_index}" + "_det.png")
                            im2show.save(result_path)
                        else:
                            im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
                            cv2.imshow("frame", im2showRGB)
                            total_toc = time.time()
                            total_time = total_toc - total_tic
                            frame_rate = 1 / total_time
                            print("Frame rate:", frame_rate)
                            if cv2.waitKey(1) & 0xFF == ord("q"):
                                break

                    hand_loc = np.asarray(hand_det_result)#[::-1]
                    x = copy.deepcopy(hand_loc[:, :, -2])
                    y = copy.deepcopy(hand_loc[:, :, -1])
                    hand_loc[:, :, -2] = copy.deepcopy(y)
                    hand_loc[:, :, -1] = copy.deepcopy(x)
                    print("hand_loc.shape: ", hand_loc.shape)  # hand_loc.shape:  (145, 1, 12)

                    shifted_hand_loc_1_array = np.concatenate((hand_loc, hand_loc[-1:]), axis=0)
                    hand_act = shifted_hand_loc_1_array[1:] - shifted_hand_loc_1_array[:-1]
                    print("hand_act.shape: ", hand_act.shape)  # hand_act.shape:  (145, 1, 12)

                    obs_group.create_dataset(f"hand_loc_{front_image_index}", data=hand_loc)
                    obs_group.create_dataset(f"hand_act_{front_image_index}", data=hand_act)
                    obs_group.create_dataset(f"front_image_{front_image_index}", data=images)

                    if front_image_index == 1:
                        obs_group.create_dataset("agentview_image", data=images)

                        hand_loc_1 = copy.deepcopy(hand_loc)
                    else:
                        obs_group.create_dataset(f"agentview_image_{front_image_index}", data=images)

                        all_hand_loc = np.concatenate((hand_loc_1[:, :, -2:], hand_loc[:, :, -2:]), axis=2)
                        obs_group.create_dataset("hand_loc", data=all_hand_loc)
                        print("all_hand_loc.shape: ", all_hand_loc.shape)  # hand_act.shape:  (145, 1, 4)

                        all_skip_hand_loc = []
                        num_future_frame = 10
                        skip_len = 2
                        T = all_hand_loc.shape[0]
                        num_samples += T-1
                        for i in range(all_hand_loc.shape[0]):
                            each_skip_hand_loc = []
                            for j in range(num_future_frame):
                                if i + j * skip_len >= all_hand_loc.shape[0]:
                                    each_skip_hand_loc.append(all_hand_loc[-1])
                                else:
                                    each_skip_hand_loc.append(all_hand_loc[i + j * skip_len])
                            all_skip_hand_loc.append(each_skip_hand_loc)
                        all_skip_hand_loc = np.asarray(all_skip_hand_loc)
                        print("all_skip_hand_loc.shape: ", all_skip_hand_loc.shape)

                        all_skip_hand_loc = all_skip_hand_loc.reshape(T, -1)
                        demo_group.create_dataset("actions", data=all_skip_hand_loc)

                        obs_group.create_dataset("robot0_eef_pos", data=all_hand_loc)
                        obs_group.create_dataset("robot0_eef_pos_future_traj", data=all_skip_hand_loc)

                        # "actions", "dones", "interventions", "next_obs", "obs", "policy_acting", "rewards", "states", "user_acting"
                        demo_group.create_dataset("dones", data=np.zeros((T-1)))
                        demo_group.create_dataset("interventions", data=np.zeros((T, 1)))
                        demo_group.create_dataset("policy_acting", data=np.zeros((T)))
                        demo_group.create_dataset("rewards", data=np.zeros((T-1)))
                        demo_group.create_dataset("states", data=np.zeros((0)))
                        demo_group.create_dataset("user_acting", data=np.zeros((T, 1)))
                        demo_group.attrs["num_samples"] = T - 1
                        print(f"Num samples: {demo_group.attrs['num_samples']}")
                        # demo_group.create_dataset("mask/train", data=[demo_name])
                        # demo_group.create_dataset("mask/valid", data=[val_demo_name])

            # split train / valid with 90% / 10%
            if i < int(0.9 * num_episodes):
                train_mask_list.append(demo_name)
            else:
                valid_mask_list.append(demo_name)

        # save into mask group
        mask_group.create_dataset("train", data=train_mask_list)
        mask_group.create_dataset("valid", data=valid_mask_list)



        data_group = hdf["data"]
        data_group.attrs["total"] = num_samples # T - 1  # num_samples + num_samples
        env_meta = {
            "env_name": "Libero_Kitchen_Tabletop_Manipulation",
            "env_version": "1.4.1",
            "type": 1,
            "env_kwargs": {
                "robots": [
                    "Panda"
                ],
                "controller_configs": {
                    "type": "OSC_POSE",
                    "input_max": 1,
                    "input_min": -1,
                    "output_max": [
                        0.05,
                        0.05,
                        0.05,
                        0.5,
                        0.5,
                        0.5
                    ],
                    "output_min": [
                        -0.05,
                        -0.05,
                        -0.05,
                        -0.5,
                        -0.5,
                        -0.5
                    ],
                    "kp": 150,
                    "damping_ratio": 1,
                    "impedance_mode": "fixed",
                    "kp_limits": [
                        0,
                        300
                    ],
                    "damping_ratio_limits": [
                        0,
                        10
                    ],
                    "position_limits": None,
                    "orientation_limits": None,
                    "uncouple_pos_ori": True,
                    "control_delta": True,
                    "interpolation": None,
                    "ramp_ratio": 0.2
                },
                "bddl_file_name": None,
                "reward_shaping": False,
                "camera_names": [
                    "agentview",
                    "robot0_eye_in_hand"
                ],
                "camera_heights": 84,
                "camera_widths": 84,
                "has_renderer": False,
                "has_offscreen_renderer": True,
                "ignore_done": True,
                "use_object_obs": True,
                "use_camera_obs": True,
                "camera_depths": False,
                "render_gpu_device_id": 0
            }
        }
        data_group.attrs["env_args"] = json.dumps(env_meta, indent=4)
        print("Save to {}".format(args.target_hdf5_path[:-5] + "_new.hdf5"))
