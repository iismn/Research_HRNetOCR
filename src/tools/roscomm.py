#!/usr/bin/env python3

import argparse
import os
import pprint
import shutil
import sys

import logging
import time
import timeit
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import _init_paths
import models
import datasets
from config import config
from config import update_config
from core.function import testval, test, test_ROS
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel

import rospy
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage, PointCloud2

import cv2
from cv_bridge import CvBridge, CvBridgeError

class HRNet_Conv:
    def __init__(self):

        # parser = argparse.ArgumentParser(description='Test segmentation network')
        # parser.add_argument('--cfg',
        #                     help='experiment configure file name',
        #                     default='/home/iismn/WorkSpace/CU11/ROS_DL/src/DL_PACK/HRNet_OCR/src/experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml',
        #                     required=False,
        #                     type=str)
        # args = parser.parse_args()

        config.defrost()
        config.merge_from_file("/home/iismn/WorkSpace/CU11/ROS_DL/src/DL_PACK/HRNet_OCR/src/experiments/cityscapes/seg_hrnet_ocr_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml")
        config.freeze()

        # cudnn related setting
        cudnn.benchmark = config.CUDNN.BENCHMARK
        cudnn.deterministic = config.CUDNN.DETERMINISTIC
        cudnn.enabled = config.CUDNN.ENABLED
        self.result = 0
        # build model
        if torch.__version__.startswith('1'):
            module = eval('models.'+config.MODEL.NAME)
            module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
        self.HRNet_OCR = eval('models.'+config.MODEL.NAME +
                     '.get_seg_model')(config)

        dump_input = torch.rand(
            (1, 3, config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
        )

        model_state_file = os.path.join('/home/iismn/WorkSpace/CU11/ROS_DL/src/DL_PACK/HRNet_OCR/src/experiments/cityscapes/hrnet_ocr_cs_trainval_8227_torch11.pth')
        pretrained_dict = torch.load(model_state_file)

        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = self.HRNet_OCR.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                            if k[6:] in model_dict.keys()}
        model_dict.update(pretrained_dict)
        self.HRNet_OCR.load_state_dict(model_dict)

        gpus = list(config.GPUS)
        self.HRNet_OCR = nn.DataParallel(self.HRNet_OCR, device_ids=gpus).cuda()

        # prepare data
        test_size = (1080,1920)
        # multi_scale_inference
        self.test_Param = eval('datasets.'+config.DATASET.DATASET)(
                            root=config.DATASET.ROOT,
                            list_path='list/cityscapes/test.lst',
                            num_samples=None,
                            num_classes=config.DATASET.NUM_CLASSES,
                            multi_scale=False,
                            ignore_label=config.TRAIN.IGNORE_LABEL,
                            flip=False,
                            base_size=config.TEST.BASE_SIZE,
                            crop_size=test_size,
                            downsample_rate=1)

        print('Setting Test Param Done')
        self.bridge = CvBridge()
        self.config = config

        self.ImgSub = message_filters.Subscriber("/Logitech_BRIO/image_rect_color/compressed", CompressedImage)
        self.LIDARSub = message_filters.Subscriber("/vlp_t/velodyne_points", PointCloud2)
        self.Sematic_ImgPub = rospy.Publisher("/Logitech_BRIO/image_semantic_color/compressed", CompressedImage)
        Time_Syncrhonizer = message_filters.ApproximateTimeSynchronizer([self.ImgSub, self.LIDARSub], 10, 0.1, allow_headerless=True)
        Time_Syncrhonizer.registerCallback(self.Image_Callback)



    def Image_Callback(self,Input_IMG,Input_PCL):
        Input_IMG_MSG = Input_IMG
        Input_IMG = self.bridge.compressed_imgmsg_to_cv2(Input_IMG)
        Input_IMG = cv2.resize(Input_IMG, (1280, 720), interpolation=cv2.INTER_AREA)
        # cv2.imshow('1',Input_IMG)
        # cv2.waitKey(0)
        # print(Input_IMG.s)
        img_Seg = test_ROS(self.config, self.test_Param, Input_IMG, self.HRNet_OCR)

        img_Seg = cv2.resize(img_Seg, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        msg = CompressedImage()
        msg.header = Input_IMG_MSG.header
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', img_Seg)[1]).tostring()
        # Publish new image
        self.Sematic_ImgPub.publish(msg)
        # print(Input_IMG.s)
        # print('callback in')

def main(args):

  rospy.init_node('roscomm', anonymous=True)
  HRNet_Conv()

  while not rospy.is_shutdown():
      try:
        rospy.spin()
      except KeyboardInterrupt:
        print("HRNET END")

if __name__ == '__main__':
    main(sys.argv)
