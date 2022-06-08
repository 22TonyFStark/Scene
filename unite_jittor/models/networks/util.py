import re
import importlib
import jittor
from argparse import Namespace
import numpy as np
from PIL import Image
import os
import sys
import argparse
import scipy.io as scio
import matplotlib


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls

def feature_normalize(feature_in):
    feature_in_norm = jittor.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature_in_norm = jittor.divide(feature_in, feature_in_norm)
    return feature_in_norm

def mean_normalize(feature, dim_mean=None):
    feature = feature - feature.mean(dim=dim_mean, keepdims=True)  # center the feature
    feature_norm = jittor.norm(feature, 2, 1, keepdim=True) + sys.float_info.epsilon
    feature = jittor.divide(feature, feature_norm)
    return feature

def vgg_preprocess(tensor, vgg_normal_correct=False):
    if vgg_normal_correct:
        tensor = (tensor + 1) / 2
    # input is RGB tensor which ranges in [0,1]
    # output is BGR tensor which ranges in [0,255]
    tensor_bgr = jittor.concat((tensor[:, 2:3, :, :], tensor[:, 1:2, :, :], tensor[:, 0:1, :, :]), dim=1)
    # tensor_bgr = tensor[:, [2, 1, 0], ...]
    # print (tensor_bgr.shape)
    # print (torch.Tensor([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1).shape)
    # print (1/0)
    tensor_bgr_ml = tensor_bgr - jittor.array([0.40760392, 0.45795686, 0.48501961]).type_as(tensor_bgr).view(1, 3, 1, 1)
    tensor_rst = tensor_bgr_ml * 255
    return tensor_rst
