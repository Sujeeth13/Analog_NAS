# -*- coding: utf-8 -*-
"""params

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1iGujGIHQ20OW37j_sA54Y-TcJwuRI1AL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import operator
import functools

import pandas as pd
import numpy as np

import gc

def parse_model_representation(df):
  """add each column element to the dictionary"""
  config = {}
  for idx,val in df.items():
    config[idx] = val
  return config

def calc_conv_params(in_channels, out_channels, kernel_size):
    return (kernel_size ** 2) * in_channels * out_channels

def calc_bn_params(num_features):
    return 2 * num_features

def calc_fc_params(in_features, out_features):
    return (in_features * out_features) + out_features

def calc_residual_branch_params(in_channels, out_channels, filter_size):
    conv1_params = calc_conv_params(in_channels, out_channels, filter_size)
    bn1_params = calc_bn_params(out_channels)
    conv2_params = calc_conv_params(out_channels, out_channels, filter_size)
    bn2_params = calc_bn_params(out_channels)
    return conv1_params + bn1_params + conv2_params + bn2_params

def calc_skip_connection_params(in_channels, out_channels):
    conv1_params = calc_conv_params(in_channels, out_channels // 2, 1)
    conv2_params = calc_conv_params(in_channels, out_channels // 2, 1)
    bn_params = calc_bn_params(out_channels)
    return conv1_params + conv2_params + bn_params

def calc_basic_block_params(in_channels, out_channels, filter_size, res_branches, use_skip):
    branches_params = sum([calc_residual_branch_params(in_channels, out_channels, filter_size) for _ in range(res_branches)])
    skip_params = calc_skip_connection_params(in_channels, out_channels) if use_skip else 0
    return branches_params + skip_params

def calc_residual_group_params(in_channels, out_channels, n_blocks, filter_size, res_branches, use_skip):
    return sum([calc_basic_block_params(in_channels if i == 0 else out_channels, out_channels, filter_size, res_branches, use_skip and i == 0) for i in range(n_blocks)])

def calc_total_params(config, input_dim=(3, 32, 32), classes=10):
    out_channel0 = config["out_channel0"]
    M = config["M"]
    R = [config[f"R{i+1}"] for i in range(M)]
    widen_factors = [config[f"widenfact{i+1}"] for i in range(M)]
    B = [config[f"B{i+1}"] for i in range(M)]

    # Initial Conv and BN layer
    total_params = calc_conv_params(3, out_channel0, 7) + calc_bn_params(out_channel0)

    in_channels = out_channel0
    for i in range(M):
        out_channels = in_channels * widen_factors[i]
        total_params += calc_residual_group_params(in_channels, out_channels, R[i], 3, B[i], in_channels != out_channels)
        in_channels = out_channels

    # Average pooling
    feature_maps_out = in_channels
    if M == 1:
      fc_len = feature_maps_out * 21 * 21
    elif M == 2:
      fc_len = feature_maps_out * 21 * 21
    else:
      fc_len = feature_maps_out * 21 * 21  # Assuming average pooling down to 1x1 feature maps

    # Fully connected layer
    total_params += calc_fc_params(fc_len, classes)

    return total_params

if __name__ == '__main__':
    df = pd.read_csv("../data/dataset_cifar10_v1.csv")
    df = df.iloc[:,:-3]
    df.head()

    input_dim = (3, 32, 32)
    classes = 10
    cnt = 0
    info = {}
    min_params = np.inf
    max_params = 0
    for i in range(len(df)):
        config = parse_model_representation(df.iloc[i,:])
        total_params = calc_total_params(config)
        if total_params < min_params:
            min_params = total_params
        if total_params > max_params:
            max_params = total_params
    print(f"Min params: {min_params}")
    print(f"Max params: {max_params}")
