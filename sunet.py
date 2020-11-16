#!/usr/bin/env python
# coding: utf-8

# In[1]:


# All imports


from __future__ import print_function, division

#!pip install monai
import matplotlib.pyplot as plt
import numpy as np
import glob

import sys
import SimpleITK as sitk
import pandas as pd
import glob
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import logging
import csv
from scipy import ndimage, misc
from tqdm import tqdm

import numba
from numba import njit, prange

import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

import scipy
from skimage.measure import label
from scipy.io import loadmat
from scipy.ndimage import zoom
#from scipy.misc import imresize
import pywt

import csv
import random
import time

from scipy import ndimage, misc

import pywt
#import hdf5storage

import scipy.io as sio
from skimage.filters import threshold_otsu

import pywt
import numpy as np
#import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk
import skimage.io as io
#from sklearn.decomposition import PCA
import collections, numpy
import warnings
from scipy import ndimage, misc
warnings.filterwarnings('ignore')
import copy

from ignite.contrib.handlers import ProgressBar
import os
import glob
import uuid
import numpy as np


import numpy
import warnings

import functools
import pickle
import time


import numpy as np

import argparse
import torch
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable

import torch.nn as nn
import shutil


import monai
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler
from monai.transforms import (
    AddChanneld,
    AsDiscreted,
    CastToTyped,
    LoadNiftid,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

np.random.seed(0)
#torch.manual_seed(0)!pip install monai


# In[2]:


# [STAR] Pytorch Models for training

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary

class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class conv_block(nn.Module):
    """
    Convolution Block 
    """
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv_3D(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv_3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up_3D(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up   = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv_3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class OutConv_3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv_3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class SUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)
        #self.out_sigmoid = nn.Sigmoid()
        self.out_softmax = nn.LogSoftmax(dim=1)
        
        self.gn1 = nn.GroupNorm(8, 16)
        self.gn2 = nn.GroupNorm(16, 32)
        self.gn3 = nn.GroupNorm(32, 64)
        self.gn4 = nn.GroupNorm(64, 128)
        self.gn5 = nn.GroupNorm(32, 64)
        self.gn6 = nn.GroupNorm(16, 32)
        self.gn7 = nn.GroupNorm(8, 16)
        
        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.gn1(x1)
        
        x2 = self.down1(x1)
        x2 = self.gn2(x2)
        
        x3 = self.down2(x2)
        x3 = self.gn3(x3)
        #x3 = self.dp1(x3)
        
        x4 = self.down3(x3)
        x4 = self.gn4(x4)
        #x4 = self.dp2(x4)
        
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.gn5(x)
       
        x = self.up2(x, x3)
        x = self.gn6(x)
            
        x = self.up3(x, x2)
        x = self.gn7(x)
        
        x  = self.up4(x, x1)
        
        logits = self.outc(x)
        #out    = self.out_softmax(logits)
        return logits

class SUNet_3D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SUNet_3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        self.inc   = DoubleConv_3D(n_channels, 16*2)
        self.down1 = Down_3D(16*2, 32*2)
        self.down2 = Down_3D(32*2, 64*2)
        self.down3 = Down_3D(64*2, 128*2)
        factor = 2 if bilinear else 1
        self.down4 = Down_3D(128*2, 256*2 // factor)
        self.up1 = Up_3D(256*2, 128*2 // factor, bilinear)
        self.up2 = Up_3D(128*2, 64*2 // factor, bilinear)
        self.up3 = Up_3D(64*2, 32*2 // factor, bilinear)
        self.up4 = Up_3D(32*2, 16*2, bilinear)
        self.outc = OutConv_3D(16*2, n_classes)
        #self.out_sigmoid = nn.Sigmoid()
        self.out_softmax = nn.LogSoftmax(dim=1)
        
        self.gn1 = nn.GroupNorm(8*2, 16*2)
        self.gn2 = nn.GroupNorm(16*2, 32*2)
        self.gn3 = nn.GroupNorm(32*2, 64*2)
        self.gn4 = nn.GroupNorm(64*2, 128*2)
        self.gn5 = nn.GroupNorm(32*2, 64*2)
        self.gn6 = nn.GroupNorm(16*2, 32*2)
        self.gn7 = nn.GroupNorm(8*2, 16*2)
        
        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.gn1(x1)
        
        x2 = self.down1(x1)
        x2 = self.gn2(x2)
        
        x3 = self.down2(x2)
        x3 = self.gn3(x3)
        #x3 = self.dp1(x3)
        
        x4 = self.down3(x3)
        x4 = self.gn4(x4)
        #x4 = self.dp2(x4)
        
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.gn5(x)
       
        x = self.up2(x, x3)
        x = self.gn6(x)
            
        x = self.up3(x, x2)
        x = self.gn7(x)
        
        x  = self.up4(x, x1)
        
        logits = self.outc(x)
        #out    = self.out_softmax(logits)
        return logits

class SUNet_3D_A(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SUNet_3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        self.inc   = DoubleConv_3D(n_channels, 16)
        self.down1 = Down_3D(16, 32)
        self.down2 = Down_3D(32, 64)
        self.down3 = Down_3D(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down_3D(128, 256 // factor)
        self.up1 = Up_3D(256, 128 // factor, bilinear)
        self.up2 = Up_3D(128, 64 // factor, bilinear)
        self.up3 = Up_3D(64, 32 // factor, bilinear)
        self.up4 = Up_3D(32, 16, bilinear)
        self.outc = OutConv_3D(16, n_classes)
        #self.out_sigmoid = nn.Sigmoid()
        self.out_softmax = nn.LogSoftmax(dim=1)
        
        self.gn1 = nn.GroupNorm(8, 16)
        self.gn2 = nn.GroupNorm(16, 32)
        self.gn3 = nn.GroupNorm(32, 64)
        self.gn4 = nn.GroupNorm(64, 128)
        self.gn5 = nn.GroupNorm(32, 64)
        self.gn6 = nn.GroupNorm(16, 32)
        self.gn7 = nn.GroupNorm(8, 16)
        
        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.gn1(x1)
        
        x2 = self.down1(x1)
        x2 = self.gn2(x2)
        
        x3 = self.down2(x2)
        x3 = self.gn3(x3)
        #x3 = self.dp1(x3)
        
        x4 = self.down3(x3)
        x4 = self.gn4(x4)
        #x4 = self.dp2(x4)
        
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.gn5(x)
       
        x = self.up2(x, x3)
        x = self.gn6(x)
            
        x = self.up3(x, x2)
        x = self.gn7(x)
        
        x  = self.up4(x, x1)
        
        logits = self.outc(x)
        #out    = self.out_softmax(logits)
        return logits

class SUNet_with_BN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SUNet_with_BN, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)
        #self.out_sigmoid = nn.Sigmoid()
        self.out_softmax = nn.LogSoftmax(dim=1)
        
        self.gn1 = nn.BatchNorm2d(16)
        self.gn2 = nn.BatchNorm2d(32)
        self.gn3 = nn.BatchNorm2d(64)
        self.gn4 = nn.BatchNorm2d(128)
        self.gn5 = nn.BatchNorm2d(64)
        self.gn6 = nn.BatchNorm2d(32)
        self.gn7 = nn.BatchNorm2d(16)
        
        self.dp1 = nn.Dropout(p=0.4)
        self.dp2 = nn.Dropout(p=0.4)
    
    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.gn1(x1)
        
        x2 = self.down1(x1)
        x2 = self.gn2(x2)
        
        x3 = self.down2(x2)
        x3 = self.gn3(x3)
       
        x4 = self.down3(x3)
        x4 = self.gn4(x4)
       
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.gn5(x)
       
        x = self.up2(x, x3)
        x = self.gn6(x)
            
        x = self.up3(x, x2)
        x = self.gn7(x)
        
        x  = self.up4(x, x1)
        
        logits = self.outc(x)
        #out    = self.out_softmax(logits)
        return logits

class SUNet_without_GN(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SUNet_without_GN, self).__init__()
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.bilinear   = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)
        
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        
        logits = self.outc(x)
        
        return logits

class AttnDecoderRNN_old(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=256, bilinear=True):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.bilinear = bilinear
        self.n_classes = 1

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size*2, self.max_length)
        
        self.attn_24 = nn.Linear(self.hidden_size*4, self.hidden_size*2)
        
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.attn_combine_bilstm = nn.Linear(self.hidden_size * 3, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
       # self.hidden = nn.Parameter(torch.randn(4,256,256).cuda()),nn.Parameter(torch.randn(4,256,256).cuda())
       
        self.lsgn_a = nn.GroupNorm(128,256)
    
        self.down5 = Down(128,256)
        
        factor = 2 if bilinear else 1
                
        self.ups4 = nn.ConvTranspose2d(256 , 256 // 2, kernel_size=2, stride=2)
        self.upsconv4 = DoubleConv(256,128)

        self.lstm = nn.LSTM(256,256,batch_first=False,bidirectional=True,num_layers=1).cuda()
    
    def forward(self, input,hidden,encoder_outputs):
        
        h = torch.unsqueeze(hidden,0)
        
        embedded = input
        
        embedded = self.dropout(embedded)

        hidden_bilstm = h[0]
        
        
        hidden_bilinn =  hidden_bilstm
        
        hidden_bilinn = self.attn(hidden_bilinn)
    
        hidden_bilinn = self.lsgn_a(hidden_bilinn)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden_bilinn), 1)), dim=1)
        
        
        
        attn_weights  = self.lsgn_a(attn_weights)
    
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))
        
   #     print('attn_applied: encoder outputs',attn_applied[0].shape,encoder_outputs[0].shape)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
  #      print('The output shape is : ',output.shape)
        
        output = self.attn_combine_bilstm(output).unsqueeze(0)
 #      print('The output shape after is : ',output.shape)
        
    
        hidden_bi = hidden_bilinn.unsqueeze(0)
        
        output = F.relu(output)
        
        #print("output and hidden before lstm ",output.shape,hidden_bi.shape)

        output, hidden = self.gru(output, hidden_bi)
        
        output = F.log_softmax(self.out(output[0]), dim=1)
        output = self.lsgn_a(output)
        
       #output = self.lsgn_a(output)
    
        return output,hidden


    def initHidden(self):
        return torch.randn(4, 256, self.hidden_size, device=device)

############### MAIN MODEL ##############
class UNetDoubleSmallGroupNormdifferent_old(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        
        super(UNetDoubleSmallGroupNormdifferent, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)

        
        self.down5 = Down(128,256)
        
        
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)
        #self.out_sigmoid = nn.Sigmoid()
        self.out_softmax = nn.LogSoftmax(dim=1)
        
        self.lsgn1 = nn.GroupNorm(128,256)
        
        self.lsgn2 = nn.GroupNorm(64,256)
        
        
        self.gn1 = nn.GroupNorm(8, 16)
        self.gn2 = nn.GroupNorm(16, 32)
        self.gn3 = nn.GroupNorm(32, 64)
        self.gn4 = nn.GroupNorm(64, 128)
        self.gn5 = nn.GroupNorm(32, 64)
        self.gn6 = nn.GroupNorm(16, 32)
        self.gn7 = nn.GroupNorm(8, 16)
   
    def forward(self, x):
        x1 = self.inc(x)
       # x1 = self.gn1(x1)
       
        x2 = self.down1(x1)
       # x2 = self.gn2(x2)
       
        x3 = self.down2(x2)
       # x3 = self.gn3(x3)
       
        x4 = self.down3(x3)
       # x4 = self.gn4(x4)
       
        x5 = self.down4(x4)
        
        #x5 = torch.squeeze(x5)
        x5 = self.down5(x5)
        #x5 = self.down6(x5)
        
        #print('x5 shape is :',x5.shape)
        
        xlst = x5.reshape([4,256,256])

        lstm = nn.LSTM(256,256,batch_first= True,bidirectional=True,num_layers=1).cuda()
                
        #print('xlst',xlst.shape)    
        
        xlst = self.lsgn1(xlst)
        
        ylst = lstm(xlst)
        
        
        #print(hidden)
        
        f = np.asarray(ylst)
        
        h  = torch.cuda.FloatTensor(ylst[0])
        
        
        h = torch.squeeze(h)
        
        encoder_o = f[0]
        
        a = np.zeros((4,256,256))

        a = torch.from_numpy(a)
        a.cuda()
        
        for i in range(4):
    
            oo,b = attn_decoder1.forward(xlst,h[i],encoder_o[i])
            oo = self.lsgn2(oo)
            a[i] = oo
        
            
        a = a.unsqueeze(0)
        a = a.reshape([4,256,16,16])
        
        
        
        x5 = a  
        x5 = x5.cuda()
        
        
        x5 = x5.type(torch.cuda.FloatTensor)
 
        
        
        x5 = self.lsgn2(x5)
        
        ups4 = nn.ConvTranspose2d(256 , 256 // 2, kernel_size=2, stride=2)
        upsconv4 = DoubleConv(256,128)

        ups4 = ups4.cuda()
        
        opt = ups4(x5)
        
        x5 = opt
        
        x = self.up1(x5, x4)
        #x = self.gn5(x)
        
        x = self.up2(x, x3)
       # x = self.gn6(x)
       
        x = self.up3(x, x2)
        #x = self.gn7(x)
       
        x = self.up4(x, x1)
        logits = self.outc(x)
        #out    = self.out_softmax(logits)
        return logits

class UNetDoubleSmallGroupNormdifferent(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        super(UNetDoubleSmallGroupNormdifferent, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc     = DoubleConv(n_channels, 16)
        self.down1   = Down(16, 32)
        self.downnew = Down(16,16)
        self.down2   = Down(32, 64)
        self.down3   = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4   = Down(128, 256 // factor) 
        self.upsam   = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.down5 = Down(128,256)
        self.ups3  = nn.ConvTranspose2d(1 , 1, kernel_size=2, stride=2)
        self.ups4  = nn.ConvTranspose2d(256 , 256 // 2, kernel_size=2, stride=2)
        
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)
        #self.out_sigmoid = nn.Sigmoid()
        self.out_softmax = nn.LogSoftmax(dim=1)
        
        self.lsgn1 = nn.GroupNorm(64,128)
        self.lsgn2 = nn.GroupNorm(64,1024)
        self.lsgn3 = nn.GroupNorm(64,1024)
        
        self.gn1 = nn.GroupNorm(8, 16)
        self.gn2 = nn.GroupNorm(16, 32)
        self.gn3 = nn.GroupNorm(32, 64)
        self.gn4 = nn.GroupNorm(64, 128)
        self.gn5 = nn.GroupNorm(32, 64)
        self.gn6 = nn.GroupNorm(16, 32)
        self.gn7 = nn.GroupNorm(8, 16)
        self.gn8 = nn.GroupNorm(4,8)
   
    def forward(self, x):
        #x = self.upsam()
        
        x1 = self.inc(x)
        #x1 = self.gn1(x1)
       
        x2 = self.down1(x1)
        #x2 = self.gn2(x2)
       
        x3 = self.down2(x2)
        #x3 = self.gn3(x3)
       
        x4 = self.down3(x3)
        #x4 = self.gn4(x4)
       
        x5 = self.down4(x4)
        #x5 = self.gn
        #x5 = torch.squeeze(x5)
        #x5 = self.down5(x5)
        #x5 = self.down6(x5)
        #print('x5:',x5.shape)
        
        xlst = x5.reshape([4,128,1024])
        

        lstm = nn.LSTM(1024,1024,batch_first= True,bidirectional=True,num_layers=1).cuda()
        
        xlst = self.lsgn1(xlst)
        ylst = lstm(xlst)
        
        f = np.asarray(ylst)
        
        h  = torch.cuda.FloatTensor(ylst[0])
        h = torch.squeeze(h)
        
        encoder_o = f[0]
        
        a = np.zeros((4,128,1024))
        #a = ndarray((4,128,1024))

        a = torch.from_numpy(a)
        a.cuda()
        
        for i in range(4):
            oo,b = attn_decoder1.forward(xlst,h[i],encoder_o[i])
            oo   = self.lsgn2(oo)
            a[i] = oo
        
            
        a = a.unsqueeze(0)
        a = a.reshape([4,128,32,32])
        
        
        x5 = a  
        x5 = x5.cuda()
        
        
        x5 = x5.type(torch.cuda.FloatTensor)
        #x5 = self.lsgn3(x5)
        
        #x5 = self.ups4(x5)
    
        x = self.up1(x5, x4)
        #x = self.gn5(x)
        
        x = self.up2(x, x3)
        #x = self.gn6(x)
       
        x = self.up3(x, x2)
        #x = self.gn7(x)
       
        x = self.up4(x, x1)
        #x = self.gn7(x)

        #x = self.downnew(x)
        
        #out    = self.out_softmax(logits)
        
        logits = self.outc(x)
        
        return logits

class UNetDoubleSmallWithoutGN(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        
        super(UNetDoubleSmallWithoutGN, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc   = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.down5 = Down(128,256)
        
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
       # x1 = self.gn1(x1)
       
        x2 = self.down1(x1)
       # x2 = self.gn2(x2)
       
        x3 = self.down2(x2)
       # x3 = self.gn3(x3)
       
        x4 = self.down3(x3)
       # x4 = self.gn4(x4)
       
        x5 = self.down4(x4)
        
        #x5 = torch.squeeze(x5)
        x5 = self.down5(x5)
        #x5 = self.down6(x5)
        
        ups4     = nn.ConvTranspose2d(256 , 256 // 2, kernel_size=2, stride=2)
        upsconv4 = DoubleConv(256,128)
        ups4 = ups4.cuda()
        
        opt = ups4(x5)
        
        x5 = opt
        
        x = self.up1(x5, x4)
        #x = self.gn5(x)
        
        x = self.up2(x, x3)
       # x = self.gn6(x)
       
        x = self.up3(x, x2)
        #x = self.gn7(x)
       
        x = self.up4(x, x1)
        logits = self.outc(x)
        #out    = self.out_softmax(logits)
        return logits

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=128, bilinear=True):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p   = dropout_p
        self.max_length  = max_length
        self.bilinear    = bilinear
        self.n_classes   = 1

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn      = nn.Linear(2048, 1024)
        
        self.attn2   = nn.Linear(1024, 128)
        
        self.attn_24 = nn.Linear(self.hidden_size*4, self.hidden_size*2)
        
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.attn_combine_bilstm = nn.Linear(3072, 1024)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru     = nn.GRU(1024, 1024)
        self.out     = nn.Linear(1024, 1024)
       # self.hidden = nn.Parameter(torch.randn(4,256,256).cuda()),nn.Parameter(torch.randn(4,256,256).cuda())
       
        #self.lsgn_a = nn.GroupNorm(512,1024)
        self.lsbn_a1 = nn.BatchNorm1d(1024)
        #self.lsgn_a2 = nn.GroupNorm(512,1024)
        
        #self.lsgn_in = nn.GroupNorm(64,128)
        self.lsbn_in1 = nn.BatchNorm1d(2048)
        self.lsbn_in2 = nn.BatchNorm1d(1024)
        
        
        self.lsbn_in3 = nn.BatchNorm1d(128)#nn.GroupNorm(64,   128)
        self.lsbn_in4 = nn.BatchNorm1d(128)#nn.GroupNorm(64,   128)
        self.lsbn_in5 = nn.BatchNorm1d(1024)#nn.GroupNorm(512,  1024)
        
        self.down5 = Down(128,256)
        
        factor = 2 if bilinear else 1
                
        self.ups4     = nn.ConvTranspose2d(256 , 256 // 2, kernel_size=2, stride=2)
        self.upsconv4 = DoubleConv(256,128)

        self.lstm = nn.LSTM(256,256,batch_first=False,bidirectional=True,num_layers=1).cuda()
    
    def forward(self, input,hidden,encoder_outputs):
        
        h        = torch.unsqueeze(hidden, 0)
        embedded = input
        #embedded = self.lsgn_in1(embedded)
        embedded = self.dropout(embedded)
        
        hidden_bilstm = h[0]
        hidden_bilinn = hidden_bilstm
        
        hidden_bilinn = self.attn(hidden_bilinn)
        hidden_bilinn = self.lsbn_a1(hidden_bilinn)
        
        hidden_bi     = hidden_bilinn.unsqueeze(0)
        
        #print(hidden_bilinn.shape)
        
        attn_weights  = torch.cat((embedded[0], hidden_bilinn), 1)
        attn_weights  = self.lsbn_in1(attn_weights)
        
        attn_weights  = self.attn(attn_weights)
        attn_weights  = self.lsbn_in2(attn_weights)
        
        attn_weights  = F.softmax(attn_weights, dim=1)
        
        attn_weights  = self.attn2(attn_weights)
        attn_weights  = self.lsbn_in3(attn_weights)
        
        #print(attn_weights.unsqueeze(0).shape,encoder_outputs.unsqueeze(0).shape)
    
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        
        #print('attn_applied: encoder outputs',attn_applied[0].shape,encoder_outputs[0].shape)

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        
        output = self.attn_combine_bilstm(output).unsqueeze(0)
        output = F.relu(output)
        output = self.lsbn_in4(output)
        
        output, hidden = self.gru(output, hidden_bi)
        
        output = self.out(output[0])
        output = self.lsbn_in5(output)
        output = F.log_softmax(output, dim=1)
        return output, hidden


    def initHidden(self):
        return torch.randn(4, 256, self.hidden_size, device=device)

class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, img_ch=1, output_ch=1):
        super(AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(img_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

        #self.active = torch.nn.Sigmoid()


    def forward(self, x):

        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #print(x5.shape)
        d5 = self.Up5(e5)
        #print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

      #  out = self.active(out)

        return out
    
class UNetNormal(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNetNormal, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        my_factor = 1
        factor    = 1
        
        self.inc   = DoubleConv(n_channels, 32*my_factor)
        self.down1 = Down(32*my_factor, 64*my_factor)
        self.down2 = Down(64*my_factor, 128*my_factor)
        self.down3 = Down(128*my_factor, 256*my_factor)
        factor = 2 if bilinear else 1
        self.down4 = Down(256*my_factor, 512*my_factor // factor)
        
        self.lsgn1 = nn.GroupNorm(256,512)
        self.lsgn2 = nn.GroupNorm(512,1024)
        
        self.up1 = Up(512*my_factor, 256*my_factor // factor, bilinear)
        self.up2 = Up(256*my_factor, 128*my_factor // factor, bilinear)
        self.up3 = Up(128*my_factor, 64*my_factor // factor, bilinear)
        self.up4 = Up(64*my_factor, 32*my_factor, bilinear)
        self.outc = OutConv(32*my_factor, n_classes)
        #self.out_sigmoid = nn.Sigmoid()
        #self.out_softmax = nn.LogSoftmax(dim=1)
       
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        #out    = self.out_softmax(logits)
        return logits



# [STAR] For training models on Challenge Dataset

import skimage
import torch.optim as optim
from tqdm import tqdm
import random
from skimage.transform import rotate, AffineTransform, warp
from scipy.stats import entropy
import numpy as np
import time

import torch.optim as optim
from tqdm import tqdm
import random
from skimage.transform import rotate, AffineTransform, warp
from scipy.stats import entropy
from scipy.ndimage import rotate

basepath         = '/home/yu-hao/SEMISUNET/Dataset/'
basepath_models  = '/home/yu-hao/SEMISUNET/Dataset/models/'


def read_training_data(read_ids):
    x_array = []
    y_array = []
    
    for p in read_ids:
        name = basepath+'masks/'
        name = name+'study_'+p+'_mask.nii.gz'
        
        mask = sitk.GetArrayFromImage(sitk.ReadImage(name))
        vol  = sitk.GetArrayFromImage(sitk.ReadImage(name.replace('_mask.nii.gz', '.nii.gz').replace('masks', 'studies/CT-1')))
        
        for t in range(mask.shape[0]):
            temp  = np.count_nonzero(mask[t].flatten())
            if temp > 0:
                x_array.append(np.expand_dims(vol[t], axis=0))
                y_array.append(np.expand_dims(mask[t], axis=0))

    x_array = (np.array(x_array)+1024.0)/1024.0
    y_array = np.array(y_array)
    
    return x_array, y_array

def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / (im1.sum() + im2.sum()+0.00001)

def dice_loss(pred, target, smooth = 1.):
    pred = F.sigmoid(pred)
    
    pred   = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def read_training_data_unlabelled(read_ids):
    x_array          = []
    x_array_lungmask = []
    
    names   = [x.split('_')[0] for x in read_ids]
    types   = [x.split('_')[1] for x in read_ids]
    count   = 0
    
    for p in names:
        name     = basepath+'studies/'+types[count]+'/'
        maskname = name+'study_'+p+'_mask.nii.gz'
        volname  = name+'study_'+p+'.nii.gz'
        
        mask = sitk.GetArrayFromImage(sitk.ReadImage(maskname))
        vol  = sitk.GetArrayFromImage(sitk.ReadImage(volname))
        mask[mask > 0] = 1
        
        for t in range(mask.shape[0]):
            if True:#t % 1 == 0:
                temp  = np.count_nonzero(mask[t].flatten())
                if temp > 0: # Check if lung region is present
                    x_array.append(np.expand_dims(vol[t], axis=0))
                    x_array_lungmask.append(np.expand_dims(mask[t], axis=0))
        
        count = count+1

    x_array          = (np.array(x_array)+1024.0)/1024.0
    x_array_lungmask = np.array(x_array_lungmask)
    
    return x_array, x_array_lungmask

def get_prediction(model, valx):
    output_array   = []
    batch_size     = 1
    
    model.eval()
    
    for ik in range(len(valx)//batch_size):
        x = valx[ik*batch_size:(ik+1)*batch_size, :, :, :]
        x = torch.tensor(x, device=device).float()

        output = model.forward(x)
        output = torch.sigmoid(output)
        output = output.data.cpu().numpy()
        #output[output > 0.5]= 1
        #output[output < 0.5]= 0
        
        for k in range(output.shape[0]):
            output_array.append(output[k, 0])
    
    output_array = np.array(output_array)
    output_array = np.expand_dims(output_array, 1)
    
    return output_array

def get_predictions(models, valx):
    output_array   = []
    batch_size     = 1
    
    for i in range(len(models)):
        models[i].eval()
    
    for ik in range(len(valx)//batch_size):
        x = valx[ik*batch_size:(ik+1)*batch_size, :, :, :]
        x = torch.tensor(x, device=device).float()
        
        outputs = []
        for k in range(len(models)):
            output = models[k].forward(x)
            output = torch.sigmoid(output)
            output = output.data.cpu().numpy()
            outputs.append(output)
        
        output_sum = np.zeros(outputs[0].shape, dtype='float16')
        for k in range(len(models)):
            output_sum = output_sum+outputs[k]
        output_sum = output_sum/5.0
        
        for k in range(output.shape[0]):
            output_array.append(output_sum[k, 0])
    
    output_array = np.array(output_array)
    output_array = np.expand_dims(output_array, 1)
    
    return output_array

def get_filtered(valx, valy):
    valxf = []
    valyf = []
    
    for i in range(valx.shape[0]):
        if np.count_nonzero(valy[i]) > 0:
            valxf.append(valx[i])
            valyf.append(valy[i])
    return np.array(valxf), np.array(valyf)

def evaluate_result(model, valx, valy):
    model.eval()
    
    val_dice       = []
    batch_size     = 1
    for ik in range(len(valx)//batch_size):
        x = valx[ik*batch_size:(ik+1)*batch_size, :, :, :]
        y = valy[ik*batch_size:(ik+1)*batch_size, :, :, :]

        x = torch.tensor(x, device=device).float()

        output = model.forward(x)

        output = torch.sigmoid(output)        
        output = output.data.cpu().numpy()

        output[output < 0.5] = 0
        output[output > 0.5] = 1
        
        for pk in range(output.shape[0]):
            dt = dice(y[pk, 0, :, :], output[pk, 0, :, :])
            val_dice.append(dt)
    return val_dice

def evaluate_result_new(pred, valy):
    val_dice       = []
    batch_size     = 1
    
    for ik in range(len(valx)//batch_size):
        output = pred[ik*batch_size:(ik+1)*batch_size, :, :, :]
        y      = valy[ik*batch_size:(ik+1)*batch_size, :, :, :]
       
        output[output < 0.5] = 0
        output[output > 0.5] = 1
       
        for pk in range(output.shape[0]):
            t1 = output[0, 0].astype('uint8')#scipy.ndimage.zoom(output[0, 0].astype('uint8'), 0.6875, order=0)
            t2 = y[0, 0]#scipy.ndimage.zoom(y[0, 0].astype('uint8'),      0.6875, order=0)
            dt = dice(y[pk, 0, :, :], output[pk, 0, :, :])
            val_dice.append(dt)
    
    return val_dice

def train_model(model, batch_size, optimizer, criterion, trainx, trainy, augment=False):
    loss_array = []
    
    model.train()
    
    for i in range(len(trainx)//batch_size):
        x = trainx[i*batch_size:(i+1)*batch_size, :, :, :]
        y = trainy[i*batch_size:(i+1)*batch_size, :, :, :]
        
        if augment:
            t1 = random.randint(0, 100)
            if t1 > 60:
                for k in range(x.shape[0]):
                    rotv = random.randint(0, 3)
                    x[k, 0, :, :] = np.rot90(x[k, 0, :, :], rotv)
                    y[k, 0, :, :] = np.rot90(y[k, 0, :, :], rotv)
        
        x = torch.tensor(x, device=device).float()
        y = torch.tensor(y, device=device).float()
        
        optimizer.zero_grad()
        output = model.forward(x)        
        loss   = criterion(output , y)
        loss.backward()
        
        loss_array.append(loss.item())
        optimizer.step()
    
    loss_array = np.mean(loss_array)
    return loss_array

def prepare_batch(batch_size, k_means, trainx_l, trainy_l, h):
    a = []
    b = []
    
    for i in range(int(batch_size/2)):
        idx = random.randint(0, trainx_l.shape[0]-1)
        c   = k_means.predict(np.reshape(trainx_l[idx].astype('float32'), [1, 512*512]))[0]
        
        a.append(trainx_l[idx])
        b.append(trainy_l[idx])
        
        idx = random.randint(0, len(h[c])-1)
        t1  = np.expand_dims(np.load(h[c][idx]), 0)
        t2  = np.expand_dims(np.load(h[c][idx].replace('-x', '-y')), 0)
        
        a.append(t1)
        b.append(t2)
   
    a1 = np.array(a).astype('float16')
    b1 = np.array(b).astype('float16')
   
    return a1, b1

def store_cluster_slices(model_teacher, k_means, version):
    epoch_array = np.arange(79)
    all_labels  = []
    step_size   = 10 
    count       = 0
    
    for epoch in epoch_array:
        temp_index               = epoch%(int(len(unlabelled_ids)/step_size))
        trainx, trainx_lungmask  = read_training_data_unlabelled(unlabelled_ids[temp_index*step_size:temp_index*step_size+step_size])
        trainy                   = get_prediction(model_teacher, trainx)
        
        #trainy = np.load('/media/pranjal/BackupPlus/SIEMENS/SIEMENS/PREDICTION-NUMPY/'+str(epoch)+'.npy')
        trainy = np.reshape(trainy, [trainy.shape[0], 512*512])
        #print(epoch, trainy.shape, trainx.shape)
        
        l1     = k_means.predict(trainy)
        
        for jt, t in enumerate(l1):
            temp  = np.reshape(trainy[jt], [512, 512]).astype('float16')
            np.save('/media/pranjal/BackupPlus/SIEMENS/SIEMENS/CLUSTER-NUMPY-'+str(version)+'/'+str(t)+'-'+str(count)+'-y.npy', temp)
            
            temp  = np.reshape(trainx[jt], [512, 512]).astype('float16')
            np.save('/media/pranjal/BackupPlus/SIEMENS/SIEMENS/CLUSTER-NUMPY-'+str(version)+'/'+str(t)+'-'+str(count)+'-x.npy', temp)
            
            count = count+1
    
    return

def prepare_hash(version):
    all_cluster_files = glob.glob('/media/pranjal/BackupPlus/SIEMENS/SIEMENS/CLUSTER-NUMPY-'+str(version)+'/*.npy')
    print('Version ', version, 'File name counts ', len(all_cluster_files))
    filename_hash = {}
    for i in range(50):
        filename_hash[i] = []

    for t in all_cluster_files:
        filename_hash[int(t.split('/')[-1].split('-')[0])].append(t)
    
    return filename_hash

def get_all_covid_lesions(valx, valy, lesion_size):
    lesion_shapes_x = []
    lesion_shapes_y = []
    
    for i in range(valy.shape[0]):
        tx           = valx[i, 0]
        blobs        = valy[i, 0]
        blobs_labels = skimage.measure.label(blobs, background=0)
        propsa       = skimage.measure.regionprops(blobs_labels)
        
        for k in range(len(propsa)):
            temp = (blobs_labels == propsa[k].label).astype('uint8')
            
            temp_size = np.count_nonzero(temp.flatten().astype('uint8'))
            if temp_size < lesion_size and temp_size > 5:
                slice_x, slice_y = ndimage.find_objects(temp == 1)[0]
                
                roi_y = 1-temp[slice_x, slice_y]
                roi_x = tx[slice_x, slice_y]*temp[slice_x, slice_y]
                
                lesion_shapes_x.append(roi_x)
                lesion_shapes_y.append(roi_y)
                
                lesion_shapes_x.append(roi_x.T)
                lesion_shapes_y.append(roi_y.T)
                
                lesion_shapes_x.append(np.rot90(roi_x, 180))
                lesion_shapes_y.append(np.rot90(roi_y, 180))
    
    return lesion_shapes_x, lesion_shapes_y

def get_augmented_slice(batch_size, read_ids, lesion_shapes_x, lesion_shapes_y):
    x_array          = []
    x_array_lungmask = []
    
    index   = random.randint(0, len(read_ids)-1)
    #print(read_ids[index])
    
    p       = read_ids[index].split('_')[0]
    types   = 'CT-1'#read_ids[index].split('_')[1]
    count   = 0
    
    name     = basepath+'studies/'+types+'/'
    maskname = name+'study_'+p+'_mask.nii.gz'
    volname  = name+'study_'+p+'.nii.gz'
    
    segmentation_mask = basepath+'masks/'
    segmentation_mask = segmentation_mask+'study_'+p+'_mask.nii.gz'
    
    mask     = sitk.GetArrayFromImage(sitk.ReadImage(maskname))
    vol      = (sitk.GetArrayFromImage(sitk.ReadImage(volname))+1024.0)/1024.0
    segmentation_mask = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_mask))
    
    mask[mask > 0] = 1
    count          = 0
    
    while(count < batch_size):
        t     = np.random.randint(0, mask.shape[0]-1)
        temp  = np.count_nonzero(mask[t].flatten())
        
        # Check if lung region is present
        if temp > 0:
            st  = vol[t]
            i,j = np.nonzero(mask[t])
            
            index = random.randint(0, len(i)-1)
            
            i = i[index]
            j = j[index]
            
            lesion_index = random.randint(0, len(lesion_shapes_x)-1)
            
            lesion_x     = lesion_shapes_x[lesion_index]
            lesion_y     = lesion_shapes_y[lesion_index]
            
            sx     = int(lesion_x.shape[0]/2)
            sy     = int(lesion_x.shape[1]/2)
            
            if st[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy].shape == lesion_x.shape:
                st[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy]  =  lesion_y*st[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy]
                st[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy]  =  lesion_x + st[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy]

                m1 = segmentation_mask[t]#np.zeros(st.shape)
                m1[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy]  += 1-lesion_y
                m1         = m1*mask[t]
                m1[m1 > 0] = 1

                x_array.append(np.expand_dims(st,          axis=0))
                x_array_lungmask.append(np.expand_dims(m1, axis=0))

                count = count+1

    x_array          = np.array(x_array)
    x_array_lungmask = np.array(x_array_lungmask)
    
    return x_array, x_array_lungmask

def get_multiple_augmented_slice(batch_size, read_ids, lesion_shapes_x, lesion_shapes_y):
    x_array          = []
    x_array_lungmask = []
    
    index   = random.randint(0, len(read_ids)-1)
    #print(read_ids[index])
    
    p       = read_ids[index].split('_')[0]
    types   = 'CT-1'#read_ids[index].split('_')[1]
    count   = 0
    
    name     = basepath+'studies/'+types+'/'
    maskname = name+'study_'+p+'_mask.nii.gz'
    volname  = name+'study_'+p+'.nii.gz'
    
    segmentation_mask = basepath+'masks/'
    segmentation_mask = segmentation_mask+'study_'+p+'_mask.nii.gz'
    
    mask     = sitk.GetArrayFromImage(sitk.ReadImage(maskname))
    vol      = (sitk.GetArrayFromImage(sitk.ReadImage(volname))+1024.0)/1024.0
    segmentation_mask = sitk.GetArrayFromImage(sitk.ReadImage(segmentation_mask))
    
    mask[mask > 0] = 1
    count          = 0
    
    while(count < batch_size):
        t     = np.random.randint(0, mask.shape[0]-1)
        temp  = np.count_nonzero(mask[t].flatten())
        
        # Check if lung region is present
        if temp > 0:
            st  = vol[t]
            #segmen
            ipl, jpl = np.nonzero(mask[t])
            
            lesion_count = random.randint(0, 5)
            temp_count   = 0
            
            while(temp_count < lesion_count):
                index = random.randint(0, len(ipl)-1)

                i = ipl[index]
                j = jpl[index]

                lesion_index = random.randint(0, len(lesion_shapes_x)-1)

                lesion_x     = lesion_shapes_x[lesion_index]
                lesion_y     = lesion_shapes_y[lesion_index]

                sx     = int(lesion_x.shape[0]/2)
                sy     = int(lesion_x.shape[1]/2)

                if st[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy].shape == lesion_x.shape:
                    st[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy]  =  lesion_y*st[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy]
                    st[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy]  =  lesion_x + st[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy]

                    m1 = segmentation_mask[t]#np.zeros(st.shape)
                    m1[i-sx:i+lesion_x.shape[0]-sx, j-sy:j+lesion_x.shape[1]-sy]  += 1-lesion_y
                    m1         = m1*mask[t]
                    m1[m1 > 0] = 1
                    segmentation_mask[t] = m1
                    temp_count           = temp_count + 1
            
            x_array.append(np.expand_dims(st,          axis=0))
            x_array_lungmask.append(np.expand_dims(m1, axis=0))
            
            count = count+1

    x_array          = np.array(x_array)
    x_array_lungmask = np.array(x_array_lungmask)
    
    return x_array, x_array_lungmask

def plot_figure_slope(model_save_name):
    N = 2
    a = val_dice_array1#np.convolve(val_dice_array1, np.ones((N,))/N, mode='valid')
    b = train_dice_array1#np.convolve(train_dice_array1, np.ones((N,))/N, mode='valid')
    c = test_dice_array1#np.convolve(test_dice_array1, np.ones((N,))/N, mode='valid')
    
    temp  = 0
    slope = 0
    #np.abs(np.abs(b[i]-b[i-1])-np.abs(a[i]-a[i-1])) < 0.1 and
    for i in range(1, len(a)):
        if b[i] >= b[i-1] and a[i] >= a[i-1]:
            temp  = i#np.argmax(a)
            slope = b[i]-b[i-1]-(a[i]-a[i-1])
            #print(i, slope, np.abs(b[i]-b[i-1]), np.abs(a[i]-a[i-1]), b[i], b[i-1])
    
    import matplotlib.pyplot as plt
    plt.plot(a)
    plt.plot(b)
    plt.plot(c)
    plt.ylabel('some numbers')
    plt.annotate('Index '+str(temp), xy=(0.75, 0.25), xycoords='axes fraction')
    plt.annotate('Train '+str(round(b[temp], 3)), xy=(0.75, 0.20), xycoords='axes fraction')
    plt.annotate('Val   '+str(round(a[temp], 3)), xy=(0.75, 0.15), xycoords='axes fraction')
    plt.annotate('Test  '+str(round(c[temp], 3)), xy=(0.75, 0.10), xycoords='axes fraction')
    plt.annotate('Slope '+str(round(slope, 3)),   xy=(0.75, 0.05), xycoords='axes fraction')
    #plt.text(6, 0, )
    #plt.text(6, 0.1, 'Val   '+str(round(a[temp], 3)))
    #plt.text(6, 0.2, 'Train '+str(round(b[temp], 3)))
    #plt.text(6, 0.3, 'Test  '+str(round(c[temp], 3)))
    
    plt.savefig(model_save_name+".png")
    
    plt.close()
    plt.clf()
    
    return

def sort_data(trainx1, trainy1):
    # Sort the data
    X = trainx1
    Y = trainy1
    r = [t for t in sorted(zip(Y,X), key=lambda pair: np.sum(pair[0].flatten()))]
    
    trainx = []
    trainy = []
    
    for i in range(len(X)):
        trainy.append(r[i][0])
        trainx.append(r[i][1])
    
    trainx = np.array(trainx)
    trainy = np.array(trainy)
    
    return trainx, trainy

# [STAR] For creating the Data Loaders

def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        LoadNiftid(keys),
        AddChanneld(keys),
        Orientationd(keys, axcodes="LPS"),
        Spacingd(keys, pixdim=(0.75, 0.75, 2.25), mode=("bilinear", "nearest")[: len(keys)]),
        ScaleIntensityRanged(keys[0], a_min=-1000.0, a_max=500.0, b_min=0.0, b_max=1.0, clip=True),
    ]
    if mode == "train":
        xforms.extend(
            [
                SpatialPadd(keys, spatial_size=(192, 192, -1), mode="reflect"),  # ensure at least 192x192
                RandAffined(
                    keys,
                    prob=0.15,
                    rotate_range=(-0.05, 0.05),
                    scale_range=(-0.1, 0.1),
                    mode=("bilinear", "nearest"),
                    as_tensor_output=False,
                ),
                RandCropByPosNegLabeld(keys, label_key=keys[1], spatial_size=(192, 192, 16), num_samples=3),
                #RandGaussianNoised(keys[0], prob=0.15, std=0.01),
                RandFlipd(keys, spatial_axis=0, prob=0.5),
                RandFlipd(keys, spatial_axis=1, prob=0.5),
                RandFlipd(keys, spatial_axis=2, prob=0.5),
            ]
        )
        dtype = (np.float32, np.uint8)
    if mode == "val":
        dtype = (np.float32, np.uint8)
    if mode == "infer":
        dtype = (np.float32,)
    xforms.extend([CastToTyped(keys, dtype=dtype), ToTensord(keys)])
    return monai.transforms.Compose(xforms)

class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy

class MyCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        # dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return cross_entropy

class myBCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice(y_pred, y_true)
        # CrossEntropyLoss target needs to have shape (B, D, H, W)
        # Target from pipeline has shape (B, 1, D, H, W)
        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
        return dice + cross_entropy

def get_inferer(_mode=None):
    """returns a sliding window inference instance."""

    patch_size = (192, 192, 16)
    sw_batch_size, overlap = 2, 0.5
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer

data_folder  = '/media/yu-hao/WindowsData/COVID-19-20_v2/Train/'
model_folder = './runs4/'

monai.config.print_config()
monai.utils.set_determinism(seed=0)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

"""run a training pipeline."""

images = sorted(glob.glob(os.path.join(data_folder, "*_ct.nii.gz")))
labels = sorted(glob.glob(os.path.join(data_folder, "*_seg.nii.gz")))
#logging.info(f"training: image/label ({len(images)}) folder: {data_folder}")

amp  = True  # auto. mixed precision
keys = ("image", "label")
train_frac, val_frac = 0.8, 0.2
n_train = int(train_frac * len(images)) + 1
n_val   = min(len(images) - n_train, int(val_frac * len(images)))
#logging.info(f"training: train {n_train} val {n_val}, folder: {data_folder}")

train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[:n_train], labels[:n_train])]
val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(images[-n_val:], labels[-n_val:])]

# create a training data loader
batch_size = 2
train_transforms = get_xforms("train", keys)
train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.1)
train_loader = monai.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

# create a validation data loader
val_transforms = get_xforms("val", keys)
val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, cache_rate=0.1)
val_loader = monai.data.DataLoader(
    val_ds,
    batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

# create BasicUNet, DiceLoss and Adam optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net    = SUNet_3D(1, 2)
net    = net.to(device)
net.load_state_dict(torch.load('/home/yu-hao/SEMISUNET/runs3/net_key_metric=0.6124.pt'))

max_epochs, lr, momentum = 500, 1e-4, 0.95
logging.info(f"epochs {max_epochs}, lr {lr}, momentum {momentum}")
opt = torch.optim.Adam(net.parameters(), lr=lr)

# create evaluator (to be used to measure model quality during training
val_post_transform = monai.transforms.Compose(
    [AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=True, n_classes=2)]
)
val_handlers = [
    ProgressBar(),
    CheckpointSaver(save_dir=model_folder, save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=3),
]



evaluator = monai.engines.SupervisedEvaluator(
    device=device,
    val_data_loader=val_loader,
    network=net,
    inferer=get_inferer(),
    post_transform=val_post_transform,
    key_val_metric={
        "val_mean_dice": MeanDice(include_background=False, output_transform=lambda x: (x["pred"], x["label"]))
    },
    val_handlers=val_handlers,
    amp=amp,
)

# evaluator as an event handler of the trainer
train_handlers = [
    ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
    StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
]
trainer = monai.engines.SupervisedTrainer(
    device=device,
    max_epochs=max_epochs,
    train_data_loader=train_loader,
    network=net,
    optimizer=opt,
    loss_function=MyCELoss(),
    inferer=get_inferer(),
    key_train_metric=None,
    train_handlers=train_handlers,
    amp=amp,
)
trainer.run()


def infer(data_folder=".", model_folder="runs", prediction_folder="output"):
    """
    run inference, the output folder will be "./output"
    """
    ckpts = sorted(glob.glob(os.path.join(model_folder, "*.pt")))
    ckpt = ckpts[-1]
    for x in ckpts:
        logging.info(f"available model file: {x}.")
    logging.info("----")
    logging.info(f"using {ckpt}.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #net = get_net().to(device)
    net    = SUNet_3D(1, 2)
    net    = net.to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()

    image_folder = os.path.abspath(data_folder)
    images = sorted(glob.glob(os.path.join(image_folder, "*_ct.nii.gz")))
    logging.info(f"infer: image ({len(images)}) folder: {data_folder}")
    infer_files = [{"image": img} for img in images]

    keys = ("image",)
    infer_transforms = get_xforms("infer", keys)
    infer_ds = monai.data.Dataset(data=infer_files, transform=infer_transforms)
    infer_loader = monai.data.DataLoader(
        infer_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    inferer = get_inferer()
    saver = monai.data.NiftiSaver(output_dir=prediction_folder, mode="nearest")
    with torch.no_grad():
        for infer_data in infer_loader:
            logging.info(f"segmenting {infer_data['image_meta_dict']['filename_or_obj']}")
            preds = inferer(infer_data[keys[0]].to(device), net)
            n = 1.0
            for _ in range(4):
                # test time augmentations
                _img = RandGaussianNoised(keys[0], prob=1.0, std=0.01)(infer_data)[keys[0]]
                pred = inferer(_img.to(device), net)
                preds = preds + pred
                n = n + 1.0
                for dims in [[2], [3]]:
                    flip_pred = inferer(torch.flip(_img.to(device), dims=dims), net)
                    pred = torch.flip(flip_pred, dims=dims)
                    preds = preds + pred
                    n = n + 1.0
            preds = preds / n
            preds = (preds.argmax(dim=1, keepdims=True)).float()
            saver.save_batch(preds, infer_data["image_meta_dict"])

    # copy the saved segmentations into the required folder structure for submission
    submission_dir = os.path.join(prediction_folder, "to_submit")
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    files = glob.glob(os.path.join(prediction_folder, "volume*", "*.nii.gz"))
    for f in files:
        new_name = os.path.basename(f)
        new_name = new_name[len("volume-covid19-A-0") :]
        new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
        to_name = os.path.join(submission_dir, new_name)
        shutil.copy(f, to_name)
    logging.info(f"predictions copied to {submission_dir}.")

#infer(data_folder="/media/yu-hao/WindowsData/COVID-19-20_v2/Validation/", model_folder="./runs3", prediction_folder="output")
#prediction_folder = './output/'
#submission_dir = os.path.join(prediction_folder, "to_submit")
#if not os.path.exists(submission_dir):
#    os.makedirs(submission_dir)
#files = glob.glob(os.path.join(prediction_folder, "volume*", "*.nii.gz"))
#for f in files:
#    new_name = os.path.basename(f)
#    new_name = new_name[len("volume-covid19-A-0") :]
#    new_name = new_name[: -len("_ct_seg.nii.gz")] + ".nii.gz"
#    to_name = os.path.join(submission_dir, new_name)
#    shutil.copy(f, to_name)
#    logging.info(f"predictions copied to {submission_dir}.")
