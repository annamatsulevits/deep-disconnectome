# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:25:39 2023

@author: Anna Matsulevits 
"""

import torch
import torch.nn.functional as F
from torch import nn


class Conv_block(nn.Module):
    def __init__(self, in_channels, first_layer=False, first_channels=None, drop_rate=0.5, kernel_size=3, padding=1, pool_size=2):
        super(Conv_block, self).__init__()
        if first_layer:
            assert first_channels is not None
        else:
            first_channels = in_channels
        self.conv = nn.Sequential(
            nn.BatchNorm3d(in_channels) if not first_layer else nn.Sequential(),
            nn.Conv3d(in_channels=first_channels, out_channels=in_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(in_channels),
            nn.Conv3d(in_channels=in_channels, out_channels=in_channels *
                      2, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True))
        self.pool = nn.Sequential(
            nn.MaxPool3d(pool_size),
            nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Sequential())

    def forward(self, input):
        conv = self.conv(input)
        pool = self.pool(conv)
        return conv, pool


class Upconv_block(nn.Module):
    def __init__(self, contract_channels, expansive_channels, out_channels, kernel_size=3, padding=1):
        super(Upconv_block, self).__init__()
        self._contract_channels = contract_channels
        self._expansive_channels = expansive_channels
        self.upsample = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.BatchNorm3d(expansive_channels + contract_channels),
            nn.Conv3d(in_channels=expansive_channels + contract_channels,
                      out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True))

    def forward(self, contract_part, expansive_part):
        assert contract_part.shape[1] == self._contract_channels
        assert expansive_part.shape[1] == self._expansive_channels
        x = self.upsample(expansive_part)
        x = torch.cat((contract_part, x), dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    # model UNET 3D
    def __init__(self, in_channels, out_channels, num_filters=24, drop_rate=0.5):
        super(UNet3D, self).__init__()

        self.conv1 = Conv_block(in_channels=num_filters, drop_rate=drop_rate,
                                first_layer=True, first_channels=in_channels)
        self.conv2 = Conv_block(
            in_channels=num_filters * 2, drop_rate=drop_rate)
        self.conv3 = Conv_block(
            in_channels=num_filters * 4, drop_rate=drop_rate)

        self.conv4_batchnorm = nn.BatchNorm3d(num_filters * 8)
        self.conv4 = nn.Conv3d(in_channels=num_filters * 8,
                               out_channels=num_filters * 16, kernel_size=3, padding=1)

        self.upconv5 = Upconv_block(contract_channels=num_filters * 8,
                                    expansive_channels=num_filters * 16, out_channels=num_filters * 8)
        self.upconv6 = Upconv_block(contract_channels=num_filters * 4,
                                    expansive_channels=num_filters * 8, out_channels=num_filters * 4)
        self.upconv7 = Upconv_block(contract_channels=num_filters * 2,
                                    expansive_channels=num_filters * 4, out_channels=num_filters * 4)

        self.conv_output = nn.Conv3d(
            in_channels=num_filters * 4, out_channels=out_channels, kernel_size=3, padding=1)

    def forward(self, input):
        #print(input.shape)
        conv1, pool1 = self.conv1(input)
       # print(conv1.shape)
        conv2, pool2 = self.conv2(pool1)
       # print(conv2.shape)
        conv3, pool3 = self.conv3(pool2)
       # print(conv3.shape)

        conv4 = self.conv4_batchnorm(pool3)
        conv4 = self.conv4(conv4)
        conv4 = F.relu(conv4)
        #print(conv4.shape)

        upconv5 = self.upconv5(conv3, conv4)
        #print(upconv5.shape)
        upconv6 = self.upconv6(conv2, upconv5)
        #print(upconv6.shape)
        upconv7 = self.upconv7(conv1, upconv6)
        #print(upconv7.shape)

        output = self.conv_output(upconv7)
        output = torch.sigmoid(output)
        #print(output.shape)
        #output = torch.clamp(output, min = 0, max = 1)
        
        #raise NotImplementedError()
    
        return output
