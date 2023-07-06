# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:00:31 2023

@author: anna matsulevits
"""
import os
import torch
import nibabel as nib
import numpy as np
from monai import transforms


img = nib.load('EXEMPLAR NIFTI IMAGE FOR CORRECT AFFINE AND HEADER')
fname = "testing"
affine = img.affine
header = img.header

# set up the transformation
tf_lesion = transforms.Resize(spatial_size=(96, 96, 96), mode='nearest')

# set up input and output directories
input_dir = 'ADD INPUT DIRECTORY PATH HERE'
output_dir = 'ADD OUTPUT DIRECTORY PATH HERE'

# make the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# iterate over all .nii.gz files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.nii.gz'):
        # load the image and get the data
        img = nib.load(os.path.join(input_dir, filename))
        data = img.get_fdata()
        
        # apply the transformation
        data_torch = torch.from_numpy(np.array(data)).unsqueeze(0)
        data_torch_tf = tf_lesion(data_torch)
        data_tf = np.squeeze(data_torch_tf).astype(np.float32)
        
        # create a new NIfTI image with the transformed data and save it
        affine = img.affine
        header = img.header
        img_tf = nib.Nifti1Image(data_tf, affine=affine, header=header)
        outfile = os.path.join(output_dir, filename)
        img_tf.to_filename(outfile)