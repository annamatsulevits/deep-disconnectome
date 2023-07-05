# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:00:31 2023

@author: Lia
"""
import os
import torch
import nibabel as nib
import numpy as np
from monai import transforms
from pathlib import Path


lesion = nib.load('D:/data/data_train/Disco_synth/vol0007.nii.gz').get_fdata()

lesion_data = torch.from_numpy(np.array(lesion)).unsqueeze(0)


tf_lesion = transforms.Resize(spatial_size=(96, 96, 96), mode = 'nearest')

test_tf = tf_lesion(lesion_data)
####


img = nib.load('D:/data/data_train/Disco_synth/vol0007.nii.gz')
fname = "testing"
affine = img.affine
header = img.header
lesion_tf = nib.Nifti1Image(np.squeeze(test_tf).astype(np.float32), affine = affine, header= header)
lesion_tf.to_filename('D:/data/lesion_transformation.nii.gz')

# u = nib.load('D:/data/lesion_transformation.nii.gz')
# print(u.shape)



# set up the transformation
tf_lesion = transforms.Resize(spatial_size=(96, 96, 96), mode='nearest')
tf_disco = transforms.Resize(spatial_size=(96, 96, 96))

# set up input and output directories
input_dir = 'D:/data/13_patients_for_validation/Lesions'
output_dir = 'D:/data/13_patients_for_validation/Lesions_transformed'

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