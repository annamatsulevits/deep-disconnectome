#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 17:40:44 2022

@author: annamatsulevits
"""
import os
import torch

import numpy as np
import nibabel as nib

from tqdm import tqdm
from pathlib import Path
from monai import transforms
from my_torch_model_fm import UNet3D
from dataset_fm import Normal_dataset
from torch.utils.data import DataLoader

from configurations import GPUS, BATCH_SIZE, TRAINED_MODEL, LESIONS

os.environ["CUDA_VISIBLE_DEVICES"] = GPUS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_image_for_affine =  nib.load('path to affine_image')


mask_data = nib.load('path to normalized_average_transformed_new.nii.gz').get_fdata()
mask_data = torch.from_numpy(np.array(mask_data)).to(device)

def test(model, test_loader, transform, save_output, device):
    Path(save_output).mkdir(exist_ok=True, mode=0o777)
    for x_val, file_info in tqdm(test_loader, leave=False, ncols=80):
        x_val = x_val.to(device).type(torch.float32)

        with torch.no_grad():
            model.eval()
            
            y_pred = model(x_val)
            y_pred = y_pred * mask_data

            for i in range(len(y_pred)):
                y_tf = transform(y_pred[i].cpu().numpy())
                fname = Path(file_info[i]).name
                affine = random_image_for_affine.affine
                header = random_image_for_affine.header
                y_tf = nib.Nifti1Image(np.squeeze(y_tf).astype(np.float32), affine = affine, header= header)
                y_tf.to_filename(os.path.join(save_output, fname))

def evaluation():
    tfX = transforms.Compose([
        transforms.Resize(spatial_size=(96, 96, 96)),
        transforms.NormalizeIntensity(),
    ])
    tf_lesion = transforms.Resize(spatial_size=(96, 96, 96), mode = 'nearest')

   
    lesions = 'EXCEL SHEET CONTAINING A LIST OF ALL LESIONS'

    tf_back = transforms.Resize(spatial_size=(91, 109, 91))
    dataset = Normal_dataset(
        LESIONS, lesions, transformX=tfX, transform_lesion=tf_lesion, return_file_info=True)

    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # define that model new!!
    model = UNet3D(
        in_channels=2,
        out_channels=1,
        num_filters=8
    ).to(device)
    model.load_state_dict(torch.load(TRAINED_MODEL))

    
    test(model, test_loader, tf_back, 'PATH TO OUTPUT DIRECTORY', device)


if __name__ == '__main__':
    evaluation()
    
