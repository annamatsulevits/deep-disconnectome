# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 13:25:39 2023

@author: Lia
"""

import torch
import numpy as np
import nibabel as nib
import pandas as pd

from os.path import join
from glob import glob
from torch.utils.data import Dataset
from pathlib import Path 



class Normal_dataset(Dataset):

	def __init__(self, data_root, csv, transformX=None, transform_lesion = None, return_file_info=False, device=None):
		super(Normal_dataset, self).__init__()
		self.csv = pd.read_csv(csv)
		self.lesion_paths = sorted(glob(join(data_root, 'Lesions_synth_transformed', '*nii.gz')))
		#self.disco_paths = sorted(glob(join(data_root, 'Disco_synth_transformed', '*.nii.gz')))
		self.wm_mask = (join(data_root, 'normalized_average_transformed_new.nii.gz'))
		self.transformX = transformX
		self.transform_lesion = transform_lesion
		#self.transformY = transformY
		self.return_file_info = return_file_info
		self.device  = device


	def __len__(self):
		return len(self.csv['lesion'])

	def __getitem__(self, idx: int):
	  	# x is lesion
# 		lesion_path = self.lesion_paths[idx]
# 		wm_mask = self.wm_mask
# 		out_path = self.disco_paths[idx]
		lesion_path = self.csv.loc[idx, 'lesion']
		#tract_path =  self.csv.loc[idx, 'tract']
		wm_mask = self.wm_mask
		out_path = self.csv.loc[idx, 'lesion']
		
		lesion = torch.from_numpy(
					np.asarray(
						nib.load(lesion_path).get_fdata()
					)
				).unsqueeze(0).type(torch.FloatTensor)
		wm_mask = torch.from_numpy(
					np.asarray(
						nib.load(wm_mask).get_fdata()
					)
				).unsqueeze(0).type(torch.FloatTensor)
        
# 		tract = torch.from_numpy(
#  					np.asarray(
# 						nib.load(tract_path).get_fdata()
#  					)
# 				).unsqueeze(0).type(torch.FloatTensor)

	  	#  for y load the disconnectomedataobj
# 		y = torch.from_numpy(
# 					np.asarray(
# 						nib.load(out_path).get_fdata()
# 					)
# 				).unsqueeze(0).type(torch.FloatTensor)
		if self.device != None:
				lesion = lesion.to(self.device)
				wm_mask = wm_mask.to(self.device)
				#tract = tract.to(self.device)
				#y = y.to(self.device)
			
		if self.transformX:
			wm_mask = self.transformX(wm_mask)
		if self.transform_lesion:
			lesion = self.transform_lesion(lesion)
		#if self.transformY:
			#y = self.transformY(y)
			#y = y*10
		#x = torch.cat([ lesion, wm_mask], dim=0)
		x = torch.cat([lesion, wm_mask], dim=0)
		print(x.shape)

		if self.return_file_info:
			
            
			return x, lesion_path
		return x


