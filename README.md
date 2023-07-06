# deep-disconnectome
This repository contains the trained 3D U-Net model and the code to compute a deep-disconnecotme from a binary lesion file.

1) Download the repository, run the files 'dataset_load.py' and 'my_torch_model_fm.py'
2) If needed, use the file image_transformation.py to get your lesion shaped into the correct size of 96x96x96 (2mm voxels).
3) Adjust the paths and run the code lesion_to_disconnectome.py
