# import numpy as np
# import nibabel as nib
# import glob
# import matplotlib.pyplot as plt
# from tifffile import imsave
# # from sklearn.preprocessing import MinMaxScaler
# # scaler = MinMaxScaler()
# import os

import os
import nibabel as nib
import numpy as np
from PIL import Image
import glob
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

data_dir = 'D:/Dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'

# Create directories for images and masks
os.makedirs(os.path.join('D:/Dataset/Brats2020_Validation', 'images'), exist_ok=True)
os.makedirs(os.path.join('D:/Dataset/Brats2020_Validation', 'masks'), exist_ok=True)

# t2_list = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii'))
t1ce_list = sorted(glob.glob('D:/Dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'))
# flair_list = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii'))
mask_list = sorted(glob.glob('D:/Dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii'))

for img in range(len(t1ce_list)):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)

    # temp_image_t2=nib.load(t2_list[img]).get_fdata()
    # temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

    temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

    # temp_image_flair=nib.load(flair_list[img]).get_fdata()
    # temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)

    temp_mask=nib.load(mask_list[img]).get_fdata()
    temp_mask=temp_mask.astype(np.uint8)
    temp_mask[temp_mask==4] = 3  #Reassign mask values 4 to 3
    #print(np.unique(temp_mask))


    # temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

    #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
    #cropping x, y, and z
    temp_image_t1ce=temp_image_t1ce[56:184, 56:184, 13:141]
    temp_mask = temp_mask[56:184, 56:184, 13:141]

    val, counts = np.unique(temp_mask, return_counts=True)

    if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
        print("Save Me")
        # temp_mask= to_categorical(temp_mask, num_classes=4)
        np.save('D:/Dataset/Brats2020_Training/images/image_'+str(img)+'.npy', temp_image_t1ce)

        np.save('D:/Dataset/Brats2020_Training/masks/mask_'+str(img)+'.npy', temp_mask)

    else:
        print("I am useless")


import splitfolders

input_folder = 'D:/Dataset/Brats2020_Training' 
output_folder = 'D:/Dataset/Brats2020_Training_mmseg'

splitfolders.ratio(input_folder, output_folder, seed=42, ratio=(.75, .25), group_prefix=None)
