import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import glob
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Define the scaler for normalization
scaler = MinMaxScaler()



def label_mapping(label):
    """Label mapping from TransUNet paper setting. It only has 9 classes, which
    are 'background', 'aorta', 'gallbladder', 'left_kidney', 'right_kidney',
    'liver', 'pancreas', 'spleen', 'stomach', respectively. Other foreground
    classes in original dataset are all set to background.

    More details could be found here: https://arxiv.org/abs/2102.04306
    """
    maped_label = np.zeros_like(label)
    maped_label[label == 0] = 0
    maped_label[label == 1] = 1
    maped_label[label == 2] = 2
    maped_label[label == 4] = 3
    return maped_label




def crop_to_divisible_by_64(image, mask):
    """
    Crops the image and mask to dimensions that are divisible by 64.
    """
    # Calculate the dimensions that are divisible by 64
    # x_start = 0
    # x_end = image.shape[0] - (image.shape[0] % 64)
    # y_start = 0
    # y_end = image.shape[1] - (image.shape[1] % 64)
    # z_start = 0
    # z_end = image.shape[2] - (image.shape[2] % 64)
    
    # Crop the image and mask
    cropped_image = image[56:184, 56:184, 13:141]
    cropped_mask = mask[56:184, 56:184, 13:141]
    
    return cropped_image, cropped_mask

def apply_custom_colormap_on_mask(mask):
    """
    Apply a custom colormap on the mask to visualize different regions.
    """
    # Define the custom colormap
    colors = [[68, 0, 84], [59, 82, 139], [24, 184, 128], [230, 215, 79]]
    cmap = mcolors.ListedColormap(colors)
    
    # Apply the colormap to the mask
    colored_mask = cmap(mask)
    
    # Convert the colored mask to RGB format
    colored_mask_rgb = np.uint8(colored_mask[:, :, :3] * 255)
    
    return colored_mask_rgb
def read_nii_file(file_path):
    img = nib.load(file_path).get_fdata()
    return img

def save_2d_slices(img, mask, file_prefix, data_split):
    for slice_idx in range(img.shape[0]): # Iterate over each slice (depth)
        # Extract 2D slice for image and mask
        slice_2d_img = img[slice_idx, :, :]
        slice_2d_mask = mask[slice_idx, :, :]
        # Normalize and save the image slice as RGB
        # slice_2d_img = np.clip(slice_2d_img, 0, 255)
        # slice_2d_img = slice_2d_img.astype(np.uint8)
        
        # img_slice_rgb = np.stack((slice_2d_img,)*3, axis=-1) # Convert to RGB
        img_file_name = f"{file_prefix}_slice{slice_idx}_image.jpg"
        img_save_path = os.path.join('D:/Dataset/BraTS2020_Training_mmseg_png', 'img_dir', data_split, img_file_name)
        Image.fromarray(slice_2d_img).save(img_save_path)
        print(f"Saved image slice: {img_save_path}")
        

        slice_2d_mask = slice_2d_mask.astype(np.uint8)
        # slice_2d_mask[slice_2d_mask==4] = 3

        # Apply custom colormap on the mask and save it as RGB
        colored_mask_rgb = apply_custom_colormap_on_mask(slice_2d_mask)
        mask_file_name = f"{file_prefix}_slice{slice_idx}_mask.png"
        mask_save_path = os.path.join('D:/Dataset/BraTS2020_Training_mmseg_png', 'ann_dir', data_split, mask_file_name)
        Image.fromarray(colored_mask_rgb).save(mask_save_path)
        print(f"Saved colored mask slice: {mask_save_path}")


def main():
    dataset_path = 'D:/Dataset/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData'
    save_path = 'D:/Dataset/BraTS2020_Training_mmseg_png'

    # List all T1c image and mask files
    image_files = sorted(glob.glob(os.path.join(dataset_path, '*/*t1ce.nii')))
    mask_files = sorted(glob.glob(os.path.join(dataset_path, '*/*seg.nii')))

    # Ensure the number of image and mask files match
    assert len(image_files) == len(mask_files), "Mismatch in the number of image and mask files."

    # Split the dataset into training and validation sets
    train_files, val_files, train_masks, val_masks = train_test_split(image_files, mask_files, test_size=0.2, random_state=42)

    # Ensure directories exist
    os.makedirs(os.path.join(save_path, 'img_dir/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'ann_dir/train'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'img_dir/val'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'ann_dir/val'), exist_ok=True)

    # Process training data
    for i, (file_path, mask_path) in enumerate(zip(train_files, train_masks)):
        img = read_nii_file(file_path) # Read the 3D image
        mask = read_nii_file(mask_path) # Read the corresponding mask
        
        img = (img + 125) / 400
        img *= 255
        img = np.transpose(img, [2, 0, 1])


        label_3d = np.transpose(mask, [2, 0, 1])
        label_3d = np.flip(label_3d, 2)
        label_3d = label_mapping(label_3d)
        # img, mask = crop_to_divisible_by_64(img, mask)
        
        file_prefix = f"train_image_{i}" # Unique file prefix for training data
        val, counts = np.unique(mask, return_counts=True)
        if (1 - (counts[0]/counts.sum())) > 0.01: # At least 1% useful volume with labels that are not 0 
            print("Save Me")
            save_2d_slices(img, label_3d, file_prefix, 'train')

    # Process validation data
    for i, (file_path, mask_path) in enumerate(zip(val_files, val_masks)):
        img = read_nii_file(file_path) # Read the 3D image
        mask = read_nii_file(mask_path) # Read the corresponding mask


        img = (img + 125) / 400
        img *= 255
        img = np.transpose(img, [2, 0, 1])


        label_3d = np.transpose(mask, [2, 0, 1])
        label_3d = np.flip(label_3d, 2)
        label_3d = label_mapping(label_3d)


        # img, mask = crop_to_divisible_by_64(img, mask)


        file_prefix = f"val_image_{i}" # Unique file prefix for validation data
        val, counts = np.unique(mask, return_counts=True)
        if (1 - (counts[0]/counts.sum())) > 0.01: # At least 1% useful volume with labels that are not 0 
            print("Save Me")
            save_2d_slices(img, label_3d, file_prefix, 'val')

if __name__ == '__main__':
    main()