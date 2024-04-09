# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
import os
import nibabel as nib
import numpy as np
from mmengine.utils import mkdir_or_exist
from PIL import Image
from sklearn.model_selection import train_test_split



def get_file_list_recursive(folder_path, extension='.npy'):
    """Recursively get a list of files with the specified extension in the given folder."""
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

def read_npy_file(file_path):
    return np.load(file_path)

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
    maped_label[label == 3] = 3
    return maped_label


def pares_args():
    parser = argparse.ArgumentParser(
        description='Convert synapse dataset to mmsegmentation format')
    parser.add_argument(
        '--dataset-path', type=str, help='synapse dataset path.')
    parser.add_argument(
        '--save-path',
        default='data/synapse',
        type=str,
        help='save path of the dataset.')
    args = parser.parse_args()
    return args


def main():
    dataset_path = 'D:/Dataset/Brats2020_Training'
    save_path = 'D:/Dataset/Brats2020_Validation_mmseg_format'

    # Get lists of image and mask files
    image_files = get_file_list_recursive(os.path.join(dataset_path, 'images'))
    mask_files = get_file_list_recursive(os.path.join(dataset_path, 'masks'))

    # Split the data into training and validation sets
    image_files_train, image_files_val, mask_files_train, mask_files_val = train_test_split(
        image_files, mask_files, test_size=0.2, random_state=42
    )

    # Create save directories for training and validation
    mkdir_or_exist(os.path.join(save_path, 'img_dir/train'))
    mkdir_or_exist(os.path.join(save_path, 'img_dir/val'))
    mkdir_or_exist(os.path.join(save_path, 'ann_dir/train'))
    mkdir_or_exist(os.path.join(save_path, 'ann_dir/val'))

    # Process training data
    for i, img_file in enumerate(image_files_train):
        idx = img_file.split('_')[-1].split('.')[0]

        img_3d = read_npy_file(os.path.join(dataset_path, 'images', img_file))
        label_3d = read_npy_file(os.path.join(dataset_path, 'masks', f'mask_{idx}.npy'))

        # Your preprocessing logic here
        # ...

        for c in range(img_3d.shape[0]):
            img = img_3d[c]
            label = label_3d[c]

            img = Image.fromarray(img).convert('RGB')
            label = Image.fromarray(label).convert('L')
            img.save(
                os.path.join(save_path, 'img_dir/train', f'case{idx.zfill(4)}_slice{str(c).zfill(3)}.jpg'))
            label.save(
                os.path.join(save_path, 'ann_dir/train', f'case{idx.zfill(4)}_slice{str(c).zfill(3)}.png'))

    # Process validation data
    for i, img_file in enumerate(image_files_val):
        idx = img_file.split('_')[-1].split('.')[0]

        img_3d = read_npy_file(os.path.join(dataset_path, 'image', img_file))
        label_3d = read_npy_file(os.path.join(dataset_path, 'masks', f'mask_{idx}.npy'))

        # Your preprocessing logic here
        # ...

        for c in range(img_3d.shape[0]):
            img = img_3d[c]
            label = label_3d[c]

            img = Image.fromarray(img).convert('RGB')
            label = Image.fromarray(label).convert('L')
            img.save(
                os.path.join(save_path, 'img_dir/val', f'case{idx.zfill(4)}_slice{str(c).zfill(3)}.jpg'))
            label.save(
                os.path.join(save_path, 'ann_dir/val', f'case{idx.zfill(4)}_slice{str(c).zfill(3)}.png'))

if __name__ == '__main__':
    main()