The document illustrate how to create conda environment for our project, and reproduce the results.

## System overview

![](/home/zane/Documents/u-mixformer/u-mixformer/system_structure.png)

## Installing

* Create virtual conda environment with python version 3.9 

  ```bash
  conda create -n computer_vision python=3.9
  ```

  

* install pytorch, cudatoolkit and etc

  * Install on Windows

  ```bash
  pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
  ```

  * Install on Ubuntu22.04 LTS

    ```bash
    conda init (shell)
    conda activate computer_vision
    conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
    ```

    

* Install the dependency packages

  ```bash
  pip install -U openmim
  mim install mmengine
  mim install 'mmcv==2.0.0rc4'
  pip install timm
  pip install imagecorruptions
  pip install future tensorboard
  ```

  

* get inside the folder u-mixformer

  ```bash
  cd u-mixformer
  ```

* install the project

  ```bash
  pip install -e .
  ```

  

## Training the project

- to train the project

  ```bash
  python tools/train.py configs/umixformer/umixformer_mit-b0_8xb2-160k_brats2020-adek_extension.py
  ```

## Testing the porject

* Testing the brain project

```bash
python tools/test.py configs/umixformer/umixformer_mit-b0_8xb2-160k_brats2020-adek_extension.py work_dirs/umixformer_mit-b0_8xb2-160k_brats2020-adek_extension/iter_20000.pth	
```

## Visualization the project

* Visualizing the prediction

  ```bash
  python demo/image_demo.py data/img_dir/visualize/image_3557.jpg  configs/umixformer/umixformer_mit-b0_8xb2-160k_brats2020-adek_extension.py work_dirs/umixformer_mit-b0_8xb2-160k_brats2020-adek_extension/iter_20000.pth --out-file data/img_dir/visualize/output_3557.jpg --device cuda:0
  ```


* Compare the difference between the ground truth mask and our prediction using the following command to visualize it.

  If you want to visualize other image, please modify the program directly.

  ```bash
  python tools/img_compare.py	
  ```

* Getting the visualization result of the training process

  ```bash
  tensorboard --logdir .	
  ```



## Data location

The location of our data is in the data folder, the mask is in the ann_dir, image in the img_dir. Some visualized the image are also kept in the img_dir/visualize and not deleted it.

## Result

The follow table shows the saved pth file performance on the corresponding iteration time on the test dataset. 

| iteration | Dice(Tumor) | Acc(Tumor) |
| --------- | ----------- | ---------- |
| 10k       | 82.48       | 80.82      |
| 20k       | 86.89       | 84.30      |
| 30k       | 85.93       | 82.30      |

Following is the visualize result we got from our program.

![](/home/zane/Documents/u-mixformer/u-mixformer/visualize_result.png)

Following is the visualization of our training process.

![](/home/zane/Documents/u-mixformer/result/aACC.png)


## Acknowledgments

This project is built on the foundation of the [U-Mixformer repository](https://github.com/julian-klitzing/u-mixformer) by [Julian Klitzing](https://github.com/julian-klitzing).

We would like to extend our gratitude to Julian Klitzing and all team for their paper and the project.


  
