  # Meta-learning Based Incremental Few-Shot Object Detection

The code of  this paper is based on [*Object as Points*](https://github.com/xingyizhou/CenterNet).

## Installation

The code was tested on Ubuntu 14.04, with [Anaconda](https://www.anaconda.com/download) Python 3.6 and [PyTorch]((http://pytorch.org/)) v0.4.1, CUDA9.2. NVIDIA GPUs are needed for both training and testing. 

Please refer to the installation steps of  [*Object as Points*](https://github.com/xingyizhou/CenterNet), which are also listed below.

 After install Anaconda and activate your virtual environment, you can do:

1. Install pytorch0.4.1:

```shell
    conda install pytorch=0.4.1 torchvision -c pytorch
```

​    	And disable cudnn batch normalization(Due to [this issue](https://github.com/xingyizhou/pytorch-pose-hg-3d/issues/16)).

```shell
    \# PYTORCH=/path/to/pytorch # usually ~/anaconda3/envs/CenterNet/lib/python3.6/site-packages/

​    \# for pytorch v0.4.0
​    sed -i "1194s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py

​    \# for pytorch v0.4.1
​    sed -i "1254s/torch\.backends\.cudnn\.enabled/False/g" ${PYTORCH}/torch/nn/functional.py
```

​     For other pytorch version, you can manually open `torch/nn/functional.py` and find the line with `torch.batch_norm` and replace the `torch.backends.cudnn.enabled` with `False`. We observed slight worse training results without doing so. 

​	Note that the version of pytorch should be consistent with your CUDA version.

2. Install the requirements

```shell
    pip install cython
    pip install -r requirements.txt       
```

3. Compile deformable convolutional (from [DCNv2](https://github.com/CharlesShang/DCNv2/tree/pytorch_0.4)).

```
    cd $CenterNet_ROOT/src/lib/models/networks/DCNv2
    ./make.sh
```

4. [Optional, only required if you are using extremenet or multi-scale testing] Compile NMS if your want to use multi-scale testing or test ExtremeNet.

```
cd $CenterNet_ROOT/src/lib/external
make
```
## Dataset preparation

### COCO

- Download the images (2017 Train, 2017 Val, 2017 Test) from [coco website](http://cocodataset.org/#download).

- Download annotation files (2017 train/val and test image info) from [coco website](http://cocodataset.org/#download). 

- Place the data (or create symlinks) to make the data folder like:

  ~~~
  ${CenterNet_ROOT}
  |-- data
  `-- |-- coco
      `-- |-- annotations
          |   |-- instances_train2017.json
          |   |-- instances_val2017.json
          |   |-- person_keypoints_train2017.json
          |   |-- person_keypoints_val2017.json
          |   |-- image_info_test-dev2017.json
          |---|-- train2017
          |---|-- val2017
          `---|-- test2017
  ~~~


### Pascal VOC

- Run

  ~~~
  cd $CenterNet_ROOT/tools/
  bash get_pascal_voc.sh
  ~~~

- The above script includes:

  - Download, unzip, and move Pascal VOC images from the [VOC website](http://host.robots.ox.ac.uk/pascal/VOC/). 
  - [Download](https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip) Pascal VOC annotation in COCO format (from [Detectron](https://github.com/facebookresearch/Detectron/tree/master/detectron/datasets/data)). 
  - Combine train/val 2007/2012 annotation files into a single json. 


- Move the created `voc` folder to `data` (or create symlinks) to make the data folder like:

  ~~~
  ${CenterNet_ROOT}
  |-- data
  `-- |-- voc
      `-- |-- annotations
          |   |-- pascal_trainval0712.json
          |   |-- pascal_test2017.json
          |-- images
          |   |-- 000001.jpg
          |   ......
          `-- VOCdevkit
  
  ~~~



## Training

Before getting started, make sure you have finished installation and datasets setting.

Run:

```shell
cd experiments
sh ctdet_coco_res101.sh
```











