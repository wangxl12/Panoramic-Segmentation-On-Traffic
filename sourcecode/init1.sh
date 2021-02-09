#!/bin/bash
cd /usr/local;sudo ln -snf /usr/local/cuda-10.0 cuda
cd /usr/local;ls -l | grep cuda
pip install --upgrade pip
pip install numpy==1.19.1
pip install cython
pip install matplotlib -U
pip install -U torch==1.4.0 
pip install torchvision==0.5
pip install --ignore-installed pyyaml==5.1
pip install --ignore-installed pillow==5.4.1
cd /home/ma-user/work/sourcecode/cocoapi-master/PythonAPI;make
cd /home/ma-user/work/sourcecode
pip install detectron2-0.1+cu100-cp36-cp36m-linux_x86_64.whl -i https://pypi.tuna.tsinghua.edu.cn/simple/ --trusted-host=pypi.tuna.tsinghua.edu.cn/simple ipython
pip install pycocotools
pip install tqdm
cd /home/ma-user
if [ ! -d ".torch" ];then
mkdir .torch
fi
cd .torch
if [ ! -d "fvcore_cache" ];then
mkdir fvcore_cache
fi
cd fvcore_cache
if [ ! -d "detectron2" ];then
mkdir detectron2
fi
cd detectron2
if [ ! -d "COCO-InstanceSegmentation" ];then
mkdir COCO-InstanceSegmentation
fi
cd COCO-InstanceSegmentation
if [ ! -d "mask_rcnn_R_50_FPN_3x" ];then
mkdir mask_rcnn_R_50_FPN_3x
fi
cd mask_rcnn_R_50_FPN_3x
if [ ! -d "137849600" ];then
mkdir 137849600
fi
cd /home/ma-user/.torch/fvcore_cache/detectron2/
if [ ! -d "Misc" ];then
mkdir Misc
fi
cd Misc
if [ ! -d "panoptic_fpn_R_101_dconv_cascade_gn_3x" ];then
mkdir panoptic_fpn_R_101_dconv_cascade_gn_3x
fi
cd panoptic_fpn_R_101_dconv_cascade_gn_3x
if [ ! -d "139797668" ];then
mkdir 139797668
fi
cd /home/ma-user/.torch/fvcore_cache/detectron2/
if [ ! -d "COCO-PanopticSegmentation" ];then
mkdir COCO-PanopticSegmentation
fi
cd COCO-PanopticSegmentation
if [ ! -d "panoptic_fpn_R_101_3x" ];then
mkdir panoptic_fpn_R_101_3x
fi
cd panoptic_fpn_R_101_3x
if [ ! -d "139514519" ];then
mkdir 139514519
fi