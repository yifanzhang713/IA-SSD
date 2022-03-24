# IA-SSD

Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds (CVPR 2022)
## Introduction
<!-- **Abstract**: -->

This is the official implementation of ***IA-SSD*** (CVPR 2022), a simple and highly efficient point-based detector for 3D LiDAR point clouds. For more details, please refer to:

**Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds** <br />
Yifan Zhang, Qingyong Hu*, Guoquan Xu, Yanxin Ma, Jianwei Wan, Yulan Guo

**[[Paper](https://arxiv.org/abs/2203.11139)]
<p align="center"> <img src="docs/imgs/IA-SSD.png" width="100%"> </p>


## Getting Started
### Installation

a. Clone this repository.
```shell
git clone https://github.com/yifanzhang713/IA-SSD.git && cd IA-SSD
```
b. Prepare the environment

We deploy the project with detailed environment follows:
* Ubuntu18.04/20.04
* Python = 3.7
* PyTorch = 1.1
* CUDA = 10.0
* CMake >= 3.13
* spconv = 1.0 (or just comment the relevant code as it is not used in our work)


You can also install higher versions above, please refer to  [OpenPCDet.md](OpenPCDet.md) and the [official github repository](https://github.com/open-mmlab/OpenPCDet) for more information. **Note that the max parallelism might be slightly lower because of the larger initial GPU memory footprint with updated `Pytorch` version.**


c. Install `pcdet` toolbox.
```shell
pip install -r requirements.txt
python setup.py develop
```

d. Prepare the dataset, please refer to [GETTING_STARTED.MD](docs/GETTING_STARTED.md)


### DEMO
We provide the pre-trained weight file so you can just run with that:
```shell
cd tools 
# To achieve fully GPU memory footprint (NVIDIA RTX2080Ti, 11GB).
python test.py --cfg_file cfgs/kitti_models/IA-SSD.yaml --batch_size 100 \
    --ckpt IA-SSD.pth --set MODEL.POST_PROCESSING.RECALL_MODE 'speed'

# To reduce the pressure on the CPU during preprocessing, a suitable batchsize is recommended, e.g. 16. (Over 5 batches per second on RTX2080Ti)
python test.py --cfg_file cfgs/kitti_models/IA-SSD.yaml --batch_size 16 \
    --ckpt IA-SSD.pth --set MODEL.POST_PROCESSING.RECALL_MODE 'speed' 
```
* Then you can get results similar to the following:
```shell
INFO  Car AP@0.70, 0.70, 0.70:
bbox AP:96.4014, 90.1601, 89.4433
bev  AP:90.3880, 88.7087, 86.7245
3d   AP:89.4686, 79.5601, 78.4456
aos  AP:96.36, 90.07, 89.27
Car AP_R40@0.70, 0.70, 0.70:
bbox AP:97.9318, 95.2306, 92.6986
bev  AP:94.8646, 91.1780, 88.7252
3d   AP:91.8687, 83.4215, 80.3901
aos  AP:97.90, 95.11, 92.49
Car AP@0.70, 0.50, 0.50:
bbox AP:96.4014, 90.1601, 89.4433
bev  AP:96.5538, 90.2454, 89.7374
3d   AP:96.4992, 90.2306, 89.6941
aos  AP:96.36, 90.07, 89.27
Car AP_R40@0.70, 0.50, 0.50:
bbox AP:97.9318, 95.2306, 92.6986
bev  AP:98.0430, 95.6010, 94.9939
3d   AP:98.0113, 95.5462, 94.8666
aos  AP:97.90, 95.11, 92.49
Pedestrian AP@0.50, 0.50, 0.50:
bbox AP:71.7005, 69.1174, 63.6240
bev  AP:67.4771, 61.4352, 56.8311
3d   AP:62.3764, 57.0564, 51.4625
aos  AP:66.24, 63.35, 58.28
Pedestrian AP_R40@0.50, 0.50, 0.50:
bbox AP:72.5850, 69.3628, 64.8827
bev  AP:67.1904, 61.9465, 55.9378
3d   AP:61.7203, 55.8031, 50.8575
aos  AP:66.30, 62.72, 58.37
Pedestrian AP@0.50, 0.25, 0.25:
bbox AP:71.7005, 69.1174, 63.6240
bev  AP:81.2598, 77.5839, 72.8308
3d   AP:81.2277, 77.5279, 72.6325
aos  AP:66.24, 63.35, 58.28
Pedestrian AP_R40@0.50, 0.25, 0.25:
bbox AP:72.5850, 69.3628, 64.8827
bev  AP:82.4582, 78.9957, 74.4619
3d   AP:82.4264, 78.9272, 73.5677
aos  AP:66.30, 62.72, 58.37
Cyclist AP@0.50, 0.50, 0.50:
bbox AP:95.8926, 77.6564, 76.0060
bev  AP:88.0179, 74.1837, 71.2950
3d   AP:86.6539, 71.4028, 66.1070
aos  AP:95.70, 77.26, 75.57
Cyclist AP_R40@0.50, 0.50, 0.50:
bbox AP:96.7458, 80.4641, 77.0962
bev  AP:93.2066, 75.6620, 72.1965
3d   AP:91.4923, 71.8465, 67.6034
aos  AP:96.54, 80.01, 76.61
Cyclist AP@0.50, 0.25, 0.25:
bbox AP:95.8926, 77.6564, 76.0060
bev  AP:94.6056, 75.6408, 73.7898
3d   AP:94.6056, 75.6408, 73.7898
aos  AP:95.70, 77.26, 75.57
Cyclist AP_R40@0.50, 0.25, 0.25:
bbox AP:96.7458, 80.4641, 77.0962
bev  AP:95.6718, 78.0718, 74.7331
3d   AP:95.6718, 78.0718, 74.7331
aos  AP:96.54, 80.01, 76.61
```



### Training
The configuration file is in tools/cfgs/kitti_models/IA-SSD.yaml, and the training scripts is in tools/scripts.

Train with single or multiple GPUs:
```shell
python train.py --cfg_file cfgs/kitti_models/IA-SSD.yaml

# or 

sh scripts/dist_train.sh ${NUM_GPUS} --cfg_file cfgs/kitti_models/IA-SSD.yaml
```


### Evaluation
The configuration file is in tools/cfgs/kitti_models/IA-SSD.yaml, and the training scripts is in tools/scripts.

Evaluate with single or multiple GPUs:
```shell
python test.py --cfg_file cfgs/kitti_models/IA-SSD.yaml  --batch_size ${BATCH_SIZE} --ckpt ${PTH_FILE}

# or

sh scripts/dist_test.sh ${NUM_GPUS} \
    --cfg_file cfgs/kitti_models/IA-SSD.yaml --batch_size ${BATCH_SIZE} --ckpt ${PTH_FILE}
```


Quantitative results of different approaches on KITTI dataset (*test* set):
<p align="center"> <img src="docs/imgs/kitti_test.png" width="100%"> </p>

Qualitative results of our IA-SSD: 
| ![z](docs/imgs/kitti_1.gif)    | ![z](docs/imgs/kitti_2.gif)   |
| -------------------------------- | ------------------------------- |
| ![z](docs/imgs/kitti_3.gif)    | ![z](docs/imgs/kitti_4.gif)   |
* Here we deploy the model on the KITTI Tracking dataset for consecutive results.

Quantitative results of different approaches on Waymo dataset (*validation* set):
<p align="center"> <img src="docs/imgs/waymo_val.png" width="100%"> </p>

Qualitative results of our IA-SSD:

| ![z](docs/imgs/waymo_1.gif)    | ![z](docs/imgs/waymo_2.gif)   |
| -------------------------------- | ------------------------------- |
| ![z](docs/imgs/waymo_3.gif)    | ![z](docs/imgs/waymo_4.gif)   |


Quantitative results of different approaches on ONCE dataset (*validation* set):
<p align="center"> <img src="docs/imgs/once_val.png" width="100%"> </p>

Qualitative result of our IA-SSD:
<p align="center"> <img src="docs/imgs/once.gif" width="90%"> </p>



## Citation 
If you find this project useful in your research, please consider citing:

```
@inproceedings{zhang2022IASSD,
  title={Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds},
  author={Zhang, Yifang and Hu, Qingyong and Xu, Guoquan and Ma, Yanxin and Wan, Jianwei and Guo, Yulan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Acknowledgement
-  This work is built upon the `OpenPCDet` (version `0.5`), an open source toolbox for LiDAR-based 3D scene perception. Please refer to [OpenPCDet.md](OpenPCDet.md) and the [official github repository](https://github.com/open-mmlab/OpenPCDet) for more information.

-  Parts of our Code refer to <a href="https://github.com/qiqihaer/3DSSD-pytorch-openPCDet">3DSSD-pytroch</a> library and the the recent work <a href="https://github.com/blakechen97/SASA">SASA</a>.



## License

This project is released under the [Apache 2.0 license](LICENSE).