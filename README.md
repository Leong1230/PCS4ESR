# MIN3dCaPose
We present a **Min**kowskiEngine-powered **Ca**nonical **3D** **Pose** estimation method via object-level classification and reimplementation of a **N**ormalized **P**art **C**oordinate **Space-based** method to object level by[GAPartNet](https://arxiv.org/pdf/2211.05272.pdf) to do 3D pose estimation. We use the newly released 3D scene dataset [Multiscan](https://github.com/smartscenes/multiscan) with over 200 scans. Our model relies on **the Min**kowskiEngine-powered U-Net backbone to get point-level features, voxelized point-level features to get object-level features, and does object-level classification. We formalize 3d Pose as a combination of the up-direction class, front-direction latitude class, and front-direction longitude class. As a result, we used the results of the NPCS method as a baseline, and our Min3dCaPose outperformed the baseline method in the Multiscan dataset. 

We trained the two model on newly released [Multiscan](https://github.com/smartscenes/multiscan) dataset

The main contribution is predicting the canonical 3d pose(front and up direction) of an object given its point cloud by object-level classification

The basic code architecture of W&B logger,  Hydra part and the Backbone model are from[MINSU3D](https://github.com/3dlg-hcvc/minsu3d)

## ObjectClassifier model introduction
- ObjectClassifier is an efficient framework(MinkowskiEngine based) for point cloud object level pose estimation. It voxelizes the per point features from UNet to obtain object-level features. It also discretizes the front/up directions into different latitude and longitude classes and then computes the directions given the predicted class. Therefore, the canonical pose estimation can be simplied as a classifiction problem and 3 layer MLP is used. 
<img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/ObjectClassifier.png" width="800"/>
The classification details in a sphere:
<img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/Lng_Lat_class.png" width="400"/>

## Normalized Object Coordinate Space regression introduction
- The Normalized Object Coordinate Space is the reimplementation of the **N**ormalized **P**art **C**oordinate **Space** model. We modified some model details and loss function to fit the [Multiscan](https://github.com/smartscenes/multiscan) dataset and the features from UNet in [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine)

### The Evalutaion metrics
* AC_(angle): the accuracy, the threshold is that the angle between prediction and ground truth direction is within [angle] degree 
* Rerr: average angle between prediction and ground truth direction, in Radian system 

### Our best results on test set
| What         | AC_5  | AC_10 | AC_20 | Rerr  |
| ------------ | ----- | ----- | ----- | ----- |
| door         | 0.244 | 0.244 | 0.268 | 1.365 |
| chair        | 0.214 | 0.243 | 0.314 | 1.287 |
| cabinet      | 0.232 | 0.261 | 0.304 | 1.527 |
| window       | 0.118 | 0.118 | 0.118 | 1.816 |
| microwave    | 0.000 | 0.000 | 0.167 | 1.864 |
| trash_can    | 0.750 | 0.750 | 0.750 | 0.589 |
| refrigerator | 0.400 | 0.400 | 0.400 | 1.302 |
| toilet       | 0.677 | 0.788 | 0.788 | 0.384 |
| average      | 0.328 | 0.349 | 0.387 | 1.267 |

The dataset is the newly released [Multiscan](https://github.com/smartscenes/multiscan) dataset using our ObjectClassifier model. Our model is only trained on the 8 object categoried with articulated parts. 

The results using the NOCS model are lower than results in our model.

### Baseline NOCS results on test set
| What         | AC_5  | AC_10 | AC_20 | Rerr  |
| ------------ | ----- | ----- | ----- | ----- |
| door         | 0.000 | 0.000 | 0.067 | NaN |
| chair        | 0.033 | 0.067 | 0.350 | NaN |
| cabinet      | 0.000 | 0.022 | 0.065 | NaN |
| window       | 0.000 | 0.000 | 0.000 | NaN|
| microwave    | 0.000 | 0.000 | 0.000 | 1.922 |
| trash_can    | 0.000 | 0.000 | 0.125 | 1.117 |
| refrigerator | 0.000 | 0.062 | 0.125 | NaN |
| toilet       | 0.000 | 0.167 | 0.677 | 0.823 |
| average      | 0.004 | 0.040 | 0.175 | 1.288 |


## Features
- Design a new object level classfier method based on latitude class and longitude class, the model architecture is as followed.
- Preprocess the [Multiscan](https://github.com/smartscenes/multiscan) to get the objects with annotated canonical poses
- Highly-modularized design enables researchers switch between the NOCS model and our ObjectClassifier model easily. 
- Better logging with [W&B](https://github.com/wandb/wandb), periodic evaluation during training, and Easy experiment configuration  with [Hydra](https://github.com/facebookresearch/hydra).

## Setup

**Environment requirements**
- CUDA 11.X
- Python 3.8

### Conda (recommended)
We recommend the use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage system dependencies.

```shell
# create and activate the conda environment
conda create -n min3dcapose python=3.8
conda activate min3dcapose

# install PyTorch 1.8.2
conda install pytorch cudatoolkit=11.1 -c pytorch-lts -c nvidia

# install Python libraries
pip install -e .

# Python libraries installation verfication
python -c "import min3dcapose"

# install OpenBLAS and SparseHash via conda
conda install openblas-devel -c anaconda
conda install -c bioconda google-sparsehash
export CPATH=$CONDA_PREFIX/include:$CPATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# install MinkowskiEngine
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
--install-option="--blas_include_dirs=${CONDA_PREFIX}/include" --install-option="--blas=openblas"

# install C++ extensions
cd minsu3d/common_ops
python setup.py develop
```

### Pip (without conda)
Note: Setting up with Pip (no conda) requires [OpenBLAS](https://github.com/xianyi/OpenBLAS) and [SparseHash](https://github.com/sparsehash/sparsehash) to be pre-installed in your system.

```shell
# create and activate the virtual environment
virtualenv --no-download env
source env/bin/activate

# install PyTorch 1.8.2
pip install torch==1.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# install Python libraries
pip install -e .

# install OpenBLAS and SparseHash via APT
sudo apt install libopenblas-dev libsparsehash-dev

# install MinkowskiEngine
pip install MinkowskiEngine

# install C++ extensions
cd minsu3d/common_ops
python setup.py develop
```

## Data Preparation

### Multiscan dataset
1. Download the [Multiscan](https://github.com/smartscenes/multiscan) dataset and repo. To acquire the access to the dataset, please refer to their instructions. The download dataset would follow this [file system structure](https://3dlg-hcvc.github.io/multiscan/read-the-docs/dataset/index.html#file-system-structure) You will get a [download script](https://docs.google.com/forms/d/e/1FAIpQLSfksFtks9YHMeQQWjZjfNbNU4bhRx0knyJ_S-OdJ-vdi2pjBw/viewform) if your request is approved:
2. Substitute the `MULTISCAN/dataset/preprocess/gen_instsegm_dataset.py` file in the downloaded [Multiscan](https://github.com/smartscenes/multiscan)repo with the [gen_instsegm_dataset.py](https://github.com/Kaola-2115/MIN3dCaPose/gen_instsegm_dataset.py), set the environment following the [Instructions](https://3dlg-hcvc.github.io/multiscan/read-the-docs/server/index.html#installation)
3. Preprocess the data, it converts the objects with annotated pose to `.pth` data, and split dataset as the default way by[Multiscan](https://github.com/smartscenes/multiscan)
```shell
# about 406.3GB in total of Multiscan raw dataset
python gen_instsegm_dataset.py
# the processed data is about 5.9GB in total
```
### Create own `.pth` dataset
Each `.pth` file should named by the scans, which contains all the objects in that scan. The objects dictionary should have the following keys:
```shell
"xyz": 
"rgb": 
"normal": 
"obb": 
      "front":
      "up":
"instance_ids":
"sem_labels":
```

### Download Multiscan objects directly
Download splitted Multiscan objects with metadata by [Multiscan_objects](https://drive.google.com/drive/folders/1bUMcDMOlSGqrfK00rHvrD6cydQM2oFmQ?usp=sharing)

## Training, Inference and Evaluation
Note: Configuration files are managed by [Hydra](https://hydra.cc/), you can easily add or override any configuration attributes by passing them as arguments.
```shell
# log in to WandB
wandb login

# train a model from scratch
python train.py model={model_name} data={dataset_name}

# train a model from a checkpoint
python train.py model={model_name} data={dataset_name} model.ckpt_path={checkpoint_path}

# test a pretrained model
python test.py model={model_name} data={dataset_name} model.ckpt_path={pretrained_model_path}

# evaluate inference results
python eval.py model={model_name} data={dataset_name} model.model.experiment_name={experiment_name}

# examples:
# python train.py model=nocs data=multiscan model.trainer.max_epochs=120
# python test.py model=object_classifier data=multiscan model.ckpt_path=Object_Classifier_best.ckpt
# python eval.py model=nocs data=multiscan model.model.experiment_name=run_1
```

## Pretrained Models

We provide pretrained models for Multiscan. The pretrained model and corresponding config file are given below.  Note that all NOCS models are trained from scratch. While the ObjectClassifier model is trained from the pretrained [HAIS-MultiScanObj-epoch=55.ckpt](https://drive.google.com/file/d/1WJCvEMicwUziwB96bqFEgSL2n0GSZ1qG/view?usp=sharing) model which is trained on Multiscan dataset. It uses the hyper-parameters in Backbone UNet to accelerate training process. After downloading a pretrained model, run `test.py` to do inference as described in the above section.

### Multiscan test set
| Model            | Code | AC_5  | AC_10 | AC_20 | Rerr  | Download |
| ---------------- | ---- | ----- | ----- | ----- | ----- | ---------|
| ObjectClassifier | [config](https://github.com/Kaola-2115/MIN3dCaPose/blob/main/config/model/object_classifier.yaml) [model](https://github.com/Kaola-2115/MIN3dCaPose/blob/main/min3dcapose/model/object_classifier.py) | 0.318 | 0.337 | 0.348 | 1.337 | [link](https://drive.google.com/file/d/19xEFrk1auE7ZhkRy6fqE3FfdMnGiYaig/view?usp=sharing) |
| NOCS  | [config](https://github.com/Kaola-2115/MIN3dCaPose/blob/main/config/model/object_classifier.yaml) [model](https://github.com/Kaola-2115/MIN3dCaPose/blob/main/min3dcapose/model/object_classifier.py) |       |       |       |       | [link]()|
## Visualization
We provide scripts to visualize the predicted and ground truth canonical 3d pose of an object. When testing and inferencing, use the following option to show visualizations
```
model.show_visualization: True
```

the default visualization results will be saved in the following file structure

``` shell
min3dcapose
├── visualization_results
# results whose average angles error is below 5 degree
│   ├── Ac5- 
# input object
│   │   ├── [object_name].png 
# object in predicted canonical pose
│   │   ├── [object_name]_r.png 
# results whose average angles error is over 30 degree
│   ├── Ac30- 
│   │   ├── [object_name].png 
│   │   ├── [object_name]_r.png
```

Some results visualizations are as followed
```shell
# the red arrow is the predicted front direction
# left one: the original object with default OBB by `pcd.get_oriented_bounding_box()` in Open3d, the pose is randomly rotated
# right one: object rotated to predicted canonicalized pose, the OBB is align to canonicalized axis
```

The good predictions when angle<5 degree:
| object name | uncanonicalized pose | predicted canonical pose |
| ----------- | ---------------------| ------------------------ |
|toilet | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac5-/toilet.png" width="400"/> |<img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac5-/toilet_r.png" width="400"/> |
|    chair    | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac5-/chair.png" width="400"/> | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac5-/chair_r.png" width="400"/> |
|   cabinet   | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac5-/cabinet.png" width="400"/> |  <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac5-/cabinet_r.png" width="400"/> |
|    door     |  <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac5-/door.png" width="400"/> | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac5-/door_r.png" width="400"/> |

The bad predictions when angle>30 degree:
| object name | uncanonicalized pose | predicted canonical pose |
| ----------- | -------------------- | ------------------------ |
|   toilet    | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac30+/toilet.png" width="400"/> |<img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac30+/toilet_r.png" width="400"/> |
|    chair    | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac30+/chair.png" width="400"/> | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac30+/chair_r.png" width="400"/> |
|   cabinet   | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac30+/cabinet.png" width="400"/> |  <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac30+/cabinet_r.png" width="400"/> |
|    door     | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac30+/door.png" width="400"/> | <img src="https://github.com/Kaola-2115/MIN3dCaPose/blob/main/visualization_results/Ac30+/door_r.png" width="400"/> |

## Performance

We report the time it takes to train on Multiscan data of 134 scans

**Test environment**
- CPU: Intel Core i7-12700 @ 2.10-4.90GHz × 12
- RAM: 32GB
- GPU: NVIDIA GeForce RTX 3090 Ti 24GB
- System: Ubuntu 20.04.2 LTS

**Training time in total (without validation)**
|      Model      | Epochs | Batch Size | Time |
|-----------------|--------|------------|------|
| ObjectClassifier | 15 | 8 | 12hr4min |
| NOCS | 91 | 8 | 30hr10min |


**Inference time per object (avg)**
| Model | Time |
| ------| -----|
| ObjectClassifier | 1.004s |

## Limitations
- It's hard to predict the canonical pose of some objects categories due to annotation limitations. For instance, the front direction of some windows is defined as pointing into room. Therefore, the front direction is hard to predict without background.
- The results of the reimplementation of NOCS model still need to be impoved.


## Acknowledgement
This repo is built upon the [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) and [Minsu3d](https://github.com/3dlg-hcvc/minsu3d).  We train our models on [Multiscan](https://github.com/ScanNet/ScanNet). If you use this repo and the pretrained models, please cite the original papers.

## Reference
```
@inproceedings{mao2022multiscan,
    author = {Mao, Yongsen and Zhang, Yiming and Jiang, Hanxiao and Chang, Angel X, Savva, Manolis},
    title = {MultiScan: Scalable RGBD scanning for 3D environments with articulated objects},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2022}
}

@article{Zhou2018,
    author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
    title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
    journal   = {arXiv:1801.09847},
    year      = {2018},
}

@article{ravi2020pytorch3d,
    author = {Nikhila Ravi and Jeremy Reizenstein and David Novotny and Taylor Gordon
                  and Wan-Yen Lo and Justin Johnson and Georgia Gkioxari},
    title = {Accelerating 3D Deep Learning with PyTorch3D},
    journal = {arXiv:2007.08501},
    year = {2020},
}

@inproceedings{choy20194d,
  title={4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks},
  author={Choy, Christopher and Gwak, JunYoung and Savarese, Silvio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3075--3084},
  year={2019}
}

@misc{https://doi.org/10.48550/arxiv.2211.05272,
  doi = {10.48550/ARXIV.2211.05272},
  url = {https://arxiv.org/abs/2211.05272},
  author = {Geng, Haoran and Xu, Helin and Zhao, Chengyang and Xu, Chao and Yi, Li and Huang, Siyuan and Wang, He},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {GAPartNet: Cross-Category Domain-Generalizable Object Perception and Manipulation via Generalizable and Actionable Parts},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```
