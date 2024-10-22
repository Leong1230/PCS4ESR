# Point Cloud Serialization for Efficient Surface Reconstruction

![PCS4ESR](assets/teaser.pdf)

**Point Cloud Serialization for Efficient Surface Reconstruction**<br>
[Zhen Li](https://colinzhenli.github.io/), [Weiwei Sun](https://m-niemeyer.github.io/](https://wsunid.github.io/)), [Shrisudhan Govindarajan](https://shrisudhan.github.io/), [Shaobo Xia](https://scholar.google.com/citations?user=eOPO9E0AAAAJ&hl=en), [Daniel Rebain](http://drebain.com/), [Kwang Moo Yi](https://www.cs.ubc.ca/~kmyi/), [Andrea Tagliasacchi](https://theialab.ca/)  
**[Paper](todo), [Project Page](todo)**

Abstract: *We present a novel approach to large-scale point cloud
surface reconstruction by developing an efficient framework
that converts an irregular point cloud into a signed distance
field (SDF). Our backbone builds upon recent transformer-
based architectures (i.e. PointTransformerV3), that serial-
izes the point cloud into a locality-preserving sequence of
tokens. We efficiently predict the SDF value at a point by ag-
gregating nearby tokens, where fast approximate neighbors
can be retrieved thanks to the serialization. We serialize
the point cloud at different levels/scales, and non-linearly
aggregate a feature to predict the SDF value. We show
that aggregating across multiple scales is critical to over-
come the approximations introduced by the serialization
(i.e. false negatives in the neighborhood). Our frameworks
sets the new state-of-the-art in terms of accuracy and effi-
ciency (better or similar performance with half the latency
of the best prior method, coupled with a simpler implemen-
tation), particularly on outdoor datasets where sparse-grid
methods have shown limited performance. To foster the
continuation of research in this topic, we will release our
complete source code, as well as our pre-trained models.

Contact [zhenli@sfu.ca](zla247@sfu.ca) for questions, comments and reporting bugs.
## News

- [2024/10/22] The code is released.
- [2024/10/22] The MinkUNet backbone version will be released soon.

## Environment setup

The code is tested on Ubuntu 20.04 LTS with PyTorch 2.0.0 CUDA 11.8 installed. Please follow the following steps to install PyTorch first.

```
# Clone the repository
git clone (Todo)
cd PCS4ESR

# create and activate the conda environment
conda create -n pcs4esr python=3.10
conda activate pcs4esr

# install cuda toolkit
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit 

# install PyTorch 2.x.x
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

```
Then, install PyTorch3D
```
# install runtime dependencies for PyTorch3D
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# install PyTorch3D
conda install pytorch3d -c pytorch3d
```

Install the necessary packages listed out in requirements.txt:
```
pip install -r requirements.txt
```

Install torch-scatter and nksr
```
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html 
pip install nksr -f https://nksr.huangjh.tech/whl/torch-2.0.0+cu118.html 
```

The detailed installation of nksr is described in the [NKSR](https://github.com/nv-tlabs/nksr).

## Reproducing results from the paper

### Data Preparation

You can download the data from the following links and put it under `PCS4ESR/data/`.
- ScanNet:
Data is available [here](https://drive.google.com/drive/folders/1JK_6T61eQ07_y1bi1DD9Xj-XRU0EDKGS?usp=sharing).
We converted original meshes to `.pth` data, and the normals are generated using the [open3d.geometry.TriangleMesh](https://www.open3d.org/html/python_api/open3d.geometry.TriangleMesh.html). The processing detailed from raw scannetv2 data is from [minsu3d](https://github.com/3dlg-hcvc/minsu3d).

- SceneNN
Data is available [here](https://drive.google.com/file/d/1d_ILfaxpJBpiiwCZtvC4jEKnixEr9N2l/view?usp=sharing).

- SyntheticRoom
Data is available [here](https://drive.google.com/drive/folders/1PosV8qyXCkjIHzVjPeOIdhCLigpXXDku?usp=sharing), it is from [ConvONet](https://github.com/autonomousvision/convolutional_occupancy_networks), which contains the processing details.

- CARLA
Data is available [here](https://drive.google.com/file/d/1BFwExw7SRJaqHJ98pqqnR-k6g8XYMAqq/view?usp=sharing), it is from [NKSR](https://github.com/nv-tlabs/nksr).


### Training
Note: Configuration files are managed by [Hydra](https://hydra.cc/), you can easily add or override any configuration attributes by passing them as arguments.
```shell
# log in to WandB
wandb login

# train a model from scratch
```bash
# ScanNet dataset
python train.py model=scannet_model data=scannet
# SyntheticRoom dataset
python train.py model=synthetic_model data=synthetic
# CARLA dataset
python train.py model=carla_model data=carla
```

In addition, you can manually specify different training settings. Common flags include:
- `--experiment_name`: Additional experiment name to specify.
- `--data.dataset_root_path`: Root path of the dataset.
- `--output_folder`: Output folder to save the results, the checkpoints will be saved in `{output_folder}/{dataset_name}/{experiment_name}/training`.
- `--model.network.default_decoder.neighboring`: Neighboring type, default is `Serial`. Options: `Serial`, `KNN`, `Mixture`

### Inference

You can either infer using your own trained models or our pre-trained checkpoints.

The pre-trained checkpoints on different datasets with different neighboring types are available [here](https://drive.google.com/drive/folders/15679CWdUmt9O8l0HxZFABHV7lYfcIcUJ?usp=sharing), you can download and put them under `PCS4ESR/checkpoints/`.

```bash
# For example, ScanNet dataset with Serialization neighboring
python eval.py model=scannet_model data=scannet model.ckpt_path=checkpoints/ScanNet_Serial_best.ckpt 
# For example, Carla dataset with Serialization neighboring, you need more than 24GB GPU memory to inferece the CARLA dataset, we recommend using a server.
python eval.py model=carla_model data=carla model.ckpt_path=checkpoints/CARLA_Serial_best.ckpt
# For example, Test on SceneNN dataset with model trained on ScanNet.
python eval.py model=scannet_model data=scenenn model.ckpt_path=checkpoints/ScanNet_Serial_best.ckpt
```

### Reconstruction
You can reconstruct a specific scene from the datasets above by specifying the scene id.
```bash
# For example, ScanNet dataset
python eval.py model=scannet_model data=scannet model.ckpt_path={path_to_checkpoint} data.over_fitting=True data.take=1 data.intake_start={scene_index}
```
In addition, you can manually specify visualization settings. Flags include:
- `--data.visualization.save`: Whether to save the results.
- `--data.visualization.Mesh`: Whether to save the mesh.
- `--data.visualization.Input_points`: Whether to save the input points.
 
The results will be saved in `{output_folder}/{dataset_name}/{experiment_name}/reconstruction/visualization`.

## License

## Citation
