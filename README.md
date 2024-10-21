# Voxel functa using meta-learning

# Installation pipeline

### Installed cuda and pytorch

```

# create and activate the conda environment
conda create -n pcs4esr python=3.10
conda activate pcs4esr
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit 

# install PyTorch 2.x.x
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# install Python libraries
pip install . 
```

### Installed MinkowskiEngine and other python library 

```
# install OpenBLAS
conda install openblas-devel --no-deps -c anaconda

# (computecanada) flexiblas
module load flexiblas

# install Minkowski Engine
cd ~/projects/MinkowskiEngine
pip install . 

#install torch-scatter, nksr, pytorch3d
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html 
pip install nksr -f https://nksr.huangjh.tech/whl/torch-2.0.0+cu118.html 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"


# compile FPT cuda source code
cd ~/projects/pcs4esrLearning_new
cd pcs4esr/cuda_ops
pip3 install .

# compile nearest_neighbor source code
cd ~/projects/pcs4esr/tools/nearest_neighbors
python setup.py install

# install PointTransformer requirement

pip install spconv-cu118 timm  
pip install flash-attn --no-build-isolation

```



