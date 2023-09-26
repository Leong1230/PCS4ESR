# Voxel functa using meta-learning

# Installation pipeline

### Installed cuda and pytorch

```
#module load cuda/11.7 or install cudatoolkit-11.7
# create and activate the conda environment
conda create -n python-3.8 python=3.8
conda activate python-3.8

# install PyTorch 2.0
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia

# installpycarus
pip install pycarus 
```

### Installed torch dependencies for pycarus(changed torch version) 

```
source ./python-3.8/lib/python3.8/site-packages/pycarus/install_torch.sh

# may need uninstall torchaudio if got error
pip uninstall torchaudio
# install Python libraries
pip install . 
```

### Installed MinkowskiEngine and other python library 

```
# install OpenBLAS
conda install openblas-devel --no-deps -c anaconda

# (computecanada) flexiblas
module load flexiblas

# install Python libraries
cd ~/projects/MinkowskiEngine
pip install . 
```

### replace pcd.py

```
# /local-scratch/localhome/zla247/anaconda3/envs/python-3.8/lib/python3.8/site-packages/pycarus/geometry/pcd.py 

```





