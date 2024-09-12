from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="hybridpc",
    version="1.0",
    author="Kaola-2115",
    url="https://github.com/Kaola-2115/hybridpc.git",
    description="",
    packages=find_packages(include=("lib", "model")),
    install_requires=["tqdm", "lightning", "lightning-bolts", "scipy", "open3d", "wandb", "hydra-core", "opencv-python", "h5py", "arrgh", "plyfile", "imageio-ffmpeg", "ninja", "gin-config", "torchviz", "trimesh", "thop", "cython", "imageio", "scikit-image", "einops"]
)
