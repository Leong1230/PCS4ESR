from setuptools import find_packages, setup

setup(
    name="min3dcapose",
    version="1.0",
    author="Kaola-2115",
    url="https://github.com/Kaola-2115/MIN3dCaPose.git",
    description="",
    packages=find_packages(include=("lib", "model")),
    install_requires=["plyfile", "tqdm", "trimesh", "pytorch-lightning==1.6.5", "scipy", "open3d", "wandb", "hydra-core", "h5py", "pyransac3d"]
)
