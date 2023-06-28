from setuptools import find_packages, setup

setup(
    name="hybridpc",
    version="1.0",
    author="Kaola-2115",
    url="https://github.com/Kaola-2115/hybridpc.git",
    description="",
    packages=find_packages(include=("lib", "model")),
    install_requires=["tqdm", "lightning", "scipy", "open3d", "wandb", "hydra-core", "ninja"]
)
