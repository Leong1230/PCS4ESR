from setuptools import setup, find_packages

setup(
    name="pcs4esr",
    version="1.0",
    author="Kaola-2115",
    url="https://github.com/Kaola-2115/pcs4esr.git",
    description="PCS4ESR Package",
    packages=find_packages(include=("lib", "model")),
    install_requires=[
        "pytorch-lightning==1.9.4",
        "lightning-bolts",
        "tqdm",
        "wandb",
        "hydra-core",
        "opencv-python",
        "arrgh",
        "plyfile",
        "imageio-ffmpeg",
        "gin-config",
        "torchviz",
        "thop",
        "imageio",
        "einops",
        "spconv-cu118",
        "timm",
        "flash-attn --no-build-isolation",
    ],
    extras_require={
        "torch-scatter": [
            "torch-scatter @ https://data.pyg.org/whl/torch-2.0.0+cu118.html"
        ],
        "nksr": [
            "nksr @ https://nksr.huangjh.tech/whl/torch-2.0.0+cu118.html"
        ],
        "pytorch3d": [
            "git+https://github.com/facebookresearch/pytorch3d.git@stable"
        ],
    },
)
