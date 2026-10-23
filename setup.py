import torch
import torchvision
from setuptools import setup, find_packages

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 9], "Requires PyTorch >= 2.5"
torchvision_ver = [int(x) for x in torchvision.__version__.split(".")[:2]]
assert torchvision_ver >= [1, 20], "Requires torchvision >= 0.20"

setup(
    name="dp2",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
                        "accelerate==1.13.0",
                        "clip@git+https://github.com/openai/CLIP.git",
                        "einops==0.7.0",
                        "huggingface-hub==1.28.0",
                        "ipykernel==6.31.0",
                        "matplotlib==3.7.2",
                        "moviepy==1.0.3",
                        "numpy==1.26.4",
                        "ollama>=0.6.3",
                        "opencv-python==4.11.0.86",
                        "pandas==1.5.3",
                        "pillow==12.3.0",
                        "pydantic==2.14.1",
                        "rfdetr==1.7.1",
                        "safetensors==0.8.0",
                        "sahi==0.11.36",
                        "scikit-learn==1.4.1.post1",
                        "scipy==1.10.1",
                        "supervision==0.28.0",
                        "tokenizers==0.22.2",
                        "torch==2.5.1",
                        "torchaudio==2.5.1",
                        "torchvision==0.20.1",
                        "tqdm==4.65.0",
                        "transformers==5.15.1",
                        "ultralytics==8.4.52"
                ],
)