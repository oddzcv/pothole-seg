import sys
import torch
import torchvision
import timm
import albumentations as A
import cv2
import yaml
import numpy as np

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Torch:", torch.__version__)
print("TorchVision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
print("timm:", timm.__version__)
print("Albumentations:", A.__version__)
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("YAML:", yaml.safe_load("ok: true"))
print("Environment check passed.")