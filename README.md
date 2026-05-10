# Pothole Instance Segmentation

A modular Mask R-CNN repository for pothole instance segmentation.

## Supported Backbones

- MobileNetV3-Large-FPN
- MobileNetV4-FPN via timm
- MobileNetV2-FPN
- ResNet50-FPN

## Main Features

- COCO instance segmentation dataset support
- Roboflow COCO-segmentation support
- MMDetection-like augmentation pipeline
- COCO bbox and segmentation mAP evaluation
- Custom pothole mIoU and Dice evaluation
- YAML-based experiment configuration