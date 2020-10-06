# Region Segmentation from microCT Scanning

## Introduction

## Training

python -u projects/microCT/main.py --config-file projects/microCT/configs/CT-Fly-No-Augmentation.yaml

## Inference  

python -u projects/microCT/main.py --config-file projects/microCT/configs/CT-Fly-No-Augmentation.yaml --inference --checkpoint path/to/checkpoint

