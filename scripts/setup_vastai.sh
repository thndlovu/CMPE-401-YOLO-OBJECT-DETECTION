#!/bin/bash

echo "=== Setting up CMPE401 YOLO Project ==="

apt-get update -q
apt-get install -y unzip wget git python3-pip

pip install ultralytics torch torchvision numpy opencv-python \
    matplotlib pandas PyYAML tqdm seaborn Pillow --quiet

git clone https://github.com/thndlovu/CMPE-401-YOLO-OBJECT-DETECTION.git
cd cmpe401-yolo-object-detection

echo "=== Setup complete ==="
echo "Now download VisDrone data and run the converter"