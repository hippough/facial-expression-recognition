# Facial Expression Recognition

This project implements facial expression recognition system using NVIDIA Jetson Inference library. The model has been retrained on a comprehensive facial emotion dataset to classify human emotions with high accuracy.

## Overview

The system can classify facial expressions into 7 distinct emotional categories, making it suitable for applications in human-computer interaction, sentiment analysis, and behavioral studies.

## The Algorithm

### Model Architecture

The model uses a ResNet-18 architecture that has been retrained on the [Face Expression Recognition Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset?resource=download). For training, 100 images per emotion category were used from the original dataset.

### Emotion Categories

The model classifies facial expressions into the following 7 categories:

1. Angry
2. Disgust
3. Fear
4. Happy
5. Neutral
6. Sad
7. Surprise

## Setup

### 1. Install Jetson Inference

```
git clone --recursive https://github.com/dusty-nv/jetson-inference
cd jetson-inference
mkdir build
cd build
cmake ../
make
sudo make install
```

### 2. Prepare Dataset

Organize images like this:
```
jetson-inference/python/training/classification/data/faces/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
├── val/
└── test/

```

### 3. Training

1. Enable more memory: `echo 1 | sudo tee /proc/sys/vm/overcommit_memory`
2. Train the model (I used batch size of 8)
  ```
  cd jetson-inference
  ./docker/run.sh
  cd python/training/classification
  python3 train.py --model-dir=models/faces data/faces
  ```
3. Export Model
  ```
  # Still in docker container:
  python3 onnx_export.py --model-dir=models/faces
  ```

## Using the Model

### Set Variables
```
cd jetson-inference/python/training/classification
NET=models/faces
DATASET=data/faces
```

### Test on Image
```
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/happy/image.jpg result.jpg
```

### Live Camera
```
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt /dev/video0
```

### Process Video
```
imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt input.mp4 output.mp4
```

## Resources
* [Dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset?resource=download)
* [ImageNet Documentation](https://github.com/dusty-nv/jetson-inference/blob/master/docs/imagenet-console-2.md)
* [Jetson Inference GitHub](https://github.com/dusty-nv/jetson-inference)
* [Video Demonstration](link)
