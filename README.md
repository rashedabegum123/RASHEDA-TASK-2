# Tree Leaf Detection using YOLOv5

This repository contains a Google Colab notebook for training a YOLOv5 model to detect tree leaves using a custom dataset. The model is trained using the Ultralytics YOLOv5 framework and Roboflow for dataset management.

# Features

Uses YOLOv5 for object detection

Downloads and prepares a dataset from Roboflow

Trains the model with specified hyperparameters

Evaluates model performance

# Installation

To set up the required environment, run the following:

!git clone https://github.com/ultralytics/yolov5  # Clone YOLOv5 repository
%cd yolov5
%pip install -qr requirements.txt  # Install dependencies
%pip install -q roboflow  # Install Roboflow

# Dataset Preparation

The dataset is obtained from Roboflow:

from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("pds12").project("task-2-pds-12")
version = project.version(1)
dataset = version.download("yolov5")

Replace YOUR_API_KEY with your actual Roboflow API key.

# Training the Model

Train the YOLOv5 model using the following command:

!python train.py --img 416 --batch 16 --epochs 15 --data {dataset.location}/data.yaml --weights yolov5s.pt --cache

Adjust parameters such as image size, batch size, and number of epochs as needed.

# Evaluation and Inference

After training, evaluate the model and run inference:

!python detect.py --weights runs/train/exp/weights/best.pt --img 416 --conf 0.4 --source test_images/

# Acknowledgments

Ultralytics YOLOv5

Roboflow

This project is developed for detecting tree leaves efficiently using deep learning.

