# Face Mask Detection using MobileNetV2

## Overview

This project implements a real-time face mask detection system using the MobileNetV2 architecture with TensorFlow and Keras. The system detects whether a person is wearing a mask or not, contributing to public health and safety initiatives, particularly in times of contagious diseases like COVID-19.

## Dependencies

- `tensorflow`
- `imutils`
- `matplotlib`
- `numpy`
- `opencv-python`
- `scikit-learn`

## Data Preprocessing

The images are preprocessed using the ImageDataGenerator from TensorFlow's Keras API. Data augmentation techniques such as rotation, zoom, and horizontal flip are applied to increase the diversity of the training dataset and improve model generalization.

## Model Architecture

The MobileNetV2 architecture is used as the base model for feature extraction. A custom classification head consisting of dense and dropout layers is added on top of the base model. The model is compiled with the Adam optimizer and binary cross-entropy loss function.

## Training

The model is trained for 20 epochs with a learning rate of 0.0001 and a batch size of 32. Training and validation accuracy and loss metrics are monitored and visualized using matplotlib.

## Model Evaluation

The trained model is evaluated on the test set using classification metrics such as accuracy, precision, recall, and F1-score.

## Real-time Detection

The trained model is deployed for real-time face mask detection using OpenCV. The system reads frames from a video stream, detects faces using a pre-trained face detector model, and classifies each detected face as wearing a mask or not.
