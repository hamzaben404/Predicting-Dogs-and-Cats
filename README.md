# Predicting Dogs and Cats

## Project Overview:
This project aims to create a model that predicts whether an image contains a dog or a cat using deep learning techniques.

## Dataset:
The dataset used for this project consists of 12500 images of dogs and 12500 images of cats sourced from [https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset].

## Model Architecture:
The model architecture is based on a convolutional neural network (CNN) designed for image classification tasks.
It consists of the following layers:

- Input Layer: Accepts RGB images resized to 128x128 pixels.

- Convolutional Layers:
    Conv1: 3 input channels (RGB), 32 output channels, kernel size 3x3.
    ReLU activation function.
    Max pooling with kernel size 2x2.
    Conv2: 32 input channels, 64 output channels, kernel size 3x3.
    ReLU activation function.
    Max pooling with kernel size 2x2.

- Fully Connected Layers:
    FC1: 643030 input features, 128 output features.
    ReLU activation function.
    FC2: 128 input features, 2 output features (for 'cat' and 'dog' classes).
    Log softmax activation function for output.


## Training:
- Data Preprocessing:
    Images resized to 128x128 pixels.
    Converted to PyTorch tensors.
    
- Training Details:
    Optimizer: Adam optimizer with a learning rate of 0.001.
    Loss Function: Cross-Entropy Loss.
    Epochs: 10.
    Batch Size: 32.

