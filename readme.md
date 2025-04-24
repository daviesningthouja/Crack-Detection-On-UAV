Crack Detection on UAV using YOLOv5

This project implements a crack detection system for UAVs using YOLOv5. The model is trained on a laptop and then exported to an ONNX format, which is further formatted to an engine file for deployment on a Jetson Nano system.
Project Overview

The primary objective of this project is to deploy a machine learning model for crack detection on UAVs. The model is initially trained on a laptop using YOLOv5. After training, the model is exported to ONNX format and transferred to the Jetson Nano system, where it is converted into an engine file for optimized performance.

System Requirements
Laptop (Training Environment)

    CUDA Version: 11.8

    cuDNN Version: 8.6

    Python Version: 3.8.20

    TensorFlow Version: Compatible with CUDA 11.8

    ONNX: To export YOLOv5 model for Jetson Nano deployment

    Dependencies:

        torch==1.12.1+cu11.6

        torchvision==0.13.1+cu11.6

        onnx==1.12.0

        protobuf==5.27.3

        numpy==1.24.4

        opencv-python==4.5.1

        matplotlib==3.5.1

Jetson Nano (Deployment Environment)

    JetPack: 4.6.1

    CUDA Version: 10.2

    cuDNN Version: 8.2

    TensorRT Version: 8.2.1

    Python Version: 3.6

    OpenCV Version: 4.5.1

    Dependencies:

        torch==2.3.1+cpu

        torchvision==0.18.1+cpu

        onnx==1.12.0

        protobuf==5.27.3

        numpy==1.24.4

Workflow

    Training on Laptop:

        Set up your environment on the laptop using the provided dependencies.

        Train the YOLOv5 model with your crack detection dataset.

        Export the model to ONNX format using the following command:

    python export.py --weights model/weights/best.pt --img-size 640 --batch-size 8 --simplify --include onnx

Deployment on Jetson Nano:

    Set up your Jetson Nano environment with the required dependencies.

    Transfer the ONNX model file to the Jetson Nano.

    Use TensorRT to optimize the ONNX model for faster inference on the Jetson Nano.

        Example for converting ONNX to TensorRT:

            trtexec --onnx=model.onnx --saveEngine=model.engine

    Inference on UAV:

        Integrate the optimized model with your UAV system to detect cracks in real-time.

Installation

To set up the environment for training on the laptop, use the following commands:

# Create an environment using conda
conda create -n crack_detection python=3.8.20
conda activate crack_detection

# Install dependencies
pip install torch==1.12.1+cu11.6 torchvision==0.13.1+cu11.6 onnx==1.12.0 protobuf==5.27.3 numpy==1.24.4 opencv-python==4.5.1 matplotlib==3.5.1

For the Jetson Nano, you’ll need to install dependencies based on the provided system environment.
Notes

    Make sure that both systems (laptop and Jetson Nano) have compatible versions of Python and libraries.

    Ensure that the TensorRT engine file is compatible with the Jetson Nano hardware for efficient inference.