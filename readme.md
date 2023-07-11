# PIFuHD Setup
This code is used to set up the PIFuHD environment and perform human pose estimation using Lightweight Human Pose Estimation and PIFuHD models. It consists of several steps that need to be executed in order to install the required dependencies and perform the pose estimation.

![image](pifuhd.gif)

## Prerequisites
Before running the code, make sure you have the following dependencies installed:

* Python 3.x: The code is written in Python, so make sure you have Python 3.x installed on your system.
* Git: Git is required to clone the PIFuHD and Lightweight Human Pose Estimation repositories.
* PyTorch: The code requires PyTorch library to run the models and perform computations.
* Torchvision: Torchvision is a PyTorch package that provides various datasets, models, and transformations for computer vision tasks.
* PyTorch3D: PyTorch3D is a library for 3D deep learning using PyTorch. It is used in the PIFuHD model.
* OpenCV: OpenCV is a library for computer vision tasks. It is used for image processing and reading image files.

## Installation
Clone the PIFuHD repository:


```!git clone https://github.com/facebookresearch/pifuhd```

This command will clone the PIFuHD repository from GitHub and download the code to your local system.

Clone the Lightweight Human Pose Estimation repository:
```!git clone https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch.git```

This command will clone the Lightweight Human Pose Estimation repository from GitHub.

Change the current directory to the Lightweight Human Pose Estimation repository:

```%cd /content/lightweight-human-pose-estimation.pytorch/```

This command will navigate to the directory where the Lightweight Human Pose Estimation code is located.

Download the pre-trained pose estimation model:
```!wget https://download.01.org/opencv/openvino_training_extensions/models/human_pose_estimation/checkpoint_iter_370000.pth```

This command will download the pre-trained pose estimation model weights.

Change the current directory to the PIFuHD repository:
```%cd /content/pifuhd/```

This command will navigate to the directory where the PIFuHD code is located.

Download the pre-trained PIFuHD model:
```!sh ./scripts/download_trained_model.sh```

This command will download the pre-trained PIFuHD model.

Install the required dependencies:

```!pip install 'torch==1.6.0+cu101' -f https://download.pytorch.org/whl/torch_stable.html```
```!pip install 'torchvision==0.7.0+cu101' -f https://download.pytorch.org/whl/torch_stable.html```
```!pip install 'pytorch3d==0.2.5'```

These commands will install the required versions of PyTorch, Torchvision, and PyTorch3D.

## Usage
Change the current directory back to the Lightweight Human Pose Estimation repository:


```%cd /content/lightweight-human-pose-estimation.pytorch/```
Import the required libraries:

import torch 
import cv2
import numpy as np
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses
import demo
from IPython.display import clear_output

Define the get_rect function:

The get_rect function performs human pose estimation and saves the bounding box coordinates for the detected poses.

Clear the output:


```clear_output()```
You can now use the get_rect function to perform human pose estimation and save the bounding box coordinates for the detected poses.

Note: Make sure you have the input images and provide the correct paths to them when calling the get_rect function.


# Example usage
```net = PoseEstimationWithMobileNet()
checkpoint = torch.load('checkpoint_iter_370000.pth', map_location='cpu')
load_state(net, checkpoint)
images = ['path/to/image1.jpg', 'path/to/image2.jpg']
height_size = 256
get_rect(net, images, height_size)```

# License
Feel free to modify and distribute it.