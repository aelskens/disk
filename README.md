# DISK

This repository implements a more flexible version of [DISK](https://doi.org/10.48550/arXiv.2006.13566) based on the kornia library's [implementation](https://github.com/kornia/kornia/blob/a041d9255f459a0034c75b20fb9ab524d7ae3983/kornia/feature/disk/disk.py).

The main differences with kornia's version are: (i) the detection and description parts can be performed independently from one to another, and (ii) the implementation of an OpenCV-like interface. Thanks to the pseudo-splitting, one can use DISK with more flexibility by using it as a detector or as a descriptor.

## Installation

Install this repository using pip:
```bash
git clone https://github.com/aelskens/disk.git && cd disk
python -m pip install .
```

## Usage

Here is a minimal example:
```python
from disk import FlexibleDISK
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

# Define the complete model and load the weights from the pretrained SuperPoint model
fdisk = FlexibleDISK(max_num_keypoints=1024).eval()
fdisk.load("https://raw.githubusercontent.com/cvlab-epfl/disk/master/depth-save.pth")

# Read an image
img = imread("path/to/image")

# Use OpenCV interface to get the keypoints and their descriptors
keypoints, descriptors = fdisk.detectAndCompute(img)
np_keypoints = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])

plt.figure()
plt.imshow(img)
plt.plot(np_keypoints[:, 0], np_keypoints[:, 1], "r.")
plt.show()
```