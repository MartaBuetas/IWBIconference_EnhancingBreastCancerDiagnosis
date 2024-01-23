''' Preparing the input images for the SinGAN model'''

# In order to use the implementation in pytorch of the official SinGAN paper:
## The dimension of the input image must be a number power of 2: the generated patches were (224,224), now are converted into (256,256).
## The input image should have three dimensions (height, width, channels): an additional channel is added to have this third dimension.

## From the 'data' folder created when generated the patch dataset, the specific patches used for training SinGAN models are preprocessed and saved into ./input/Images folder.

import cv2 as cv
import matplotlib.pyplot as plt
import os
import skimage.io as skio
import pandas as pd
import imageio
from PIL import Image
import numpy as np


input_files=['13.png', '77.png', '3941.png', '2789.png', '11.png', '75.png', '3939.png', '2787.png', '2180.png', '168.png', '3884.png', '51.png']
dir_file=input("Path of generated patch dataset folder: ")

dir_save=r"./input/Images"
for i in range(len(input_files)):
    img1 = Image.open(os.path.join(dir_file, input_files[i])).convert('L')
    img_rgb = img1.convert('RGB')
    size = (256, 256)
    img_resized = img_rgb.resize(size, resample=Image.BILINEAR)
    # Convert the image to a numpy array
    img_gray_arr = np.array(img_resized)

    # Add a third dimension to the array
    img_gray_arr = img_gray_arr.reshape((256, 256, 3))
    imageio.imwrite(os.path.join(dir_save, 'resized_'+ input_files[i]), img_gray_arr)
    img_gray_arr.shape


