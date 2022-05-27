import torch
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
import json

# image size
WIDTH, HEIGHT = 160, 120

# Read the dataset, containing the name of images and corresponding vehicle steering wheel angle and throttle
home_dir = os.path.dirname(__file__) 
data = pd.read_csv(os.path.join(home_dir,'data/dataset.csv'))
data_aug = data.copy()

'''Probability of different type of image augments'''
p_fliped = 0.2
p_shifted = 1.0
p_shadow = 1.0
p_bright_blur = 0.2


# Brightness&Blur
transform = T.RandomApply(transforms=torch.nn.ModuleList([
    T.ColorJitter(brightness=[0.5,1.5],contrast=[0.5,1.5],saturation=[0.5,1.],hue=0.),  
    T.GaussianBlur(kernel_size=5, sigma=(0.1,2.0))
    ]), p=p_bright_blur)

# Shadow
def random_shadow(img, w_low=0.3, w_high=0.7): 
    '''# where the input 'img' is PIL.Image image
    [w_low,w_high) is the randomly choiced opacity of this particular 'img'
    this method returns np.array image'''

    cols, rows = (img.size[0], img.size[1])

    top_y = np.random.random_sample() * rows 
    bottom_y = np.random.random_sample() * rows
    bottom_y_right = bottom_y + np.random.random_sample() * (rows - bottom_y)
    top_y_right = top_y + np.random.random_sample() * (rows - top_y)
    if np.random.random_sample() <= 0.5:
        bottom_y_right = bottom_y - np.random.random_sample() * (bottom_y)
        top_y_right = top_y - np.random.random_sample() * (top_y)

    poly = np.asarray([[ [top_y,0], [bottom_y, cols], [bottom_y_right, cols], [top_y_right,0]]], dtype=np.int32)

    mask_weight = np.random.uniform(w_low, w_high)
    origin_weight = 1 - mask_weight

    mask = np.copy(img).astype(np.int32)
    cv2.fillPoly(mask, poly, (0, 0, 0)) 

    shadow_img=cv2.addWeighted(np.copy(img).astype(np.int32), origin_weight, mask, mask_weight, 0).astype(np.uint8)
    return Image.fromarray(shadow_img)

# Shift
def random_shift(img,ori_angle,width=WIDTH,height=HEIGHT,low_x_range=-20,high_x_range=20,low_y_range=-2,high_y_range=2,delta_st_angle_per_px=0.009):
    ''' 'ori_angle' is the original vehicle steering wheel angle when capturing the current 'img'
    [low_x_range,high_x_range), [low_y_range,high_y_range) are the shift arange in horizontal and vertical direction, respectively
    delta_st_angle_per_px is the coefficient of steering angle changing per pixel, this parameter could vary for different sets of images
    this function returns the new image and steering angle after shifted
    '''

    translation_x = np.random.randint(low_x_range, high_x_range)
    translation_y = np.random.randint(low_y_range, high_y_range) 

    angle_new = ori_angle + translation_x * delta_st_angle_per_px # Only the shifting in horizontal direction could affect the steering angle

    translation_matrix = np.float32([[1, 0, translation_x],[0, 1, translation_y]])
    img_new = cv2.warpAffine(img, translation_matrix, (width, height)) 
    
    return img_new, angle_new


for i,ele in enumerate(data.image):
    img = Image.open(os.path.join(home_dir,'data/images',ele)) # PIL Image to PIL Image
    img = transform(img)  # Brightness&Blur  # PIL Image to PIL Image

    if np.random.rand() < p_shadow: 
        img = random_shadow(img) # Shadow # PIL Image to PIL Image

    if np.random.rand() < p_fliped:  # fliped # PIL Image to PIL Image
        img = T.RandomHorizontalFlip(p=1.0)(img)
        data_aug.loc[i,'angle'] = -data_aug.loc[i,'angle']

    img.save(os.path.join(home_dir,'data/images_aug',ele)) 


for i,ele in enumerate(data.image): # shadow, cv2 image to cv2 image
    img = cv2.imread(os.path.join(home_dir,'data/images_aug',ele))
    if np.random.rand() < p_shifted: 
        img_new,angle_new = random_shift(img,ori_angle=data_aug.loc[i,'angle'])
        data_aug.loc[i,'angle'] = angle_new
        cv2.imwrite(os.path.join(home_dir,'data/images_aug',ele),img_new) 

# save the dataset
data_aug.to_csv(os.path.join(home_dir,'data/data_aug.csv'), index=False, encoding='utf8')

