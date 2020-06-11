# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 18:46:57 2019

import os
src_img=os.listdir(src)
import skimage.io

for img in src_img:
    image = skimage.io.imread(os.path.join(src, img))
    
    

@author: prajwal mj
"""
import cv2 
import os
#import numpy as np
src1 = os.getcwd()
img1 = cv2.imread(os.path.join(src1, 'frame_1_300new.jpg-objects/person-3.jpg'))

def crop(img1):
           
    
    #height, width, color components C:\Users\prajwal mj\.spyder-py3\frame_4_300new.jpg-objects
    h ,w ,c = img1.shape
    (h1 , w1) = (0,0)
    h2 = h // 4 
    w2 = w
    img = img1[ h1:h2, w1:w2 ]
    cv2.imwrite("C:\\Users\\prajwal mj\\.spyder-py3\\Project\\Helmet_det\\person-2_crop.jpg",img)
    return 'person-2_crop.jpg'

#crop(img1)