#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 14:04:04 2020

@author: Yogesh Upadhyay
"""
import os
import numpy as np
from PIL import Image
import nltk

#import cv2


data_path = '/Users/Apple/Documents/SkecthtoCode/data/all_data'

data = os.listdir(data_path)
y=[]
x=[]
#Seperating png and gui files


for i in data:
    if i.endswith('.png'):
        x.append(i)
    else:
        y.append(i)

x1 =[]
#Converting the image files into array of(256,256)and creating the final training array for the model
for i in x:
    im = Image.open(data_path+'/'+i)
    im = im.resize((256,256), Image.ANTIALIAS)
    im = np.array(im)
    im = np.reshape(im  , (256,256,3))
    x1.append(im)
    
#Tokenizing GUI files

sample = y[0]
file = open(data_path+'/'+y[0] ,'r')
text = file.read()
tokens = nltk.line_tokenize(text)

    
