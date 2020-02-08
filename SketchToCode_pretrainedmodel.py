#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 16:12:46 2020

@author: Yogesh_Upadhyay
"""
import numpy as np
from PIL import Image
from keras.models import model_from_json


data_path = '/Users/admin/Documents/SketchToCode'



json_file = open('model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('weights.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

img_features = np.array(Image.open(data_path+'/all_data/1A4A0B67-2481-49AF-9E74-1AAC30F88AF4.png'))
img_features=np.reshape((None,256,256,3) , img_features)
