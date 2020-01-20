#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:33:20 2020

@author: Yogesh Upadhyay
"""

import os
import numpy as np
from PIL import Image
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv2D,Dense,Dropout,Flatten,GRU,Embedding,concatenate
from keras.models import Model,Input
from keras import Sequential
from keras.optimizers import RMSprop


#import cv2


data_path = '/Users/trycatch2/Documents/SketctoCode/data'

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
"""
sample = y[0]
file = open(data_path+'/'+y[0] ,'r')
text = file.read()
tokens = nltk.line_tokenize(text)
"""

y1 = []
for i in y:
    file = open(data_path + '/' + i , 'r')
    text = file.read()
    text = '<START> ' + text + ' <END>' 
    text = ' '.join(text.split())
    text = text.replace(',', ' ,')
    y1.append(text)
 


#Tokenize the GUI Files for creating the vocabulary
token_file  = open('/Users/trycatch2/Documents/SketctoCode' + '/' + 'vocabulary.vocab')        #Creating token file
token_file = token_file.read()


tokenizer = Tokenizer(filters= '' , split = ' ',lower = False)
tokenizer.fit_on_texts(token_file)
vocab_size = len(tokenizer.word_index) + 1
train_sequences = tokenizer.texts_to_sequences(y1)
max_sequence = max(len(s) for s in train_sequences)
max_length = 48





image_model = Sequential()
image_model.add(Conv2D(16, (3, 3), padding='valid', activation='relu', input_shape=(256, 256, 3,)))
image_model.add(Conv2D(16, (3,3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
image_model.add(Conv2D(32, (3,3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
image_model.add(Conv2D(64, (3,3), activation='relu', padding='same', strides=2))
image_model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
image_model.add(Flatten())
image_model.add(Dense(1024, activation='relu'))
image_model.add(Dropout(0.3))
image_model.add(Dense(1024, activation='relu'))
image_model.add(Dropout(0.3))



visual_input = Input(shape=(256,256,3,))
encoded_image = image_model(visual_input)

language_input = Input(shape=(3,))
language_model = Embedding(vocab_size, 50, input_length=3, mask_zero=True)(language_input)
language_model = GRU(128, return_sequences=True)(language_model)
language_model = GRU(128, return_sequences=True)(language_model)
# Decoder
decoder = concatenate([encoded_image, language_model])
decoder = GRU(512, return_sequences=True)(decoder)
decoder = GRU(512, return_sequences=False)(decoder)
decoder = Dense(vocab_size, activation='softmax')(decoder)

model = Model(inputs = visual_input ,outputs = decoder)
model.compile(loss = 'categorical_crossentropy' , optimizer=RMSprop)






