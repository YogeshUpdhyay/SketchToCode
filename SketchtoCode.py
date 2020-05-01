#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:33:20 2020
@author: Yogesh Upadhyay
"""

import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.layers import Conv2D,Dense,Dropout,Flatten,GRU,Embedding,concatenate,RepeatVector
from keras.models import Model,Input
from keras import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from PIL import Image


#import cv2


data_path = "data/"

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
token_file  = open('/Users/trycatch/Documents/SketchToCode' + '/' + 'vocabulary.vocab')        #Creating token file
token_file = token_file.read()

#Create tokenizer object and train on sample vocab file
tokenizer = Tokenizer(filters= '' , split = ' ',lower = False)
tokenizer.fit_on_texts(token_file)
vocab_size = len(tokenizer.word_index) + 1 #+1 for the unknown extra term
#Train the tokenizer on all GUI texts
train_sequences = tokenizer.texts_to_sequences(y1)
max_sequence = max(len(s) for s in train_sequences)
#max_length = 48



#Creating batches of GUI tokens for training process and splitting the intial tokens
#to a input and output file for creating input of contextual tokens which will be updated at every timestep

image_data , X ,Y = list() ,list(),list()


for img_no , seq in enumerate(train_sequences):
    for i in range(1,len(seq)):
        in_seq, out_seq = seq[:i], seq[i]
        in_seq = pad_sequences([in_seq], maxlen=max_sequence)[0]
        out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
        image_data.append(x1[img_no])
        X.append(in_seq[-48:])
        Y.append(out_seq)


image_data = np.array(image_data)
X = np.array(X)
Y = np.array(Y)

#print(len(image_data))
"""
Ximages,Xseq,y=list(),list(),list()
for k in range(len(image_data)):
    Ximages.append(image_data[k])
    Xseq.append(X[k])
    y.append(Y[k])

Ximages=np.array(Ximages)
Xseq=np.array(Xseq)
y=np.array(y)
"""

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
image_model.add(RepeatVector(48))



visual_input = Input(shape=(256,256,3,))
encoded_image = image_model(visual_input)

language_input = Input(shape=(48,))
language_model = Embedding(vocab_size, 50, input_length=48, mask_zero=True)(language_input)
language_model = GRU(128, return_sequences=True)(language_model)
language_model = GRU(128, return_sequences=True)(language_model)

print(encoded_image.shape)
print(language_model.shape)
# Decoder
decoder = concatenate([encoded_image, language_model])
decoder = GRU(512, return_sequences=True)(decoder)
decoder = GRU(512, return_sequences=False)(decoder)
decoder = Dense(vocab_size, activation='softmax')(decoder)

model = Model(inputs = [visual_input,language_input] ,outputs = decoder)
optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
model.compile(loss = 'categorical_crossentropy' , optimizer=optimizer )


input_param = [image_data,X]
"""
from sklearn.model_selection import train_test_split
x1_train,x1_test,x2_train,x2_test,y_train,y_test = train_test_split(np.array(image_data),np.array(X),np.array(Y),stratify=Y,test_size = 0.2,shuffle=True)
"""

model.fit( generator = [[np.array(image_data),np.array(X)],Y] , epochs = 100,steps_per_epoch = 1000)

x = model.predict_generator([image_data[0],X[0]])







