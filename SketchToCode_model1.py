#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 15:11:41 2020

@author: Yogesh Upadhyay
"""

tokenizer, vocab_size = Dataset.load_vocab()
self.vocab_size = vocab_size
        
        # Image encoder
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
image_model.add(RepeatVector(MAX_LENGTH))
visual_input = Input(shape=(256, 256, 3,))
encoded_image = image_model(visual_input)
# Language encoder
language_input = Input(shape=(MAX_LENGTH,))
language_model = Embedding(vocab_size, 50, input_length=MAX_LENGTH, mask_zero=True)(language_input)
language_model = GRU(128, return_sequences=True)(language_model)
language_model = GRU(128, return_sequences=True)(language_model)
# Decoder
decoder = concatenate([encoded_image, language_model])
decoder = GRU(512, return_sequences=True)(decoder)
decoder = GRU(512, return_sequences=False)(decoder)
decoder = Dense(vocab_size, activation='softmax')(decoder)

# Compile the model
self.model = Model(inputs=[visual_input, language_input], outputs=decoder)
optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)


--------------------

import nltk
file_content = open("myfile.txt").read()
tokens = nltk.word_tokenize(file_content)
print tokens