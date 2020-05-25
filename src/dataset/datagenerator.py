from __future__ import absolute_import

import os
import shutil
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from .imageprocessor import *

BATCH_SIZE              = 64


class datagenerator():
    def __init__(self,data_input_folder,vocab_path):
        self.data_input_folder = data_input_folder
        self.vocab_path = vocab_path

    def split_datasets(self,texts,images,validation_split):
        train_texts,val_texts,train_imgaes,val_images = train_test_split(texts,images,test_size = validation_split)

        return train_texts,train_imgaes,val_texts,val_images

    

    def load_vocab(self):
        file = open(self.vocab_path,'r')
        text = file.read().splitlines()[0]
        file.close()
        tokenizer = Tokenizer(filters='', split=" ", lower=False)
        tokenizer.fit_on_texts([text])
        vocab_size = len(tokenizer.word_index) + 1
        self.vocab_size = vocab_size

        return tokenizer,vocab_size


    def load_data(self):
        all_files = os.listdir(self.data_input_folder)
        guis , image_files = list(),list()
        for i in all_files:
            if(i.endswith('png')):
                image_files.append(i)
            else:
                file = open(self.data_input_folder+i,'r')
                texts = file.read()
                file.close()
                i = '<START> ' + texts + ' <END>'
                i = ' '.join(i.split())
                i = i.replace(',', ' ,')
                guis.append(i)
        
        return image_files,guis

    def get_steps_per_epoch(self,texts):
        total_sequences = 0
        for i in texts:
            total_sequences += len(i.split())
        steps_per_epoch = total_sequences

        return steps_per_epoch//BATCH_SIZE


    def create_generator(self,max_sequence,validation_split):
        image_files , texts = self.load_data()

        tokenizer ,vocab_size = self.load_vocab()

        image_processor = imageprocessor()
        images = image_processor.processing_images(image_files,self.data_input_folder)

        train_texts,train_imgaes,val_texts,val_images = self.split_datasets(texts,images,validation_split)

        train_generator = self.data_generator(train_texts, train_imgaes, max_sequence, tokenizer, vocab_size)
        val_generator   = self.data_generator(val_texts,val_images,max_sequence,tokenizer,vocab_size)

        train_steps_per_epoch = self.get_steps_per_epoch(train_texts)
        val_steps_per_epoch   = self.get_steps_per_epoch(val_texts)

        return train_steps_per_epoch,train_generator,val_steps_per_epoch,val_generator


    def data_generator(self, text_features, img_features, max_sequences, tokenizer, vocab_size):
        while 1:
            for i in range(0, len(text_features), 1):
                Ximages, XSeq, y = list(), list(),list()
                for j in range(i, min(len(text_features), i+1)):
                    image = img_features[j]
                    desc = text_features[j]
                    in_img, in_seq, out_word = self.process_data_for_generator([desc], [image], max_sequences, tokenizer, vocab_size)
                    for k in range(len(in_img)):
                        Ximages.append(in_img[k])
                        XSeq.append(in_seq[k])
                        y.append(out_word[k])
                yield [[np.array(Ximages), np.array(XSeq)], np.array(y)]


    def process_data_for_generator(self, texts, features, max_sequences, tokenizer, vocab_size):
        X, y, image_data = list(), list(), list()
        sequences = tokenizer.texts_to_sequences(texts)
        for img_no, seq in enumerate(sequences):
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_sequences)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                image_data.append(features[img_no])
                X.append(in_seq[-48:])
                y.append(out_seq)
        return np.array(image_data), np.array(X), np.array(y)
