#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 15:39:42 2020

@author: Yogesh Upadhyay 
"""
def data_generator(cls, text_features, img_features, max_sequences, tokenizer, vocab_size):
        while 1:
            for i in range(0, len(text_features), 1):
                Ximages, XSeq, y = list(), list(),list()
                for j in range(i, min(len(text_features), i+1)):
                    image = img_features[j]
                    desc = text_features[j]
                    in_img, in_seq, out_word = Dataset.process_data_for_generator([desc], [image], max_sequences, tokenizer, vocab_size)
                    for k in range(len(in_img)):
                        Ximages.append(in_img[k])
                        XSeq.append(in_seq[k])
                        y.append(out_word[k])
                yield [[np.array(Ximages), np.array(XSeq)], np.array(y)]

    @classmethod
    def process_data_for_generator(cls, texts, features, max_sequences, tokenizer, vocab_size):
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

@classmethod
    def process_data_for_generator(cls, texts, features, max_sequences, tokenizer, vocab_size):
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