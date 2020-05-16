import cv2
import sys
from os import listdir
from os.path import join
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.layers import Conv2D,Dense,Dropout,Flatten,GRU,Embedding,concatenate,RepeatVector
from keras.models import Model,Input
from keras import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences


def resize_img(png_file_path):
    img_rgb = cv2.imread(png_file_path)
    img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_adapted = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 9)
    img_stacked = np.repeat(img_adapted[...,None],3,axis=2)
    resized = cv2.resize(img_stacked, (200,200), interpolation=cv2.INTER_AREA)
    bg_img = 255 * np.ones(shape=(256,256,3))
    bg_img[27:227, 27:227,:] = resized
    bg_img /= 255
    return bg_img
    
def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

def process_data_for_generator(texts, features, max_sequences, tokenizer, vocab_size):
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

def load_directory(data_dir):
    all_filenames = listdir(data_dir)
    all_filenames.sort()
    image_filenames,texts = list(),list()
    for filename in (all_filenames):
        if filename[-3:] == "png":
            image_filenames.append(filename)
        else:
            text = '<START> ' + load_doc(data_dir+filename) + ' <END>'
            text = ' '.join(text.split())
            text = text.replace(',', ' ,')
            texts.append(text)
    return image_filenames,texts

def load_tokenizer(texts):
    tokenizer = Tokenizer(filters='', split=" ", lower=False)
    
    tokenizer.fit_on_texts([load_doc('C:/Users/Yogesh Upadhyay/Documents/MachineLearningProjects/SketchToCode/vocabulary.vocab')])
   
    vocab_size = len(tokenizer.word_index) + 1

    train_sequences = tokenizer.texts_to_sequences(texts)

    max_sequence = max(len(s) for s in train_sequences)

    return tokenizer,vocab_size,train_sequences,max_sequence

def load_images(image_filenames,data_dir):
    images = list()
    for image in image_filenames:
        images.append(resize_img(data_dir+image))
    return images

def data_generator(text_features, img_features, max_sequences, tokenizer, vocab_size):
    while 1:
        for i in range(0, len(text_features), 1):
            Ximages, XSeq, y = list(), list(),list()
            for j in range(i, min(len(text_features), i+1)):
                image = img_features[j]
                desc = text_features[j]
                in_img, in_seq, out_word = process_data_for_generator([desc], [image], max_sequences, tokenizer, vocab_size)
                for k in range(len(in_img)):
                    Ximages.append(in_img[k])
                    XSeq.append(in_seq[k])
                    y.append(out_word[k])
            yield [[np.array(Ximages), np.array(XSeq)], np.array(y)]

"""        
class Dataset():
    def __init__(self, data_dir, input_transform=None, target_transform=None):
        self.data_dir = data_dir
        self.image_filenames = []
        self.texts = []
        all_filenames = listdir(data_dir)
        all_filenames.sort()
        for filename in (all_filenames):
            if filename[-3:] == "png":
                self.image_filenames.append(filename)
            else:
                text = '<START> ' + load_doc(self.data_dir+filename) + ' <END>'
                text = ' '.join(text.split())
                text = text.replace(',', ' ,')
                self.texts.append(text)
        self.input_transform = input_transform
        self.target_transform = target_transform
        
        # Initialize the function to create the vocabulary 
        tokenizer = Tokenizer(filters='', split=" ", lower=False)
        # Create the vocabulary 
        tokenizer.fit_on_texts([load_doc('C:/Users/Yogesh Upadhyay/Documents/MachineLearningProjects/SketchToCode/vocabulary.vocab')])
        self.tokenizer = tokenizer
        # Add one spot for the empty word in the vocabulary 
        self.vocab_size = len(tokenizer.word_index) + 1
        # Map the input sentences into the vocabulary indexes
        self.train_sequences = tokenizer.texts_to_sequences(self.texts)
        # The longest set of boostrap tokens
        self.max_sequence = max(len(s) for s in self.train_sequences)
        # Specify how many tokens to have in each input sentence
        self.max_length = 48
        images = list()
        vocab_size = len(tokenizer.word_index) + 1

        for image in self.image_filenames:
            images.append(resize_img(data_dir+image))

        Ximages, XSeq, y = list(), list(),list()
        for i in range(0, len(self.texts), 1):
            for j in range(i, min(len(self.texts), i+1)):
                image = images[j]
                desc = self.texts[j]
                in_img, in_seq, out_word = process_data_for_generator([desc], [image], 48, self.tokenizer, vocab_size)
                for k in range(len(in_img)):
                    Ximages.append(in_img[k])
                    XSeq.append(in_seq[k])
                    y.append(out_word[k])
"""
dir_name = 'C:/Users/Yogesh Upadhyay/Documents/MachineLearningProjects/SketchToCode/data/'
#batch_size = 32

image_filenames,texts = load_directory(dir_name)
tokenizer,vocab_size,train_sequences,max_sequence= load_tokenizer(texts)
images = load_images(image_filenames,dir_name)

print(images.shape)
print(texts.shape)
data_gen = data_generator(texts, images, max_sequence, tokenizer, vocab_size)

#Model 

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
model.summary()



model.fit_generator(generator = generator,epochs=2,steps_per_epoch=10 )

