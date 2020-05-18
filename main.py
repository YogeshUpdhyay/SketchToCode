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

def load_tokenizer(texts):
    tokenizer = Tokenizer(filters='', split=" ", lower=False)
    tokenizer.fit_on_texts([load_doc('C:/Users/Yogesh Upadhyay/Documents/MachineLearningProjects/SketchToCode/vocabulary.vocab')])
    vocab_size = len(tokenizer.word_index) + 1
    return tokenizer,vocab_size

def load_images(image_filenames,data_dir):
    images = list()
    for image in image_filenames:
        images.append(resize_img(data_dir+image))
    return np.array(images)

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

dir_name = 'C:/Users/Yogesh Upadhyay/Documents/MachineLearningProjects/SketchToCode/data/'
img = '1A4A0B67-2481-49AF-9E74-1AAC30F88AF4.png'
batch_size = 32

image_filenames,texts = load_directory(dir_name)
tokenizer,vocab_size= load_tokenizer(texts)
images = load_images(image_filenames,dir_name)

print(images.shape)

data_gen = data_generator(texts, images, 150, tokenizer, vocab_size)
total_sequences = 0
for text_set in texts: 
    total_sequences += len(text_set.split())
print(total_sequences)
steps_per_epoch = total_sequences // batch_size
print(steps_per_epoch)

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



model.fit_generator(generator = data_gen,epochs = 1,steps_per_epoch = steps_per_epoch/2)


img_features = np.array([resize_img(dir_name+img)])
in_text = '<START> '

def word_for_id(integer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

for i in range(150):
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], maxlen=48)
    yhat = model.predict([img_features, sequence], verbose=0)
    yhat = np.argmax(yhat)
    word = word_for_id(yhat)
    if word is None:
        break
    in_text += word + ' '
    if word == '<END>':
        break

generated_gui = in_text.split()


print(generated_gui)

