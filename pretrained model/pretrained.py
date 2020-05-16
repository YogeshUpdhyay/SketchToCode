import cv2
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.engine.saving import model_from_json
from keras.preprocessing.sequence import pad_sequences
from Compiler import *

data_dir = 'data/'
img = 'data/1A4A0B67-2481-49AF-9E74-1AAC30F88AF4.png'
def resize(img):
    im = cv2.imread(img)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    img_adapted = cv2.adaptiveThreshold(im, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 9)
    img_stacked = np.repeat(img_adapted[...,None],3,axis=2)
    resized = cv2.resize(img_stacked, (200,200), interpolation=cv2.INTER_AREA)
    bg_img = 255 * np.ones(shape=(256,256,3))
    bg_img[27:227, 27:227,:] = resized
    bg_img /= 255
    return bg_img

img_features = np.array([resize(img)])

file = open('vocabulary.vocab', 'r')
text = file.read().splitlines()[0]
file.close()
tokenizer = Tokenizer(filters='', split=" ", lower=False)
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index) + 1


json_file = open('model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('weights.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
loaded_model.summary()


in_text = '<START> '

def word_for_id(integer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

for i in range(150):
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    sequence = pad_sequences([sequence], maxlen=48)
    yhat = loaded_model.predict([img_features, sequence], verbose=0)
    yhat = np.argmax(yhat)
    word = word_for_id(yhat)
    if word is None:
        break
    in_text += word + ' '
    if word == '<END>':
        break


generated_gui = in_text.split()


print(generated_gui)


style = 'default'
compiler = Compiler(style)
compiled_website = compiler.compile(generated_gui)


print("\nCompiled HTML:")
print(compiled_website)

sample_id  = '1A4A0B67-2481-49AF-9E74-1AAC30F88AF4'
output_folder = 'C:/Users/Yogesh Upadhyay/Documents/MachineLearningProjects/SketchToCode'
if compiled_website != 'HTML Parsing Error':
    output_filepath = "{}/{}.html".format(output_folder, sample_id)
    with open(output_filepath, 'w') as output_file:
        output_file.write(compiled_website)
        print("Saved generated HTML to {}".format(output_filepath))

print('Complete')