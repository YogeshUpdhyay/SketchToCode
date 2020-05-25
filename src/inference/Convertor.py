
import os
import shutil
import json
import numpy as np

from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences

from ..dataset.imageprocessor import *
from ..dataset.datagenerator import *
from .Compiler import *

VOCAB_PATH = "../vocabulary.vocab"
class Convertor():


    def load_vocab(self):
        file = open(VOCAB_PATH,'r')
        text = file.read().splitlines()[0]
        file.close()
        tokenizer = Tokenizer(filters='', split=" ", lower=False)
        tokenizer.fit_on_texts([text])
        vocab_size = len(tokenizer.word_index) + 1

        return tokenizer,vocab_size


    def convert_single_image(self,png_path,model,output_folder,style):

        png_filename = os.path.basename(png_path)
        if png_filename.find('.png') == -1:
            raise ValueError("Image is not a png!")
        sample_id = png_filename[:png_filename.find('.png')]


        imageprocessor = imageprocessor()
        img_features = np.array([imageprocessor.get_image_features(png_path)])

        tokenizer,vocab_size = self.load_vocab()

        generated_gui = self.generate_guis(tokenizer,img_features,model)
        self.generate_html(generated_gui,sample_id,output_folder,style)
        



    def word_for_id(self,integer,tokenizer):
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None

    def generate_guis(self,tokenizer,img_features,model):
        in_text = '<START> '
        for i in range(150):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=48)
            yhat = model.predict([img_features, sequence], verbose=0)
            yhat = np.argmax(yhat)
            word = self.word_for_id(yhat,tokenizer)
            if word is None:
                break
            in_text += word + ' '
            if word == '<END>':
                break

        generated_gui = in_text.split()

        return generated_gui

    def generate_html(self, gui_array, sample_id, output_folder, style='default'):

        compiler = Compiler(style)
        compiled_website = compiler.compile(gui_array)

        
        print("\nCompiled HTML:")
        print(compiled_website)

        if compiled_website != 'HTML Parsing Error':
            output_filepath = "{}/{}.html".format(output_folder, sample_id)
            with open(output_filepath, 'w') as output_file:
                output_file.write(compiled_website)
                print("Saved generated HTML to {}".format(output_filepath))