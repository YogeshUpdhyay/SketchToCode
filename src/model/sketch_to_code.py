from keras.models import Model, Sequential, model_from_json
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Embedding, GRU, TimeDistributed, RepeatVector, LSTM, concatenate , Input, Reshape, Dense
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop

from ..dataset.datagenerator import *
from keras.engine.saving import model_from_json




class sketch_to_code():

    def __init__(self,vocab_size):
        self.vocab_size = vocab_size

    def create_model(self):

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
        language_model = Embedding(self.vocab_size, 50, input_length=48, mask_zero=True)(language_input)
        language_model = GRU(128, return_sequences=True)(language_model)
        language_model = GRU(128, return_sequences=True)(language_model)

        # Decoder
        decoder = concatenate([encoded_image, language_model])
        decoder = GRU(512, return_sequences=True)(decoder)
        decoder = GRU(512, return_sequences=False)(decoder)
        decoder = Dense(self.vocab_size, activation='softmax')(decoder)

        model = Model(inputs = [visual_input,language_input] ,outputs = decoder)
        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        model.compile(loss = 'categorical_crossentropy' , optimizer=optimizer )
        model.summary()

        return model

    def load_model(self):
        json_file = open('model_json.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights('weights.h5')
        loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        loaded_model.summary()

        return loaded_model
    


