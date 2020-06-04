from keras.models import Model, Sequential, model_from_json
from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Embedding, GRU, TimeDistributed, RepeatVector, LSTM, concatenate , Input, Reshape, Dense
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop

from ..dataset.datagenerator import *
from keras.engine.saving import model_from_json


MAX_SEQUENCE = 150

class sketch_to_code():

    def __init__(self,model_output_path,data_input_path,vocab_path):
        self.model_output_path = model_output_path
        self.data_input_path = data_input_path
        self.vocab_path = vocab_path
         

    def create_model(self):

        self.data_generator = datagenerator(self.data_input_path,self.vocab_path)
        tokenizer , vocab_size = self.data_generator.load_vocab()
        self.vocab_size = vocab_size

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

        # Decoder
        decoder = concatenate([encoded_image, language_model])
        decoder = GRU(512, return_sequences=True)(decoder)
        decoder = GRU(512, return_sequences=False)(decoder)
        decoder = Dense(vocab_size, activation='softmax')(decoder)

        model = Model(inputs = [visual_input,language_input] ,outputs = decoder)
        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        model.compile(loss = 'categorical_crossentropy' , optimizer=optimizer )
        model.summary()

        return model
    
    def save_model(self,model):
        model_json = model.to_json()
        with open("{}/model_json.json".format(self.model_output_path), "w") as json_file:
            json_file.write(model_json)
        model.save_weights("{}/weights.h5".format(self.model_output_path))


    def train(self,model,data_input_path,validation_split,epochs):

        train_steps_per_epoch,train_generator,val_steps_per_epoch,val_generator = self.data_generator.create_generator(MAX_SEQUENCE,validation_split)

        print("Training started")
        model.fit_generator(generator=train_generator, validation_data=val_generator, epochs=epochs, shuffle=False, validation_steps=val_steps_per_epoch,steps_per_epoch=train_steps_per_epoch)
        print("Finished Training")

        self.save_model(model)


