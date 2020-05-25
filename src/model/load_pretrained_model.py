from keras.engine.saving import model_from_json
from keras.optimizers import RMSprop

class load_pretrainedd_model():

    def load_model(self, model_json_file, model_weights_file):
        json_file = open(model_json_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights_file)
        optimizer = RMSprop(lr=0.0001, clipvalue=1.0)
        loaded_model.compile(loss = 'categorical_crossentropy' , optimizer=optimizer )
        loaded_model.summary()
        return loaded_model