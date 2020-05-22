from src.dataset.datagenerator import *  
from src.model.sketch_to_code import *

data_path = 'C:/Users/Yogesh Upadhyay/Documents/MachineLearningProjects/SketchToCode/data/'
vocab_path = 'C:/Users/Yogesh Upadhyay/Documents/MachineLearningProjects/SketchToCode/vocabulary.vocab'
MAX_SEQ = 150

data_generator = datagenerator(data_path)
steps_per_epoch,data_gen = data_generator.create_generator(data_path,MAX_SEQ,vocab_path)

tokenizer,vocab_size = data_generator.load_vocab(vocab_path)
#model_output_path = 'C:/Users/Yogesh Upadhyay/Documents/MachineLearningProjects/SketchToCode/'

sketch_to_code = sketch_to_code(vocab_size)
model = sketch_to_code.create_model()
batch_size = 64

model.fit_generator(generator = data_gen,steps_per_epoch = steps_per_epoch/batch_size,epochs=1)


