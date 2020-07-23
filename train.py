from src.dataset.datagenerator import *  
from src.model.sketch_to_code import *
import os
from argparse import ArgumentParser
VAL_SPLIT = 0.2
MAX_SEQ = 150
EPOCHS = 1

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--data_input_path', type=str,
                        dest='data_input_path', help='directory containing images and guis',
                        required=True)
    parser.add_argument('--vocab_path', type=str,
                        dest='vocab_path', help='directory containing vocabulary',
                        required=True)
    parser.add_argument('--validation_split', type=float,
                        dest='validation_split', help='portion of training data for validation set',
                        default=VAL_SPLIT)
    parser.add_argument('--epochs', type=int,
                        dest='epochs', help='number of epochs to train on',
                        default=EPOCHS)
    parser.add_argument('--model_output_path', type=str,
                        dest='model_output_path', help='directory for saving model data',
                        required=True)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    data_input_path = options.data_input_path
    validation_split = options.validation_split
    epochs = options.epochs
    model_output_path = options.model_output_path
    vocab_path = options.vocab_path

    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)


    sketchtocode = sketch_to_code(model_output_path,data_input_path,vocab_path)
    model = sketchtocode.create_model()

    print("Created new model")

    sketchtocode.train(model,data_input_path,validation_split,epochs)

    print("Model_training_complete and saved at {}".format(model_output_path))


if __name__ == "__main__":
    main()

