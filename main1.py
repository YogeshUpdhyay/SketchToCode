from src.dataset.datagenerator import *  
from src.model.sketch_to_code import *
import os
from argparse import ArgumentParser
VAL_SPLIT = 0.2
MAX_SEQ = 150

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
                        required=True)
    parser.add_argument('--model_output_path', type=str,
                        dest='model_output_path', help='directory for saving model data',
                        required=True)
    parser.add_argument('--model_json_file', type=str,
                        dest='model_json_file', help='pretrained model json file',
                        required=False)
    parser.add_argument('--model_weights_file', type=str,
                        dest='model_weights_file', help='pretrained model weights file',
                        required=False)
    return parser


def main():
    parser = build_parser()
    options = parser.parse_args()
    data_input_path = options.data_input_path
    validation_split = options.validation_split
    epochs = options.epochs
    model_output_path = options.model_output_path
    model_json_file = options.model_json_file
    model_weights_file = options.model_weights_file
    vocab_path = options.vocab_path

    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)


    sketchtocode = sketch_to_code(model_output_path,data_input_path,vocab_path)


    if model_json_file is not None and model_weights_file is not None:
        model = sketchtocode.load_model(model_json_file, model_weights_file)
        print("Loaded pretrained model from disk")

    else:
        model = sketchtocode.create_model()
        print("Created new model")

    sketchtocode.train(model,data_input_path,validation_split,epochs)

    print("Model_training_complete and saved at {}".format(model_output_path))


if __name__ == "__main__":
    main()
