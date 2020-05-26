import sys
import os
from argparse import ArgumentParser
from os.path import basename

from src.inference.convertor import *
from src.model.load_pretrained_model import *

def build_parser():
  parser = ArgumentParser()
  parser.add_argument('--png_path', type=str,
                      dest='png_path', help='png filepath to convert into HTML',
                      required=True)
  parser.add_argument('--output_folder', type=str,
                      dest='output_folder', help='dir to save generated gui and html',
                      required=True)
  parser.add_argument('--model_json_file', type=str,
                      dest='model_json_file', help='trained model json file',
                      required=True)
  parser.add_argument('--model_weights_file', type=str,
                      dest='model_weights_file', help='trained model weights file', required=True)
  parser.add_argument('--style', type=str,
                      dest='style', help='style to use for generation', default='default')
  return parser
  
def main():
    parser = build_parser()
    options = parser.parse_args()
    png_path = options.png_path
    output_folder = options.output_folder
    model_json_file = options.model_json_file
    model_weights_file = options.model_weights_file
    style = options.style

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    load_model = load_pretrained_model()
    model = load_model.load_model(model_json_file, model_weights_file)

    convert = convertor()
    convert.convert_single_image(png_path,model,output_folder, style=style)
    print("Converted sucessfully")

if __name__ == "__main__":
  main()