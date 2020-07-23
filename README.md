# SketchToCode

![](https://img.shields.io/badge/python-3-brightgreen.svg) ![](https://img.shields.io/badge/tensorflow-1.1.0-orange.svg)

*Generating HTML Code from a hand-drawn wireframe*

![Preview](https://github.com/ashnkumar/sketch-code/blob/master/header_image.png)

SketchCode is a deep learning model that takes hand-drawn web mockups and converts them into working HTML code. It uses an [image captioning](https://towardsdatascience.com/image-captioning-in-deep-learning-9cd23fb4d8d2) architecture to generate its HTML markup from hand-drawn website wireframes.

For more information, check out this post: [Automating front-end development with deep learning](https://blog.insightdatascience.com/automated-front-end-development-using-deep-learning-3169dd086e82)

This project is based on [Ashwin Kumar's](https://github.com/ashnkumar/sketch-code) sketch to code.

<b>Note:</b> This project can only give HTML for inputs resembling to the core dataset and not genralized data.


## Setup
### Prerequisites

- Python 3 (not compatible with python 2)
- pip

### Install dependencies

```sh
pip install -r requirements.txt
```
### Download [data](http://sketch-code.s3.amazonaws.com/data/all_data.zip) to train the model
 
### Downlaod [weights](http://sketch-code.s3.amazonaws.com/model_json_weights/weights.h5) and [model](http://sketch-code.s3.amazonaws.com/model_json_weights/model_json.json)

Unzip the data and use it for the training of the model.

Train the model:
```sh
python train.py --data_input_path {path/to/folder/with/pngs/guis} \
      --vocab_path {path/to/vocabulary.vocab} \
      --validation_split 0.2 \
      --epochs 10 \
      --model_output_path {path/to/output/model} 
```


Converting an example drawn image into HTML code, using pretrained weights:
```sh
python convert_single_image.py --png_path {path/to/img.png} \
      --output_folder {folder/to/output/html} \
      --model_json_file {path/to/model/json_file.json} \
      --model_weights_file {path/to/model/weights.h5} \
      --vocab_path {path/to/vocabulary.vocab}
```

