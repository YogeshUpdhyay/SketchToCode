import tensorflow as tf
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt  
import os
import sys
import shutil
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from opencv import cv2



class imagedataprocessing:
    def __init__(self):
        pass

    def buildimagedata( self, data_folder, agument_data = True):
        print("Converting image data to arrays from {} , augmentation: {}".format(data_folder,agument_data))
        