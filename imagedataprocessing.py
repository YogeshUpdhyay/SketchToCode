import tensorflow as tf
import scipy as scipy
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt  
from numba import jit, cuda 
import os
from timeit import default_timer as timer
from PIL import Image
import IPython.display as display

def loadimages(data_path):
    images = []
    gui = []
    data = os.listdir(data_path)
    for i in data:
        if i.endswith("png"):
            temp_image =  Image.open(str(os.path.join("all_data/" + i)))
            images.append(temp_image)
        else:
            gui.append(str(i))

data_path = os.path.join("all_data")
start = timer()
loadimages(data_path)
print("with GPU:", timer()-start) 
