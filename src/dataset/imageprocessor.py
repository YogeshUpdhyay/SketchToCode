import os
import cv2
import numpy as np 
import matplotlib.pyplot as pyplot


from keras.preprocessing.image import ImageDataGenerator


class imageprocessor():

    def __init__(self):
        pass

  
    def processing_images(self, all_filenames ,data_path, augmentation = True,):
        print("Converting images to array ")
        resized_images = self.get_images(all_filenames,data_path)
        return resized_images
    

    def resize_img(self,png_file_path):
        img_rgb = cv2.imread(png_file_path)
        img_grey = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_adapted = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 101, 9)
        img_stacked = np.repeat(img_adapted[...,None],3,axis=2)
        resized = cv2.resize(img_stacked, (200,200), interpolation=cv2.INTER_AREA)
        bg_img = 255 * np.ones(shape=(256,256,3))
        bg_img[27:227, 27:227,:] = resized
        bg_img /= 255
        return bg_img


    def get_images(self,all_filenames,data_path):
        resized_images =list()
        for i in all_filenames:
            img = self.resize_img(data_path+i)
            resized_images.append(img)
        return np.array(resized_images)




