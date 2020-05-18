import os
import cv2
import numpy as np 
import numba


from keras.preprocessing.image import ImageDataGenerator


class imageprocessor():

    def __init__(self):
        pass

  
    def processing_images(self, all_filenames , augmentation = True):
        print("Converting images to array and augmentation")
        resized_images , labels = self.get_images_labels(all_filenames)

        if augmentation == True:
            images = self.get_augment_images(resized_images)
        else:
            self.save_images(resized_images,labels)
        
        return resized_images, labels

    def get_augment_images(self,resized_images):
        generator = ImageDataGenerator(rotation_range=2,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 zoom_range=0.05
                                )
        keras_generator = generator.flow(resized_images,batch_size=1)
        images = list()
        for i in  range(0,len(resized_images)):
            image = next(keras_generator)
            images.append(image)
        
        return images
    

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

    def get_images_labels(self,all_filenames):
        resized_images =list()
        for i in all_filenames:
            img = self.resize_img(i)
            resized_images.append(img)


#data_path = 'C:/Users/Yogesh Upadhyay/Documents/MachineLearningProjects/SketchToCode/data/1A4A0B67-2481-49AF-9E74-1AAC30F88AF4.png'


