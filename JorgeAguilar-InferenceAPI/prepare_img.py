import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array,load_img

def read_load_image(file_path):
    image = load_img(file_path, target_size=(64,64))
    test_img = img_to_array(image) / 255.0
    test_img = np.expand_dims(test_img,axis=0)
    return test_img

def read_load_image(file_path):
    image = load_img(file_path, target_size=(128,128))
    test_img = img_to_array(image) / 255.0
    test_img = np.expand_dims(test_img,axis=0)
    return test_img