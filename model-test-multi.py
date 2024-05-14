import os.path
from PIL import Image

import keras
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from tensorflow import keras
from class_names import class_names

IMAGE_DIRECTORY = 'test_multi_images'

print("Testing image files in: ", IMAGE_DIRECTORY)
print("Specified classes: ", class_names)

# set image sizes
img_height = 180
img_width = 180

images = []
for filename in os.listdir(IMAGE_DIRECTORY):

    filepath = os.path.join(IMAGE_DIRECTORY, filename)
    #loading the test image into a Python Imaging Library format
    img = tf.keras.utils.load_img(
        filepath, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img) #turn image into 3D numpy(numerical python) array
    img_array = tf.expand_dims(img_array, 0) # Fill in the array if it's missing dimensions to make up a tensor

    new_model = tf.keras.models.load_model('exported_model/model.keras')#load the trained model
    #returns the prediction as a list of arrays in probabilities for each category in the model
    predictions = new_model.predict(img_array)
    print("Image: ", filename)
    print("Model predictions: ", predictions)

    for i, predictions in enumerate(predictions[0]):
        print(f"Score for class {class_names[i]}: {predictions}")

    #show the image
    plt.imshow(Image.open(filepath))
    plt.show()