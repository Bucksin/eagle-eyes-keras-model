import os.path

import numpy as np
import tensorflow as tf
from tensorflow import keras


IMAGE_FILE = 'test_images/log_2.png'

class_names = ['logs', 'mossy_rocks']

print("Testing image file: ", IMAGE_FILE)
print("Specified classes: ", class_names)

# Prepare image
img_height = 180
img_width = 180

#loading the test image into a Python Imaging Library format
img = tf.keras.utils.load_img(
    IMAGE_FILE, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img) #turn image into 3D numpy(numerical python) array
img_array = tf.expand_dims(img_array, 0) # Fill in the array if it's missing dimensions to make up a tensor

new_model = tf.keras.models.load_model('exported_model/model.keras')#load the trained model
#returns the prediction as a list of arrays in probabilities for each category in the model
#it may be a list of raw data from the model called logits if the model is using softmax function in the last layer
predictions = new_model.predict(img_array)
print(predictions)
#sigmoid function is to convert predictions to probabilities between 0 to 1 (if there's only two possible outcomes)
# no need if last layer is not softmax
score = float(keras.ops.sigmoid(predictions[0][0]))
print(score)

# Apply softmax function to predictions, it's another function to convert logits to probabilities
# (multi-class classification problems, where there are more than two possible outcomes)
probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)

# Convert probabilities to percentages
scores_percentages = probabilities * 100
print(scores_percentages)
for i, predictions in enumerate(scores_percentages[0]):
    print(f"Score for class {class_names[i]}: {predictions}")