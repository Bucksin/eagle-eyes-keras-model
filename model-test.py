import numpy as np
import tensorflow as tf
from tensorflow import keras


IMAGE_FILE = 'test_images/daisy3.jpg'

class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

print("Testing image file: ", IMAGE_FILE)
print("Specified classes: ", class_names)

# Prepare image
img_height = 180
img_width = 180

img = tf.keras.utils.load_img(
    IMAGE_FILE, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

new_model = tf.keras.models.load_model('exported_model/model.keras')
predictions = new_model.predict(img_array)
print(predictions)
score = float(keras.ops.sigmoid(predictions[0][0]))
print(score)

# Apply softmax function to predictions
probabilities = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)

# Convert probabilities to percentages
scores_percentages = probabilities * 100
print(scores_percentages)
for i, score in enumerate(scores_percentages[0]):
    print(f"Score for class {class_names[i]}: {score}")


 

