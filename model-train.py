import matplotlib.pyplot as plt
import pathlib
import os
import numpy as np

import tensorflow as tf
import keras
from keras import layers

ACTIVATION_FUNCTION = 'softmax' # set to either "sigmoid" or "softmax"
DATASET_PATH = 'model_training/dataset/objects'

data_dir = pathlib.Path(DATASET_PATH).with_suffix('')

image_count = len(list(data_dir.glob('*/*.png')))
print(image_count)

batch_size = 32
img_height = 180
img_width = 180

#Yields batches of images and labels
#loads 80% of the images for training
#generate td.data.Dataset from image files in a directory
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # set proportion of images to be used for validation
  subset="training", #smaller group of dataset and categorize as "training" dataset
  seed=123,#random number for shuffling and transformation
  image_size=(img_height, img_width),#standarized the image size
  batch_size=batch_size)#number of samples being propogated through the next work


#loads 20% of the images for training
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2, # set proportion of images to be used for validation
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names #the categories
print(class_names)

# puts the dataset into memory to improve performance (used for tensorflow to prefetch data)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)#caches the elements
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data normalization
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# print(np.min(first_image), np.max(first_image))


num_classes = len(class_names)

#Defining the structure of the model (using Convolutional Neural Network)
#Sequential means 'step-by-step' building
model = keras.Sequential([
  layers.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),  # rescales the image values from 0-255 to 0-1
  layers.Conv2D(16, 3, padding='same', activation='relu'),
    # convolutional layers, basically filters to create feature maps that reps the features in the image
  layers.MaxPooling2D(),  # downsample the above so it's less to calculate
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),#2D feature maps into 1D vector for the next layer
  layers.Dense(128, activation='relu'),#dense layer that uses 128 neurons and relu activation function
  layers.Dense(num_classes),#same num of neurons as the classes
  layers.Activation(ACTIVATION_FUNCTION)
])

#The Dense layer is a type of layer in neural networks where each neuron is connected to all neurons in the previous layer
#this layer learns the global patterns from all pixels and not just the specified section
#So it takes the feature id'd by the convolutional layer AND the entire image and makes a final decision


#Config the learning process before training
#loss function: evaluating how well the algorithm models your dataset. If the predictions off, loss function outputs a higher number
#optimizer:tweaks the model's parameters based on the result of loss function
#metric:used to judge the performance of your model (but not used when training the model), we're judging the modal based on accuracy
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs=10 #train how many times over the whole dataset
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Visualizing the training performance, aka. to print things out, that's it
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Data Augmentation: by flipping or rotating the images, and retrain the model again so we don't need new dataset
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                   img_width,
                                   3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Dropout
model = keras.Sequential([
  data_augmentation,#the data augumentation layer is added
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),#dropout layer randomly sets a fraction of the input to 0 to prevent overfitting
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, name="outputs"),
  layers.Activation(ACTIVATION_FUNCTION)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

epochs = 15
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Visualizing the training performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save the model
model_path = 'exported_model/model.keras'
model.save(model_path)

# Test saving of the model and loading it back
def test_save_and_load_submission():
    model = tf.keras.models.load_model(model_path)

    # Load a single image for testing
    test_image_path = next(data_dir.glob('*/*.png'))
    img = tf.keras.utils.load_img(
        test_image_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Run detection with the original model
    original_result = model.predict(img_array)
    print("Original model predictions:", original_result)

    # Save the model to a file
    saved_model_path = os.path.expanduser("~/Downloads/test_model.keras")
    model.save(saved_model_path)

    # Load the model from the file
    loaded_model = tf.keras.models.load_model(saved_model_path)

    # Run detection with the loaded model
    new_result = loaded_model.predict(img_array)
    print("Loaded model predictions:", new_result)

    # Verify that the number of detections is the same
    assert len(original_result) == len(new_result), "Number of detections differ"

    # Verify that the predictions are close
    for orig, new in zip(original_result[0], new_result[0]):
        assert np.isclose(orig, new), f"Prediction mismatch: {orig} vs {new}"

# Run the test for saving and loading the model
test_save_and_load_submission()