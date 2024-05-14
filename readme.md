# Introduction
Team CABBI was tasked to implement a new model to identify anomalies in aerial images taken by drones.
The team implemented an existing colour-based detection model as the first stage of this model.
The second stage utilizes machine learning image classification techniques to rule out certain objects that were picked as anomalies in the first stage model.

This repository includes scripts to train and test the image classification model used for the second stage.


# Installing dependencies with Conda

Run the following command in your conda environment.
````
pip install -r requirements.txt
````

# Code source

The majority of the code in model-train.py was copied from:

https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb

# Loading training data

Put training data into the model_training/dataset
The dataset needs to be arranged in a hierarchy
All photos need to be separated into folders as classes. See the sample dataset flower_photo for reference.

# Training a new model
Before training, in model-train.py,
To set sigmoid or softmax functions., change the variable ACTIVATION_FUNCTION to either "sigmoid" or "softmax"

Set the variable DATASET_PATH to model_training/dataset/your_dataset

run the script python model-train.py

# Testing the model
Make sure all your classes are listed in class_names.py.

To run the model on a single image in the test_images folder:

run python model-test-single.py

To run the model on multiple images in the test_multi_images folder:

run python model-test-multi.py

