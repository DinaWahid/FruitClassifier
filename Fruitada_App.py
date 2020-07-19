#Name: Dina Wahid Salamah Elgohary
#ID: 025661

#importing the needed packages
from __future__ import division, print_function
from keras_preprocessing.image import ImageDataGenerator
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.preprocessing import image
from keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

#defining the flask app
app = Flask(__name__)


#loading the trained models
model_fruit = load_model('models/fruits-classifier-model.h5')
model_orange = load_model("models/oranges-classifier-model.h5")


#setting the size to which the images will be resized and the batch size
img_rows, img_cols = 32, 32
batch_size = 20

#getting the labels of the fruit classifier model
train_dataset_fruit = 'datasets/fruits/fruits/Train'
test_dataset_fruit = 'datasets/fruits/fruits/Test'

train_data_fruit= ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator_fruit = train_data_fruit.flow_from_directory(
    train_dataset_fruit,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

test_data_fruit = ImageDataGenerator(rescale=1. / 255)

test_generator = test_data_fruit.flow_from_directory(
    test_dataset_fruit,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

class_labels_fruit = test_generator.class_indices
class_labels_fruit = {v: k for k, v in class_labels_fruit.items()}
classes_fruit = list(class_labels_fruit.values())

#getting the labels of the orange classifier model
train_dataset_orange = 'datasets/oranges/oranges/train'
test_dataset_orange = 'datasets/oranges/oranges/test'

train_data_orange = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

test_data_orange = ImageDataGenerator(rescale=1. / 255)

train_generator_orange = train_data_orange.flow_from_directory(
    train_dataset_orange,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

test_generator_orange = test_data_orange.flow_from_directory(
    test_dataset_orange,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

class_labels_orange = test_generator_orange.class_indices
class_labels_orange = {v: k for k, v in class_labels_orange.items()}
classes_orange = list(class_labels_orange.values())


#routing for the home page
@app.route('/', methods=['GET'])
def index():
    return render_template('home.html')

#routing for the fruit classifier page
@app.route("/fruits_classifier_page")
def classifier():
  return render_template("fruits_classifier_page.html")

#routing to the orange classifier page
@app.route("/orange_classifier_page")
def orange():
    return render_template("orange_classifier_page.html")

#routing to the about page
@app.route("/about_page")
def about():
  return render_template("about_page.html")

#allowing the user to upload an image, save it in the upload file, then feed it into the fruit classifier model
#then returning the result obtained from the model
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        #getting the image and saving it
        #to the uploads folder
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        #preparing the image for the model
        img = image.load_img(file_path, target_size=(32, 32))
        predictions = []
        x = image.img_to_array(img)
        x = x * 1. / 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        #predicting the class of the image fed into the model then returning the result
        classes_fruit = model_fruit.predict_classes(images, batch_size=10)
        predictions.append(classes_fruit)
        result = class_labels_fruit[predictions[0][0]]
        print(class_labels_fruit[predictions[0][0]])

        return result

    return None

#allowing the user to upload an image, save it in the upload file, then feed it into the fruit classifier model
#then returning the result obtained from the model
@app.route('/predict2', methods=['GET', 'POST'])
def upload2():
    if request.method == 'POST':

        # getting the image and saving it
        #to the uploads folder
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # preparing the image for the model
        img = image.load_img(file_path, target_size=(32, 32))
        predictions = []
        x = image.img_to_array(img)
        x = x * 1. / 255
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])
        # predicting the class of the image fed into the model then returning the result
        classes_2 = model_orange.predict_classes(images, batch_size=10)
        predictions.append(classes_2)
        result = class_labels_orange[predictions[0][0]]
        print(class_labels_orange[predictions[0][0]])

        return result

    return None

#running the web app
if __name__ == '__main__':
    app.run(debug=False, threaded=False)