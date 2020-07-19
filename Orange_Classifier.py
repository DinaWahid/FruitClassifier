#Name: Dina Wahid Salamah Elgohary
#ID: 025661

#importing the needed packages
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

#setting the total numbers of classes for classification
num_classes = 2
#setting the size to which the images will be resized and the batch size
img_rows, img_cols = 32, 32
batch_size = 20

#loading the dataset into training dataset and testing variables
train_dataset = 'datasets/oranges/train'
test_dataset = 'datasets/oranges/test'

#setting a random seed
np.random.seed(0)

#using image augmentation to increase the size of training data
train_data = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

train_generator = train_data.flow_from_directory(
    train_dataset,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

#using image augmentation to increase the size of testing data
test_data = ImageDataGenerator(rescale=1. / 255)
test_generator = test_data.flow_from_directory(
    test_dataset,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

#creating the sequential model and adding
#the layers line by line
model = Sequential()

#creating the convolutional layers, pooling layers, and dense layers
#the parameters were set according to trial and error processes
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape= (img_rows, img_cols, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

#printing the model summary for observation
print(model.summary())

#saving the model to an .h5 file
checkpoint = ModelCheckpoint("models/oranges-classifier-model.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only=True,
                             verbose=1)

#setting early stopping to prevent overfitting
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=3,
                          verbose=1,
                          restore_best_weights=True)
#reducing the learning rate when a metric stops imporving
reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

#setting the callback list
callbacks = [earlystop, checkpoint, reduce_lr]

#setting the learning rate
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

#training the model
nb_train_samples = 3062
nb_validation_samples = 791
epochs = 20
history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=test_generator,
    validation_steps=nb_validation_samples // batch_size)

#calculating the accuracy
avg_accuracy=np.mean(history.history['accuracy'])
print(avg_accuracy)

#plotting training loss and validation loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, color='yellow', label='Training loss')
plt.plot(epochs, val_loss, color='blue', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plotting the training accuracy and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, color='yellow', label='Training acc')
plt.plot(epochs, val_acc, color='blue', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#printing the confusion matrix
predictions = model.predict_generator(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
results = confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix')
print(results)

#printing the classification report of the model
class_labels = test_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
target_names = list(class_labels.values())
print('Classification Report')
print(classification_report(test_generator.classes, predicted_classes, target_names=target_names))


