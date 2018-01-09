'''
This script goes along with the blog post "Building powerful image classification
models using very little data" from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
Data preparation steps are as follows:
- create data folder structure
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1399 in data/validation/cats
- put the dogs pictures index 0-999 in data/train/dogs
- put the dog pictures index 1000-1399 in data/validation/dogs
So that we have the first 1000 training examples for each class, unshuffled, and
400 validation examples for each class. In summary, this is our directory
structure:

data/
    train/
        dogs/
            dog.0.jpg
            ...
            dog.999.jpg
        cats/
            cat.0.jpg
            ...
            cat.999.jpg
    validation/
        dogs/
            dog.1000.jpg
            ...
            dog.1399.jpg
        cats/
            cat.1000.jpg
            ...
            cat.1399.jpg
'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import datetime as dt

print("Begin setup ... ...")
start_time = dt.datetime.now()

# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = '../data/train'
validation_data_dir = '../data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Report completion of setup and duration. Start next steps.
stop_time = dt.datetime.now()
print("Elapsed time for setup = ", (stop_time - start_time).total_seconds(), "s.")
print("\nCreating the ImageDataGenerator object ... ...")
start_time = dt.datetime.now()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Report completion of generator and duration. Start next steps.
stop_time = dt.datetime.now()
print("Elapsed time for creation of generator = ", (stop_time - start_time).total_seconds(), "s.")
print("\nFitting the model ... ...")
start_time = dt.datetime.now()

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# Report completion of fitting and duration. Start next steps.
stop_time = dt.datetime.now()
print("Elapsed time for fitting the model = ", (stop_time - start_time).total_seconds(), "s.")
print("\nSaving the weights ... ...")
start_time = dt.datetime.now()

model.save_weights('first_try.h5')

stop_time = dt.datetime.now()
print("Elapsed time for saving the weights = ", (stop_time - start_time).total_seconds(), "s.")
print("\nComplete")
