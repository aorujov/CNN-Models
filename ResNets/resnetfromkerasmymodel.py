
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.layers import Input
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import np_utils
import matplotlib.pyplot as plt
from skimage.transform import resize
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator



(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#plt.imshow(x_train[0]) 

x_train,y_train = shuffle(x_train,y_train, random_state=7)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=7)


num_classes = 10
batch_size=32



## Model 1

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))


model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


def img_resize(img):
    resized_img = resize(img, (224, 224))
    return resized_img

train_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=img_resize)
validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow(x_train, y_train,
        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels

validation_generator = validation_datagen.flow(x_val, y_val,
        batch_size=batch_size)

model.fit_generator(
        train_generator,
        steps_per_epoch=50000 // batch_size,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=10000 // batch_size)


model.save_weights('first_try.h5')


## Model2

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)





image_input = Input(shape=(224, 224, 3))

res50 = applications.ResNet50(weights='imagenet', include_top=True, input_tensor = image_input)

last_layer = res50.get_layer('avg_pool').output
A = Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(A)

custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

custom_resnet_model.fit_generator(my_generator(), epochs = 1, verbose=1, steps_per_epoch=50000//32 )






def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()




































b_size = 32

import random


def size_generator(x):
    imgs = []
    for j in range(x.shape[0]):
        img = x[j]
        resized_img = resize(img, (224, 224))
        imgs.append(resized_img)
        #yield np.expand_dims(resized_img,axis=0)
    return np.array(imgs)
    

def my_generator():
    #idxs = random.sample(range(img_arr.shape[0]), batch_size)
#    while 1:
    for i in range(50000 // 32):
        imgs = size_generator(x_train[i*32:(i+1)*32])
        yield (imgs, y_train[i*32: (i+1)*32])#.reshape(1,10))





# Resize image arrays
#x_train_gen = resize_image_arr(x_train, y_train)
#x_test_gen = resize_image_arr(x_test, y_test)

image_input = Input(shape=(224, 224, 3))

res50 = ResNet50(weights='imagenet', include_top=True, input_tensor = image_input)

last_layer = res50.get_layer('avg_pool').output
A = Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(A)

custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

custom_resnet_model.fit_generator(my_generator(), epochs = 1, verbose=1, steps_per_epoch=50000//32 )



for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


"""
C:\Users\abido\Desktop\Transfer-Learning-in-keras---custom-data

https://gist.github.com/abearman/916673e9a0f12bcf5ac2723b8b5eb819
"""

#analyze how generator works in keras models using yield
