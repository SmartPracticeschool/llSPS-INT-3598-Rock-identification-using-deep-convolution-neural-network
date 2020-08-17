# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 15:26:09 2020

@author: lenovo
"""

#importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import numpy as np

#initialize model
model = Sequential()
model.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(output_dim=1,activation='sigmoid',init='uniform'))

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1)

x_train = train_datagen.flow_from_directory(r'C:/Dataset/training_set',target_size=(64,64),batch_size=8,class_mode='binary')
x_test = test_datagen.flow_from_directory(r'C:/Dataset/test_set',target_size=(64,64),batch_size=8,class_mode='binary')
print(x_train.class_indices)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(x_train,steps_per_epoch=7,epochs=4,validation_data=x_test,validation_steps=2)

model.save('cnn.h5')

from keras.models import load_model
from keras_preprocessing import image


model = load_model('cnn.h5')

img = image.load_img(r'C:\Users\lenovo\Music\Desktop\igneous.jpg',target_size=(64,64))
x = image.img_to_array(img)
x = np.expand_dims(x,axis = 0)
pred=model.predict_classes(x)
pred
