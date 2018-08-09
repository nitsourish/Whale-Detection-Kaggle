# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:20:52 2018

@author: Sourish
"""
from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
import cv2
from glob import glob
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import tqdm
import cv2
from keras import applications
from keras.models import Model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D,LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.layers import Input, Dense, Flatten, Dropout, Activation, Lambda, Permute, Reshape
import numpy as np
import keras.backend as K
from scipy.spatial import distance
from PIL import Image
from keras.preprocessing import image
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from hyperas.distributions import choice, uniform, conditional
import hyperopt
from sklearn.cross_validation import train_test_split

print('all package loaded')

def data():
    train = pd.read_csv('D:/data_science\kaggle_whale_detection/train.csv')
    train = train.iloc[1:100,:]
    def read_img(img_path):
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img,(128,128))
        return img

    TRAIN_PATH = 'D://data_science/kaggle_whale_detection/train/'
    TEST_PATH = 'D://data_science/kaggle_whale_detection/test.zip/test/'
    TEST_PATH = glob('D:/data_science/kaggle_whale_detection/test/*jpg')

    train_img = []
    test_img = []
    for img_path in (train['Image'].values):
        train_img.append(read_img(TRAIN_PATH + img_path))
    x_train = np.array(train_img)
    x_train = x_train/255

    train["Image"] = train["Image"].map(lambda x : "D:/data_science/kaggle_whale_detection/train/"+x)
    ImageToLabelDict = dict( zip( train["Image"], train["Id"]))

    train_images = train['Image'].tolist() 
    y = list(map(ImageToLabelDict.get, train_images))

    le = LabelEncoder()
    y = le.fit_transform(y)
    import keras
    y_cat = keras.utils.to_categorical(y, num_classes=4251)
    train_X,test_X,train_y,test_y = train_test_split(x_train,y_cat, test_size = 0.05, random_state = 20) 
    return train_X, test_X, train_y, test_y

print('data ready for model')

def create_model(train_X, train_y, test_X, test_y):
    model = Sequential()
    model.add(Conv2D({{choice([48,64])}}, kernel_size=(3, 3),
                 input_shape=(128,128,3),data_format="channels_last"))
    model.add(LeakyReLU(alpha={{uniform(0.5, 1)}}))
    model.add(Conv2D(48, (3,3)))
    model.add(Activation({{choice(['relu','sigmoid','tanh'])}}))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout({{uniform(0, 0.5)}}))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense({{choice([18, 36, 72])}}, activation='relu'))
    model.add(Dropout(0.33))
    model.add(Dense({{choice([36,72])}}))
    model.add(Activation({{choice(['relu','sigmoid','tanh'])}}))
    model.add(Dense(4251, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer={{choice(['rmsprop', 'adam'])}},
              metrics=['accuracy'])

    filepath='keras_model_wgtcorr'
    checkpoint = ModelCheckpoint(filepath,mode='max',monitor='val_acc',save_best_only=True,verbose=1)
    train_datagen = ImageDataGenerator(rotation_range=15, 
        width_shift_range=0.15,
        height_shift_range=0.15, 
        horizontal_flip=True)
    train_datagen.fit(train_X,augment = True)
    weight = [0.21022410381342865,	 0.5946035575013605,	 0.5946035575013605,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	 1.0,	1.0]   
    cl = [0,	27,	31,	8,	30,	76,	71,	36,	3,	11,	58,	62,	61,	9,	10,	49,	33,	67,	64,	82,	13,	17,	63,	7,	73,	16,	32,	85,	42,	28,	4,	20,	35,	1,	41,	68,	12,	15,	45,	86,	72,	34,	46,	37,	84,	74,	69,	56,	23,	79,	51,	80,	59,	19,	6,	50,	40,	88,	66,	70,	47,	38,	53,	75,	52,	48,	77,	18,	39,	25,	26,	2,	57,	14,	65,	21,	89,	22,	55,	5,	54,	83,	87,	24,	29,	81,	44,	78,	43,	60] 
    class_weight_dic = dict(zip(cl,weight))
    model.fit_generator(train_datagen.flow(train_X, train_y, batch_size=32),
          steps_per_epoch=  train_X.shape[0]//32,
          epochs={{choice([10, 20])}},validation_data=(test_X, test_y),callbacks=[checkpoint],class_weight=class_weight_dic,
          verbose=1)
      
    score, acc = model.evaluate(test_X, test_y, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

print('model optimization started')

from hyperas import optim
from hyperopt import Trials, STATUS_OK, tpe
if __name__ == '__main__':
    import gc; gc.collect()
    best_run, best_model = optim.minimize(model=create_model,data = data,
                                          algo=tpe.suggest,
                                          max_evals=10,
                                          trials=Trials())
    train_X, test_X, train_y, test_y = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(test_X, test_y))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)