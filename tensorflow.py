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
K.set_image_data_format('channels_first')
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
from keras.layers import Convolution2D, MaxPooling2D
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

train = pd.read_csv('D:/data_science\kaggle_whale_detection/train.csv')

#REad-image
# function to read image
def read_img(img_path):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img,(128,128))
    return img
## set path for images
TRAIN_PATH = 'D://data_science/kaggle_whale_detection/train/'
TEST_PATH = 'D://data_science/kaggle_whale_detection/test.zip/test/'
TEST_PATH = glob('D:/data_science/kaggle_whale_detection/test/*jpg')
# load data
train_img = []
for img_path in (train['Image'].values):
    train_img.append(read_img(TRAIN_PATH + img_path))
    
x_train = np.array(train_img)
x_train = x_train/255
#One hot encoding - Id
class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()
    def fit_transform(self, x):
        features = self.le.fit_transform( x)
        return self.ohe.fit_transform( features.reshape(-1,1))
    def transform( self, x):
        return self.ohe.transform( self.la.transform( x.reshape(-1,1)))
    def inverse_tranform( self, x):
        return self.le.inverse_transform( self.ohe.inverse_tranform( x))
    def inverse_labels( self, x):
        return self.le.inverse_transform( x)

train["Image"] = train["Image"].map(lambda x : "D:/data_science/kaggle_whale_detection/train/"+x)
ImageToLabelDict = dict( zip( train["Image"], train["Id"]))
train_images = train['Image'].tolist() 
y = list(map(ImageToLabelDict.get, train_images))
lohe = LabelOneHotEncoder()
y_cat = lohe.fit_transform(y)


#tensorflow implementation
#placeholder
tf.reset_default_graph()
tf.__version__
import __future__

nclass = y_cat.toarray().shape[1]
X = tf.placeholder(tf.float32, shape=[None, 128,128,1])
Y = tf.placeholder(tf.float32, [None, nclass])
y_true_cls = tf.argmax(Y, dimension=1)
#Block function
import tensorflow.contrib.layers as layers
def conv_layer(input, num_channel, filter_size, no_filter,name):
    with tf.variable_scope(name) as scope:
        shape = [filter_size,filter_size,num_channel,no_filter]
        weight = tf.Variable(tf.truncated_normal(shape,dtype = tf.float32, stddev=0.05))
        bias = tf.Variable(tf.constant(0.1,shape = [no_filter]))
        layer = tf.nn.conv2d(input, filter=weight,strides=[1,1,1,1], padding = 'SAME')
        layer+=bias
        return layer, weight
    
def pool_layer(input,name):
    with tf.variable_scope(name) as scope:
        layer = tf.nn.max_pool(value=input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    return layer    

def relu_activation(input,name):
    with tf.variable_scope(name) as scope:
        layer = tf.nn.relu(input)
    return layer

def fully_connected(input,ninput,noutput,name):
    with tf.variable_scope(name) as scope:
        shape = [ninput,noutput]
        weight = tf.Variable(tf.truncated_normal(shape,dtype = tf.float32, stddev=0.05))
        bias = tf.Variable(tf.constant(0.1,shape = [noutput]))
        layer = tf.matmul(input, weight)+bias
        return layer
    
#Create CNN
#conv layer-1
cov1_layer, cov1_wgt = conv_layer(X,1,3,48,'conv1')        
#pool layer-1
pool1_layer = pool_layer(cov1_layer, name = 'pool1')
#relu-1
relu_1 = relu_activation(pool1_layer,'relu1')
#conv layer-2
conv2_layer, conv2_wgt = conv_layer(relu_1,num_channel = 48,filter_size = 5,no_filter =48,name = 'conv2')        
#pool layer-1
pool2_layer = pool_layer(conv2_layer, name = 'pool2')
#relu-1
relu_2 = relu_activation(pool2_layer,'relu2')
#drop out
relu_2 = tf.nn.dropout(relu_2, keep_prob = 0.7)
#Flaten
feature = relu_2.get_shape()[1:4].num_elements()
flatten1 = tf.reshape(relu_2, shape = [-1,feature], name = 'flatten1')
#fully connected
fc_layer1 = fully_connected(flatten1, ninput = feature, noutput = 36,name = 'fc1')
relu_3 = relu_activation(fc_layer1,'relu3')
fc_layer1 = tf.nn.dropout(fc_layer1, keep_prob = 0.7)
fc_layer2 = fully_connected(fc_layer1, ninput = 36, noutput = 36,name = 'fc2')
fc_layer3 = fully_connected(fc_layer2, ninput = 36, noutput = nclass,name = 'fc3')
#Soft max
# Use Softmax function to normalize the output
with tf.variable_scope("Softmax"):
    y_pred = tf.nn.softmax(fc_layer3)
    y_pred_cls = tf.argmax(y_pred, dimension=1)
cost = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y_pred)
cost = tf.reduce_mean(weighted_losses)

with tf.name_scope("optimizer"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

import keras
le = LabelEncoder()
y = le.fit_transform(y)
y_cat = keras.utils.to_categorical(y, num_classes=nclass)
from sklearn.cross_validation import train_test_split
train_X,test_X,train_y,test_y = train_test_split(x_train,y_cat, test_size = 0.05, random_state = 20) 

display_step = 100
n_epoch = 30
init = tf.global_variables_initializer()
step = 0
batch_size = 32
# Start training
no_of_batches = int(len(train_X)/batch_size)    
for step in range(1,n_epoch):
    ptr = 0
    for j in range(1, no_of_batches):
        batch_x, batch_y = train_X[ptr:ptr+batch_size], train_y[ptr:ptr+batch_size]
        ptr+=batch_size
        sess = tf.Session()
        tf.global_variables_initializer().run()
        batch_x = batch_x.reshape((batch_size,128,128,1))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, accuracy = sess.run([cost, acc], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " +                   "{:.4f}".format(loss) + ", Training Accuracy= " +                   "{:.3f}".format(accuracy))
    print("Epoch -",str(step))       
    loss, accuracy = sess.run([cost, acc], feed_dict={x: test_X,
                                                                 y: test_y})
    print("Step " + str(step) + ", step Loss= " +                   "{:.4f}".format(loss) + ", step Test Accuracy= " +                   "{:.3f}".format(accuracy))
print('optimization complete')
sess.close()