import librosa
import librosa.display
import scipy.io.wavfile
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
from keras import utils
from keras import applications,models, losses,optimizers
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
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm import tqdm,tqdm_pandas



print('all package loaded')

os.chdir('D:/data_science/kaggle_whale_detection')
os.getcwd()
train = pd.read_csv('train.csv')
class Config(object):
    def __init__(self,
                 n_classes=4251,n_channel = 3,width = 64,depth = 64,
                 n_folds=3, learning_rate=0.01, 
                 max_epochs=10):
        self.n_classes = n_classes
        self.n_folds = n_folds
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.n_channel = n_channel
        self.width = width
        self.depth = depth
        self.dim = (self.width, self.width, self.n_channel)

class DataGenerator(Sequence):
    def __init__(self, config, data_dir, list_IDs, labels=None, 
                 batch_size=32):
        self.config = config
        self.data_dir = data_dir
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.on_epoch_end()
        self.dim = self.config.dim

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
      
    def read_img(img_path):
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img,(128,128))
        return img
    
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            data = cv2.imread(file_path, 1)
            data = cv2.resize(data,(128,128))/255
            X[i,] = data
            # Store class
        if self.labels is not None:
            y = np.empty(cur_batch_size, dtype=int)
            for i, ID in enumerate(list_IDs_temp):
                y[i] = self.labels[ID]
            return X, to_categorical(y, num_classes=self.config.n_classes)
        else:
            return X

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
config = Config(n_folds=3, learning_rate=0.01)    
import keras
from keras import losses, models, optimizers
from keras.activations import relu, softmax
def conv_2D_mod(config):
    nclass = config.n_classes
    inp = Input(shape = config.dim)
    x = Conv2D(48, kernel_size=(3, 3),
                 activation='relu',
                 data_format="channels_last")(inp)
    x = Conv2D(48, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(48, (5, 5), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3))(x)
    x = Dropout(0.33)(x)
    x = Flatten()(x)
    x = Dense(36, activation='relu')(x)
    x= Dropout(0.33)(x)
    x = Dense(36, activation='relu')(x)
    out = Dense(4251, activation=softmax)(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    model.summary() 
    return model

test = pd.read_csv('sample_submission.csv')
LABELS = list(train['Id'].unique())
label_idx = {label: i for i,label in enumerate(LABELS)}
train.set_index('Image', inplace = True)
test.set_index("Image", inplace=True)
train['label_idx'] = train.Id.apply(lambda x: label_idx[x])        
COMPLETE_RUN = True

PREDICTION_FOLDER = "predictions_2dconv_generator"
if not os.path.exists(PREDICTION_FOLDER):
    os.mkdir(PREDICTION_FOLDER)

skf = StratifiedKFold(train.label_idx, n_folds=config.n_folds)

for i, (train_split, val_split) in enumerate(skf):
    train_set = train.iloc[train_split]
    val_set = train.iloc[val_split]
    checkpoint = ModelCheckpoint('best_%d.h5'%i, monitor='val_loss', verbose=1, save_best_only=True)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    #tb = TensorBoard(log_dir='./logs/' + PREDICTION_FOLDER + '/fold_%d'%i, write_graph=True)

    callbacks_list = [checkpoint, early]
    print("Fold: ", i)
    print("#"*50)
    
    model = conv_2D_mod(config)

    train_generator = DataGenerator(config, 'D:/data_science/kaggle_whale_detection/train/', train_set.index, 
                                    train_set.label_idx, batch_size=32)
    val_generator = DataGenerator(config, 'D:/data_science/kaggle_whale_detection/train/', val_set.index, 
                                  val_set.label_idx, batch_size=32)

    
    history = model.fit_generator(train_generator, callbacks=callbacks_list, validation_data=val_generator,
                                  epochs=config.max_epochs, use_multiprocessing=True, workers=6)

    model.load_weights('best_%d.h5'%i)
    test_generator = DataGenerator(config, 'D:/data_science/kaggle_whale_detection/test/', test.index, batch_size=32)
    predictions = model.predict_generator(test_generator, use_multiprocessing=True, 
                                          workers=6, max_queue_size=20, verbose=1)
    np.save(PREDICTION_FOLDER + "/test_predictions_%d.npy"%i, predictions)

    # Make a submission file
    top_3 = np.array(LABELS)[np.argsort(-predictions, axis=1)[:, :3]]
    predicted_labels = [' '.join(list(x)) for x in top_3]
    test['label'] = predicted_labels
    test[['label']].to_csv(PREDICTION_FOLDER + "/predictions_%d.csv"%i)