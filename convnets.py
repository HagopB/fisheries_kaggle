from __future__ import division,print_function
import math, os, json, sys, re
import _pickle as pickle
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain
from imp import reload

import pandas as pd
import PIL
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from scipy.misc import imread, imresize, imsave

from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import BaggingClassifier 
from sklearn.metrics import label_ranking_average_precision_score, accuracy_score, f1_score, fbeta_score, confusion_matrix

from sklearn.decomposition import PCA
from sklearn.utils import resample

from IPython.lib.display import FileLink
import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, l1
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import *
from keras.layers.merge import *
from keras.optimizers import SGD, RMSprop, Adam
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras import applications
import tensorflow as tf


def get_VGG16(trainable=False,
              pop=True):
    '''It calls the convolutional part of the vgg model. 
    The model will mainly serve as feature extractor from the images
    Parameters:
    -------------
    trainable: Boolean, optional (default=False) 
        If to train the convolutional layers or not
        
    pop: Boolean, optional (default=True)
        if to pop the Maxpooling layer of not
    
    Output:
    -------------
    model: Keras Sequential model 
        The full model, compiled, ready to be fit.
    '''
    
    #importing convolutional layers of vgg16 from keras
    model = VGG16(include_top=False, weights='imagenet',input_shape=(224,224,3))
    if pop == True:
        model.layers.pop()
    #setting the convolutional layers to non-trainable 
    for layer in model.layers:
        layer.trainable = trainable
    return(model)

def top_model_vgg(n_classes,
                  X_shape=(7,7,512),
                  dense_neurons=512,
                  do=0.5,
                  lr=0.01, 
                  loss_function = 'categorical_crossentropy'):
    """ Top model multi:MLP 
    The top model corresponds to the VGG16's classification layers.
    The model is adapted for MULTICLASS classification tasks.
    
    Parameters:
    -------------
    n_classes: int 
        How many classes are you trying to classify ? 
        
    X_shape: tuple, optional (default=(7,7,512))
        The input shape for the first layer.
    
    dense_neurons: int, optional (default=512)
        The number of neurons in the hidden dense layers
        
    do: float, optional (default=0.5) 
        Dropout probability
    
    lr: float, optional (default=0.01)
        The learning rate (between 0 and 1)
    
    loss_function: str, optional (default='categorical_crossentropy')
        The loss function (keras object)
        
    Output:
    -------------
    model: Keras Sequential model 
        The full model, compiled, ready to be fit.
    """
    
    ### top_model takes output from VGG conv and then adds 2 hidden layers
    top_model = Sequential()
    top_model.add(MaxPooling2D(input_shape=X_shape,name = 'top_maxpooling'))
    top_model.add(BatchNormalization())
    top_model.add(Dropout(do))
    
    top_model.add(Flatten(name='top_flatten'))
    top_model.add(Dense(dense_neurons, activation='relu', name='top_relu_1'))
    top_model.add(BatchNormalization())
    top_model.add(Dropout(do))
    top_model.add(Dense(dense_neurons, activation='relu', name='top_relu_2'))
    top_model.add(BatchNormalization())
    top_model.add(Dropout(do))
    
    ### the last multilabel layers with the number of classes
    top_model.add(Dense(n_classes, activation='softmax'))
    
    adam = keras.optimizers.Adam(lr=lr)
    top_model.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])
    
    return(top_model)

def heat_layers(n_classes,
                X_shape=(7,7,512),
                n_fm = 128,
                do=0.5,
                lr=0.01, 
                loss_function = 'categorical_crossentropy'):
    """ Top model - fully convolutional layers 
    The fully convolutional model is composed of full convo layers.
    The model is adapted for MULTICLASS classification tasks.
    
    Parameters:
    -------------
    n_classes: int 
        How many classes are you trying to classify ? 
        
    X_shape: tuple, optional (default=(7,7,512))
        The input shape for the first layer.
        
    n_fm: int, optional (default=128)
        The number of feature maps for the the convo layers 
        
    do: float, optional (default=0.5) 
        Dropout probability
    
    lr: float, optional (default=0.01)
        The learning rate (between 0 and 1)
    
    loss_function: str, optional (default='categorical_crossentropy')
        The loss function (keras object)
        
    Output:
    -------------
    model: Keras Sequential model 
        The full model, compiled, ready to be fit.
    """
    
    ### top_model takes output from VGG conv and then adds 2 hidden layers
    top_model = Sequential()
    top_model.add(BatchNormalization(input_shape=X_shape))
    top_model.add(Conv2D(n_fm,(3,3), activation='relu', padding='same'))
    top_model.add(BatchNormalization())
    top_model.add(Conv2D(n_fm,(3,3), activation='relu', padding='same'))
    top_model.add(BatchNormalization())
    top_model.add(Conv2D(n_fm,(3,3), activation='relu', padding='same'))
    top_model.add(BatchNormalization())
    top_model.add(Conv2D(n_classes,(3,3), padding='same'))
    top_model.add(Dropout(do))
    top_model.add(GlobalAveragePooling2D())
    top_model.add(Activation(activation='softmax'))
    
    adam = keras.optimizers.Adam(lr=lr)
    top_model.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])
    
    return(top_model)
                
def get_batches(dirname, 
                gen=image.ImageDataGenerator(rescale=1./255), 
                shuffle=False, 
                batch_size=1, 
                class_mode='categorical',
                target_size=(224,224)):
    """ Batches of images 
    Batch data generator, ready to be fit or predict on (generator).
    
    Parameters:
    -------------
    dir_name: str 
        The name of the directory where all images are stored in subfolders
        
    gen: object, optional (default=image.ImageDataGenerator(rescale=1./255))
        The image Data generator of keras
    
    shuffle: Boolean, optional (default=True)
        Wheter to shuffle of not the images
        
    batch_size: int, optional (default=1)
        The size of the Batch
    
    class_mode: str, optional (default='categorical')
        The class mode, refer to keras
    
    target_size: tuple, optional (default=(224,224))
        The size of the target, this will resize all images to this given size
        
    Output:
    -------------
    model: keras object 
        The keras data generator compiled ready to be fit or predict on
    """
    return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)