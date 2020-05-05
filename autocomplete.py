# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 13:05:46 2019

@author: deept
"""


#Imports
print("[INFO] Importing dependencies.")
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.layers import LSTM, Dropout
from keras.layers import TimeDistributed
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams

print("[INFO] Dependencies imported.")

print("[INFO] Variables initialized to defaults.")

SEQ_LEN = 40
STEP = 3
sns.set(style = 'whitegrid', palette = 'muted', font_scale = 1.5)
rcParams['figure.figsize'] = 12, 5
chars = []
sentences = []
next_chars = []
def load_train_data(path='Data/autocomplete_train.txt', verbose = 1):
    text = open(path).read().lower()
    if len(chars) ==0:
        chars = sorted(list(set(text)))
    else:
        for c in sorted(list(set(text))):
            if c in chars:
                continue
            else:
                chars.append(c)
                    
    indices_to_char = dict((i, c) for i, c in enumerate(chars))
    char_to_indices = dict((c, i) for i, c in enumerate(chars))
    if verbose == 1:
        print("[INFO] The corpus length is ", len(text)," words.")
        print("[INFO] The number of unique characters is ", len(chars),".")
    char_len = len(chars)
    return text, chars, indices_to_char, char_to_indices


def prepare_training_sentences(text, chars, verbose = 0):
        for i in range(0, len(text) - SEQ_LEN, STEP):
            sentences.append(text[i: i + SEQ_LEN])
            next_chars.append(text[i + SEQ_LEN])
        X = np.zeros((len(sentences), SEQ_LEN, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), char_len), dtype = np.bool)
        if verbose == 1:
            print("The number of training samples is ", len(sentences),".")
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, c2i[char]] =- 1
            y[i, c2i[next_chars[i]]] = 1
        X = X
        y = y
        return X, y, next_chars, sentences