# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 12:14:45 2018

@author: deept
"""

### Variational AutoEncoder using Keras ###

#imports
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline
from scipy.stats import norm

import keras
from keras import layers
from keras.models import Model
from keras import metrics
from keras import backend as K

K.clear_session()
np.random.seed(237)

#Loading Dataset
train_orig = pd.read_csv('Data/MNIST/train.csv')
test_orig = pd.read_csv('Data/MNIST/test.csv')
train_orig.head()

#Combining Train and Test:
test_orig['label'] = 11
testCols = test_orig.columns.tolist()
testCols = testCols[-1:] + testCols[:-1]
test_orig = test_orig[testCols]

combined = pd.concat([train_orig, test_orig], ignore_index = True)
combined.head()
combined.tail()


# Split into training and val:
val = combined.sample(n=5000, random_state = 555)
train = combined.loc[~combined.index.isin(val.index)]
del train_orig, test_orig, combined

val.head()


#reshape and normalize

X_train = train.drop(['label'], axis = 1)
X_val = val.drop(['label'], axis = 1)

# Labels
y_train = train['label']
y_val = val['label']


# Normalize and reshape
X_train = X_train.astype('float32') / 255.
X_train = X_train.values.reshape(-1,28,28,1)

X_val = X_val.astype('float32') / 255.
X_val = X_val.values.reshape(-1,28,28,1)

plt.figure(1)
plt.subplot(221)
plt.imshow(X_train[13][:,:,0])

plt.subplot(222)
plt.imshow(X_train[690][:,:,0])

plt.subplot(223)
plt.imshow(X_train[2375][:,:,0])

plt.subplot(224)
plt.imshow(X_train[42013][:,:,0])
plt.show()

img_shape = (28,28,1)
batch_size = 16
latent_dim = 2

# Encoder architecture: Input --> Conv2D*4 --> Faltten --> Dense
input_img = keras.Input(shape = img_shape)

x = layers.Conv2D(32, 3, padding = 'same', activation = 'relu')(input_img)
x = layers.Conv2D(64, 3, padding = 'same', activation = 'relu', strides = (2,2))(x)
x = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(x)
x = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(x)

shape_before_flattening = K.int_shape(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation = 'relu')(x)

# Two output, latent mean and (log)variance
z_mu = layers.Dense(latent_dim)(x)
z_log_sigma = layers.Dense(latent_dim)(x)

def sampling(args):
    z_mu, z_log_sigma = args
    epsilon = K.random_normal(shape = (K.shape(z_mu)[0], latent_dim), mean=0., stddev = 1.)
    return z_mu + K.exp(z_log_sigma) * epsilon

z = layers.Lambda(sampling)([z_mu, z_log_sigma])

## decoder arch - reverse of encoder.
decoder_input = layers.Input(K.int_shape(z)[1:])

x = layers.Dense(np.prod(shape_before_flattening[1:]),
                         activation = 'relu')(decoder_input)

# reshape
x = layers.Reshape(shape_before_flattening[1:])(x)

x = layers.Conv2DTranspose(32, 3, padding = 'same', activation = 'relu', strides = (2,2))(x)
x = layers.Conv2D(1, 3, padding = 'same', activation = 'sigmoid')(x)

decoder = Model(decoder_input, x)
z_decoded = decoder(z)

class CustomVariationalLayer(keras.layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + z_log_sigma - K.square(z_mu) - K.exp(z_log_sigma), axis = -1)
        return K.mean(xent_loss + kl_loss)
    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x,z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x
y = CustomVariationalLayer()([input_img, z_decoded])

vae = Model(input_img, y)
vae.compile(optimizer = 'rmsprop', loss = None)
vae.summary()

vae.fit(x = X_train, y = None, shuffle= True, epochs = 7, batch_size = batch_size, validation_data = (X_val, None))

valid_noTest = val[val['label']!=11]

#Xs and Ys
X_valid_noTest = valid_noTest.drop('label', axis = 1)
y_valid_noTest = valid_noTest['label']

X_valid_noTest = X_valid_noTest.astype('float32') / 255.
X_valid_noTest = X_valid_noTest.values.reshape(-1, 28, 28, 1)

encoder = Model(input_img, z_mu)
x_valid_noTest_encoded = encoder.predict(X_valid_noTest, batch_size = batch_size)
plt.figure(figsize = (10,10))
plt.scatter(x_valid_noTest_encoded[:, 0], x_valid_noTest_encoded[:, 1],
            c = y_valid_noTest, cmap='brg')
plt.colorbar()
plt.show()

custom_cmap = matplotlib.cm.get_cmap('brg')
custom_cmap.set_over('gray')
x_valid_encoded = encoder.predict(X_val, batch_size = batch_size)
plt.figure(figsize = (10, 10))
gray_marker = mpatches.Circle(4, radius = 0.1, color='gray', label = 'Test')
plt.legend(handles = [gray_marker], loc = 'best')
plt.scatter(x_valid_encoded[:, 0], x_valid_encoded[:, 1], c = y_val, cmap=custom_cmap)
plt.clim(0, 9)
plt.colorbar()

n = 20
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
        x_decoded = decoder.predict(z_sample, batch_size = batch_size)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit
plt.figure(figsize = (10,10))
plt.imshow(figure, cmap = 'gnuplot2')
plt.show()



































