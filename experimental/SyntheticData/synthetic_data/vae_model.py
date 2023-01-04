#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 12:05:11 2018

@author: stelios
"""

'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''



from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils.vis_utils import plot_model
from keras import backend as K

import numpy as np
import argparse
import os
import pandas as pd
from helpers import reverse_categoricals, reverse_dates,calc_error
import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def run_vae(df,num_rows=100):
    means=df.mean().values
    std=df.std().values
    
    x_train=(df-means)/std
    original_dim = df.shape[1]
    # network parameters
    input_shape = (original_dim, )
    intermediate_dim = int(x_train.shape[1]*10)
    batch_size = 128
    latent_dim = (int(x_train.shape[1]/2)+1)*2
    epochs = 500
    
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim*3, activation='relu')(inputs)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(intermediate_dim*2, activation='relu')(inputs)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.BatchNormalization()(x)
    x = Dense(intermediate_dim, activation='relu')(inputs)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.BatchNormalization()(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='linear')(x)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')
    
    models = (encoder, decoder)
    
    reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + 2*kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    
    X_train, X_test, y_train, y_test = train_test_split(x_train, x_train, test_size=0.1)
    # train the autoencoder
    vae.fit(X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, None),
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=0)])
            
    
    preds=[]
    for n in range(num_rows):
        
        pred=decoder.predict(np.array([np.random.randn(latent_dim)*10]))[0]
        preds.append(pred)
    preds=pd.DataFrame(np.array(preds))
    preds=preds*std+means
    preds.columns = df.columns
    error=calc_error(preds,df)
    print(error)
      
    return preds, error
   
