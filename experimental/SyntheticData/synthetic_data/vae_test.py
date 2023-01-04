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
from helpers import *
import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from vae_model import run_vae


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



# MNIST dataset
df = pd.read_excel('tokenomics_data.xlsx')
df.drop(['Name','Month'],inplace=True,axis=1)

df=df.replace([np.inf, -np.inf], np.nan)
df.dropna(inplace=True)
df = df[df.iloc[:,0]>10**6]
df = df[df['Token Sale Price']>0]
df['year3']=df['Diff days']>=365*2
df['year2']=np.logical_and(df['Diff days']>365,df['Diff days']<=365*2)
df['year1']=df['Diff days']<=365
df['years_survived']='One'
df.loc[df['year1'],'years_survived']='One'
df.loc[df['year2'],'years_survived']='Two'
df.loc[df['year3'],'years_survived']='Three'

df['somedate']=pd.date_range('1/1/2011', periods=len(df), freq='H')

df_original=df.copy()
df,params,date_params = process_dataset(df)

run_vae(df)




