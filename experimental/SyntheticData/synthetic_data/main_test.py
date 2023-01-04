#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 13:20:53 2018

@author: stelios
"""


import pandas as pd
import scipy as sp
import scipy.stats
from scipy.stats import burr
import numpy as np
from helpers import *

iters=10
num_rows=100

df = readData('tokenomics_data.xlsx',type_of_file="excel")
df=impute_missing(df)
df.drop(['Name','Month'],inplace=True,axis=1)


#THESE ARE SOME BASIC MANIPULATIONS which are created for test purposes

#df=df.replace([np.inf, -np.inf], np.nan)
#df.dropna(inplace=True)
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
df['somedate']=df['somedate'].astype('object')

#copying the original so we can compare the original dataset against the synthetic
df_original=df.copy()


#main functions. Produces an artificial dataset (df_art) and some test metrics (error_total)
df_art, error_total = core_function(df,num_rows=None,iters=10,optim_iters=100,use_vae=True)