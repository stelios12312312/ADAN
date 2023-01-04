#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:14:28 2022

@author: stelios
"""
import numpy as np

import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)


import cdt
import adan
from adan.aiem.genetics import *
from adan.aidc.utilities import *
from adan.aidc.feature_selection import *


from adan.metrics.metrics_utilities import *
from adan.aipm.optimizers.hyperparam_optimization import *

from adan.aiem.symbolic_modelling import *
from adan.aiem.genetics.genetic_programming import *
from adan.protocols import *
from matplotlib import pyplot as plt
from adan.aist.mappers import *
from adan.aipd.aipd_main import *
from adan.aiem.symbolic_modelling import *
from adan.aiem.optimise import reverse_optimise
import os
import sklearn

import random
#random.seed(27)
pd.set_option('display.max_columns', 500)

from Pyfhel import Pyfhel, PyPtxt, PyCtxt
# Pyfhel class contains most of the functions.
# PyPtxt is the plaintext class
# PyCtxt is the ciphertext class


ngen=3
quant=0.9
ngen=10
quant=0.5
task='regression'
path="auto_mpg/auto_mpg.csv"
sample_n=1000
              
choice_method='quant'
fix_skew=True
              
allowed_time=6

n_pop=300

df=pd.read_csv(path)

k = run_all(path, target_name='V1', task=task, quant_or_num=quant,ngen=ngen,
             sample_n=sample_n,choice_method=choice_method,causal=False,
             fix_skew=fix_skew,allowed_time=allowed_time,n_pop=n_pop,ngen_for_second_round=5,
             pca_criterion_thres=1)



print("==============================================================")
print("================ Pyfhel with Numpy and Pickle ================")
print("==============================================================")


print("1. Creating Context and KeyGen in a Pyfhel Object ")
HE = Pyfhel()           # Creating empty Pyfhel object
HE.contextGen(p=65537)  # Generating context. The value of p is important.
                        #  There are many configurable parameters on this step
                        #  More info in Demo_ContextParameters.py, and
                        #  in the docs of the function (link to docs in README)
HE.keyGen()             # Key Generation.
print(HE)

print("2. Encrypting two arrays of integers.")
print("    For this, you need to create empty arrays in numpy and assign them the cyphertexts")
array1 = np.array([1.3,3.4,5.8,7.1,9.2])
array2 = np.array([-2., 4., -6., 8.,-10.])
arr_gen1 = np.empty(len(array1),dtype=PyCtxt)
arr_gen2 = np.empty(len(array1),dtype=PyCtxt)

# Encrypting! This can be parallelized!
for i in np.arange(len(array1)):
    arr_gen1[i] = HE.encryptFrac(array1[i])
    arr_gen2[i] = HE.encryptFrac(array2[i])