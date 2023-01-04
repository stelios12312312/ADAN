#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:37:44 2021

@author: stelios
"""

import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
from datetime import datetime

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
from sklearn.metrics import classification_report

import random



n_pop=[10,10,10]
path="data/banknote/banknotes2.csv"
# path='iris_1/iris_1.csv'
df=readData(path)
df=pd.read_csv(path,encoding='latin1')

res = run_all(target_name='E',\
                      ngen=10,quant_or_num=0.9,task='classification',path_or_data=path,\
                          sample_n=-1,n_pop=n_pop,test_perc=0.0,ngen_for_second_round=10,n_folds=10,
                          variables_to_keep=['A','B','D','E'])

conf=classification_report(np.round(res['predicted_train_values']),res['ground_truth_train'])
print(conf)

print(str(res['model'].model))
    

    