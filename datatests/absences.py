#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 18:44:55 2022

@author: stelios
"""
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
from sklearn.metrics import classification_report


import random
#random.seed(27)

df=pd.read_csv('data/HRDataset_v14.csv')

res=run_all(df,'Absences',causal=False)


plt.scatter(res['predicted_train_values'],res['ground_truth_train'])
print(conf)

print(str(res['model'].model))