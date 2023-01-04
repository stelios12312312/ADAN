#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 17:56:07 2020

@author: stelios
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 11:11:17 2020

@author: stelios
"""


import os, sys
lib_path = os.path.abspath(os.path.join('..'))+"/adan"
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('../..'))
sys.path.append(lib_path)


from adan.aiem.genetics import *
from adan.aidc.utilities import *
from adan.aidc.feature_selection import *
#from adan.aipm.optimizers.xgboost_wrapper import *
#from adan.aipm.optimizers.optimizers import *
#from adan.aipm.optimizers.xgbOptimizer import *

from adan.metrics.metrics_utilities import *
from adan.aipm.optimizers.hyperparam_optimization import *

from adan.aiem.symbolic_modelling import *
from adan.aist.mappers import *
from adan.aidc.utilities import *
#from adan.aipm.optimizers import *
from adan.aidc.utilities import *
from adan.aiem.genetics.genetic_programming import *
#from adan.modellingDeprecated.estimation_utilities import *
#from adan.aipm.optimizers.xgbOptimizer import *
from matplotlib import pyplot as plt
from adan.protocols import *
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix,classification_report

import cdt
import sklearn
cdt.SETTINGS.rpath='/usr/local/bin/Rscript'

lr=sklearn.linear_model.LogisticRegression()
import pandas as pd

df=pd.read_csv('credit_train.csv')
target='SeriousDlqin2yrs'
df.dropna(inplace=True)
preds=sklearn.model_selection.cross_val_predict(lr,df.drop(target,axis=1),df[target],cv=5)

conf=classification_report(preds,df[target])
print(conf)
