#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:39:31 2018

@author: stelios
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 11:56:29 2018

@author: stelios
"""

import os,sys,inspect
from os.path import isfile,join

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir2= os.path.dirname(parentdir)
sys.path.insert(0,parentdir) 
sys.path.insert(0,parentdir2) 

import adan
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
from adan.aiem.symbolic_modelling import *
from adan.protocols import symbolic_regression_protocol
from matplotlib import pyplot as plt


print('Community crimes')


#df.drop(['communityname','state','assaultPerPop','larcPerPop'],inplace=True,axis=1)
import requests,json


path="/Users/stelios/Dropbox/ADAN/adan/datatests/community crimes/crimedata.csv"
ngen=50

df=readData(path)
df.drop(['communityname','state'],inplace=True,axis=1)

data={}
data['target_name']='ViolentCrimesPerPop'
data['dataset']=df.to_csv()
data['task']='regression'
data['quant']=0.95
data['ngen']=10
data['max_tree']=5
data['features']=5
data['population']=200
data=json.dumps(data)
r = requests.post('http://127.0.0.1:5000/analyze', data = data)
