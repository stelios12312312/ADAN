#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:31:10 2018

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
import numpy as np

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

import flask, json

from functools import wraps
from flask import Flask, flash, request, session, redirect, url_for,render_template,json as fjson,send_from_directory
from werkzeug import secure_filename
#from random_words import RandomWords

from io import StringIO
import base64
import pandas as pd



################
#### config ####
################

app = Flask(__name__)
#app.config.from_object('_config')

                

@app.route('/analyze',methods=['POST'])    
def analyze():   
    data = json.loads(request.data)
    #df = StringIO(unicode(dataset.data))
    df = readData(StringIO(data['dataset']))
    
    target_name=data['target_name']
    
    df2,target,scaler,centerer,categorical_vars,filler=prepareTrainData(df,target_name,center=False)
    #feature selection
    df2=chooseBest(df2,target,limit_n=1000,method="num",quant=data['quant'])[0]
    g=findFeaturesGP(df=df2,target=target,ngen=data['ngen'],max_tree=data['max_tree'],
                     population=data['population'],
                     features=data['features'],n_processes=1,evaluator=evalPearsonCorNumba, allowed_time=None)
    
    k = findSymbolicExpression(df2,target,g,scaler,task=data['task'],
                     features_type='best_features')
    
    perfs=[]
    for c in k:
        perfs.append(c[1])
    perfs=np.array(perfs)
    
    final_models=[]
    for m in k:
        final_models.append(Model(m[0],m[1],g,scaler,centerer,categorical_vars,filler))
    
    final_dict={}
    solution=0
    for m, in zip(final_models,perfs):
        equation = m.model
        performance = m.performance
        solution+=1
        dummy1="Explained {0}% of variance".format(str(performance*100))
        dummy2=str(equation)
        #final_dict[solution]={'performance':dummy1,'equation':dummy2,'predictions':m.evaluate(df)}
        final_dict[solution]={'performance':dummy1,'equation':dummy2}

    return flask.jsonify(final_dict)
    

    
if __name__=="__main__":
    from models import *
    app.run()