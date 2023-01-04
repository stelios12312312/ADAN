# -*- coding: utf-8 -*-
#from minepy import MINE
import os, sys
lib_path = os.path.abspath(os.path.join('..','..'))
sys.path.append(lib_path)
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import *
from sklearn.feature_selection import f_classif
from adan.aidc.feature_selection import f_classifNumba
from numba import jit
from adan.metrics.regression import corrNumba
import re


def evalSymbRegCV(individual, targets,toolbox,cv=3):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    if(np.logical_not(all(np.isfinite(func)))):
        return 0.0,

    model=linear_model.LinearRegression()
    scores=sklearn.model_selection.cross_val_score(estimator=model, X=np.expand_dims(func,1), y=targets, cv=cv,scoring="r2")
    
    return np.mean(scores),
 
        
def evalPearsonCor(individual, targets,toolbox):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    if(np.logical_not(all(np.isfinite(func)))):
        return 0.0,

    score=np.corrcoef(func,targets )[0][1]
    if np.isnan(score):
        score=0.0
    final=score

    return final,


def evalPearsonCorNumba(individual, targets,toolbox,sampling=0.9):
    # Transform the tree expression in a callable function
    #sampling takes a subset of the targets each time when calculating the metric
    #this should help with overfitting
    try:
        func = toolbox.compile(expr=individual)
        if(np.logical_not(all(np.isfinite(func)))):
            return -2.0,
        
        if any(abs(func)>3.4028235e+38):
            return -2.0,
        
        indices=np.random.choice(len(targets),int(sampling*len(targets)))
        score=corrNumba(func[indices],targets[indices])
    
        if np.isnan(score):
            return -2.0, 
    
        return abs(score),   
    except:
        0,



def evalANOVA(individual,targets,toolbox):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    if(np.logical_not(all(np.isfinite(func)))):
        return 0.0,   
    #this returns the p-value but we use 1-x so that greater values are better
    #we have to use reshape(-1,1) because scikit learn needs arrays in the form [[0],[1.34],..etc.]
    score=1-f_classif(func.reshape(-1,1),targets)[1][0]
    if np.isnan(score):
        score=0.0

    return score,

def evalANOVANumba(individual,targets,toolbox,sampling=0.9):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    if(np.logical_not(all(np.isfinite(func)))):
        return 0.0,
    
    #this returns the p-value but we use 1-x so that greater values are better
    #we have to use reshape(-1,1) because scikit learn needs arrays in the form [[0],[1.34],..etc.]
    indices=np.random.choice(len(targets),int(sampling*len(targets)))
    score=1-f_classifNumba(func.reshape(-1,1)[indices],targets[indices])[1][0]
    if np.isnan(score):
        score=-2.0

    return score,

def convert_model_to_executable(atoms,model_expression,task='regression'):

    final_matches = []
    for a in atoms:
        try:
            float(a)
        except:
            final_matches.append(str(a))
    #sort in order the length
    final_matches=sorted(final_matches,key=len)
    final_matches.reverse()
    
    model_string=str(str(model_expression))
    
    for f in final_matches:
        if f.find('_over_')>-1:
            model_string=model_string.replace(f,"df['{}']".format(f))
        else:
            finds=list(re.finditer(str(f),model_string))
            l=len(finds)
            for i in range(l):
                find = finds[i]
                span1 = find.span()[0]
                span2 = find.span()[1]
                #avoid cases where the variable's name is part of a larger variable, like
                #for example var1_std_over_var2
                if model_string[span1-1]!="'" and model_string[span1-5:span1]!='over_':
                    toreplace="df['{}']".format(find.group())
                    model_string=model_string[:span1]+toreplace+model_string[span2:]
                i+=1
                finds=list(re.finditer(str(f),model_string))
                
                    
    model_string = model_string.replace('Abs','abs') 
    model_string = model_string.replace('sqrt','squareroot') 
    model_string = model_string.replace('log','makelog')
    
    return model_string

