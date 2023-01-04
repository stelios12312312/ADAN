# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:31:09 2016

@author: stelios
"""
import numpy as np
import scipy
from scipy import stats


def checkWrongNumberFormat(x):
    if((any(np.isnan(x))) or (any(np.isinf(x)))):
        return True
    else:
        return False

def add(x,y):
    return x+y
    
def sub(x,y):
    return x-y
 
def mul(x,y):
    return (x*y)   
    
def div(left, right):
    res=np.array(left * 1.0 / right)
    k=np.isfinite(res)
    #if the result is not finite return nan
    res[np.logical_not(k)]=np.nan
    return res

def cube(x):
    return (x)**3

def recipr(x):
    return 1.0/x
#    if(not any(x==0)):
#        return 1/x
#    else:
#        #return np.zeros(len(x))+np.max(x)
#        return np.nan

    
def squareroot(x):
    # res=np.sqrt(x)
    res=(x)**(1/2)
    if type(res)==type((1.129761046562897+0.4679623477271737j)):        
        res=np.nan
    # try:
    #     #this will fail if x is a constant (E.g. 2.0)
    #     res[np.logical_not(k)]=0    
    # except:
    #     pass
    # return np.sqrt(res)      
    return res

    
def makelog(x):
    return np.log(x+1)
#    if(all(x>=0)):
#        return np.log(x+1)
#    else:
#        return np.zeros(len(x))+np.min(x)

def kurtosis(x):
    try:
        scipy.stats.kurtosis(x)
    except:
        #we have to do this, because parallel pandas breaks down otherwise if
        #we just return 0
        return np.sum(x)*0
    
def skew(x):
    try:
        scipy.stats.skew(x)
    except:
        return np.sum(x)*0


        


singlefunctions=[np.square,cube,recipr,makelog,squareroot,np.cos,np.sin,np.abs]
twopartfunctions=[add,sub,mul,div]
#aggregationfunctions=[np.min,np.max,scipy.mean,scipy.stats.hmean,np.std,sum,scipy.stats.kurtosis,scipy.stats.skew]
aggregationfunctions=[np.min,np.max,scipy.mean,np.std,np.sum,kurtosis,skew]