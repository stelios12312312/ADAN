import os, sys
lib_path = os.path.abspath(os.path.join('..'))+"/adan"
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
lib_path='/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/adan'
sys.path.append(lib_path)
lib_path='/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/adan/'
sys.path.append(lib_path)
lib_path='/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan/'
sys.path.append(lib_path)
#lib_path='/Users/stelios/Dropbox/ADAN/adan_scikitlearn0p20/adan2/adan'
#sys.path.append(lib_path)

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
from adan.protocols import symbolic_regression_protocol,tidyup_find_best
from matplotlib import pyplot as plt
from sklearn.metrics import cohen_kappa_score, confusion_matrix

print('Barcelona accidents file')

target_name='target'
path="/accidents_2017.csv"
task='regression'
ngen=15

df=readData(os.path.dirname(os.path.abspath(__file__))+path,encoding='utf-8-sig')
#df=readData(os.path.dirname(os.path.abspath(__file__))+path)

#y=(df['Mild injuries']+2*df['Serious injuries']+df['Vehicles involved'])+df['Victims']
#y=np.log(y+1)
y=(df['Victims']+df['Serious injuries']+df['Mild injuries'])
y=np.log(y+1)
#y[y>=1]=1
#y[y<1]=0

#y=df['Mild injuries']+df['Serious injuries']
df.drop(['Id','Vehicles involved','Mild injuries','Street','Neighborhood Name',
         'Serious injuries','Victims','Latitude','Longitude'],axis=1,inplace=True)
df['target']=y
#this is where the cleaning takes place
#this function
df2a,target,scaler,centerer,categorical_vars,filler=prepareTrainData(df,target_name,center=False)
#feature selection
print('choosing best features')
df2=chooseBest(df2a,target,limit_n=1000,method="num",quant=0.95)[0]
g=findFeaturesGP(df=df2,target=target,ngen=ngen,max_tree=4,population=250,
                 features=50,n_processes=1,#evaluator=evalPearsonCorNumba, 
                 allowed_time=None)

k = findSymbolicExpression(df2,target,g,scaler)

#model returns the best performing model, and the models on the pareto frontier
#do model.model to get the equation. Do this for the final_models
model,final_models = tidyup_find_best(k,g,scaler,centerer,categorical_vars,filler,task=task)
#model_exec = model._convert_model_to_executable()


res_train = model.evaluate(df)
error = np.mean(np.abs(res_train-target))
print(error)

plt.scatter(res_train,target)
realizer=sentenceRealizerSymbolic()
realizer.interpretSymbolic(k,task='regression')
res=realizer.realizeAll()
print(res)
#classification
#res_train=np.round(res_train)
#cohen_kappa_score(y,res_train)
#print(confusion_matrix(y,res_train))