import os, sys
lib_path = os.path.abspath(os.path.join('..'))+"/adan"
sys.path.append(lib_path)
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)

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


print('Kaggle bicycle')

target_name='registered'
path="/kaggle_bicycle/train.csv"
path_test = "/kaggle_bicycle/test.csv"
task='regression'
ngen=50

df=readData(os.path.dirname(os.path.abspath(__file__))+path)
df.season=df.season.astype('category')
df.weather=df.weather.astype('category')
df.drop(['casual','count'],axis=1,inplace=True)
df=df.sample(5000)
#this is where the cleaning takes place
#this function
df2,target,scaler,centerer,categorical_vars,filler=prepareTrainData(df,target_name,center=False)
#feature selection
df2=chooseBest(df2,target,limit_n=1000,method="num",quant=0.75)[0]
g=findFeaturesGP(df=df2,target=target,ngen=ngen,max_tree=3,population=200,
                 features=10,n_processes=1,evaluator=evalPearsonCorNumba, allowed_time=None)

k = findSymbolicExpression(df2,target,g,scaler,task=task)

#model returns the best performing model, and the models on the pareto frontier
#do model.model to get the equation. Do this for the final_models
model,final_models = tidyup_find_best(k,g,scaler,centerer,categorical_vars,filler)
#model_exec = model._convert_model_to_executable()

df=df[df[target_name].notnull()]
df3=prepareTestData(df,scaler,centerer,categorical_vars_match=categorical_vars,
                    filler=filler,model=model.model)

res_train = model.evaluate(df)


error = np.mean(np.abs(res_train-df[target_name]))
print(error)
log_error = np.sqrt(np.mean((np.log(res_train+1)-np.log(df[target_name]+1))**2))
print('competition error:'+str(log_error))
print(log_error)
