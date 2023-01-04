# -*- coding: utf-8 -*-

from sklearn import linear_model
from sklearn import *
import numpy as np
from adan.aiem.symbolic_conversion import convertIndividualsToEqs,convertToEquation
from sympy import sympify,simplify,together
import time
from adan.aiem.genetics.genetic_programming import calcNewFeatures,findEquationFeatures
from adan.aidc.utilities import prepareTestData
import re
from adan.functions import *
from numpy import sqrt, sin, cos, log
import shap
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import pandas as pd
import scipy as sp
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import joblib


def pickle_model(model,filename):
    dummy=model._genetic_result_set
    model._genetic_result_set=None
    joblib.dump(model,filename)
    model._genetic_result_set=dummy
    
    

def keep_only_features(model_string):
    
    model_string2=model_string.replace(' ','')
    exps=re.findall("([0-9]+\.[0-9]+\*)" ,model_string2)
    for expression in exps:
        model_string2=model_string2.replace(expression,'')
     
    exps=re.findall("[-+][0-9]+\.[0-9]+" ,model_string2)
    for expression in exps:
        model_string2=model_string2.replace(expression,'')
        
    return model_string2.split('+')

class Model:
    model=None
    performance=None
    genetic_result_set=None
    sklearn_model=None
    
    def __init__(self,model,performance,genetic_result_set,scaler,centerer,
                 categorical_vars,filler,task,target,sklearn_model,
                 original_results=None,target_category_mapper=None,numerical_variables=None,
                 pca_object=None,components=None,eq_breakdown=None):
        self.model=model
        self.performance=performance
        self._genetic_result_set={'best_individuals_equations':genetic_result_set['best_individuals_equations'],
                                  'best_perindividuals_object':genetic_result_set['best_individuals_object']}
        
        self._scaler=scaler   
        self._centerer=centerer                       
        self._filler=filler
        self._categorical_vars=categorical_vars
        self._task=task
        self._sklearn_model=sklearn_model
        self._original_results=original_results
        self._target_category_mapper=target_category_mapper
        self._numerical_variables=numerical_variables
        self._pca_object=pca_object
        #weightings of the components in case PCA was executed
        self._components=components
        self._eq_breakdown=eq_breakdown
        
        #calculate complexity of the model
        length,atoms=self._get_length_atoms()
        complexity=default_complexity_metric(length,atoms)
        
        self._complexity=complexity
        
        if task=='regression':
            #if NaN is detected during the calculation, then default to mean
            self._mean_target=np.mean(target)
            #if -Inf detected during the calculation, then default to minimum
            self._min_target=np.min(target)
            #if Inf detected during the calculation, then default to maximum
            self._max_target=np.max(target)
        else:
            self._mode_target=sp.stats.mode(target)[0][0]
            
            

        
    
#    def _te_old(self,df):
#        """
#        deprecated. used sympy which was super slow
#        """
#        res=[]                    
#        df2=prepareTestData(df,scaler=self._scaler,model=str(self.model),categorical_vars_match=self._categorical_vars,filler=self._filler)
#       
#        for i in range(0,len(df2)):
#            result=self.model.evalf(subs=df2.ix[i,:].to_dict())
#            try:
#                float(result)
#            except:
#                prepareTestData(df,scaler=self._scaler,centerer=self._centerer,model=str(self.model),
#                                categorical_vars_match=self._categorical_vars,filler=self._filler)
#            res.append(result)
#        return res
    
    def _get_length_atoms(self):
        sol=self.model
        try:
            length=len(sol.args)
        except:
            length=len(sol[0].args)
        try:
            atoms=len(sol.atoms())
        except:
            atoms=len(sol[0].atoms())
        
        return length, atoms
    
    def return_variables(self):
        if self._task=='regression':
            atoms=self.model.atoms()
        else:
            atoms=[]
            for mod in self.model:
                for a in mod.atoms():
                    atoms.append(a)
        final_list=[]
        for atom in atoms:
            try:
                dummy=float(atom)
            except:
                final_list.append(str(atom))
        #return only unique atoms, otherwise it might double count some atoms
        return np.unique(final_list)
    
    # def evaluate2(self,df_or_datapoint,convert_class_output=False,final_features=None,
    #               fix_margins_high=False,fix_margins_low=False):
    #     """
    #     This is a copy of evaluate() for debugging purposes. This function
    #     takes the equation breakdown and uses this for evaluation.
    #     The equation breakdown is a list of the equatins:
    #         ['2*X','3*X2'], etc.
    #     """
    #     if type(df_or_datapoint)==type({}):
    #         df_original=pd.DataFrame([df_or_datapoint])
    #     else:
    #         df_original=df_or_datapoint
        
    #     res=[]    

    #     if self._task=='regression':
    #         atoms = self.model.atoms()
    #         new_eqs=[]
    #         for eq in self._eq_breakdown:
    #             model_string = self._convert_model_to_executable(atoms=atoms,model_expression=eq,task='regression')      
    #             new_eqs.append(model_string)
    #         #df is created as a variable, before calling eval
    #         df=prepareTestData(df_original,scaler=self._scaler,centerer=self._centerer,
    #                            model=str(self.model),
    #                    categorical_vars_match=self._categorical_vars,filler=self._filler,
    #                    numerical_variables=self._numerical_variables,
    #                    pca_object=self._pca_object)   
            
    #         #in some cases in the test set, a category might be missing
    #         #we just fill it in with 0
    #         for variable in self.return_variables():
    #             if variable not in df.columns:
    #                 df[variable]=0
    #                 print('\n VARIABLE '+variable+' WAS MISSING FROM THE TEST SET. IT IS REPLACED WITH 0\n')
    #         # for col in df.columns:
    #         #     df[col]=df[col].astype('float64')
    #         results=np.zeros(df.shape[0])
    #         for eq in new_eqs:
    #             try:
    #                 if eq.find('*'):
    #                     dummy=eval(eq[:-1])
    #                 else:
    #                     dummy=eval(eq)
    #             except:
    #                 asdasd=2322
    #             try:
    #                 results=results+dummy.values
    #             except:
    #                 #here we add the intercept
    #                 results=results+dummy
    #         # results=pd.Series(results)
    #         #fix results that are NaN, or Inf, etc.
    #         results[np.isnan(results)]=self._mean_target
    #         for i in range(len(results)):
    #             if results[i]==-np.inf:
    #                 results[i]=self._min_target
    #             elif results[i]==np.inf:
    #                 results[i]=self._max_target
    #             elif results[i]<self._min_target and fix_margins_low:
    #                 results[i]=self._min_target
    #             elif results[i]>self._max_target and fix_margins_high:
    #                 results[i]=self._max_target
    #     return results
    


        
                        
    
    def evaluate(self,df_or_datapoint,convert_class_output=False,
                  fix_margins_low=False,fix_margins_high=False,ignore_pca=False):
        """
        df_or_datapoint: either a dataframe, formatted in exactly the same way
        as the original dataframe, or a dictionary (which will then be converted 
        into a dataframe internally)
        
        convert_class_output: If True, then the returned result will be a pandas Series object (if regression) with predictions for each row
        or a Pandas dataframe oobject (if classification) with probabilities for each category
        
        If false, then the results will be numpy arrays. The former should be used for displaying
        output in the frontend. The latter shouldbe used for handling metrics calculation internally.
        
        fix_margins_low/high: If True, then the predicted values will never be higher or lower than the
        maximum and minimum values that existed in the original dataset.
        
        ignore_pca: If True, then then evaluate function will not run PCA. This is useful
        when using the run_equation_model_cv. Because of the way the model is set up
        the PCA is calculated before doing the split.
        
        """
        #NOTE: THIS REGULAR EXPRESSION WILL CHOOSE AL THE COEFFICIENTS FROM A MODEL STRING
        #([0-9]+\.[0-9]+\*)|([-+][0-9]+\.[0-9]+)
        #WE CAN USE THIS TO TAKE THE MODEL_STRING AS A BASE TO THEN CALCULATE FEATUERES AGAIN
        #AND THEN USE IT FOR CROSS-VALIDATION. You need to remove empty space before you run this regex.
        
        if type(df_or_datapoint)==type({}):
            df_original=pd.DataFrame([df_or_datapoint])
        else:
            df_original=df_or_datapoint
        
        res=[]   
        
        if ignore_pca:
            pca_object=None
        else:
            pca_object=self._pca_object

        if self._task=='regression':
            atoms = self.model.atoms()
            model_string = self._convert_model_to_executable(atoms=atoms,model_expression=self.model,task='regression')      
            #df is created as a variable, before calling eval
            df=prepareTestData(df_original,scaler=self._scaler,centerer=self._centerer,
                               model=str(self.model),
                       categorical_vars_match=self._categorical_vars,filler=self._filler,
                       numerical_variables=self._numerical_variables,
                       pca_object=pca_object)   
            
            #in some cases in the test set, a category might be missing
            #we just fill it in with 0
            for variable in self.return_variables():
                if variable not in df.columns:
                    df[variable]=0
                    print('\n VARIABLE'+variable+' WAS MISSING FROM THE TEST SET. IT IS REPLACED WITH 0\n')
            results=eval(model_string).values
            
            #fix results that are NaN, or Inf, etc.
            results[np.isnan(results)]=self._mean_target
            for i in range(len(results)):
                if results[i]==-np.inf:
                    results[i]=self._min_target
                elif results[i]==np.inf:
                    results[i]=self._max_target
                elif (results[i]<self._min_target) and fix_margins_low:
                    results[i]=self._min_target
                elif results[i]>self._max_target and fix_margins_high:
                    results[i]=self._max_target

                    
            results=pd.Series(results)
            
        elif self._task=='classification':
            results=[]
            for class_model in self.model:
                atoms=class_model.atoms()
                model_string = self._convert_model_to_executable(atoms=atoms,model_expression=class_model,
                                                                 task='classification') 
                #df is created as a variable, before calling eval
                df=prepareTestData(df_original,scaler=self._scaler,centerer=self._centerer,model=str(class_model),
                           categorical_vars_match=self._categorical_vars,filler=self._filler,
                           numerical_variables=self._numerical_variables,pca_object=pca_object) 
                
                 #in some cases in the test set, a category might be missing
                 #we just fill it in with 0
                for variable in self.return_variables():
                    if variable not in df.columns:
                        df[variable]=0
                        print('\n VARIABLE'+variable+' WAS MISSING FROM THE TEST SET. IT IS REPLACED WITH 0\n')
                
                res=eval(model_string)
                results.append(res)
                
            #if the length of results is 1, then use a logistic function
            if len(results)==1:
                #if NaN is produced, turn it to -Inf, so that we get a 0 after applying the exponential function.
                results=results[0].values
                results[np.isnan(results)]=-np.inf
                results=1/(1+np.exp(-results))
                results[results==-np.inf]=0
                results[results==np.inf]=1
            #otherwise use the softmax function
            else:
                try:
                    results=np.column_stack(results)
                except:
                    results=[[np.nan]*(results[0])]
                #if NaN is produced, turn it to -Inf, so that we get a 0 after applying the exponential function.
                results[np.isnan(results)]=-np.inf
                denominator=np.sum((np.exp(results)),axis=1)
                denominator=denominator.reshape(-1,1)
                results=np.exp(results)/denominator
                results[np.isnan(results)]=0
                for i in range(results.shape[0]):
                    if sum(results[i])==0:
                        results[i][self._mode_target]=1
                   
            #this happens only in the case of binary classification where the target
                        #variable was a number, e.g. [0, 1]
            if convert_class_output:
                if self._target_category_mapper=={}:
                    #binary classification, simply predict the probability of positive
                    results = pd.DataFrame(results,columns=[1])
                else:
                    results = pd.DataFrame(results,columns=self._target_category_mapper.keys())
                
        return results
    
    def _convert_model_to_executable(self,atoms,model_expression,task='regression'):

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
                    
                        
        model_string = model_string.replace('absolute','abs') 
        model_string = model_string.replace('Abs','abs') 
        model_string = model_string.replace('sqrt','squareroot') 
        model_string = model_string.replace('log','makelog')
        
        return model_string
    
    
    def _explain_component(self,component_name,quant=0.95):
        component = self._components.loc[component_name]

        negative=component[component<0]
        positive=component[component>0]
        positive=positive[positive>=positive.quantile(quant)]
        negative=negative[negative<=negative.quantile((1-quant))]
        
        pos_text=''
        neg_text=''
        if positive.shape[0]>0:
            pos_text='The following variables affect {0} in a positive manner: '.format(component_name)
            pos_text+=', '.join(positive.index.tolist())+'.'
        
        if negative.shape[0]>0:
            neg_text='The following variables affect {0} in a negative manner: '.format(component_name)
            neg_text+=', '.join(negative.index.tolist())+'.'
            
        final=pos_text+' '+neg_text
        return final
    
    def explain_components_from_variables(self,components_or_variables=None):
        if self._components is None:
            return None
        
        #if components is None, then interpret all components
        if components_or_variables is None:
            components=self._components.index
        else:
        #read the variables, and find which components are used
            components_used=[]
            for comp in components_or_variables:
                components_used+=re.findall("var_PCA_[0-9]",comp)
            components=np.unique(components_used)
        
        explanation=''
        for component in components:
            explanation+=self._explain_component(component)+'\n'
        
        return explanation
         
    
    def explain_new_data(self,df_original):
        """
        df_original: The result of a readData function
        """
        try:
            if type(df_original)==tuple or type(df_original)==list:
                df_original=df_original[0]
        except:
            pass
        values=self.explain(df_original).copy()
        #this gets triggered when there are many classes and we have a classification problem
        if type(values)==list:
            cols=values[0].columns
            values=np.mean(values,axis=0)
            values=pd.DataFrame(values,columns=cols)
        percentages=values.abs().sum()/values.abs().sum().sum()

        average_contribution=values.mean()
        
        return {'percentage_contribution':percentages,'expected_mean_contribution':average_contribution,
                'full_values':values}
        
    
    def explain(self,df_original,plot=False):
        """
        Uses a decision tree surrogate model, alongside shapley values in order
        to explain the importance of the various features.
        
        Returns: A single dataframe for regression, multiple dataframes for classification.
        
        Warning: In case ADAN produces a model that does not predict every category (e.g.
        there are 3 categories, but the model predicts always categories 1 or 3), then shapley values
        will return with less categories than the ones that exist in the original dataset.
        """

        
        variables=self.return_variables()

        results = self.evaluate(df_original)
        
        df=prepareTestData(df_original,scaler=self._scaler,centerer=self._centerer,model=str(self.model),
           categorical_vars_match=self._categorical_vars,filler=self._filler,
           numerical_variables=self._numerical_variables,pca_object=self._pca_object) 

        remaining_data=df.loc[:,variables]
             
        remaining_data.fillna(remaining_data.mean(), inplace=True)

        if self._task=='regression':
            tree=RandomForestRegressor()
        else:
            tree=RandomForestClassifier()
            if len(results.shape)>1:
                results=np.argmax(results,axis=1)
            else:
                results=results.round()
                
        if len(np.unique(results))==1:
            print('Issue with the model. Returns only one class :'+str(np.unique(results)[0]))
            return []
                
        
        tree.fit(remaining_data,results)
        explainer = shap.TreeExplainer(tree)
        shap_values = explainer.shap_values(remaining_data)
        
        # plot the SHAP values for the Setosa output of the first instance
        if plot:
            shap.force_plot(explainer.expected_value, shap_values, remaining_data)
            
        if self._task=='regression':
            return pd.DataFrame(shap_values,columns=variables)
        else:
            dfs=[]
            for i in range(len(shap_values)):
                dfs.append(pd.DataFrame(shap_values[i],columns=variables))
            return dfs
    

def dominates(row, candidateRow):
    return sum([row[x] > candidateRow[x] for x in range(len(row))]) == len(row)   

def simple_cull(inputPoints):
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints

def trailing_zeros(num):
    num = str(num)
    numzeroes=0
    
    for i in range(0,len(num)):
        if num[i]=='0':
            numzeroes=numzeroes+1
        if num[i]!='.' and num[i]!='0':
            break
    
    return numzeroes
    
def volume_complexity_metric(length,atoms):
    return -1*length*np.log2(atoms)
    
def default_complexity_metric(length,atoms,a=3,b=2,c=1):
    try:
        return -1*(a*length+b*atoms+c*(atoms*1.0/length))
    except:
        #this is used in case of error, e.g. division by zero
        -100000
    
    
def _get_length_and_atoms(sol):
    
    #the exception applies to multiclass problems
    
    try:
        length=len(sol[0].args)
    except:
        length=len(sol[0][0].args)
    try:
        atoms=len(sol[0].atoms())
    except:
        atoms=len(sol[0][0].atoms())
        
    return length, atoms
    
def findAcceptable(sols,complexity_metric=volume_complexity_metric,rounding=2,median_deviations=4):
    """
    Args:
        rounding: The number of decimals to round the score to. A default value of 3 is used. This is required
        so that the pareto optimal frontier classifies as equal solutions with similar scores. Otherwise, it can give an advantage
        to a solution with a small non-significant advantage (e.g. to the 4th decimal place).
        
        median_deviations: results are excluded if they are above the median deviation threshold for complexity.
        This prevents very complicated solutions on dominating the problem.
    returns: a list of tuples (equation, variance explained,)
    """
    solution_expansion=[]

    if(type(sols[0])==type([1,2,3])):
        for sol in sols:
            solution_expansion.append(sol[0])
    else:
        for sol in sols:
            solution_expansion.append(sol)
    
    new_sols=[]
    performance_scores=[k[1] for k in sols]
    complexity_scores=[]
    new_perf_scores=[]
    # if (max(performance_scores))>0:
    for sol in solution_expansion:
        # if sol[1]>0:
        new_sols.append(sol) 
        #print(sol)
        start=time.time()
        
        length,atoms = _get_length_and_atoms(sol)
        #print('time required for length:'+str(time.time() - start))
        start=time.time()
        #print('time required for atoms:'+str(time.time() - start))
            
        complexity_scores.append(complexity_metric(length,atoms))
        new_perf_scores.append(sol[1])
                
    median_complexity = np.median(complexity_scores)
    
    filtered_complexity_scores = []
    filtered_perf_scores = []
    threshold = median_complexity - median_deviations*np.std(complexity_scores)
    for comp,scores in zip(complexity_scores,new_perf_scores):
        if comp >= threshold:
            filtered_complexity_scores.append(comp)
            filtered_perf_scores.append(scores)
            
    if len(complexity_scores)==0:
        best_index=np.where(new_perf_scores==max(new_perf_scores))[0][0]
        filtered_perf_scores.append(new_perf_scores[best_index])
        filtered_complexity_scores.append(complexity_scores[best_index])
          
    print('running pareto')
    paretoPoints, dominatedPoints = simple_cull(list(zip(filtered_complexity_scores,np.around(filtered_perf_scores,rounding))))

    #add the solutions
    new_sols=[]
    for sol in solution_expansion:
        for point in paretoPoints:
            length,atoms = _get_length_and_atoms(sol)
            metric=complexity_metric(length,atoms)
            if metric==point[0] and np.around(sol[1],rounding)==point[1]:
                new_sols.append(sol)
    
    print('removing duplicates')
    #remove duplicates
    toremove=[]
    for i in range(0,len(new_sols)):
        for j in range(i,len(new_sols)):
            if i!=j:
                if(np.around(new_sols[i][1],rounding)==np.around(new_sols[j][1],rounding)):
                    if(type(new_sols[j][0])==type([1,2,3])):
                        length1=(len(new_sols[i][0][0].args))
                        atoms1=len(new_sols[i][0][0].atoms())
                        length2=(len(new_sols[j][0][0].args))
                        atoms2=len(new_sols[j][0][0].atoms())
                    else:
                        length1=(len(new_sols[i][0].args))
                        atoms1=len(new_sols[i][0].atoms())
                        length2=(len(new_sols[j][0].args))
                        atoms2=len(new_sols[j][0].atoms())
                    if length1==length2 and atoms1==atoms2:
                        if len(str(new_sols[i][0])) > len(str(new_sols[j][0])):
                            toremove.append(i)
                        else:
                            toremove.append(j)
    
    toremove=list(set(toremove))  
    
    for rem in sorted(toremove,reverse=True):
        del new_sols[rem]

    return new_sols


def _get_features_for_symbolic_reg(df,result_object,features_type,max_length):
    """
    Gets the output of the genetic programming feature generation process
    and then create a dataframe with all the features. Column names are the equation objects.
    """
    columns=df.columns.values
    X=result_object[features_type]
    individuals=result_object['best_individuals_equations']

    
    final=[]
    eqs=convertIndividualsToEqs(individuals,columns)
    eqs2=[]
    if max_length>0:
        for eq in eqs:
            if len(eq)<max_length:
                eqs2.append(eq)           
        eqs=eqs2
    
    x=np.column_stack(X)   
    features_to_be_used=pd.DataFrame(x)
    #this is where the simple features start (like individual variables)
    simple_features_start = len(result_object['best_features'])
    names_simple=[k.name for k in X[simple_features_start:]]
    
    #features_to_be_used is used in order to test that the results of the model are indeed correct
    #these are the features output that the equation is using
    features_to_be_used.to_csv('features_test.csv',index=False,header=individuals+names_simple)
    features_to_be_used.columns=eqs+names_simple
    
    return features_to_be_used

def _interpret_coefs(models,features_to_be_used,columns):
    """
    Get the models, and then convert them into equations. Also, add the score and the predictions.
    
    The score in this case is the default score used by a model in scikit-learn. So, for example,
    for regression it is the R squared
    """
    final=[]
    for model,score,predictions in models: 
     #there are cases where the results are all 0, or the algorithm only predicts the same output
     if sum(predictions)!=0 and len(np.unique(predictions))>1:
         coefs = model.coef_
        #This line detects whether this is a multiclass classification problem or not. If this condition is true, then the coefficients
        #are of dimension 1, so there is only one class or 1 value to regress on.
         if(len(model.coef_.shape)==1):                              
             eq=""
             eq_breakdown=[]
             #category represents a class. In logistic regression with scikit learn                
             for j in range(0,len(features_to_be_used.columns)):
                     sympyterm=convertToEquation(features_to_be_used.columns[j],columns)
                     expression=str(coefs[j])+"*("+str(sympify(sympyterm))+"))"
                     eq=eq+"+("+expression
                     eq_breakdown.append(expression)
             eq = eq+'+'+str(model.intercept_)
             eq_breakdown.append(model.intercept_)
             #model.predict(features_to_be_used)
             final.append((together(sympify(eq)),score,predictions,model,eq_breakdown))
         else:
             dummy=[]
             intercept_number=0
             for coefs in model.coef_:               
                 eq=""
                 eq_breakdown=[]
                 #category represents a class. In logistic regression with scikit learn
                 
                 for j in range(0,len(features_to_be_used.columns)):    
                     #eq=eq+"+("+str(coefs[j])+"*("+str(sympify(eqs[j]))+"))"
                     sympyterm=convertToEquation(features_to_be_used.columns[j],columns)
                     expression=str(coefs[j])+"*("+str(sympify(sympyterm))+"))"
                     # eq=eq+"+("+str(coefs[j])+"*("+str(sympify(features_to_be_used.columns[j]))+"))"
                     eq=eq+"+("+expression
                     eq_breakdown.append(expression)

                 eq = eq+'+'+str(model.intercept_[intercept_number])
                 eq_breakdown.append(model.intercept_)
                 intercept_number+=1
                 dummy.append(together(sympify(eq)))
                 final.append((dummy,score,predictions,model,eq_breakdown))
             
    return final

def findSymbolicExpression(df,target,result_object,scaler,task="regression",
                           logged_target=False,find_acceptable=True,
                           features_type='best_features_plus_cols',max_length=-1,
                           complexity_tolerance=4):
    """
    features_type: choice is between 'best features', 'all features' , and 'best
    features plus columns'. The last one, adds a column only if it exists in a 
    complex feature.
    
    complexity_tolerance: create a threshold= std of complexity scores times this value.
    If complexity penalty is lower than this threshold, then discard the solution
    
    returns:
        A list of tuples, where each tuple is (model equation,score,predictions,model object)
    
    """
    
    # features_to_be_used=_get_features_for_symbolic_reg(df=df,result_object=result_object,
    #                                                    features_type=features_type,max_length=max_length)
    
    features_to_be_used=result_object['best_all_feats_df']
    if task=='regression':
        models=findSymbolicExpressionL1_regression_helper(features_to_be_used.values,target)
    elif task=='classification':
        models=findSymbolicExpressionL1_classification_helper(features_to_be_used.values,target)  
    
    print('interpreting coefficients')
    final=_interpret_coefs(models,features_to_be_used,df.columns)
    
    if len(final)==0:
        print('ISSUE WITH SYMBOLIC REGRESSION. ALL MODELS PREDICT CONSTANT RESULTS.')
        raise Exception('Execution problem.')
    
    print('finding acceptable solutions')
    if find_acceptable:
        final_2=findAcceptable(final)
    else:
        final_2=final
    
    return final_2    


def findSymbolicExpression_genetic(df,target,result_object,scaler,task="regression",
                           logged_target=False,find_acceptable=True,
                           features_type='best_features_plus_cols',max_length=-1,
                           complexity_tolerance=4,ngen=5,population=10,
                           crossover_prob=0.5,mut_prob=0.1,individual_mut=0.1):
    """
    features_type: choice is between 'best features', 'all features' , and 'best
    features plus columns'. The last one, adds a column only if it exists in a 
    complex feature.
    
    complexity_tolerance: create a threshold= std of complexity scores times this value.
    If complexity penalty is lower than this threshold, then discard the solution
    
    returns:
        A list of tuples, where each tuple is (model equation,score,predictions,model object)
    
    """
    
    #Get the features_to_be_used dataframe, that contains variables with names
    # features_to_be_used=_get_features_for_symbolic_reg(df=df,result_object=result_object,
    #                                                    features_type=features_type,max_length=max_length)
    
    features_to_be_used=result_object['best_all_feats_df']
    final_choice=findEquationFeatures(features_to_be_used,task=task,target=target,
                                      ngen=ngen,population=population,
                           crossover_prob=crossover_prob,mut_prob=mut_prob,
                           individual_mut=individual_mut)
    
    #If all variables are false, then return all of them, as this means that
    #we couldn't find a good solution
    if sum(final_choice)==0:
        final_choice=final_choice+True
        print('The 2nd level genetic feature search could not discover a \
              meaningful set of features. Returning all of them. Consider re-reunning with \
                  different parameters.')
    
    final_features=features_to_be_used.loc[:,final_choice]
    
    if task=='regression':
        models=findSymbolicExpressionL1_regression_helper(final_features.values,target)
    elif task=='classification':
        models=findSymbolicExpressionL1_classification_helper(final_features.values,target)  
    

    print('interpreting coefficients')
    final=_interpret_coefs(models,final_features,df.columns)
    
    # if len(final)==0:
    #     print('ISSUE WITH SYMBOLIC REGRESSION. ALL MODELS PREDICT CONSTANT RESULTS.')
    #     raise Exception('Execution problem.')
    
    print('finding acceptable solutions')
    if find_acceptable:
        final_2=findAcceptable(final)
    else:
        final_2=final
    
    return final_2,final_features
    
    
def round_coefs(coefs):
    
    if(np.max(np.abs(coefs))>=1):
        return np.round(coefs)
    else:
        dummy=np.max(np.abs(coefs))
        zeroes=trailing_zeros(dummy)
        return np.around(coefs,zeroes+1)

def findSymbolicExpressionL1_general_helper(models,x,target):
    """
    Returns a tuple of (model, score, coefficients)
    """
    results=[]
    for model in models:
        try:
            predictions=cross_val_predict(model,x,target,cv=3)
            # model.fit(x,target)
            # score=model.score(x,target)
            # predictions=model.predict(x)
            score=r2_score(target,predictions)
            model.fit(x,target)
            res=(model,score,predictions)
            results.append(res)
            #create two versions, the second one having rounded coefficients which look easier to interpret
            #LAME IDEA, requires different optimization objective, deprecated!
            #model.coef_=round_coefs(model.coef_)
            #res=(model,model.score(x,target),model.coef_,predictions)
            results.append(res)
        except:
            pass

    return results
    
def _generator_elastic_net(values=[0.01]+np.arange(0.1,1.1,0.1).tolist()):
    res=[]
    for val in values:
        result = linear_model.ElasticNetCV(l1_ratio=val)
        #a result of None is returned if there is some infinite or nan value 
        #in the dataset. This will happen if symbolic regression screws up.
        if result is not None:
            res.append(result)
    return res
    
def _generator_sgd(values=[0.01]+np.arange(0.1,1.1,0.1).tolist()):
    res=[]
    for val in values:
        res.append(linear_model.SGDClassifier(l1_ratio=val,loss='log'))
    return res
    
def findSymbolicExpressionL1_regression_helper(x,target):
    #models=[]
    models = _generator_elastic_net()
    models.append(linear_model.LinearRegression())
    models.append(linear_model.OrthogonalMatchingPursuit())
    return findSymbolicExpressionL1_general_helper(models,x,target)
    

def findSymbolicExpressionL1_classification_helper(x,target):
    models = _generator_sgd()
    #models=[]
    models.append(linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs'))
    models.append(linear_model.RidgeClassifier())
    return findSymbolicExpressionL1_general_helper(models,x,target)    
    

    

    