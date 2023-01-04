import numpy as np
from scipy import stats


quant=['highest','higher','high','average','low','lower','lowest']
qual=['excellent','good','decent','average','mediocre','poor','dismal']


class labelMapper():    
    def __init__(self,mapping={},labels=[],lower=-1,upper=1):
        if len(mapping)>0:
            self.mapping=mapping
        else:
            self.mapping=dict()
            split=np.linspace(lower,upper,len(labels)+1)
            for i,lab in enumerate(labels):
                self.mapping[lab]=[split[i],split[i+1]]
            
            self.mapping[labels[0]]=[-np.inf,self.mapping[labels[0]][1]]
            self.mapping[labels[len(labels)-1]]=[self.mapping[labels[len(labels)-1]][0],np.inf]
    
    def read_map_function(self,x):    
        for label,value in self.mapping.items():
            if value[0]<=x<=value[1]:
                return label    
        
        return "no"


def threshold(x,lower=0,upper=1):
    if x>upper:
        x=upper  
    elif x<lower:
        x=lower
    return x

class sentenceRealizer(object):
    def __init__(self):
        self.mapper=dict()
        self.results={}
    
    def realize(self,term,value):
        dummy=self.mapper[term]
        dummy=dummy.replace('<WORD>',value)
        return dummy
        
    def realizeAll(self):
        dummy=""
        self.interpretation={}
        for key,val in self.results.items():
            res=self.realize(key,val)
            dummy+=res+"\n"
            self.interpretation[key]=res
            
        return dummy
    


class sentenceRealizerExpert(sentenceRealizer):
    def __init__(self):
        super(sentenceRealizerExpert, self).__init__()
        self.mapper['fit_quality_regression']="The quality of the fit is <WORD>."
        self.mapper['bias']="The model has <WORD> bias."
        self.mapper['kappa']="The kappa statistic is <WORD>"
        
    
    def interpretRegression(self,correlation,concordance):
        cor_conc_diff=threshold(correlation-concordance,lower=0,upper=1)  
        bias_labels=['no','low','average','above average','high','very high','extremely high']
        if cor_conc_diff<0.05:
            bias=bias_labels[0]
        else:
            bias_mapper=labelMapper(labels=bias_labels,lower=-0.1,upper=1)
            bias=bias_mapper.read_map_function(cor_conc_diff)        
        
        qual=['dismal', 'poor', 'mediocre', 'average', 'decent', 'good', 'excellent']      
        if correlation<=0 or concordance<=0:
            fitness_quality=qual[0]
        else:
            m=labelMapper(labels=qual,lower=-0.1,upper=1)
            fitness_quality=m.read_map_function(concordance)
              
        self.results={'fit_quality_regression':fitness_quality,'bias':bias}
        
        

        
class sentenceRealizerSymbolic(sentenceRealizer):
    def __init__(self):
        super(sentenceRealizerSymbolic, self).__init__()
        self.mapper['complexity']="The solutions are on average <WORD>."
        self.mapper['average_eq_size']="The size of the equations is <WORD> on average."
        self.mapper['average_atom_size']="The terms are <WORD> on average."
        self.mapper['atoms_performance']="Increasing the complexity of the terms seems to have <WORD> effect on performance."
        self.mapper['num_terms_performance']="Increasing the number of the terms seems to have <WORD> effect on performance."     
        self.mapper['performance_general']='<WORD>'
        self.mapper['dataset_score']=''
        
    def interpretSymbolic(self,res_object,task='regression'):  
        
        self.results = {}
        if task=='classification':
            #classification
            length_sols=np.array([len(x[0][0].expand().args) for x in res_object])
            performances=np.array([x[1] for x in res_object])
            length_atoms=np.array([len(x[0][0].atoms()) for x in res_object])   
        else:
            #regression
            length_sols=np.array([len(x[0].expand().args) for x in res_object])
            performances=np.array([x[1] for x in res_object])
            length_atoms=np.array([len(x[0].atoms()) for x in res_object])   

        
        complexity_mapper={}
        complexity_mapper['very simple']=[-1.0*np.Inf,8]
        complexity_mapper['simple']=[8,15]
        complexity_mapper['fairly simple']=[15,30]
        complexity_mapper['complicated']=[30,70]
        complexity_mapper['very complicated']=[70,np.Inf]
        
        complexity=np.median((length_atoms**2+length_sols**1.03)/length_sols)
        #print("The complexity is:"+str(np.median((length_atoms**2)/length_sols)))
        complexity_map=labelMapper(mapping=complexity_mapper)
        complexity_mean_read=complexity_map.read_map_function(complexity)
        
        self.results['complexity'] = complexity_mean_read
        
        slope_labels=['a strong negative','a negative','a small negative','little','a small positive','a positive','a strong positive']
        slope_mapper=labelMapper(labels=slope_labels,lower=-0.3,upper=0.3)
                
        if(len(length_atoms)>1):
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(length_atoms,performances)
            except:
                slope=0
                intercept=0
                p_value=1
            atoms_performance=slope_mapper.read_map_function(slope)
            self.results['atoms_performance'] = atoms_performance
        
        if (len(length_sols)>1):
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(length_sols,performances)
            except:
                slope=0
                intercept=0
                p_value=1
            num_terms_performance=slope_mapper.read_map_function(slope)
            self.results['num_terms_performance'] = num_terms_performance
            
            
        mean_length_sols=np.mean(length_sols)
        mean_length_atoms=np.mean(length_atoms)
        
        equation_length_labels=['very short','short','normal','long','very long']
        equation_length_mapper=labelMapper(labels=equation_length_labels,lower=1,upper=10)
        average_eq_size=equation_length_mapper.read_map_function(mean_length_sols)
        self.results['average_eq_size'] = average_eq_size
        
        atoms_length_labels=['simple','average','complicated','very complicated']
        atoms_length_mapper=labelMapper(labels=atoms_length_labels,lower=1,upper=7)
        average_atoms_size=atoms_length_mapper.read_map_function(mean_length_atoms)
        self.results['average_atom_size'] = average_atoms_size
        
        mean_perf=np.mean(performances)
        dataset_score=mean_perf*150/(np.mean([np.log(mean_length_atoms),np.log(mean_length_sols)]))
        dataset_score=np.round(dataset_score/10,decimals=1)
        
        if dataset_score>10:
            dataset_score=10
        if dataset_score<0:
            dataset_score=0
            
        if mean_perf<=0.25:
            self.results['performance_general']="""Performance is low. Maybe try using exploratory data analysis and visualizations instead?"""
            
            
        elif mean_perf>0.25 and mean_perf<=0.5:
            self.results['performance_general']="""Performance is below average. Maybe try getting more data?"""
        elif mean_perf>0.5 and mean_perf<=0.75:
            self.results['performance_general']="""Performance is good."""
        elif mean_perf>0.75:
            self.results['performance_general']="""Performance is excellent."""
        self.results['dataset_score']='Dataset score is {}/10'.format(dataset_score)

#        self.results={'complexity':complexity_mean_read,'atoms_performance':atoms_performance,
#        'num_terms_performance':num_terms_performance,'average_eq_size':average_eq_size,
#        'average_atom_size':average_atoms_length}

class shapleyRealizer(sentenceRealizer):
    def read_shapley(self,shapley_values,target,variable_index=0):
        """
        variable: This argument is used only for classification tasks, where we have more than 1
        output models
        
        """
        
        #If a classification model returns only 1 class, then the model is 'wrong'. 
        #No functions are going to run in this case.
        if len(shapley_values)==0:
            print("Shapley Realizer warning: model returns only 1 class. Failing execution!")
            self.wrong_model=True
            return None
        else:
            self.wrong_model=False
        
        
        try:
            self.columns=shapley_values.columns
        except:
            #error displayed here AttributeError: 'list' object has no attribute 'columns'
            #IN classification problems we get one set of shapley values per class
            #Note that if there are 3+ classes, but the model is not good, and only predicts 1 or 2 of them
            #then this object will contain a smaller number of shapley values sets, than the total number of classes
            #in the original datatests parameter settings ,there was this issue with the Iris dataset model
            #returning only the class 2 or 1 (but not 3)
            shapley_values=shapley_values[variable_index]
            self.columns=shapley_values.columns
    
        #calculate the core variables
        self.average=np.nanmean(target)
        shapley_values=shapley_values.copy()/self.average
        
        self.maximum=shapley_values.max().sort_values(ascending=False)
        self.minimum=shapley_values.min().sort_values(ascending=True)
        
        self.median_abs=shapley_values.median().abs().sort_values(ascending=False)
        self.mean=shapley_values.mean().sort_values(ascending=False)
        
        self.mean_positive=self.mean[self.mean>0]
        self.mean_negative=self.mean[self.mean<0]
        
        self.stand=shapley_values.std().sort_values(ascending=False)
       
        self.range=(shapley_values.max()-shapley_values.min()).sort_values(ascending=False)
        
        self.negative_influence=(shapley_values<0).sum()/shapley_values.shape[0]
        self.positive_influence=(shapley_values>0).sum()/shapley_values.shape[0]
        self.no_influence=(shapley_values==0).sum()/shapley_values.shape[0]
        
        self.contribution=(1-self.no_influence)*(self.positive_influence-self.negative_influence)
        self.clarity=np.abs(self.contribution)
        self.relative_importance=shapley_values.abs().divide(shapley_values.abs().sum(axis=1),axis=0).mean()
        
        
        
    def getImportantVariableNames(self,n=3,quant=None):
        """
        Returns the most important variables, for visualization purposes.
        You have to either choose number of variables (n) or quantile. If 
        quantile is not set to None, then the default option is the median quantile.
        """
        if self.wrong_model:
            return None
        if quant is None:    
            return self.median_abs.index[0:n].values
        else:
            threshold=self.median_abs.quantile(quant)
            return self.median_abs[self.median_abs>=threshold].index.values
    

class variableExplainer(shapleyRealizer):
    
    def __init__(self):
        super(shapleyRealizer, self).__init__()      
        self.mapper['general_importance']="The average importance of the variable is <WORD>."
        self.mapper['standard_dev']="The variable's variance is <WORD>."
        self.mapper['clear_role']="The variable's role is <WORD> (variables without a clear role can have positive and negative contributions at the same time)."
        self.mapper['contribution']="The variable has a <WORD> contribution overall."
        #self.mapper['no_contribution']="There are cases where the 
        
    def explainSingleVariable(self,variable):
        if self.wrong_model:
            return None
        
        importance_mapper={}
        importance_mapper['none']=[0,0.05]
        importance_mapper['very low']=[0.05,0.1]
        importance_mapper['low']=[0.1,0.15]
        importance_mapper['fair']=[0.15,0.3]
        importance_mapper['high']=[0.3,0.4]
        importance_mapper['very high']=[0.4,0.6]
        importance_mapper['extremely high']=[0.6,np.inf]
        
        importance_map=labelMapper(mapping=importance_mapper)
        #this used to be self.median_abs but it was replaced by relative importance
        importance=importance_map.read_map_function(self.relative_importance[variable])
        
        #standard deviaton
        stand_mapper={}
        stand_mapper['very low']=[0,0.1]
        stand_mapper['low']=[0.1,0.2]
        stand_mapper['average']=[0.2,0.3]
        stand_mapper['high']=[0.3,0.5]
        stand_mapper['very high']=[0.5,0.7]
        stand_mapper['extremely high']=[0.7,np.inf]
        
        stand_map=labelMapper(mapping=stand_mapper)
        standard_dev=stand_map.read_map_function(self.stand[variable])
        
        #clarity of role
        clarity_mapper={}
        clarity_mapper['unclear']=[0,0.1]
        clarity_mapper['pretty clear']=[0.1,0.2]
        clarity_mapper['clear']=[0.2,0.3]
        clarity_mapper['very clear']=[0.3,np.inf]
        
        clarity_map=labelMapper(mapping=clarity_mapper)
        clarity=clarity_map.read_map_function(self.clarity[variable])
        
        #clarity of role
        contribution_mapper={}
        contribution_mapper['negative']=[-np.inf,-0.1]
        contribution_mapper['neutral']=[-0.1,0.1]
        contribution_mapper['positive']=[0.1,np.inf]
        
        contribution_map=labelMapper(mapping=contribution_mapper)
        contribution=contribution_map.read_map_function(self.contribution[variable])
        
        self.results={'general_importance':importance,'standard_dev':standard_dev,
                      'clear_role':clarity,'contribution':contribution}
        



class allVariablesExplainer(shapleyRealizer):
      def __init__(self):
        super(shapleyRealizer, self).__init__()      
        self.mapper['most_important']="The most important variable is <WORD>."
        self.mapper['most_positive']="The variable with the most positive contribution is <WORD>."
        self.mapper['most_negative']="The variable with the most negative contribution is <WORD>."
        #self.mapper['no_contribution']="There are cases where the 
      
      def explainVariablesSummary(self):
        most_important_variable = self.median_abs.index[0]
        try:
            most_positive_average = self.mean_positive.index[0]
        except:
            most_positive_average = 'none'
        
        try:
            most_negative_average = self.mean_negative.index[0]
        except:
            most_negative_average = 'none'
        
        self.results={'most_important':most_important_variable,'most_positive':most_positive_average,
                      'most_negative':most_negative_average}
        
        
def interpret_generalisation(results_em,score_from_ml):
    """
    compares the performance of the ADAN engine against an ML model and tries to figure out whether
    the performance is good or not. If the ML model does well, when theengine does not
    then this means that the dataset is probably too complicated.
    
    There are also cases where both models don't perform well on the test set.
    this can often happen due to a lownumber of datapoints.
    
    In this case we can simply use the train results, but we also 
    need to add some significance test to make them more meaningful.
        
    
    returns:
        ratio: The performance of ADAN vs ML
        comparison: How deos ADAN fair against ML
        can_trust_test: Natural language description of whether the models might generalize
        use_test: Use test performance or train performance?

    """
    
    score=np.mean(score_from_ml)
    perf_test=np.mean(results_em['performance_test'])
    
    ratio=perf_test/score
    
    if ratio>0.9:
        comparison='excellent'
    elif ratio>0.7:
        comparison='very good'
    elif ratio>0.5:
        comparison='fair'
    elif ratio>0.3:
        comparison='poor'
    else:
        comparison='bad'
    
    can_trust_test=None
    use_test=False
    if score<0 and perf_test<0:
        can_trust_test="No, models cannot generalise"
    elif score>0 and perf_test<0:
        can_trust_test="Problem too complex, ADAN can't generalize"
    elif score>0 and perf_test>0:
        can_trust_test="Can generalize"
        use_test=True
        
    return ratio,comparison,can_trust_test,use_test
    
    
    
    
    