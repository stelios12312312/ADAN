
import numpy as np
import skfuzzy as fuzz


quant=['highest','higher','high','average','low','lower','lowest']
qual=['excellent','good','decent','average','mediocre','poor','dismal']
def negator(term,names,negate):
    dummy=[]
    for nam in names:
        if not (nam in negate):
            dummy.append(term[nam])    
    return tuple(dummy)


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
        for key,val in self.results.items():
            dummy+=self.realize(key,val)+" "
            
        return dummy
    


class sentenceRealizerExpert(sentenceRealizer):
    def __init__(self):
        super(sentenceRealizerExpert, self).__init__()
        self.mapper['fit_quality_regression']="The quality of the fit is <WORD>."
        self.mapper['bias']="The model has <WORD> bias."
        self.mapper['kappa']="The kappa statistic is <WORD>"
        
    
    def interpretRegression(self,correlation,concordance):
        metric_range=np.arange(-1000, 1000, 1)*0.001
        concor = fuzz.Antecedent(metric_range, 'concordance')
        pearson = fuzz.Antecedent(metric_range, 'pearson')
        
        pear_sub_cor = fuzz.Antecedent(metric_range, 'pear_sub_cor')
        
        bias = fuzz.Consequent(metric_range, 'bias')
        bias.defuzzify_method='centroid'
        fit_quality_regression = fuzz.Consequent(metric_range, 'fit_quality_regression') 
        
        # Auto-membership function population is possible with .automf(3, 5, or 7)
        concor.automf(7,variable_type='quant')
        pearson.automf(7,variable_type='quant')
        bias.automf(5,variable_type='quant')
        fit_quality_regression.automf(7,variable_type='quality')
        pear_sub_cor.automf(7,variable_type='quant')
        
        
        rule1_bias = fuzz.Rule(pear_sub_cor['highest'], bias['higher'])
        rule2_bias = fuzz.Rule(pear_sub_cor['higher'], bias['higher'])
        rule3_bias = fuzz.Rule(pear_sub_cor['high'], bias['high'])
        rule4_bias = fuzz.Rule(pear_sub_cor['average'], bias['high'])
        rule5_bias = fuzz.Rule(pear_sub_cor['low'], bias['average'])
        rule6_bias = fuzz.Rule(pear_sub_cor['lower'], bias['low'])
        rule7_bias = fuzz.Rule(pear_sub_cor['lowest'], bias['lower'])
        
        rule1_fitness_quality = fuzz.Rule((pearson['highest'] & concor['highest']), fit_quality_regression['excellent']%0.5)
        rule2_fitness_quality = fuzz.Rule((pearson['highest'] & ~concor['highest']), (fit_quality_regression['excellent']%0.6, fit_quality_regression['decent']%0.8, fit_quality_regression['average']%0.4))
        
        rule3_fitness_quality = fuzz.Rule((pearson['higher']), fit_quality_regression['good']%0.5)
        rule4_fitness_quality = fuzz.Rule((pearson['higher'] & ~concor['higher']), (fit_quality_regression['good']%1.0, fit_quality_regression['decent']%0.8, fit_quality_regression['average']%0.6))
                
        rule5_fitness_quality = fuzz.Rule((pearson['high']), fit_quality_regression['decent']%0.5)
        rule6_fitness_quality = fuzz.Rule((pearson['high'] & ~concor['high']), (fit_quality_regression['decent']%1.0, fit_quality_regression['average']%0.8, fit_quality_regression['mediocre']%0.6))
        
        rule7_fitness_quality = fuzz.Rule((pearson['average']), fit_quality_regression['average']%0.5)
        rule8_fitness_quality = fuzz.Rule((pearson['average'] & ~concor['average']), (fit_quality_regression['average']%1.0, fit_quality_regression['mediocre']%0.8, fit_quality_regression['poor']%0.6))
        
        rule9_fitness_quality = fuzz.Rule((pearson['low']), fit_quality_regression['mediocre']%0.5)
        rule10_fitness_quality = fuzz.Rule((pearson['low'] & ~concor['low']), fit_quality_regression['poor'])
        
        rule11_fitness_quality = fuzz.Rule((pearson['lower']), fit_quality_regression['poor']%0.5)
        
        rule12_fitness_quality = fuzz.Rule((pearson['lowest'] | concor['lowest']), fit_quality_regression['dismal'])
    
    
        interp_ctrl = fuzz.ControlSystem([rule1_bias,rule2_bias,rule3_bias,rule4_bias,rule5_bias,rule6_bias,rule7_bias,
                                       rule1_fitness_quality,rule3_fitness_quality,rule5_fitness_quality,rule7_fitness_quality,rule9_fitness_quality,rule11_fitness_quality,
                                       rule4_fitness_quality,rule2_fitness_quality,rule6_fitness_quality,rule8_fitness_quality,
                                       rule10_fitness_quality,rule12_fitness_quality])
                                       
        
        interp = fuzz.ControlSystemSimulation(interp_ctrl)
        interp.input['pear_sub_cor'] = correlation-concordance
        interp.input['concordance']=concordance
        interp.input['pearson']=correlation
        
        interp.compute()
        
        fitness_quality=resolve_fuzzy(fit_quality_regression,metric_range,interp.output['fit_quality_regression'])   
        bias=resolve_fuzzy(bias,metric_range,interp.output['bias'])   
        
        self.results={'fit_quality_regression':fitness_quality,'bias':bias}
        
        

        
class sentenceRealizerSymbolic(sentenceRealizer):
    def __init__(self):
        super(sentenceRealizerExpert, self).__init__()
        self.mapper['fit_quality_regression']="The quality of the fit is <WORD>."
        self.mapper['bias']="The model has <WORD> bias."
        self.mapper['kappa']="The kappa statistic is <WORD>"
        
    




def resolve_fuzzy(member,universe,output):
    res={}
    for name,mfx in member.terms.items():
        res[name]=fuzz.interp_membership(universe, mfx.mf, output)

    res_name=max(res.iterkeys(), key=(lambda key: res[key]))
    return res_name


realizer=sentenceRealizerExpert()

for i in (np.arange(0,1000)*0.1):
    cor=np.random.rand(1)[0]
    conc=cor-np.random.rand(1)[0]*np.random.rand(1)[0]*np.random.rand(1)[0]
    if conc<0:
        conc=cor
    
    print('pear_sub_cor: '+str(cor-conc))
    print('concordance: '+str(conc))
    print('pearson: '+str(cor))                              

    realizer.interpretRegression(cor,conc)
    
    res=realizer.realizeAll()
    
    print(res)
