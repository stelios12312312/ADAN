import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)


#import cdt
import adan
from adan.aiem.genetics import *
from adan.aidc.utilities import *
from adan.aidc.feature_selection import *


from adan.metrics.metrics_utilities import *
from adan.aipm.optimizers.hyperparam_optimization import *

from adan.aiem.symbolic_modelling import *
from adan.aiem.genetics.genetic_programming import *
from adan.protocols import *
from matplotlib import pyplot as plt
from adan.aist.mappers import *
from adan.aipd.aipd_main import *
from adan.aiem.symbolic_modelling import *
from adan.aiem.optimise import reverse_optimise
import os
import sklearn

import random
#random.seed(27)
pd.set_option('display.max_columns', 500)

def describe_dataset(dataset):
    stats={}
    stats['num_rows']=dataset.shape[0]
    stats['num_columns']=dataset.shape[1]
    stats['num_categorical']=dataset.dtypes[dataset.dtypes=='object'].shape[0]

    stats['std_of_means']=dataset.mean().std()
    stats['std_of_std']=dataset.std().std()
    stats['std_of_skew']=dataset.skew().std()
    
    stats['std_correlation']=dataset.corr().mean().std()
    stats['average_correlation']=dataset.corr().mean().mean()
    
    return stats
    

    
    
def load_test(name,target_name,ngen,quant,task,path,sample_n=1000,
              choice_method='quant',fix_skew=True,
              allowed_time=6,n_pop=300):
    print('\n\n')
    print('************RUNNING TEST************')
    print(name)
    print('************************************\n\n')

    if "DATATESTS_PATH" in os.environ:
        path = os.environ["DATATESTS_PATH"] + path
    else:
        path = path[1:]
    
    try:
        k = run_all(path, target_name=target_name, task=task, quant_or_num=quant,ngen=ngen,
                     sample_n=sample_n,choice_method=choice_method,causal=False,
                     fix_skew=fix_skew,allowed_time=allowed_time,n_pop=n_pop,ngen_for_second_round=5,
                     pca_criterion_thres=1)
        return(k)
    except:
        print('Test failed : '+name)
    
    
    

ngen=3
quant=0.9
 
##Works 

##regression tests
for x in range(1):
    ngen = random.randint(10,30)
    n_pop = random.randint(10,30)
    sample_n = -1


    auto=load_test('AUTO MPG','V1',ngen,quant,'regression',"/data/auto_mpg/auto_mpg.csv", n_pop=n_pop)
    house=load_test('HOUSING',13,ngen,quant,'regression',"/data/housing/housing.csv",sample_n=sample_n,n_pop=n_pop)

    servo=load_test('SERVO',4,ngen,quant,'regression',"/data/servo/servo.csv",sample_n=sample_n,n_pop=n_pop)

    fires=load_test('FOREST FIRES','area',ngen,quant,'regression',"/data/forestfires/forestfires.csv",sample_n=sample_n,n_pop=n_pop)

    bike_demand=load_test('BICYCLE DEMAND','count',ngen,quant,'regression',"/data/bicycle/train.csv",sample_n=sample_n,n_pop=n_pop)

    prop=load_test('PROPERTY INSPECTION','Hazard',ngen,quant,'regression',"/data/liberty_property_inspection/train.csv",sample_n=sample_n,n_pop=n_pop)

    lymphography=load_test('LYMPHOGRAPHY',0,ngen,quant,'classification',"/data/lymphography/lymphography.csv",sample_n=sample_n,n_pop=n_pop)

    page_blocks=load_test('PAGE BLOCKS',10,ngen,quant,'classification',"/data/page_blocks/page_blocks.csv",sample_n=sample_n, n_pop=n_pop)

    post_operative=load_test('POST OPERATIVE',8,ngen,quant,'classification',"/data//post_operative/post_operative.csv",sample_n=sample_n,n_pop=n_pop)
        
    

    airparticles=load_test('air particules','Attributable Deaths at coefft (change for 10 µg/m3 PM2.5) 12%',ngen,quant,'regression',"/airpollution/particulate-air-pollution-mortality.csv",sample_n=sample_n,n_pop=n_pop)

    

    iris=load_test('IRIS',4,ngen,quant,'classification',"/data/iris/iris.csv",sample_n=sample_n,n_pop=n_pop)

    german=load_test('GERMAN CREDIT DATA',20,ngen,quant,'classification',"/data/german_credit_data/german_credit_data.csv",sample_n=sample_n,n_pop=n_pop)
    
    balance=load_test('BALANCE',0,ngen,quant,'classification',"/data/balance/balance.csv",sample_n=sample_n,n_pop=n_pop)

    abalone=load_test('ABALONE',0,ngen,quant,'classification',"/data/abalone/abalone.csv",sample_n=sample_n,n_pop=n_pop)

    balance_scale=load_test('BALANCE SCALE',0,ngen,quant,'classification',"/data/balance_scale/balance_scale.csv",sample_n=sample_n,n_pop=n_pop)


    breast=load_test('BREAST CANCER','class',ngen,quant,'classification',"/breast_cancer/breast_cancer.csv",sample_n=sample_n,n_pop=n_pop)


    iris_1=load_test('IRIS',4,ngen,quant,'classification',"/data/iris_1/iris_1.csv",sample_n=sample_n,n_pop=n_pop)
    lenses=load_test('LENSES',4,ngen,quant,'classification',"/data/lenses/lenses.csv",sample_n=sample_n,n_pop=n_pop)
    letter=load_test('LETTER',0,ngen,quant,'classification',"/data/letter/letter.csv",sample_n=sample_n,n_pop=n_pop)
    spect=load_test('SPECT',0,ngen,quant,'classification',"/data/spect/spect.csv",sample_n=sample_n,n_pop=n_pop)

#does not work
#tae=load_test('TAE',0,ngen,quant,'classification',"/tae/tae.csv")


    wine=load_test('WINE',0,ngen,quant,'classification',"/data/wine/wine.csv",sample_n=sample_n,n_pop=n_pop)

    yeast=load_test('YEAST',9,ngen,quant,'classification',"/data/yeast/yeast.csv",sample_n=sample_n,n_pop=n_pop)

    poker_hand=load_test('POKER HAND',10,ngen,quant,'classification',"/data/poker_hand/poker_hand.csv",sample_n=sample_n,n_pop=n_pop)

    skin_segmentation=load_test('SKIN SEGMENTATION',3,ngen,quant,'classification',"/data/skin_segmentation/skin_segmentation.csv",sample_n=sample_n,n_pop=n_pop)

    seismic_bumps=load_test('SEISMIC BUMPS',18,ngen,quant,'classification',"/data/seismic_bumps/seismic_bumps.csv",sample_n=sample_n,n_pop=n_pop)

    avila=load_test('AVILA',10,ngen,quant,'classification',"/data/avila/avila.csv",sample_n=sample_n,n_pop=n_pop)

    naval_propulsion_compressor=load_test('NAVAL PROPULSION COMPRESSOR',16,ngen,quant,'regression',"/data/naval_propulsion_compressor/naval_propulsion_compressor.csv",sample_n=sample_n,n_pop=n_pop)

    naval_propulsion_turbine=load_test('NAVAL PROPULSION TURBINE',17,ngen,quant,'regression',"/data/naval_propulsion_compressor/naval_propulsion_compressor.csv",sample_n=sample_n,n_pop=n_pop)


    car=load_test('CAR',6,ngen,quant,'classification',"/data/car/car.csv",sample_n=sample_n,n_pop=n_pop)

    nursery=load_test('NURSERY',8,ngen,quant,'classification',"/data/nursery/nursery.csv",sample_n=sample_n,n_pop=n_pop)

    glass=load_test('GLASS',10,ngen,quant,'classification',"/data/glass/glass.csv",sample_n=sample_n,n_pop=n_pop)

    lung_cancer=load_test('LUNG CANCER',0,ngen,quant,'classification',"/data/lung_cancer/lung_cancer.csv",sample_n=sample_n,n_pop=n_pop)

    primary_tumor=load_test('PRIMARY TUMOR',0,ngen,quant,'classification',"/data/primary_tumor/primary_tumor.csv",sample_n=sample_n,n_pop=n_pop)

    ionosphere=load_test('IONOSPHERE',34,ngen,quant,'classification',"/data/ionosphere/ionosphere.csv",sample_n=sample_n,n_pop=n_pop)

    waveform=load_test('WAVEFORM',40,ngen,quant,'classification',"/data/waveform/waveform.csv",sample_n=sample_n,n_pop=n_pop)

    horse=load_test('HORSE',24,ngen,quant,'classification',"/data/horse/horse.csv",sample_n=sample_n,n_pop=n_pop)




# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    Attrition = load_test('Attrition', 'Attrition', ngen, quant, 'classification', '/data/attrition/attrition.csv',sample_n=sample_n, n_pop=n_pop)

    scores = load_test('scores', 'status', ngen, quant, 'classification', '/data/scores/scores.csv',sample_n=sample_n,n_pop=n_pop)

    diamonds = load_test('diamonds', 'price', ngen, quant, 'regression', '/data/diamonds/diamonds.csv',sample_n=sample_n,n_pop=n_pop)

    voice = load_test('voices', 'label', ngen, quant, 'classification', '/data/voices/voice.csv',sample_n=sample_n,n_pop=n_pop)

    cereal = load_test('cereal', 'rating', ngen, quant, 'regression', '/data/cereal/cereal.csv',sample_n=sample_n,n_pop=n_pop)
    cereal_calories = load_test('cereal_calories', 'calories', ngen, quant, 'regression', '/cereal/cereal_calories.csv',sample_n=sample_n,n_pop=n_pop)

    cars_price = load_test('cars_price', 'price', ngen, quant, 'regression', '/data/cars_price/prices.csv',sample_n=sample_n,n_pop=n_pop)
    
    macdonalds = load_test('macdonalds', 'Calories', ngen, quant, 'regression', '/data/macdonalds/menu.csv',sample_n=sample_n,n_pop=n_pop)

    financial = load_test('financial', 'Financial Distress', ngen, quant, 'regression', '/data/financial/financial.csv',sample_n=sample_n,n_pop=n_pop)

    mobileprice = load_test('mobileprice', 'price_range', ngen, quant, 'classificatvary ion', '/data/mobileprice/train.csv',sample_n=sample_n,n_pop=n_pop)

    biopsy = load_test('biopsy', 'Biopsy', ngen, quant, 'classification', '/Biopsy/biopsy.csv',sample_n=sample_n,n_pop=n_pop)
    complications = load_test('complications', 'complication', ngen, quant, 'classification', '/data/complications/complications.csv',sample_n=sample_n,n_pop=n_pop)


    ###This also has the random problem - with str, but also has issues whn comparing sklearn results and adan results
    Mushrooms = load_test('Mushrooms', 'category', ngen, quant, 'classification', '/data/Mushrooms/mushrooms.csv')

    # Takes a very long time 
    Digits=load_test('Digits','label',ngen,quant,'classification',"/data/digit-recognizer/train.csv")
    
    #There might be issues with the dataset structure in this one, so will have to ask about this 
    # Titanic=load_test('Titanic','Survived',ngen,quant,'classification','/titanic/train.csv')
    NLP=load_test('NLP','target',ngen,quant,'classification','/data/nlp-getting-started/train.csv')



# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    ##does not work - random function
    credit=load_test('CREDIT SCREENING',15,ngen,quant,'classification',"/data/credit_screening/crx.csv")
    haberman=load_test('HABERMAN',3,ngen,quant,'classification',"/data/haberman/haberman.csv")
    hayes_roth=load_test('HAYES ROTH',5,ngen,quant,'classification',"/data/hayes_roth/hayes_roth.csv")
    mammographic=load_test('MAMMOGRAPHIC',5,ngen,quant,'classification',"/data/mammographic/mammographic.csv")
    blood_transfusion=load_test('BLOOD TRANSFUSION',4,ngen,quant,'classification',"/data/blood_transfusion/blood_transfusion.csv")
    banknote=load_test('BANKNOTE',4,ngen,quant,'classification',"/data/banknote/banknote.csv")
    tic_tac_toe=load_test('TIC TAC TOE',9,ngen,quant,'classification',"/data/tic_tac_toe/tic_tac_toe.csv")
    congressional=load_test('CONGRESSIONAL',0,ngen,quant,'classification',"/data/congressional/congressional.csv")



    # Issues which used to give an off performance between sklearn and ADAN
    heart=load_test('HEART DISEASE',13,ngen,quant,'classification',"/data/heart_disease/heart_disease.csv")
    contraceptive=load_test('CONTRACEPTIVE',9,ngen,quant,'classification',"/data/contraceptive/contraceptive.csv")
    forest=load_test('FORESvT',54,ngen,quant,'classification',"/data/forest/forest.csv")
    adult=load_test('ADULT',14,ngen,quant,'classification',"/data/adult/adult.csv")
    soybean_small=load_test('SOYBEAN SMALL',35,ngen,quant,'classification',"/data/soybean_small/soybean_small.csv")
    spectf=load_test('SPECTF',0,ngen,quant,'classification',"/data/spectf/spectf.csv")
    zoo=load_test('ZOO',17,ngen,quant,'classification',"/data/zoo/zoo.csv")
    hepatitis=load_test('HEPATITIS',0,ngen,quant,'classification',"/data/hepatitis/hepatitis.csv")






#####arcene has a huge number of features!
####arcene=load_test('ARCENE','Class',ngen,quant,'classification',"/arcene/arcene.csv")
###
###
##
##
############# ---------------- MY CLASSIFICATION ---------------- ############
##
##





#
##TAKES A LONG TIME
#spambase=load_test('SPAMBASE',57,ngen,quant,'classification',"/spambase/spambase.csv")
#
#acute_inflammations_d1=load_test('ACUTE INFLAMMATIONS D1',6,ngen,quant,'classification',"/acute_inflammations/acute_inflammations.csv")
#acute_inflammations_d2=load_test('ACUTE INFLAMMATIONS D2',7,ngen,quant,'classification',"/acute_inflammations/acute_inflammations.csv")

## ---  Key Error --- ###

# ---  No Results --- ###



### ------ Key Error

air_quality=load_test('AIR QUALITY',14,ngen,quant,'regression',"/data/air_quality/air_quality.csv")
beijing=load_test('BEIJING',5,ngen,quant,'regression',"/data/beijing/beijing.csv")
communities_crime=load_test('COMMUNITIES CRIME',0,ngen,quant,'regression',"/data/communities_crime/communities_crime.csv")
##useless dataset
##car_park=load_test('CAR PARK','Occupancy',ngen,quant,'regression',"/car_park/car_park.csv")
energy_efficiency_y1=load_test('ENERGY EFFICIENCY Y1',8,ngen,quant,'regression',"/data/energy_efficiency/energy_efficiency.csv")
energy_efficiency_y2=load_test('ENERGY EFFICIENCY Y2',9,ngen,quant,'regression',"/data/energy_efficiency/energy_efficiency.csv")
#
###this error occurs only with pandas 0.25.0. Sean please downgrade the version.
#### ------ "TypeError: __init__() got an unexpected keyword argument 'tupleize_cols'" ####
#
computer_hardware=load_test('COMPUTER HARDWARE',9,ngen,quant,'regression',"/data/computer_hardware/computer_hardware.csv")
solar_flare_c=load_test('SOLAR FLARE C CLASS',10,ngen,quant,'regression',"/data/solar_flare/solar_flare.csv")
solar_flare_m=load_test('SOLAR FLARE M CLASS',11,ngen,quant,'regression',"/data/solar_flare/solar_flare.csv")
solar_flare_x=load_test('SOLAR FLARE X CLASS',12,ngen,quant,'regression',"/data/solar_flare/solar_flare.csv")
us_space=load_test('US SPACE',1,ngen,quant,'regression',"/data/us_space/us_space.csv")
forest_fires=load_test('FOREST FIRES',12,ngen,quant,'regression',"/data/forest_fires/forest_fires.csv")
concrete_strength=load_test('CONCRETE STRENGTH',8,ngen,quant,'regression',"/data/concrete_strength/concrete_strength.csv")
concrete_slump=load_test('CONCRETE SLUMP',7,ngen,quant,'regression',"/data/concrete_data/concrete_data.csv")
concrete_flow=load_test('CONCRETE FLOW',8,ngen,quant,'regression',"/data/concrete_data/concrete_data.csv")
household_metering_1=load_test('HOUSEHOLD METERING 1',6,ngen,quant,'regression',"/data/household_metering/household_metering.csv")
household_metering_2=load_test('HOUSEHOLD METERING 2',7,ngen,quant,'regression',"/data/household_metering/household_metering.csv")
household_metering_3=load_test('HOUSEHOLD METERING 3',8,ngen,quant,'regression',"/data/household_metering/household_metering.csv")
yacht_hydrodynamics=load_test('YACHT HYDRODYNAMICS',6,ngen,quant,'regression',"/data/yacht_hydrodynamics/yacht_hydrodynamics.csv")
physicochemical=load_test('PHYSIOCHEMICAL',8,ngen,quant,'regression',"/data/physicochemical/physicochemical.csv")
bike_sharing=load_test('BIKE SHARING',15,ngen,quant,'regression',"/data/bike_sharing/bike_sharing.csv")
airfoil=load_test('AIRFOIL',5,ngen,quant,'regression',"/data/airfoil/airfoil.csv")
power_plant=load_test('POWER PLANT',4,ngen,quant,'regression',"/data/power_plant/power_plant.csv")
o_ring=load_test('O-RING',1,ngen,quant,'regression',"/data/o_ring/o_ring.csv")
cycle_powerplant=load_test('CYCLE POWER PLANT',4,ngen,quant,'regression',"/data/cycle_powerplant/cycle_powerplant.csv")
cpu_performance=load_test('CPU PERFORMANCE',9,ngen,quant,'regression',"/data/cpu_performance/cpu_performance.csv")
fertility=load_test('FERTILITY',9,ngen,quant,'classification',"/data/fertility/fertility.csv")
real_estate=load_test('REAL ESTATE',7,ngen,quant,'regression',"/data/real_estate/real_estate.csv")
#
######## ------- TypeError: Both weights and assigned values must be a sequence of numbers when assigning to values of <class 'deap.creator.FitnessMax'>. Currently assigning value(s) None of <class 'NoneType'> to a fitness with weights (1,).
##
brazil_traffic=load_test('BRAZIL TRAFFIC',17,ngen,quant,'regression',"/data/brazil_traffic/brazil_traffic.csv")
##
##
###### ------- UnicodeDecodeError: 'utf-8' codec can't decode byte 0xca in position 4: unexpected end of data
##DATASET has corrupt encoding. Ignore.
crane=load_test('CRANE',2,ngen,quant,'regression',"/data/crane/crane.csv")


######LONDON CRIME CASE STUDY
london_crime=load_test('LONDON CRIME','value',ngen=150,task='regression',path="/data/london_crime/london_crime.csv",
                      sample_n=-1,choice_method='quant',quant=0.75,
                      fix_skew=False,allowed_time=12,n_pop=300)


"""
Analysis of why the results were badly produced

///////////////////////////////////////////////////////////////////////////////////////////
auto=load_test('AUTO MPG','V1',ngen,quant,'regression',"/auto_mpg/auto_mpg.csv")

Firstly the number of rows is signifcantly low - only 398 rows, so there is not a lot of data points to deal with 
There are only 10 relationships which have a relationship of over absiltue value of 0.7
average correlation is very low

///////////////////////////////////////////////////////////////////////////////////////////
house=load_test('HOUSING',13,ngen,quant,'regression',"/housing/housing.csv")

This also has very little number of rows 506

Even though the model produced a slightly worse model, it produced very similar results to the random forest model
average correlation is very bad, approx 0.1


///////////////////////////////////////////////////////////////////////////////////////////

servo=load_test('SERVO',4,ngen,quant,'regression',"/servo/servo.csv")

Only 167 rows, however, produces much better model than the random forest 

- average correlation between variables 0.9 
#however, bear in mind the test size will be very small, so could potentially get a good test case 


///////////////////////////////////////////////////////////////////////////////////////////
fires=load_test('FOREST FIRES','area',ngen,quant,'regression',"/forestfires/forestfires.csv")

500 odd rows produced much better results than random forest, however, still not that good results, 
variables are not strongly correlated with one another 


///////////////////////////////////////////////////////////////////////////////////////////

bike_demand=load_test('BICYCLE DEMAND','count',ngen,quant,'regression',"/bicycle/train.csv")

procudes very good results, the 2 most important variables have a very strong relationship with the target 


"""