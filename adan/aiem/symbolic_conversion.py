import os, sys
lib_path = os.path.abspath(os.path.join('..','..'))
sys.path.append(lib_path)


def makelog(x):
    res='log('+x+')'
    return res
    
def add(x,y):
    res='Add('+x+','+y+')'
    return res
    
def sub(x,y):
    res='Add('+x+',Mul(Integer(-1),'+y+'))'
    return res

def mul(x,y):
    res='Mul('+x+','+y+')'
    return res
    
def div(x,y):
    res='Mul('+x+',Pow('+y+',Integer(-1)))'
    return res

def cube(x):
    res='Pow('+x+',3)'
    return res
    
def square(x):
    res='Pow('+x+',2)'
    return res
 

def squareroot(x):
    res='sqrt('+x+')'
    return res

def recipr(x):
    res='Pow('+x+',Integer(-1))'
    return res
    
def absolute(x):
    res='Abs('+x+')'
    return res
    
def sin(x):
    res='sin('+x+')'
    return res
    
def cos(x):
    res='cos('+x+')'
    return res


def stringifyVars(ind,column_names):
    for col in column_names:
        ind=ind.replace("("+col+")","('"+col+"')")
        ind=ind.replace("("+col+",","('"+col+"',")
        ind=ind.replace(","+col+")",",'"+col+"')")
        ind=ind.replace(", "+col+")",",'"+col+"')")
    return ind

def replaceNames(ind,column_names):
    for col in column_names:
        ind=ind.replace("("+col+")","(Symbol('"+col+"'))")
        ind=ind.replace("("+col+",","(Symbol('"+col+"'),")
        ind=ind.replace(","+col+")",",Symbol('"+col+"'))")
        ind=ind.replace(", "+col+")",",Symbol('"+col+"'))")
        # ind=ind.replace("("+col+")","(df['"+col+"'])")
        # ind=ind.replace("("+col+",","(df['"+col+"']),")
        # ind=ind.replace(","+col+")",",df['"+col+"'])")
        # ind=ind.replace(", "+col+")",",df['"+col+"'])")
    return ind
        
def checkIfColumn(ind,column_names):
    for col in column_names:
        if col==ind:
            return True
    return False

def convertToEquation(individual,column_names):    
    ind=stringifyVars(individual,column_names)
    if not checkIfColumn(ind,column_names):
        ind=eval(ind)
        ind=replaceNames(ind,column_names)

    else:
        ind="Symbol('"+ind+"')"
        # ind="df['"+ind+"']"
    return ind
    
def convertIndividualsToEqs(individuals,column_names):
    inds2=[]
    for ind in individuals:
        inds2.append(convertToEquation(ind,column_names))
    return inds2
    
    
#colnames=['var1', 'var2', 'var7', 'var10', 'var13', 'var14', 'var1_amin_over_var0', 'var1_amax_over_var0', 'var1_mean_over_var0', 'var1_std_over_var0', 'var1_sum_over_var0', 'var1_kurtosis_over_var0', 'var1_skew_over_var0', 'var2_amax_over_var0', 'var2_mean_over_var0', 'var2_std_over_var0', 'var2_sum_over_var0', 'var2_kurtosis_over_var0', 'var2_skew_over_var0', 'var7_amax_over_var0', 'var7_mean_over_var0', 'var7_std_over_var0', 'var7_sum_over_var0', 'var7_kurtosis_over_var0', 'var7_skew_over_var0', 'var10_amax_over_var0', 'var10_mean_over_var0', 'var10_std_over_var0', 'var10_sum_over_var0', 'var10_kurtosis_over_var0', 'var10_skew_over_var0', 'var13_amax_over_var0', 'var13_mean_over_var0', 'var13_std_over_var0', 'var13_sum_over_var0', 'var13_kurtosis_over_var0', 'var13_skew_over_var0', 'var14_amax_over_var0', 'var14_mean_over_var0', 'var14_std_over_var0', 'var14_sum_over_var0', 'var14_kurtosis_over_var0', 'var14_skew_over_var0', 'var1_amin_over_var3', 'var1_amax_over_var3', 'var1_mean_over_var3', 'var1_std_over_var3', 'var1_sum_over_var3', 'var1_kurtosis_over_var3', 'var1_skew_over_var3', 'var2_amin_over_var3', 'var2_amax_over_var3', 'var2_mean_over_var3', 'var2_std_over_var3', 'var2_sum_over_var3', 'var2_kurtosis_over_var3', 'var2_skew_over_var3', 'var7_amax_over_var3', 'var7_mean_over_var3', 'var7_std_over_var3', 'var7_sum_over_var3', 'var7_kurtosis_over_var3', 'var7_skew_over_var3', 'var10_amax_over_var3', 'var10_mean_over_var3', 'var10_std_over_var3', 'var10_sum_over_var3', 'var10_kurtosis_over_var3', 'var10_skew_over_var3', 'var13_amin_over_var3', 'var13_amax_over_var3', 'var13_mean_over_var3', 'var13_std_over_var3', 'var13_sum_over_var3', 'var13_kurtosis_over_var3', 'var13_skew_over_var3', 'var14_amax_over_var3', 'var14_mean_over_var3', 'var14_std_over_var3', 'var14_sum_over_var3', 'var14_kurtosis_over_var3', 'var14_skew_over_var3', 'var1_amin_over_var4', 'var1_amax_over_var4', 'var1_mean_over_var4', 'var1_std_over_var4', 'var1_sum_over_var4', 'var1_kurtosis_over_var4', 'var1_skew_over_var4', 'var2_amin_over_var4', 'var2_amax_over_var4', 'var2_mean_over_var4', 'var2_std_over_var4', 'var2_sum_over_var4', 'var2_kurtosis_over_var4', 'var2_skew_over_var4', 'var7_amax_over_var4', 'var7_mean_over_var4', 'var7_std_over_var4', 'var7_sum_over_var4', 'var7_kurtosis_over_var4', 'var7_skew_over_var4', 'var10_amax_over_var4', 'var10_mean_over_var4', 'var10_std_over_var4', 'var10_sum_over_var4', 'var10_kurtosis_over_var4', 'var10_skew_over_var4', 'var13_amin_over_var4', 'var13_amax_over_var4', 'var13_mean_over_var4', 'var13_std_over_var4', 'var13_sum_over_var4', 'var13_kurtosis_over_var4', 'var13_skew_over_var4', 'var14_amax_over_var4', 'var14_mean_over_var4', 'var14_std_over_var4', 'var14_sum_over_var4', 'var14_kurtosis_over_var4', 'var14_skew_over_var4', 'var1_amin_over_var5', 'var1_amax_over_var5', 'var1_mean_over_var5', 'var1_std_over_var5', 'var1_sum_over_var5', 'var1_kurtosis_over_var5', 'var1_skew_over_var5', 'var2_amin_over_var5', 'var2_amax_over_var5', 'var2_mean_over_var5', 'var2_std_over_var5', 'var2_sum_over_var5', 'var2_kurtosis_over_var5', 'var2_skew_over_var5', 'var7_amin_over_var5', 'var7_amax_over_var5', 'var7_mean_over_var5', 'var7_std_over_var5', 'var7_sum_over_var5', 'var7_kurtosis_over_var5', 'var7_skew_over_var5', 'var10_amax_over_var5', 'var10_mean_over_var5', 'var10_std_over_var5', 'var10_sum_over_var5', 'var10_kurtosis_over_var5', 'var10_skew_over_var5', 'var13_amax_over_var5', 'var13_mean_over_var5', 'var13_std_over_var5', 'var13_sum_over_var5', 'var13_kurtosis_over_var5', 'var13_skew_over_var5', 'var14_amin_over_var5', 'var14_amax_over_var5', 'var14_mean_over_var5', 'var14_std_over_var5', 'var14_sum_over_var5', 'var14_kurtosis_over_var5', 'var14_skew_over_var5', 'var1_amin_over_var6', 'var1_amax_over_var6', 'var1_mean_over_var6', 'var1_std_over_var6', 'var1_sum_over_var6', 'var1_kurtosis_over_var6', 'var1_skew_over_var6', 'var2_amin_over_var6', 'var2_amax_over_var6', 'var2_mean_over_var6', 'var2_std_over_var6', 'var2_sum_over_var6', 'var2_kurtosis_over_var6', 'var2_skew_over_var6', 'var7_amin_over_var6', 'var7_amax_over_var6', 'var7_mean_over_var6', 'var7_std_over_var6', 'var7_sum_over_var6', 'var7_kurtosis_over_var6', 'var7_skew_over_var6', 'var10_amax_over_var6', 'var10_mean_over_var6', 'var10_std_over_var6', 'var10_sum_over_var6', 'var10_kurtosis_over_var6', 'var10_skew_over_var6', 'var13_amax_over_var6', 'var13_mean_over_var6', 'var13_std_over_var6', 'var13_sum_over_var6', 'var13_kurtosis_over_var6', 'var13_skew_over_var6', 'var14_amin_over_var6', 'var14_amax_over_var6', 'var14_mean_over_var6', 'var14_std_over_var6', 'var14_sum_over_var6', 'var14_kurtosis_over_var6', 'var14_skew_over_var6', 'var1_amin_over_var8', 'var1_amax_over_var8', 'var1_mean_over_var8', 'var1_std_over_var8', 'var1_sum_over_var8', 'var1_kurtosis_over_var8', 'var1_skew_over_var8', 'var2_amax_over_var8', 'var2_mean_over_var8', 'var2_std_over_var8', 'var2_sum_over_var8', 'var2_kurtosis_over_var8', 'var2_skew_over_var8', 'var7_amax_over_var8', 'var7_mean_over_var8', 'var7_std_over_var8', 'var7_sum_over_var8', 'var7_kurtosis_over_var8', 'var7_skew_over_var8', 'var10_amax_over_var8', 'var10_mean_over_var8', 'var10_std_over_var8', 'var10_sum_over_var8', 'var10_kurtosis_over_var8', 'var10_skew_over_var8', 'var13_amax_over_var8', 'var13_mean_over_var8', 'var13_std_over_var8', 'var13_sum_over_var8', 'var13_kurtosis_over_var8', 'var13_skew_over_var8', 'var14_amax_over_var8', 'var14_mean_over_var8', 'var14_std_over_var8', 'var14_sum_over_var8', 'var14_kurtosis_over_var8', 'var14_skew_over_var8', 'var1_amin_over_var9', 'var1_amax_over_var9', 'var1_mean_over_var9', 'var1_std_over_var9', 'var1_sum_over_var9', 'var1_kurtosis_over_var9', 'var1_skew_over_var9', 'var2_amax_over_var9', 'var2_mean_over_var9', 'var2_std_over_var9', 'var2_sum_over_var9', 'var2_kurtosis_over_var9', 'var2_skew_over_var9', 'var7_amax_over_var9', 'var7_mean_over_var9', 'var7_std_over_var9', 'var7_sum_over_var9', 'var7_kurtosis_over_var9', 'var7_skew_over_var9', 'var10_amin_over_var9', 'var10_amax_over_var9', 'var10_mean_over_var9', 'var10_std_over_var9', 'var10_sum_over_var9', 'var10_kurtosis_over_var9', 'var10_skew_over_var9', 'var13_amax_over_var9', 'var13_mean_over_var9', 'var13_std_over_var9', 'var13_sum_over_var9', 'var13_kurtosis_over_var9', 'var13_skew_over_var9', 'var14_amax_over_var9', 'var14_mean_over_var9', 'var14_std_over_var9', 'var14_sum_over_var9', 'var14_kurtosis_over_var9', 'var14_skew_over_var9', 'var1_amin_over_var11', 'var1_amax_over_var11', 'var1_mean_over_var11', 'var1_std_over_var11', 'var1_sum_over_var11', 'var1_kurtosis_over_var11', 'var1_skew_over_var11', 'var2_amax_over_var11', 'var2_mean_over_var11', 'var2_std_over_var11', 'var2_sum_over_var11', 'var2_kurtosis_over_var11', 'var2_skew_over_var11', 'var7_amax_over_var11', 'var7_mean_over_var11', 'var7_std_over_var11', 'var7_sum_over_var11', 'var7_kurtosis_over_var11', 'var7_skew_over_var11', 'var10_amax_over_var11', 'var10_mean_over_var11', 'var10_std_over_var11', 'var10_sum_over_var11', 'var10_kurtosis_over_var11', 'var10_skew_over_var11', 'var13_amax_over_var11', 'var13_mean_over_var11', 'var13_std_over_var11', 'var13_sum_over_var11', 'var13_kurtosis_over_var11', 'var13_skew_over_var11', 'var14_amax_over_var11', 'var14_mean_over_var11', 'var14_std_over_var11', 'var14_sum_over_var11', 'var14_kurtosis_over_var11', 'var14_skew_over_var11', 'var1_amin_over_var12', 'var1_amax_over_var12', 'var1_mean_over_var12', 'var1_std_over_var12', 'var1_sum_over_var12', 'var1_kurtosis_over_var12', 'var1_skew_over_var12', 'var2_amin_over_var12', 'var2_amax_over_var12', 'var2_mean_over_var12', 'var2_std_over_var12', 'var2_sum_over_var12', 'var2_kurtosis_over_var12', 'var2_skew_over_var12', 'var7_amax_over_var12', 'var7_mean_over_var12', 'var7_std_over_var12', 'var7_sum_over_var12', 'var7_kurtosis_over_var12', 'var7_skew_over_var12', 'var10_amax_over_var12', 'var10_mean_over_var12', 'var10_std_over_var12', 'var10_sum_over_var12', 'var10_kurtosis_over_var12', 'var10_skew_over_var12', 'var13_amin_over_var12', 'var13_amax_over_var12', 'var13_mean_over_var12', 'var13_std_over_var12', 'var13_sum_over_var12', 'var13_kurtosis_over_var12', 'var13_skew_over_var12', 'var14_amax_over_var12', 'var14_mean_over_var12', 'var14_std_over_var12', 'var14_sum_over_var12', 'var14_kurtosis_over_var12', 'var14_skew_over_var12', 'var0_a', 'var0_b', 'var3_l', 'var3_u', 'var3_y', 'var4_g', 'var4_gg', 'var4_p', 'var5_aa', 'var5_c', 'var5_cc', 'var5_d', 'var5_e', 'var5_ff', 'var5_i', 'var5_j', 'var5_k', 'var5_m', 'var5_q', 'var5_r', 'var5_w', 'var5_x', 'var6_bb', 'var6_dd', 'var6_ff', 'var6_h', 'var6_j', 'var6_n', 'var6_o', 'var6_v', 'var6_z', 'var8_f', 'var8_t', 'var9_f', 'var9_t', 'var11_f', 'var11_t', 'var12_g', 'var12_p', 'var12_s']
#
#fasi=['sin(var1_amax_over_var8)', 'sin(absolute(cube(square(var1_sum_over_var9))))', 'div(var10_mean_over_var6, div(sin(cube(squareroot(var14_skew_over_var3))), absolute(sub(add(var5_cc, var7_kurtosis_over_var0), div(var13_skew_over_var8, var2_skew_over_var12)))))', 'sub(var13_std_over_var9, var2_skew_over_var4)', 'mul(sub(square(recipr(sin(var2_std_over_var9))), div(recipr(var10_sum_over_var4), add(var14_sum_over_var6, var7_amax_over_var6))), absolute(div(makelog(var10_std_over_var8), mul(var13_kurtosis_over_var11, var1_mean_over_var9))))', 'mul(var14_sum_over_var8, var14_skew_over_var4)', 'sub(var13_skew_over_var12, var2_mean_over_var9)', 'div(cube(square(var7_skew_over_var8)), var10_amax_over_var9)', 'div(div(var1_mean_over_var12, var8_t), squareroot(var13_skew_over_var8))', 'add(absolute(squareroot(makelog(var9_t))), cube(var1_amax_over_var9))', 'makelog(var13_amax_over_var8)', 'sub(cos(var1_mean_over_var12), cos(var2_mean_over_var8))', 'square(var7_mean_over_var5)', 'cube(var13_mean_over_var5)', 'square(var7_amax_over_var9)', 'makelog(cube(div(var11_f, var1_std_over_var8)))', 'sub(makelog(cos(squareroot(cos(var13_mean_over_var4)))), div(recipr(recipr(mul(var7_sum_over_var9, var2_kurtosis_over_var3))), sin(add(recipr(var7_std_over_var5), makelog(var10_std_over_var9)))))', 'sub(sin(var7_mean_over_var5), var10_skew_over_var9)', 'sub(makelog(add(var12_s, var14_skew_over_var5)), squareroot(cos(var14_skew_over_var9)))', 'recipr(square(var7_std_over_var8))', 'add(var14_kurtosis_over_var12, var7_kurtosis_over_var8)', 'squareroot(recipr(mul(cos(var7_skew_over_var6), cos(var8_f))))', 'div(var13_mean_over_var9, var14_skew_over_var8)', 'sub(var1_skew_over_var3, absolute(squareroot(makelog(var9_t))))', 'makelog(var13_kurtosis_over_var8)', 'makelog(var8_f)', 'sin(var14_sum_over_var8)', 'sin(cube(var1_skew_over_var8))', 'cos(cos(div(sub(absolute(var14_mean_over_var8), add(var9_t, var2_amax_over_var4)), add(absolute(var10_skew_over_var8), sub(var7_mean_over_var3, var7_amax_over_var9)))))', 'sin(var7_amax_over_var9)', 'sub(var8_t, var10_mean_over_var5)', 'cube(square(var7_std_over_var8))', 'sub(var2_std_over_var8, var7_std_over_var9)', 'sin(var7_skew_over_var8)', 'cos(var7_skew_over_var8)', 'mul(var10_std_over_var12, var1_std_over_var9)', 'mul(var1_skew_over_var8, var10_kurtosis_over_var12)', 'cube(var13_sum_over_var8)', 'sub(var2_std_over_var8, recipr(cos(sin(var14_mean_over_var12))))', 'sub(cube(div(var10_amax_over_var9, cos(var2_skew_over_var9))), cos(add(makelog(var7_sum_over_var9), squareroot(var10_sum_over_var12))))', 'div(var14_std_over_var8, square(var7_std_over_var8))', 'recipr(var14_sum_over_var8)', 'mul(var7_skew_over_var8, var14_skew_over_var0)', 'absolute(mul(squareroot(squareroot(var13_sum_over_var4)), absolute(var2_mean_over_var9)))', 'div(var1_sum_over_var9, var2_std_over_var0)', 'div(var10_mean_over_var6, sin(var8_t))', 'div(var10_mean_over_var6, sin(var1_amax_over_var8))', 'mul(var7_kurtosis_over_var3, add(sin(var13_amax_over_var4), cos(var7_mean_over_var5)))', 'sub(var6_ff, var14_kurtosis_over_var4)', 'div(var8_f, var10_skew_over_var9)']
#t=convertIndividualsToEqs(fasi,colnames)
#print(sympify(t))