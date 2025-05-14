import numpy as np
from numpy import random
from scipy import stats

def resampling(weight , num = None):
    weight_ = np.nan_to_num(weight)
    weight = weight_/np.sum(weight_)
    if num is None:
        num = weight.shape[0]
    index = random.multinomial(n=num, pvals=weight)
    index2 = np.zeros([num],dtype='uint32')
    s_top = 0
    s_bot = 0
    for i in range(weight.shape[0]):
        if index[i] != 0 :
            s_top += index[i]
            index2[s_bot:s_top] = i
            s_bot = s_top
    return index2


def resampling_exp(ln_weight , num = None):
    m = np.nanmax(ln_weight)
    try:
        weight = np.exp(ln_weight-m+690)
        if num is None:
            num = weight.shape[0]
        return resampling(weight , num = num)
    except:
        print("weight error")
        print(ln_weight)
        print("-----------")
        print(weight)



