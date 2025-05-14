from Particle_filters.L96 import L96
from Particle_filters import PF,IPF, UPF,EPF
from CINN_for_PF import proposal_distribution, CINN_PF
from scipy import stats

import numpy as np
from time import time

dim = 50
nods = 50
layer = 5

#---training--#

for s_v in [1,2]:
    for p in [2,1]:
        sys = L96(sigma_x = 0.01,sigma_v = 0.1*s_v,sigma_e = 0.01,p=p,N=dim)
        q = proposal_distribution(sys,nodes = [nods]*layer)
        u = np.zeros( [5000,1])
        start_time = time()
        q.train( u = u ,terms=20000)
        stop_time = time()
        print('Time:')
        print(stop_time-start_time)
        #q.save( "models_for_L96_test//model_"+str(s_v)+"_"+str(p) )
        print('==========================')

#--------------#

s_v = 1
p = 1
sys = L96(sigma_x = 0.01,sigma_v = 0.1*s_v,sigma_e = 0.01,p=p,N=dim)
location = "models_for_L96_test//model_"+str(s_v)+"_"+str(p)
q = proposal_distribution(sys,load = location)


u = np.zeros( [200,1])
data,x = sys.generate(u = u)

'''
start_time = time()
x_p = PF(data,sys,num=10000)
stop_time = time()
print('PF10000')
print('Time:')
print(stop_time-start_time)
print('RMSE:')
print( '%.3f'%(np.sqrt(dim*np.mean(np.square(x-x_p)))) )
print('-----------------------------')


start_time = time()
x_p = EPF(data,sys,num=50)
stop_time = time()
print('EPF')
print('Time:')
print(stop_time-start_time)
print('RMSE:')
print( '%.3f'%(np.sqrt(dim*np.mean(np.square(x-x_p)))) )
print('-----------------------------')

start_time = time()
x_p = UPF(data,sys,num=50)
stop_time = time()
print('UPF')
print('Time:')
print(stop_time-start_time)
print('RMSE:')
print( '%.3f'%(np.sqrt(dim*np.mean(np.square(x-x_p)))) )
print('-----------------------------')


start_time = time()
x_p = IPF(data,sys,num=10)
stop_time = time()
print('IPF')
print('Time:')
print(stop_time-start_time)
print('RMSE:')
print( '%.3f'%(np.sqrt(dim*np.mean(np.square(x-x_p)))) )

print('-----------------------------')
'''
start_time = time()
x_p = CINN_PF(data,sys,q,num=10)
stop_time = time()
print('CINN_PF')
print('Time:')
print(stop_time-start_time)
print('RMSE:')
print( '%.3f'%(np.sqrt(dim*np.mean(np.square(x-x_p)))) )
print('-----------------------------')
