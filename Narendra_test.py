from Particle_filters.Narendra_Li import Narendra_Li
from Particle_filters import PF,IPF, UPF,EPF
from CINN_for_PF import proposal_distribution, CINN_PF
from scipy import stats

import numpy as np
from time import time

nods = 50
layer = 5
dim = 2
#---training--#

s_v = 100
s_e = 1

sys = Narendra_Li(sigma_x = 0.01,sigma_v = 0.01*s_v,sigma_e = 0.01 * s_e)
q = proposal_distribution(sys)
u = stats.norm.rvs(size = [2000,1])
start_time = time()
q.train( u = u ,terms=50000)
stop_time = time()
print('Time:')
print(stop_time-start_time)
q.save( "models_for_Narendra_test//model_"+str(s_v)+"_"+str(s_e) )
print('==========================')

#--------------#

s_v = 100
s_e = 1
sys = Narendra_Li(sigma_x = 0.01,sigma_v = 0.01*s_v,sigma_e = 0.01 * s_e)
location =  "models_for_Narendra_test//model_"+str(s_v)+"_"+str(s_e)
q = proposal_distribution(sys,load = location)


t = np.linspace(0,200,201).reshape([201,1])
u = np.sin(2*np.pi*t/10) + np.sin(2*np.pi*t/25)
data,x = sys.generate(u = u)

N = [10,20,50,100,200]

for i in N:
    start_time = time()
    x_p = PF(data,sys,num=i)
    stop_time = time()
    print('PF')
    print('Particle size:')
    print(i)
    print('Time:')
    print(stop_time-start_time)
    print('RMSE:')
    print( '%.3f'%(np.sqrt(dim*np.mean(np.square(x-x_p)))) )
    print('-----------------------------')


    start_time = time()
    x_p = EPF(data,sys,num=100)
    stop_time = time()
    print('EPF')
    print('Time:')
    print(stop_time-start_time)
    print('Particle size:')
    print(i)
    print('RMSE:')
    print( '%.3f'%(np.sqrt(dim*np.mean(np.square(x-x_p)))) )
    print('-----------------------------')

    start_time = time()
    x_p = UPF(data,sys,num=100)
    stop_time = time()
    print('UPF')
    print('Time:')
    print(stop_time-start_time)
    print('Particle size:')
    print(i)
    print('RMSE:')
    print( '%.3f'%(np.sqrt(dim*np.mean(np.square(x-x_p)))) )
    print('-----------------------------')


    start_time = time()
    x_p = IPF(data,sys,num=100)
    stop_time = time()
    print('IPF')
    print('Time:')
    print(stop_time-start_time)
    print('Particle size:')
    print(i)
    print('RMSE:')
    print( '%.3f'%(np.sqrt(dim*np.mean(np.square(x-x_p)))) )

    print('-----------------------------')

    start_time = time()
    x_p = CINN_PF(data,sys,q,num=100)
    stop_time = time()
    print('CINN_PF')
    print('Time:')
    print(stop_time-start_time)
    print('Particle size:')
    print(i)
    print('RMSE:')
    print( '%.3f'%(np.sqrt(dim*np.mean(np.square(x-x_p)))) )
    print('-----------------------------')
