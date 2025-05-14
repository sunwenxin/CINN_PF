from Particle_filters.L96 import L96
from Particle_filters import PF,IPF, UPF,EPF
from CINN_for_PF import proposal_distribution, CINN_PF
from scipy import stats
from scipy import io
import numpy as np
from time import time
import os

dim = 50
num_tests = 100

list_PF = [10,50,500,10000,50000]
list_EPF = [1,10,50,100,200,500,2000]
list_UPF = [1,10,20,30,50,100,500]
list_IPF = [1,3,5,7,10,50]
list_CINN_PF = [1,10,50,500,1000,5000,10000,20000,30000]


if os.path.exists('result.mat'):
    data = io.loadmat('result.mat')
    result_PF = data["result_PF"]
    result_EPF = data["result_EPF"]
    result_UPF = data["result_UPF"]
    result_IPF = data["result_IPF"]
    result_CINN_PF = data["result_CINN_PF"]
    t_PF = data["t_PF"]
    t_EPF = data["t_EPF"]
    t_UPF = data["t_UPF"]
    t_IPF = data["t_IPF"]
    t_CINN_PF = data["t_CINN_PF"]
    term = data["term"]
else:
    result_PF = np.zeros([num_tests,4, len(list_PF)])*np.nan
    result_EPF = np.zeros([num_tests,4, len(list_EPF)])*np.nan
    result_UPF = np.zeros([num_tests,4, len(list_UPF)])*np.nan
    result_IPF = np.zeros([num_tests,4, len(list_IPF)])*np.nan
    result_CINN_PF = np.zeros([num_tests,4, len(list_CINN_PF)])*np.nan
    t_PF = np.zeros([num_tests,4, len(list_PF)])*np.nan
    t_EPF = np.zeros([num_tests,4, len(list_EPF)])*np.nan
    t_UPF = np.zeros([num_tests,4, len(list_UPF)])*np.nan
    t_IPF = np.zeros([num_tests,4, len(list_IPF)])*np.nan
    t_CINN_PF = np.zeros([num_tests,4, len(list_CINN_PF)])*np.nan
    term = np.array( [[-1]] )

for k in range(term[0,0] + 1,num_tests):
    print('No: '+str(k))
    for s_v in [1,2]:
        for p in [1,2]:
            j = ( s_v - 1 )*2 + p - 1
            print("="*50)
            print("s_v = "+str(s_v)+'; '+"p = "+str(p))
            print("="*50)
            sys = L96(sigma_x = 0.01,sigma_v = 0.1*s_v,sigma_e = 0.01,p=p,N=dim)
            data,x = sys.generate(u = np.zeros([200,1]))
            location = "models_for_L96_test//model_"+str(s_v)+"_"+str(p)
            q = proposal_distribution(sys,load = location)

            for i in range(len(list_PF)):
                num = list_PF[i]
                print('PF')
                print("Particel size: "+str(num))
                start_time = time()
                x_p = PF(data,sys,num=num)
                stop_time = time()
                result_PF[ k,j,i ] = np.sqrt(dim*np.mean(np.square(x-x_p)))
                t_PF[ k,j,i ] = stop_time-start_time
                print("Time: {:.5f}, RMSE: {:.5f}".format(stop_time - start_time, np.sqrt(dim * np.mean(np.square(x - x_p)))))

            print('-'*36)

            for i in range(len(list_EPF)):
                num = list_EPF[i]
                print('EPF')
                print("Particel size: "+str(num))
                start_time = time()
                x_p = EPF(data,sys,num=num)
                stop_time = time()
                result_EPF[ k,j,i ] = np.sqrt(dim*np.mean(np.square(x-x_p)))
                t_EPF[ k,j,i ] = stop_time-start_time
                print("Time: {:.5f}, RMSE: {:.5f}".format(stop_time - start_time, np.sqrt(dim * np.mean(np.square(x - x_p)))))
            print('-'*36)

            for i in range(len(list_UPF)):
                num = list_UPF[i]
                print('UPF')
                print("Particel size: "+str(num))
                start_time = time()
                x_p = UPF(data,sys,num=num)
                stop_time = time()
                result_UPF[ k,j,i ] = np.sqrt(dim*np.mean(np.square(x-x_p)))
                t_UPF[ k,j,i ] = stop_time-start_time
                print("Time: {:.5f}, RMSE: {:.5f}".format(stop_time - start_time, np.sqrt(dim * np.mean(np.square(x - x_p)))))
            print('-'*36)

            for i in range(len(list_IPF)):
                num = list_IPF[i]
                print('A-IPF')
                print("Particel size: "+str(num))
                start_time = time()
                x_p = IPF(data,sys,num=num)
                stop_time = time()
                result_IPF[ k,j,i ] = np.sqrt(dim*np.mean(np.square(x-x_p)))
                t_IPF[ k,j,i ] = stop_time-start_time
                print("Time: {:.5f}, RMSE: {:.5f}".format(stop_time - start_time, np.sqrt(dim * np.mean(np.square(x - x_p)))))
            print('-'*36)


            for i in range(len(list_CINN_PF)):
                num = list_CINN_PF[i]
                print('CINN_PF')
                print("Particel size: "+str(num))   
                start_time = time()
                x_p = CINN_PF(data,sys,q,num=num)
                stop_time = time()
                result_CINN_PF[ k,j,i ] = np.sqrt(dim*np.mean(np.square(x-x_p)))
                t_CINN_PF[ k,j,i ] = stop_time-start_time
                print("Time: {:.5f}, RMSE: {:.5f}".format(stop_time - start_time, np.sqrt(dim * np.mean(np.square(x - x_p)))))
            print('-'*36)


            io.savemat("result.mat",{"list_PF":list_PF,
                                                     "list_EPF":list_EPF,
                                                     "list_UPF":list_UPF,
                                                     "list_IPF":list_IPF,
                                                     "list_CINN_PF":list_CINN_PF,
                                                     "result_PF":result_PF,
                                                     "result_EPF":result_EPF,
                                                     "result_UPF":result_UPF,
                                                     "result_IPF":result_IPF,
                                                     "result_CINN_PF":result_CINN_PF,
                                                     "t_PF":t_PF,
                                                     "t_EPF":t_EPF,
                                                     "t_UPF":t_UPF,
                                                     "t_IPF":t_IPF,
                                                     "t_CINN_PF":t_CINN_PF,
                                                     "term":k,})
