from Narendra_Li import Narendra_Li
from system import PF,UPF,EPF,IPF
from matplotlib import pyplot as plt
from time import time
import numpy as np

mod = L96(sigma_x = 0.01,sigma_v = 0.01,sigma_e = 0.01,N = 20)
#mod = Narendra_Li(sigma_x = 0.01,sigma_v = 1,sigma_e = 0.01)
length = 200
t = np.linspace(0,length,length+ 1)
u = np.c_[np.sin(2*np.pi*t/10) + np.sin(2*np.pi*t/25)]
data,x = mod.generate(u)


x_hat1 = PF(data,mod,num = 10)
x_hat2 = EPF(data,mod,num = 10)
x_hat3 = UPF(data,mod,num = 10)
x_hat4 = IPF(data,mod,num = 10)

print( np.mean( np.square(x_hat1 - x) ) )
print( np.mean( np.square(x_hat2 - x) ) )
print( np.mean( np.square(x_hat3 - x) ) )
print( np.mean( np.square(x_hat4 - x) ) )
