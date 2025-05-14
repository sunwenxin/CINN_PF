from Particle_filters import Narendra_Li,L96,PF,EPF,UPF,IPF
from matplotlib import pyplot as plt
from time import time
import numpy as np

sys = L96(sigma_x = 0.01,sigma_v = 0.01,sigma_e = 0.01,N = 50)
#mod = Narendra_Li(sigma_x = 0.01,sigma_v = 1,sigma_e = 0.01)
length = 200
t = np.linspace(0,length,length+ 1)
u = np.c_[np.sin(2*np.pi*t/10) + np.sin(2*np.pi*t/25)]
data,x = sys.generate(u)


x_hat1 = PF(data,sys,num = 10)
print( np.mean( np.square(x_hat1 - x) ) )

x_hat2 = EPF(data,sys,num = 10)
print( np.mean( np.square(x_hat2 - x) ) )

x_hat3 = UPF(data,sys,num = 10)
print( np.mean( np.square(x_hat3 - x) ) )

x_hat4 = IPF(data,sys,num = 10)
print( np.mean( np.square(x_hat4 - x) ) )
