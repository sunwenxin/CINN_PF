from Particle_filters.L96 import L96
from Particle_filters.Narendra_Li import Narendra_Li
import numpy as np
from Particle_filters import E_approximator,U_approximator,I_approximator
from CINN_for_PF import proposal_distribution


dim = 2
s_v = 1
pp = 2

sys = L96(sigma_x = 0.1,sigma_v = 0.1*s_v,sigma_e = 0.01,p=pp,N=dim)

x = np.array([[7.976, 8.600]])
y = np.array([[8.237, 7.944]])
'''
x = np.array([[7.97557542, 8.59971765]])
y = np.array([[8.23672972, 7.94428108]])

x = np.array([[5.03522691, 4.55856093]])
y = np.array([[4.42796357, 5.35747103]])

x = np.array([[7.64267617, 6.84462676]])
y = np.array([[7.62450095, 6.97412114]])

x = np.array([[ 9.08662553, 10.04274968]])
y = np.array([[9.70470639, 9.59857897]])



x = np.array([[7.50899946, 7.36580989]])
y = np.array([[6.90248047, 7.47986339]])
'''

x_til = sys.f.noise_free(x = x,u = [])

'''
result,x = sys.generate(u=np.zeros([2000,1]))
x = x[1001:1002,:]
y = result["y"][1002:1003,:]
x_til = sys.f.noise_free(x = x,u = [])
'''
xx, yy = np.meshgrid(  np.linspace(-2, 2, 500,dtype='float32'),np.linspace(-2, 2, 500,dtype='float32') )
xx += x_til[0,0]
yy += x_til[0,1]
z = np.c_[xx.ravel(), yy.ravel()]

Z = sys.F(x_tild = x_til , y = y , v = z - x_til)/2
p = np.exp(-Z).reshape(xx.shape)
optimal_q = p/( np.mean(p) * 16)

q = {}

q["Original"] =sys.f.p(x = x,u = [], x_next = z).reshape(xx.shape)

EPF_approx = E_approximator(sys, x = x,u = [], y_next = y).p(z)
q["q_EPF"] = EPF_approx.reshape(xx.shape)

UPF_approx = U_approximator(sys, x = x,u = [], y_next = y).p(z)
q["q_UPF"] = UPF_approx.reshape(xx.shape)

IPF_approx = I_approximator(sys, x = x,u = [], y_next = y).p(z)
q["q_IPF"] = IPF_approx.reshape(xx.shape)

'''
mod = proposal_distribution(sys,nodes = [50]*5)
mod.train(u = np.zeros([2000,1]),terms = 20000)
mod.save( "models_for_L96_test//double_test_"+str(s_v)+"_"+str(pp) )
'''
mod = proposal_distribution(sys,load = "models_for_L96_test//double_test_"+str(s_v)+"_"+str(pp) )
q["q_CINN_PF"] = mod.p(x = x,y_next = y,x_next = z).reshape(xx.shape)

from matplotlib import pyplot as plt
import matplotlib
from matplotlib import rcParams
matplotlib.use("svg")
pgf_config = {
    "font.family":'serif',
    "font.size": 10,
    "pgf.rcfonts": False,
    "text.usetex": True,
    "pgf.preamble": str([
        r"\usepackage{unicode-math}",
        #r"\setmathfont{XITS Math}", 
        # 这里注释掉了公式的XITS字体，可以自行修改
        r"\setmainfont{Times New Roman}",
        r"\usepackage{xeCJK}",
        r"\xeCJKsetup{CJKmath=true}",
        r"\setCJKmainfont{SimSun}",
    ]),
}
rcParams.update(pgf_config)





p = optimal_q
fig, ax = plt.subplots(figsize=[6, 4])
plt.subplots_adjust(top=0.9,bottom=0.15,left=0.13,right=0.97,hspace=0,wspace=0)
ax.set_xlim([7,9.5])
ax.set_ylim([6.5,9.5])
strs1 = np.linspace(0.001,np.max(p)+0.05,8)
strs1 = ((strs1*1000)//1)/1000
CS1 = ax.contourf(xx, yy, p ,strs1, alpha=0.8, cmap=plt.cm.Reds)
h1,_ = CS1.legend_elements()
n = np.size(h1)
strs = []
for j in range(n):
    strs += ["%.3f"%CS1.levels[j]+"$\sim$"+"%.3f"%CS1.levels[j+1]]

plt.legend(h1, strs,ncol=1,fontsize=9,loc='lower right' )
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("${\mathbf x}_{1,t+1}$",fontsize=16)
plt.ylabel("${\mathbf x}_{2,t+1}$",fontsize=16)
plt.savefig("optimal_q.pdf")




for i in ["Original","q_EPF","q_UPF","q_IPF","q_CINN_PF"]:
    p = q[i]
    fig, ax = plt.subplots(figsize=[6, 4])
    plt.subplots_adjust(top=0.9,bottom=0.15,left=0.13,right=0.97,hspace=0,wspace=0)
    ax.set_xlim([7,9.5])
    ax.set_ylim([6.5,9.5])
    ax.contourf(xx, yy, optimal_q ,strs1, alpha=0.8, cmap=plt.cm.Reds)
    strs2 = np.linspace(0.001,np.max(p)+0.05,8)
    strs2 = ((strs2*1000)//1)/1000
    CS1 = ax.contourf(xx, yy, p ,strs2, alpha=0.8, cmap=plt.cm.Blues)
    h1,_ = CS1.legend_elements()
    n = np.size(h1)
    strs = []
    for j in range(n):
        strs += ["%.3f"%CS1.levels[j]+"$\sim$"+"%.3f"%CS1.levels[j+1]]

    plt.legend(h1, strs,ncol=1,fontsize=9,loc='lower right' )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("${\mathbf x}_{1,t+1}$",fontsize=16)
    plt.ylabel("${\mathbf x}_{2,t+1}$",fontsize=16)
    plt.savefig(i+".pdf")










