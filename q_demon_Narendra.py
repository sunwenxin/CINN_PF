from Particle_filters.L96 import L96
from Particle_filters.Narendra_Li import Narendra_Li
import numpy as np
from Particle_filters import E_approximator,U_approximator,I_approximator
from CINN_for_PF import proposal_distribution
import os

s_v = 100
s_e = 1

x = np.array([[1.861,-2.669]])
u = np.array([[0.410]])
y = np.array([[2.993]])

if not os.path.exists("q_demo_Narendra"):
    os.makedirs("q_demo_Narendra")

sys = Narendra_Li(sigma_x = 0.01,sigma_v = 0.01*s_v,sigma_e = 0.01 * s_e)


x_til = sys.f.noise_free(x = x,u = u)

lim = [3.5,-2.5,4,0]##

xx, yy = np.meshgrid(  np.linspace(lim[0], lim[1], 500,dtype='float32'),np.linspace(lim[2], lim[3], 500,dtype='float32')  )
z = np.c_[xx.ravel(), yy.ravel()]


Z = sys.F(x_tild = x_til , y = y , v = z - x_til)/2
p = np.exp(-Z).reshape(xx.shape)
optimal_q = p/( np.mean(p) * 24)



q = {}

q["Original"] =sys.f.p(x = x,u = u, x_next = z).reshape(xx.shape)

EPF_approx = E_approximator(sys, x = x,u = u, y_next = y).p(z)
q["q_EPF"] = EPF_approx.reshape(xx.shape)

UPF_approx = U_approximator(sys, x = x,u = u, y_next = y).p(z)
q["q_UPF"] = UPF_approx.reshape(xx.shape)

IPF_approx = I_approximator(sys, x = x,u = u, y_next = y).p(z)
q["q_IPF"] = IPF_approx.reshape(xx.shape)


mod = proposal_distribution(sys,load = "models_for_Narendra_test\model_100_1" )
q["q_CINN_PF"] = mod.p(x = x,u = u,y_next = y,x_next = z).reshape(xx.shape)

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
ax.set_xlim([lim[1],lim[0]])
ax.set_ylim([lim[3],lim[2]])
strs1 = np.linspace(0.001,np.max(p)+0.05,8)
strs1 = ((strs1*1000)//1)/1000
CS1 = ax.contourf(xx, yy, p ,strs1, alpha=0.8, cmap=plt.cm.Reds)
h1,_ = CS1.legend_elements()
n = np.size(h1)
strs = []
for j in range(n):
    strs += ["%.3f"%CS1.levels[j]+"$\sim$"+"%.3f"%CS1.levels[j+1]]

ax.text(0.6, 0.9, '$\mathbf x_t=[ %.3f,%.3f ]$;\n$\mathbf u_t=%.3f$;\n$\mathbf y_{t+1}=%.3f$.' % (x[0,0],x[0,1],u[0,0],y[0,0]),
        horizontalalignment='left',
        verticalalignment='center',
        transform=ax.transAxes,fontsize=16)

plt.legend(h1, strs,ncol=2,fontsize=9,loc='lower center' )
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel("${\mathbf x}_{1,t+1}$",fontsize=16)
plt.ylabel("${\mathbf x}_{2,t+1}$",fontsize=16)
plt.savefig("q_demo_Narendra\\optimal_q.pdf")



for i in ["Original","q_EPF","q_UPF","q_IPF","q_CINN_PF"]:
    p = q[i]
    fig, ax = plt.subplots(figsize=[6, 4])
    plt.subplots_adjust(top=0.9,bottom=0.15,left=0.13,right=0.97,hspace=0,wspace=0)
    ax.set_xlim([lim[1],lim[0]])
    ax.set_ylim([lim[3],lim[2]])
    ax.contourf(xx, yy, optimal_q ,strs1, alpha=0.8, cmap=plt.cm.Reds)
    strs2 = np.linspace(0.001,np.max(p)+0.05,8)
    strs2 = ((strs2*1000)//1)/1000
    CS1 = ax.contourf(xx, yy, p ,strs2, alpha=0.5, cmap=plt.cm.Blues)
    h1,_ = CS1.legend_elements()
    n = np.size(h1)
    strs = []
    for j in range(n):
        strs += ["%.3f"%CS1.levels[j]+"$\sim$"+"%.3f"%CS1.levels[j+1]]

    plt.legend(h1, strs,ncol=2,fontsize=9,loc='lower center' )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel("${\mathbf x}_{1,t+1}$",fontsize=16)
    plt.ylabel("${\mathbf x}_{2,t+1}$",fontsize=16)
    plt.savefig("q_demo_Narendra\\"+i+".pdf")







