from scipy import io
from numpy import c_
from matplotlib import pyplot as plt
import numpy as np

dim=50
s_v = 2
pp = 1

data = io.loadmat('result2.mat')
term = data["term"][0,0]
l = ( data["list_EPF"][0,:]==500 )
result_EPF_ = data["result_EPF"][0:term,:,l]

l = ( data["list_UPF"][0,:]==500 )
result_UPF_ = data["result_UPF"][0:term,:,l]

l = ( data["list_CINN_PF"][0,:]==500 )
result_CINN_PF_ = data["result_CINN_PF"][0:term,:,l]





for s_v in [1,2]:
    j = ( s_v - 1 )*2 + pp - 1
    result_EPF = result_EPF_[:,j,0]
    result_UPF = result_UPF_[:,j,0]
    result_CINN_PF = result_CINN_PF_[:,j,0]

    
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
    flier_props = dict(marker=r'$\ast$', markerfacecolor='red',markeredgecolor='red', markersize=4)
    rcParams.update(pgf_config)
    fig = plt.figure(figsize=(6, 3))
    gs = fig.add_gridspec(
                      left=0.1, right=0.95, bottom=0.17, top=0.9,
                      wspace=0.1, hspace=0.08)
    ax = fig.add_subplot(gs[0])
    box = ax.boxplot([result_EPF,result_UPF,result_CINN_PF],
                     labels=['EPF', 'UPF','CINN-PF'],
                     flierprops=flier_props)


    for median in box['medians']:
        median.set(color='red', linewidth=0.5)
    plt.grid(ls="--")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('RMSE',fontsize = 16)
    plt.xticks(fontsize=16)
    #plt.subplots_adjust(top=0.9,bottom=0.17,left=0.1,right=0.97,hspace=0,wspace=0)
    plt.savefig("usetex"+str(s_v+6)+".pdf")



