import numpy as np
from scipy import stats
from numpy.random import rand,randn,seed,standard_t,multinomial
from copy import deepcopy
from numpy import c_,r_
from .Basic_functions import resampling_exp

from scipy import io
from time import time

from scipy.optimize import minimize


from scipy import io
from scipy.spatial import distance_matrix

import os

# State Space System
# x_{t} = f(x_{t-1},u_{t-1})+v ;  y_t = g(x_t)+e ; x_{0}\sim p
# arg: f,g,init,v_sig,e_sig
class system:
    def __init__(self,**arg):
        self.f = arg['f']
        self.g = arg['g']
        self.init = arg['init']
        self.x = self.init.sample(1) # x_{0}

    def generate(self,u= []):
        num = u.shape[0]
        x = np.zeros([ num+1,self.x.shape[1] ])
        x[0:1,:] = self.x
        y_ = self.g.sample(x = self.x)
        y = np.zeros([num+1,y_.shape[1]])
        y[0:1,:] = y_
        for t in range(num):
            self.x = self.f.sample( x = self.x , u = u[t:t+1,:] )
            y[t+1:t+2,:] = self.g.sample(x = self.x)
            x[t+1:t+2,:] = self.x
        return {'u':u,'y':y}, x
    
    def reset(self):
        self.x = self.init.sample(1)



#---------------A Basic function for EPF and UPF--------#
#Generate samples form Gaussian distributions
# data[i,:]  <-   N( expec[i,:] , cov[i,:,:] )
def multi_G(expec,cov):
    a,b = np.linalg.eig(cov)
    a = a.real
    b = b.real
    a[a<1e-24] = 1e-24 #for numerical safety
    e = stats.norm.rvs(size = a.shape)
    data = np.einsum('kj,klj->kl',e*np.sqrt(a), b) + expec
    a,b = np.linalg.slogdet(cov)
    c = a*b + expec.shape[1] * np.log(2*np.pi)
    ln_p = -(np.sum(np.square(e),1) + c)/2  
    return data,ln_p
#--------------------------------Filer----------------------------#

def PF(data,system,num=100):
    u = data['u']
    y = data['y']
    n = u.shape[0]
    x = system.init.sample(num)
    result = np.zeros([ n + 1,x.shape[1] ])*np.nan
    for i in range(n+1):
        ln_p = system.g.ln_p(x,y[ [i]*num ,:])
        m = np.max(ln_p)
        exp_weight = np.exp(ln_p-m+690)
        result[i,:] = exp_weight @ x/np.sum(exp_weight)
        index = resampling_exp(ln_p)
        x = x[index,:]
        if i<n:
            x =  system.f.sample(x,u[i,:])
    return result






def EPF(data,system,num=100):
    u = data['u']
    y = data['y']
    n = u.shape[0]
    P = system.f.noise.cov
    R = system.g.noise.cov
    x = system.init.sample(num)
    result = np.zeros([ n+1 ,x.shape[1] ])*np.nan
    ln_p = system.g.ln_p(x,y[ [0]*num ,:])
    ma = np.max(ln_p)
    exp_weight = np.exp(ln_p-ma+690)
    result[0,:] = exp_weight @ x/np.sum(exp_weight)
    index = resampling_exp(ln_p)
    x = x[index,:]
    for i in range(1,n+1):
        try:
            expectation_x = system.f.noise_free(x,u[i-1,:])
            expectation_y = system.g.noise_free(expectation_x)
            e =  y[[i]*num,:] - expectation_y
            Jacobian = system.g.Jacobian(expectation_x)#
            HTP = np.einsum('kli,lj->kij',Jacobian,P)
            M = np.einsum('kij,kjl->kil',HTP,Jacobian) + R
            K = np.linalg.solve(M,HTP)     
            exp_x = expectation_x + np.einsum('ki,kij->kj',e,K)
            COV = P - np.einsum('kji,kjl->kil',HTP,K)
            x_next,lnp = multi_G( exp_x , COV )
            lnp1 = system.f.ln_p(x,x_next,u[ [i-1]*num,: ]).real
            lnp2 = system.g.ln_p(x_next,y[[i]*num,:]).real
            ln_p = (lnp1 + lnp2 - lnp).real
            m = np.max(ln_p)
            exp_weight = np.exp(ln_p-m+690)
            result[i,:] = exp_weight @ x_next/np.sum(exp_weight)  
            index = resampling_exp(ln_p)
            x = x_next[index,:]
        except:
            print('Diverge')
            break
    return result





def UPF(data,system,num=100,a=1e-2,b=2,k = 3):
    u = data['u']
    y = data['y']
    n = u.shape[0]
    x = system.init.sample(num)
    result = np.zeros([ n + 1 , x.shape[1] ])*np.nan
    if np.sum(y[0,:]) is np.nan:
        result[0,:] = np.mean(x,0)
    else:
        ln_p = system.g.ln_p(x,y[ [0]*num ,:])
        ma = np.max(ln_p)
        exp_weight = np.exp(ln_p-ma+690)
        result[0,:] = exp_weight @ x/np.sum(exp_weight)
        index = resampling_exp(ln_p)
        x = x[index,:]
    m = x.shape[1]
    m_y = y.shape[1]
    P = system.f.noise.cov
    R = system.g.noise.cov
    lamb = np.square(a)*(n+k)-m
    [e,g] = np.linalg.eig(P)
    sqrt_P = np.diag( np.sqrt(e * (m+lamb) ) )@g.T
    sqrt_P = np.r_[sqrt_P,-sqrt_P]
    w_0_m = lamb/(lamb+m)
    w_0_c = w_0_m + 1 - np.square(a) + b
    w_i = 1/(2 * (m+lamb) )
    pred_y = np.zeros([num,m_y,2*m])
    for i in range(1,n+1):
        try:
            expectation_x = system.f.noise_free(x,u[i-1,:])
            expectation_y = system.g.noise_free(expectation_x)
            for j in range(2*m):
                pred_y[:,:,j] = system.g.noise_free(expectation_x + sqrt_P[j:j+1,:])
            y_predict = np.sum(pred_y,2)*w_i + expectation_y*w_0_m  ##
            d_y = (pred_y.T - y_predict.T).T
            d_y_0 = expectation_y - y_predict
            P_ZX = (d_y @ sqrt_P)*w_i
            P_ZZ = np.einsum('kji,kli->kjl',d_y,d_y)*w_i+ np.einsum('kj,kl->kjl',d_y_0,d_y_0)*w_0_c + R
            K = np.linalg.solve(P_ZZ,P_ZX)
            e =  y[[i]*num,:] - y_predict
            exp_x =  expectation_x + np.einsum('ki,kij->kj',e,K)
            COV = P - np.einsum('kjm,kji,kil->kml',K,P_ZZ,K)
            x_next,lnp = multi_G( exp_x , COV )
            lnp1 = system.f.ln_p(x,x_next,u[ [i-1]*num,: ]).real
            lnp2 = system.g.ln_p(x_next,y[[i]*num,:]).real
            ln_p = (lnp1 + lnp2 - lnp).real
            ma = np.max(ln_p)
            exp_weight = np.exp(ln_p-ma+690)
            result[i,:] = exp_weight @ x_next/np.sum(exp_weight)
            index = resampling_exp(ln_p)
            x = x_next[index,:]
        except:
            print('Diverge')
            break
    return result


def IPF(data,system,num = 10):
    u = data['u']
    y = data['y']
    n = u.shape[0]
    x = system.init.sample(num)
    result = np.zeros([ n + 1,x.shape[1] ])*np.nan
    v_q = np.zeros(x.shape)
    lnq = np.zeros([num])
    x_tild = np.zeros(x.shape)

    if np.sum(y[0,:]) is np.nan:
        result[i,:] = np.mean(x,0)
    else:
        ln_p = system.g.ln_p(x,y[ [0]*num ,:])
        m = np.max(ln_p)
        exp_weight = np.exp(ln_p-m+690)
        result[0,:] = exp_weight @ x/np.sum(exp_weight)
        index = resampling_exp(ln_p)
        x = x[index,:]
        
    for i in range(1,n+1):
        if np.sum(y[i,:]) is np.nan:
            result[i,:] = np.mean(x,0)
            x = system.f.sample(x,u[i-1:i,:])
        else:
            yi = y[i,:]
            x_tild = system.f.noise_free(x = x, u  = u[i-1:i,:])
            norm = stats.norm.rvs(size = x.shape)
            lnq = -np.sum( np.square(norm),1 )/2
            for j in range(num):
                F = lambda c: system.F(x_tild = x_tild[j,:],y=yi,v=c)
                v0 = np.zeros([x.shape[1]])
                res = minimize(F, v0, method='BFGS')
                v0 = res.x
                H = res.hess_inv
                [e,g] = np.linalg.eig(H)
                v_q[j:j+1,:] = (norm[j:j+1,:] * np.sqrt(e)) @ g.T + v0
                lnq[j] -= np.sum(np.log(e))/2
            x = x_tild + v_q
            lnp = -system.F(x_tild = x_tild , y = yi , v = v_q)/2
            ln_p = lnp - lnq
            m = np.max(ln_p)
            exp_weight = np.exp(ln_p-m+690)
            result[i,:] = exp_weight @ x/np.sum(exp_weight)
            index = resampling_exp(ln_p)
            x = x[index,:]
    return result





    







