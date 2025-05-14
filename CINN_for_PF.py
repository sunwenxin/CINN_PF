import numpy as np
from scipy import stats
from CINN_model import PCINN
from Particle_filters.Basic_functions import resampling_exp

class proposal_distribution:
    def __init__(self,system, nodes = [50]*5, a_f = ['LR'], load = False):
        self.f = system.f
        self.g = system.g
        self.system = system
        self.system.reset()
        self.sqrt_cov = self.system.f.noise.sqrt_cov
        self.inv_sqrt_cov = np.linalg.inv(self.sqrt_cov)
        self.constant = np.linalg.det(self.sqrt_cov)
        if load:
            self.mod = PCINN(load = load)
        else:
            self.mod = PCINN(nodes , a_f)

    def train(self,u = None,terms = 2000,batch = 100):
        input_value = []
        v_ = []
        for i in range(batch):
            result,x = self.system.generate(u = u)
            y = result['y']
            x_p = self.f.noise_free(x = x[0:-1,:],u = u)
            v = x[1::,:] - x_p
            v_ += [v @ self.inv_sqrt_cov]
            input_value += [ np.c_[x_p,y[1::,:]] ]
        input_value = np.vstack(input_value)
        v_ = np.vstack(v_)
        self.mod.train(input_value,v_,terms)

    def ln_p(self,u = None,x = None,y_next = None,x_next = None):
        x_p = self.f.noise_free(x = x,u = u)
        v = x_next-x_p
        v_ = v@self.inv_sqrt_cov
        input_value = np.c_[x_p,y_next]
        return self.mod.ln_p(input_value , v_ ) - self.constant
    def p(self,u = None,x = None,y_next = None,x_next = None):
        x_p = self.f.noise_free(x = x,u = u)
        v = x_next-x_p
        v_ = v@self.inv_sqrt_cov
        input_value = np.c_[x_p,y_next]
        return self.mod.p(input_value , v_ )
    def sample(self,u = None,x = None,y_next = None,return_lnp = False):
        x_p = self.f.noise_free(x = x,u = u)
        input_value = np.c_[x_p,y_next]
        if return_lnp:
            v_,lnp = self.mod.sample( input_value , return_lnp = return_lnp )
            v = v_@self.sqrt_cov
            return x_p+v , lnp - self.constant
        else:
            v_ = self.mod.sample( input_value , return_lnp = return_lnp )
            v = v_@self.sqrt_cov
            return x_p+v
    def save(self,Location = None):
        self.mod.save(Location)



def CINN_PF(data,system,p,num = 10):
    u = data['u']
    y = data['y']
    n = u.shape[0]
    x = system.init.sample(num)
    result = np.zeros([ n + 1,x.shape[1] ])*np.nan
    v_q = np.zeros(x.shape)
    lnq = np.zeros([num])
    x_tild = np.zeros(x.shape)

    if np.sum(y[0,:]) is np.nan:
        result[0,:] = np.mean(x,0)
    else:
        ln_p = system.g.ln_p(x,y[ [0]*num ,:])
        m = np.max(ln_p)
        exp_weight = np.exp(ln_p-m+690)
        result[0,:] = exp_weight @ x/np.sum(exp_weight)
        index = resampling_exp(ln_p)
        x = x[index,:]
        
    for i in range(1,n+1):
        try:
            x_tild = system.f.noise_free(x = x, u  = u[[i-1]*num,:])
            x,lnq = p.sample(u[ [i-1]*num ,:],x = x,y_next = y[ [i]*num ,:],return_lnp = True)
            v_q = x - x_tild

            lnp = -system.F(x_tild = x_tild , y = y[ [i]*num ,:] , v = v_q)
            ln_p = lnp - lnq

            index = resampling_exp(ln_p)
            x = x[index,:]
            result[i,:] = np.mean(x,0)
        except:
            print('Diverge')
            break
    return result


    
