from scipy import stats
from numpy import c_
import numpy as np

#---------------Joint Gaussian Distribution--------------
class Gaussian:
    # arg[0]:mean; arg[1]:cov
    def __init__(self,*arg):
        self.mean = arg[0]
        self.cov = arg[1]
        self.dim = self.mean.shape[1]
        a,b = np.linalg.slogdet(self.cov)
        self.ln_c = (a*b + self.dim * np.log(2*np.pi))
        [e,g] = np.linalg.eigh(self.cov)
        self.inv_cov = g@np.diag(1/e)@g.T
        self.sqrt_cov = np.diag(np.sqrt(e))@g.T
        self.sqrt_invcov = np.diag(1/np.sqrt(e))@g.T
        self.cov_size = self.cov.size

        
    def ln_p(self,x):
        e = x - self.mean
        return (-np.sum( (e@self.inv_cov) *e,1)-self.ln_c)/2
    def p(self,x):
        return np.exp(self.ln_p(x))
    def sample(self,num,return_lnp = False):
        z = stats.norm.rvs(size = [num,self.dim])
        result = z@self.sqrt_cov+self.mean
        if return_lnp:
            lnp = -(  np.sum( np.square(z),1) + self.ln_c )/2
            return result,lnp
        else:
            return result

