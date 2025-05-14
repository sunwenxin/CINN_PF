#from Basic_functions.Gaussian import Gaussian
from .Basic_functions import Gaussian
import numpy as np
from .system import system



class f:
    def __init__(self,sigma,par = np.array([1.8,8,0.5])):
        cov = sigma*np.eye(2)
        self.par = np.r_[par,cov.reshape(cov.size)]
        self.noise = Gaussian( np.array([[0,0]]) , self.par[3::].reshape(2,2) )
    def noise_free(self,x = [],u = []):
        x1_ = (x[:,0:1]/(1+np.square(x[:,0:1]))+self.par[0])*np.sin(x[:,1:2])
        x2_ = x[:,1:2]*np.cos(x[:,1:2]) + x[:,0:1]*np.exp(-(np.square(x[:,0:1])+np.square(x[:,1:2]))/self.par[1]) + (u**3)/(1+u**2+self.par[2]*np.cos(x[:,0:1]+x[:,1:2]))
        return np.c_[x1_,x2_]
    def sample(self,x = [],u = [],return_lnp = False):
        if not return_lnp:
            v = self.noise.sample(x.shape[0])
            return self.noise_free(x,u)+v
        else:
            v,lnp = self.noise.sample(x.shape[0],return_lnp = True)
            return self.noise_free(x,u)+v,lnp
    def p(self,x = [],x_next =[],u = []):
        x = self.noise_free(x,u)
        return self.noise.p(x_next - x)
    def ln_p(self,x = [],x_next = [],u = []):
        x = self.noise_free(x,u)
        return self.noise.ln_p(x_next - x)

        
        
            

class g:
    def __init__(self,sigma,par = np.array([0.5,0.5])):
        cov = sigma*np.eye(1)
        self.par = np.r_[par,cov.reshape(cov.size)]
        self.noise = Gaussian(np.array([[0]]),self.par[2::].reshape([1,1]))

    def noise_free(self,x = []):
        y = x[:,0:1]/( 1+np.sin(x[:,1:2])*self.par[0] ) + x[:,1:2]/(1+np.sin(x[:,0:1])*self.par[1])
        return y

    def noise_free_torch(self, x = []):
        y = x[:,0:1]/( 1+torch.sin(x[:,1:2])*self.par[0] ) + x[:,1:2]/(1+torch.sin(x[:,0:1])*self.par[1])
        return y
        
        
    def sample(self , x = []):
        y = self.noise_free(x) + self.noise.sample(x.shape[0])
        return y
    
    def p(self,x = [],y = []):
        e = y - self.noise_free(x)
        return self.noise.p(e)
    
    def ln_p(self,x = [],y = []):
        e = y - self.noise_free(x)
        return self.noise.ln_p(e)
    
    def Jacobian(self,x = []):
        Jo = np.zeros([x.shape[0], 2,1])
        Jo[:,0:1,0] = 1/( 1+np.sin(x[:,1:2])*self.par[0] ) - x[:,1:2]*self.par[1]*np.cos(x[:,0:1])/( np.square(1+np.sin(x[:,0:1])*self.par[1]) )
        Jo[:,1:2,0] = 1/( 1+np.sin(x[:,0:1])*self.par[1] ) - x[:,0:1]*self.par[0]*np.cos(x[:,1:2])/( np.square(1+np.sin(x[:,1:2])*self.par[0]) )
        return Jo

    

        
class Narendra_Li(system):
    def __init__(self,sigma_v = [],sigma_e = [],sigma_x = [],par_f = np.array([1.8,8,0.5]),par_g=np.array([0.5,0.5])):
        self.inv_sigma_v = 1/(2*sigma_v)
        self.inv_sigma_e = 1/(2*sigma_e)
        f_ = f(sigma_v,par=par_f )
        g_ = g(sigma_e,par=par_g)
        init = Gaussian( np.array([[0,0]]) , sigma_x*np.eye(2) )
        super(Narendra_Li,self).__init__(f = f_, g = g_, init = init)

    def F(self, x_tild = [], y = [],v = []):
        x = x_tild + v
        if x.ndim == 1:
            y_p = x[0]/( 1+np.sin(x[1])*self.g.par[0] ) + x[1]/(1+np.sin(x[0])*self.g.par[1])
            result = np.sum( np.square(v) )* self.inv_sigma_v + np.sum( np.square(y-y_p) )* self.inv_sigma_e
        else:
            y_p = self.g.noise_free(x = x)
            result = np.sum(np.square(v),1)* self.inv_sigma_v / 2 + np.sum(np.square(y-y_p),1)* self.inv_sigma_e / 2
        return result




    
