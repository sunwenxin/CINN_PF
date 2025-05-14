#from Basic_functions.Gaussian import Gaussian
import numpy as np
from .system import system
from .Basic_functions import Gaussian


class f:
    def __init__(self,sigma, N = 20, F = 8,dt = 0.01):
        self.noise = Gaussian( np.array(np.zeros([1,N])) , sigma*np.eye(N) )
        self.F = F
        self.index1 = np.array([i for i in range(N)]) - 1
        self.index2 = (np.array([i for i in range(N)]) + 1)%N
        self.index3 = np.array([i for i in range(N)]) - 2
        self.N = N
        self.dt = dt
    def gradient(self, x = []):
        d = np.zeros([1,self.N])
        d = ( x[:,self.index2] - x[:,self.index3] ) * x[:,self.index1] - x + self.F
        return d

        
    def noise_free(self,x = [],u = []):
        h1 = self.gradient(x = x)
        h2 = self.gradient(x = x+h1*self.dt/2)
        h3 = self.gradient(x = x+h2*self.dt/2)
        h4 = self.gradient(x = x+h3*self.dt)
        return x + self.dt*(h1+2*h2+2*h3+h4)/6
    
    def sample(self,x = [],u = [],return_lnp = False):
        if not return_lnp:
            v = self.noise.sample(x.shape[0])
            return self.noise_free(x,u)+v
        else:
            v,lnp = self.noise.sample(x.shape[0],return_lnp = True)
            return self.noise_free(x,u)+v,lnp
    def p(self,x = [],x_next =[],u = []):
        x_ = self.noise_free(x = x,u = u)
        return self.noise.p(x_next - x_)
    def ln_p(self,x = [],x_next = [],u = []):
        x_ = self.noise_free(x = x,u = u)
        return self.noise.ln_p(x_next - x_)
        
        
            

class g:
    def __init__(self,sigma, N = 20,p = 2):
        self.noise = Gaussian( np.zeros([1,N]), sigma*np.eye(N))
        self.N = N
        self.index_ = np.array([i for i in range(N)])
        self.index = np.array([i for i in range(N)])-1
        self.index2 = ( np.array([i for i in range(N)])+1 )%N
        self.p = p

    def noise_free(self,x = []):
        y = x + np.sin(self.p*x[:,self.index]+self.p*x[:,self.index2])
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
        num = x.shape[0]
        d = self.p*np.cos(self.p*x[:,self.index]+self.p*x[:,self.index2])
        A = np.array( [np.eye( self.N )]*num )
        A[ :,self.index,self.index_ ] = d
        A[ :,self.index2,self.index_ ] = d
        return A
    
        
class L96(system):
    def __init__(self,sigma_v,sigma_e,sigma_x, N = 20,p = 1):
        self.inv_sigma_v = 1/(2*sigma_v)
        self.inv_sigma_e = 1/(2*sigma_e)
        f_ = f(sigma_v, N = N)
        g_ = g(sigma_e, N = N,p = p)
        init = Gaussian( np.zeros([1, f_.N]) , sigma_x*np.eye(f_.N) )
        super(L96,self).__init__(f = f_, g = g_, init = init)

    def F(self, x_tild = [], y = [],v = []):
        x = x_tild + v
        if x.ndim == 1:
            y_p = x + np.sin(self.g.p*x[self.g.index]+self.g.p*x[self.g.index2])
            result = np.sum( np.square(v) )* self.inv_sigma_v + np.sum( np.square(y-y_p) )* self.inv_sigma_e
        else:
            y_p = self.g.noise_free(x = x)
            result = np.sum(np.square(v),1)* self.inv_sigma_v / 2 + np.sum(np.square(y-y_p),1)* self.inv_sigma_e / 2
        return result
        


    
