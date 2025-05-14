import torch
import numpy as np
import warnings
import os
from torch.nn import Linear as Li
from torch import nn
#-------------------------Basic CINN Structure -------------------------#
class cond_INN(nn.Module):
    def __init__(self,*arg,location = False):
        super(cond_INN, self).__init__()
        Act = {'LR':nn.LeakyReLU,'T':nn.Tanh,'R':nn.ReLU,'S':nn.Sigmoid}
        [nods,a_f,x1_dim,x2_dim,u_dim] = arg
        self.layers = len(nods)
        if len(a_f) != self.layers:
            if len(a_f)==1:
                a_f *= self.layers
            else:
                warnings.warn("the length of (a_f) and (nods) should be equal")
                a_f = [ a_f[0] ]*self.layers
        self.modelstructure = {'x1_dim':x1_dim,'x2_dim':x2_dim,'a_f':a_f,'u_dim':u_dim,'nods':nods}
        self.t1 = nn.ModuleList( [torch.nn.Sequential( Li(x2_dim+u_dim, nods[i]),Act[a_f[i]](),Li(nods[i], x1_dim) ) for i in range(self.layers)] )
        self.g1 = nn.ModuleList( [torch.nn.Sequential( Li(x1_dim+u_dim, nods[i]),Act[a_f[i]](),Li(nods[i], x2_dim) ) for i in range(self.layers)] )#
        self.t2 = nn.ModuleList( [torch.nn.Sequential( Li(x2_dim+u_dim, nods[i]),Act[a_f[i]](),Li(nods[i], x1_dim) ) for i in range(self.layers)] )
        self.g2 = nn.ModuleList( [torch.nn.Sequential( Li(x1_dim+u_dim, nods[i]),Act[a_f[i]](),Li(nods[i], x2_dim) ) for i in range(self.layers)] )#

        self.S_input = torch.nn.Sequential( Li(u_dim, nods[0]*2),Act[a_f[0]]() )
        self.S_f = nn.ModuleList( [torch.nn.Sequential( Li(nods[i]*2, nods[i+1]*2),nn.BatchNorm1d(nods[i+1]*2),Act[a_f[i+1]]() ) for i in range(self.layers-1)] )
        self.S_output = Li(nods[-1]*2, u_dim)

    def forward(self, x1, x2,u):
        u_ =  self.S_input(u)
        for i in range(self.layers-1):
            u_ = self.S_f[i](u_)
        u = self.S_output(u_) + u
        
        for i in range(self.layers):
            input2 = torch.hstack( (x2,u) )
            s2 = self.g2[i]( input2 )*0.1
            y1 = x1 * torch.exp( s2 ) + self.t2[i]( input2 )
            input1 = torch.hstack( (y1,u) ) 
            s1 = self.g1[i]( input1 )*0.1
            y2 = x2 * torch.exp( s1 ) + self.t1[i]( input1 )
            x1 = y1
            x2 = y2
            if i==0:
                J = torch.sum(s1,dim = 1,keepdims=True) + torch.sum(s2,dim = 1,keepdims=True)
            else:
                J += torch.sum(s1,dim = 1,keepdims=True) + torch.sum(s2,dim = 1,keepdims=True)
        return torch.hstack( (x1,x2) ), J


    def inverse(self, y1, y2, u):
        u_ =  self.S_input(u)
        for i in range(self.layers-1):
            u_ = self.S_f[i](u_)
        u = self.S_output(u_) + u
        
        for i in range(self.layers):
            j = self.layers - i - 1
            input1 = torch.hstack( (y1,u) )
            s1 = self.g1[j]( input1 )*0.1
            x2 = ( y2 - self.t1[j]( input1 ) ) * torch.exp( -s1 )
            input2 = torch.hstack( (x2,u) )
            s2 = self.g2[j]( input2 )*0.1
            x1 = ( y1 - self.t2[j]( input2 ) ) * torch.exp( -s2 )
            y1 = x1
            y2 = x2
            if i==0:
                J = torch.sum(s1,dim = 1,keepdims=True) + torch.sum(s2,dim = 1,keepdims=True)
            else:
                J += torch.sum(s1,dim = 1,keepdims=True) + torch.sum(s2,dim = 1,keepdims=True)
        return torch.hstack( (x1,x2) ), J

#-------------------------------------------------------------------------------------
class model_structure(cond_INN):
    def __init__(self, *arg, c = [] ,Name = False):
        # White initialize
        self.m_c = torch.mean(c,0,keepdim=True)
        c_ = c-self.m_c 
        k = c_.T@c_/(c_.shape[0]-1) 
        [a,b] = torch.linalg.eigh(k)
        a = torch.sqrt(torch.abs(a + 1e-12))
        self.w_c = b/a 
        c_dim = self.w_c.shape[1]
        [nods,a_f,input_dim] = arg
        self.x1_dim = input_dim//2
        self.x2_dim = input_dim-self.x1_dim
        super(model_structure, self).__init__(nods,a_f,self.x1_dim,self.x2_dim,c_dim)
        self.__constant = np.log(2*np.pi)*input_dim
        self.m = input_dim
    def forward(self, x, c):
        c= (c - self.m_c)@self.w_c
        return super(model_structure, self).forward(x[:,0:self.x1_dim],x[:,self.x1_dim::], c) 

    def inverse(self,y, c):
        c= (c - self.m_c)@self.w_c
        return super(model_structure, self).inverse(y[:,0:self.x1_dim], y[:,self.x1_dim::], c) 

    def sample(self,c,return_lnp = False):
        c= (c - self.m_c)@self.w_c
        num = c.shape[0]
        z = torch.randn([num,self.m])
        x,J = self.inverse(z, c) 
        if return_lnp:
            log_p = J - (  torch.sum( torch.square(z),1,keepdims=True) + self.__constant)/2
            return x,log_p[:,0]
        else:
            return x

    def ln_p(self,x,c):
        c= (c - self.m_c)@self.w_c
        z,J = self.forward(x,c)
        result = J - (  torch.sum( torch.square(z),1,keepdims=True) +  self.__constant)/2
        result[torch.isnan(result[:,0]),0] = -torch.inf
        return result

    def p(self,x,c):
        return torch.exp(self.ln_p(x,c))
         

#-------------------CINN for probability density approximation------------#
class PCINN:
    def __init__(self,*arg,lam = 1e-12,lr = 1e-2,load = False):
        if load:
            self.mod = torch.load(load + "\\model.pth", weights_only=False)
            self.mod.eval()
        else:
            [nods,a_f] = arg
            self.lr = lr
            self.lam = lam
            self.nods = nods
            self.a_f = a_f
    def train(self,c,x,terms=2000):
        x = np.c_[x]
        c = np.c_[c]
        [num,m] = x.shape
        self.m = m
        self.__constant = np.log(2*np.pi)*m

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.astype('float32')
        x = torch.from_numpy(x).to(device)
        c = c.astype('float32')
        c = torch.from_numpy(c).to(device)

        best_mod = None
        best_result = float('inf')
        self.mod = model_structure(self.nods, self.a_f, m, c=c).to(device)

        self.optimizer = torch.optim.Adam(self.mod.parameters(), lr=self.lr, weight_decay=self.lam)
        # back up#
        epoch = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.mod.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, "checkpoint.pth")
        self.mod.train()
        #======#
        while epoch<terms:
            result = -torch.mean( self.mod.ln_p(x,c) )
            self.optimizer.zero_grad()
            result.backward()
            self.optimizer.step()
            if epoch%100 == 0:
                print('%d'%epoch+':%f'%result)
                if epoch == 0:
                    bottomline = result
                elif (result>bottomline) or (torch.isnan(result)):
                    checkpoint = torch.load("checkpoint.pth", map_location=device)
                    self.mod.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch = checkpoint['epoch']
                    self.lr *= 0.5
                    print("Learning rate is too large, let lr = "+str(self.lr))
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr
                else:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.mod.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                    }, "checkpoint.pth")
                    if result>0:
                        bottomline = result * 1.5
                    else:
                        bottomline = result / 1.5
            epoch += 1

        self.mod = self.mod.to("cpu")
        self.mod.m_c = self.mod.m_c.to("cpu")
        self.mod.w_c = self.mod.w_c.to("cpu")
        if os.path.exists("checkpoint.pth"):
            os.remove("checkpoint.pth")
        self.mod.eval()
    
    def ln_p(self,c,x):
        c = np.c_[c]
        x = np.c_[x]
        if c.shape[0] == 1 and x.shape[0] > 1:
            c = c[[0]*x.shape[0],:]
        x = x.astype('float32')
        x = torch.from_numpy(x)
        c = c.astype('float32')
        c = torch.from_numpy(c)
        return self.mod.ln_p(x,c).data.numpy()[:,0]

    def p(self,c,x):
        ln_p = self.ln_p(c,x)
        return np.exp(ln_p)


    def save(self, load):
        import os
        if not os.path.exists(load):
            os.makedirs(load)
        torch.save(self.mod, load + "\\model.pth")

    def load(self, load):
        self.mod = torch.load(load + "\\model.pth", weights_only=False)

    def sample(self,c,return_lnp = False):
        c = np.c_[c]
        c = c.astype('float32')
        c = torch.from_numpy(c)

        if return_lnp:
            x,lnp =  self.mod.sample(c,return_lnp = return_lnp)
            return x.data.numpy() , lnp.data.numpy()
        else:
            x =  self.mod.sample(c,return_lnp = return_lnp)
            return x.data.numpy()

        



if __name__ == '__main__':    
    from my_toolbox.benchmark import Cond_P_benchmark2 as Cond_P_benchmark
    from matplotlib import pyplot as plt
    mod = PCINN([50]*5,['LR'])

    num = 100000
    u = np.linspace(-np.pi,np.pi,num)
    x = Cond_P_benchmark(u.reshape([num,1]))
    mod.train( u,x, terms=10000 )

    given = 1.5
    k = 1.5#1.5
    u = np.zeros([250000,1]) + given
    xx, yy = np.meshgrid(  np.linspace(-k, k, 500,dtype='float32'),np.linspace(-k, k, 500,dtype='float32')  )
    z = np.c_[xx.ravel(), yy.ravel()]
    z = torch.from_numpy(z)
    ln_Z = mod.ln_p(u,z)
    Z = mod.p(u,z)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    plt.title('The distribution of $x$')
    plt.show()


    u = np.zeros([2500,1]) + given
    x = Cond_P_benchmark(u)
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.scatter(x[:,0],x[:,1],s=0.5,alpha=0.5)
    plt.show()

    mod.save('test')
    mod = PCINN(load = 'test')

    given = 0.5
    u = np.zeros([250000,1]) + given
    xx, yy = np.meshgrid(  np.linspace(-1.5, 1.5, 500,dtype='float32'),np.linspace(-1.5, 1.5, 500,dtype='float32')  )
    z = np.c_[xx.ravel(), yy.ravel()]
    z = torch.from_numpy(z)
    Z = mod.p(u,z)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    plt.title('The distribution of $x$')
    plt.show()


    u = np.zeros([2500,1]) + given
    x = Cond_P_benchmark(u)
    plt.scatter(x[:,0],x[:,1],s=0.5,alpha=0.5)
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.show()




    
    
