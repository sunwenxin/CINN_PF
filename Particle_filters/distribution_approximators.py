from .Basic_functions import Gaussian
import numpy as np

from scipy.optimize import minimize


def E_approximator(system,x = [],u = [],y_next = []):
    P = system.f.noise.cov
    R = system.g.noise.cov
    expectation_x = system.f.noise_free(x,u)
    expectation_y = system.g.noise_free(expectation_x)
    e =  y_next - expectation_y
    Jacobian = system.g.Jacobian(expectation_x)#
    HTP = np.einsum('kli,lj->kij',Jacobian,P)
    M = np.einsum('kij,kjl->kil',HTP,Jacobian) + R
    K = np.linalg.solve(M,HTP)     
    exp_x = expectation_x + np.einsum('ki,kij->kj',e,K)
    COV = P - np.einsum('kji,kjl->kil',HTP,K)
    f = Gaussian(exp_x,COV[0,:,:])
    return f

def U_approximator(system,x = [],u = [],y_next = []):
    transform = system.f
    observer = system.g
    P = transform.noise.cov
    R = observer.noise.cov
    a=1e-2
    b=2
    k = 3
    n,m = x.shape
    m_y = y_next.shape[1]
    lamb = np.square(a)*(n+k)-m
    [e,g] = np.linalg.eig(P)
    sqrt_P = np.diag( np.sqrt(e * (m+lamb) ) )@g.T
    sqrt_P = np.r_[sqrt_P,-sqrt_P]
    w_0_m = lamb/(lamb+m)
    w_0_c = w_0_m + 1 - np.square(a) + b
    w_i = 1/(2 * (m+lamb) )
    pred_y = np.zeros([1,m_y,2*m])
    expectation_x = transform.noise_free(x,u)
    expectation_y = observer.noise_free(expectation_x)
    for j in range(2*m):
        pred_y[:,:,j] = observer.noise_free(expectation_x + sqrt_P[j:j+1,:])
    y_predict = np.sum(pred_y,2)*w_i + expectation_y*w_0_m
    d_y = pred_y - y_predict.reshape([n,m_y,1])
    d_y_0 = expectation_y - y_predict
    P_ZX = (d_y @ sqrt_P)*w_i
    P_ZZ = np.einsum('kji,kli->kjl',d_y,d_y)*w_i+ (d_y_0.T@d_y_0)*w_0_c + R ###
    P_ZZ = P_ZZ[0,:,:]
    P_ZX = P_ZX[0,:,:]
    K = np.linalg.solve(P_ZZ,P_ZX)
    e =  y_next - y_predict
    exp_x =  expectation_x + e@K
    COV = P - K.T@P_ZZ@K
    f = Gaussian(exp_x,COV)
    return f


def I_approximator(system,x = [],u = [],y_next = []):
    x_tild = system.f.noise_free(x = x, u  = u)
    F = lambda c: system.F(x_tild = x_tild[0,:],y = y_next[0,:],v=c)
    v0 = np.zeros([x.shape[1]])
    res = minimize(F, v0, method='BFGS')
    v0 = res.x
    v0 = v0.reshape(1,v0.size)
    H = res.hess_inv
    f = Gaussian(v0+x_tild , H)
    return f



