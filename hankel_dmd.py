from scipy.linalg import *
from step import *
#def edmd():
'''
    Plots eigenvalues of an nxn matrix K that solves 
    the least squares problem,
        K_edmd = argmin_{K in R^{nxn}} || Y - K X ||^2, 
    where X and Y are nxn matrices with elements,
        X[i,j] = psi_i(u_j)
        Y[i,j] = psi_i(u_{j+1})
    u_{j+1} = step(u_j, 1)
    psi_i is a scalar function.
    Note that if psi_i were generic different 
    functions for each i, K would be the EDMD 
    matrix with the basis functions being psi_i. 

    Here we do Hankel DMD, which is a special case 
    of EDMD with different but interrelated psi_i. 
    Specifically, psi_i are functions 
    of the form 
        psi_i(u) = psi_k o step(u, i)
    where psi_k = psi_0 is a fixed function defined 
    inline. That is,
        psi_{i+1}(u) = psi_i o step(u,1)
     

'''
if __name__=="__main__":
    psi_k = lambda u: u[0]*u[1]#mean(u, axis=0)
    n = 1000
       
    u_trj = empty((2,2*n),dtype='complex')
    
    X = empty((n,n),dtype='complex')
    Y = empty((n,n),dtype='complex')
    G = empty((n,n),dtype='complex')
    A = empty((n,n),dtype='complex')


    n_bins = 10
    radius = hstack([[0.,0.25],linspace(0.5, 1, n_bins-4),[1.2,2.,12]])
    dist_eig = zeros(n_bins)
    
    fig, ax = subplots()
    
    circle = e**(1j*linspace(0,2*pi-1.e-6,1000))
    ax.plot(real(circle), imag(circle), 'g.', label=r'Unit circle on $\mathbb{C}^2$', ms=1)

    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.set_xlabel('Re axis', fontsize=30)
    ax.set_ylabel('Im axis', fontsize=30)
    ax.set_title('Eigenvalues of the Hankel DMD matrix',\
            fontsize=30)
    n_repeat = 1
    s = [0.5,0.5]
    for index in range(n_repeat):
        u = rand(2)
        u_trj = step(u, s, 2*n, 1)[:,:,0].T 
        f_trj = psi_k(u_trj)
        X = hankel(f_trj[:n],f_trj[n-1:-1])
        Y = hankel(f_trj[1:n+1],f_trj[n:])
        G = dot(X,X.T)
        A = dot(Y,X.T)
        K_edmd = dot(G, inv(A))
        l, _ = eig(K_edmd)
        indices = digitize(abs(l), radius)
        for i in range(indices.size):
            binno = indices[i] - 1
            if(binno >= n_bins):
                binno = n_bins - 1
            dist_eig[binno] += 1
        l = l.compress(abs(l) < 1.1) 
        ax.plot(real(l), imag(l), 'k.', \
                label=r'eigs of $K_{\rm h-dmd}$', ms=10)
    legend(fontsize=30,loc='lower left')
    dist_eig /= sum(dist_eig)
    print('Length of trajectory used for Hankel-DMD ',\
            n)
    print('Size of matrix obtained from Hankel-DMD,',\
            'K_{h-dmd}', K_edmd.shape)
    print('Distribution of eigenvalues of K_{h-dmd}:')
    for i in range(1,radius.size):
        print('%f < l < %f, %f' %(radius[i-1], \
                radius[i], dist_eig[i-1]*100),'%')
        
