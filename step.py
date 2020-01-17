from numpy import *
from scipy import interpolate
def step(u, s=[0.,0.], n=1, m=1):
    # n: number of timesteps
    # m: number of initial conditions
    # s[0] = abs(lambda), s[1] = alpha
    # output size: (n+1)xdxm
    theta = lambda phi: 2*pi*phi - s[1]
    Psi = lambda phi: (1/pi)*arctan(s[0]*sin(theta(phi))\
            /(1. - s[0]*cos(theta(phi))))
    u_trj = empty((n+1,2,m))
    u_trj[0,0] = u[0]
    u_trj[0,1] = u[1]
    for i in range(n):
        x = u_trj[i,0]
        y = u_trj[i,1]

        psix = Psi(x)
        u_trj[i+1,0] = (2*x + y + psix) % 1
        u_trj[i+1,1] = (x + y + psix) % 1
    return u_trj
def dstep(u, s=[0.,0.], m=1):
    """
    Input info:
    m: number of initial conditions
    u.shape = (d, m)
    
    Output:
    Jacobian matrices at each u
    shape: mxdxd
    """

    d = u.shape[0]
    theta = lambda phi: 2*pi*phi - s[1]
    dtheta = 2*pi
    num_t = lambda phi:  s[0]*sin(theta(phi))
    den_t = lambda phi: 1. - s[0]*cos(theta(phi))
    t = lambda phi: num_t(phi)/den_t(phi)
    Psi = lambda t: (1/pi)*arctan(t)
    dnum_t = lambda phi: s[0]*cos(theta(phi))*dtheta
    dden_t = lambda phi: s[0]*sin(theta(phi))*dtheta
    dt = lambda phi: (den_t(phi)*dnum_t(phi) - \
            num_t(phi)*dden_t(phi))/\
            (den_t(phi)**2.0) 
    dPsi_dt = lambda t: 1.0/pi/(1 + t*t)
    dPsi = lambda phi: dPsi_dt(t(phi))*dt(phi)
   
    dPsix = dPsi(u[0])
    du1_1 = 2.0 + dPsix
    du2_1 = 1.0 + dPsix
    dTu_u = ravel(vstack([du1_1, ones(m), du2_1, \
            ones(m)]), order='F')
    dTu_u = dTu_u.reshape([-1,2,2])
    return dTu_u

def clvs(n,s=[0.,0.]):
    """
    Outputs are: 
    u_trj: primal trjectory shape:dx(n+1),
    v1_trj: 1st CLVs along u_trj, shape:dx(n+1)
    v2_trj: 2nd CLVs along u_trj, shape:dx(n+1)
    """

    u = rand(2)
    u_trj = step(u, s, n) #u_trj.shape = (n+1)x2x1
    u_trj = u_trj[:,:,0].T #u_trj.shape = 2x(n+1)
    n = n+1
    dstep_trj = dstep(u_trj, s, n) #u_trj.shape = 2xn

    v = rand(2,2)
    v /= linalg.norm(v, axis=0)
     
    v1_trj = empty((n,2))
    v2_trj = empty((n,2))
    r_trj = empty((n,2,2))
    l = zeros(2)
    for i in range(n):
        v = dot(dstep_trj[i],v)
        v,r = linalg.qr(v)
        l += log(abs(diag(r)))/n
        v1_trj[i] = v[:,0]
        v2_trj[i] = v[:,1]
        r_trj[i] = r

    c = array([0.,1.])
    for i in range(n-1,-1,-1):
        v2_trj[i] = c[0]*v1_trj[i] + c[1]*v2_trj[i]
        v2_trj[i] /= norm(v2_trj[i])
        c /= norm(c)
        c = linalg.solve(r_trj[i], c)

    print('Lyapunov exponents: ', l)
    return u_trj, v1_trj.T, v2_trj.T

def plot_clvs():
    fig, ax = subplots(1,1)
    s = [0.7, 0.3]
    eps = 5.e-2
    u, v1, v2 = clvs(1000,s)
    ax.plot(u[0], u[1], 'k.', ms=1)
    ax.plot([u[0] - eps*v1[0], u[0] + eps*v1[0]],\
            [u[1] - eps*v1[1], u[1] + eps*v1[1]],\
            lw=2.0, color='red')
    ax.plot([u[0] - eps*v2[0], u[0] + eps*v2[0]],\
            [u[1] - eps*v2[1], u[1] + eps*v2[1]],\
            lw=2.0, color='black')
    return fig,ax

def growth_factor(n=100,s=[0.,0.]):
    """
    z^i(u) := ||Df(u) V^i(u)||, where 
    V^i is the i-th CLV at u.
    Output:
    u : primal trajectory of shape dxn
    z1 : shape: n
    z2 : shape: n
    """
    u, v1, v2 = clvs(4*n,s) 
    # Take n from the middle
    u = u[:,2*n:3*n]
    v1 = v1[:,2*n:3*n] #shape:dxn
    v2 = v2[:,2*n:3*n] #shape:dxn
    v1 = reshape(v1.T, (n,2,1))
    v2 = reshape(v2.T, (n,2,1))
    df_trj = dstep(u,s,n) #shape:nxdxd
    df_v1 = matmul(df_trj, v1).reshape((\
            n,2))
    df_v2 = matmul(df_trj, v2).reshape((\
            n,2))
    z1 = norm(df_v1, axis=1) #shape:n 
    z2 = norm(df_v2, axis=1)
    return u,z1,z2

def plot_growth_factor(n=100,s=[0.,0.]):
    fig, ax = subplots(2,1)
    u, z1, z2 = growth_factor(n,s)
    ax[0].plot(u[1]+u[0],z1,'r.',ms=5.0)
    ax[1].plot(u[1]+u[0],z2,'b.',ms=5.0)


    #z1_fit = interpolate.interp2d(\
    #        u[0],u[1],z1)
    #z2_fit = interpolate.interp2d(\
    #        u[0],u[1],z2)
    #theta = linspace(0.,1.,20)
    #phi = linspace(0.,1.,20)
    #plot_z1 = ax[0].contourf(theta, phi, z1_fit(\
    #        theta,phi),linspace(0,1,20))
    #plot_z2 = ax[1].contourf(theta, phi, z2_fit(\
    #        theta,phi),linspace(0,1,20))
    #colorbar(plot_z1)
    stop
    return fig,ax

if __name__ == "__main__":
#def unstable_manifold():
    """
    Constructs a piece of 
    an unstable manifold
    """
    n = 100
    u = rand(2,n)
    s_r = [0.7,0.3]
    s_c = s_r[0]*exp(1j*s_r[1])
    s_cc = s_c.conjugate()
    z1_fixed = (1.0 + s_c)/\
            (1 + s_cc)
    z2_fixed = 1 - 1/z1_fixed
    u0_fixed = (1/pi)*arctan2(\
            z1_fixed.imag,\
            z1_fixed.real) % 1
    u1_fixed =(1/pi)*arctan2(\
            z2_fixed.imag,\
            z2_fixed.real) % 1
    u_fixed = array([u0_fixed,\
            u1_fixed]).reshape(2,1)
    print(step(u_fixed, s_r,\
            10,1))
    #m = 1000
    #u_trj = step(u,s,n,m) 
    

    









