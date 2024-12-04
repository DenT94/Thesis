import numpy as np
import scipy.special as sps
import scipy.optimize as spo
from scipy.integrate import odeint
from numba import jit
import scipy

from itertools import product
import inspect
###############################################
#antigenic functions 
##antigenic effect distribution
rho_d = lambda d,p_alpha: p_alpha*(1-p_alpha)**(d-1)

#scale of the antigenic effect distribution
def xi_d(p_alpha):
    return -1/np.log(1-p_alpha)

c_d = lambda d, xi: np.exp(-d/xi)

###############################################
#epidemiological functions
def y_peak(R0,kappa=0):
    if kappa==0:
        return (R0 - 1 - np.log(R0))/R0
    else:
        return (R0-1)/kappa

def z_peak(R0, kappa=0):
    if kappa==0:
        return np.log(R0)/R0
    else:
        return R0/kappa * np.log( (R0-1)/(R0**2)*kappa)-(R0-1)/kappa

def x_peak(R0, kappa=0):
    if kappa ==0:
        return 1/R0
    else:
        return 1- R0/kappa * np.log( (R0-1)/(R0)*kappa)

def p_est_neutral(R0):
    return (R0-1)/R0

def x_inf_bigR0(R0):
    return np.exp(-R0)

def z_inf_bigR0(R0):
    return 1- np.exp(-R0)

def t_peak (R0, N):
    return np.log(1+ N *(1-1/R0) *np.log(R0))/(R0-1)
t_peaks = np.vectorize(t_peak)

def x_inf(R0, k):
    if k>0:
        c = (R0-k+R0*k)/(k*(R0-k))
        rho = 1/(R0/k)
        y_x = lambda x: c*x**rho + x/(rho-1) - 1/k

        roots = spo.fsolve(y_x, 0)
        return roots[0]
    else:
        return -1/R0*np.real(sps.lambertw(-R0*np.exp(-R0)))

def T_ext_deterministic(R0,kappa, N):
    return np.log(N)*(1/(R0-1) + 1/(1-R0*x_inf(R0,kappa)))


#define s_d_infinity as R0*(1-x_inf(R0,kappa))*(1-exp(-d/xi))
def s_d_infinity(R0, d, xi,kappa=0):
    return R0*(1-x_inf(R0,kappa))*(1-np.exp(-d/xi))


#intersection time functions
def tx_func(x,s):
    return np.log((1-x)/x)/s

def binary_entropy(x):
    return -x*np.log(x) - (1-x)*np.log(1-x)

def integral_tx(xmax,xmin,):
    return binary_entropy(xmax)-binary_entropy(xmin)

def averaged_tx(xmax,xmin,s,Z):
    #Z is the partition function of x^-2
    return integral_tx(xmax,xmin)/(s*Z)

def averaged_tx(xmax,xmin,s):
    return integral_tx(xmax,xmin)/s

def tx_functional(d,R0,mu,xi,xi_d, N, x_func, s_func = s_d_infinity):
    #check that x_func is a function
    if not inspect.isfunction(x_func):
        raise TypeError('x_func is not a function')    
    return tx_func(x_func(d,R0,mu,xi,xi_d, N),s_func(R0, d, xi))

#vectorize tx_functional for d
tx_functional_vec = np.vectorize(tx_functional, excluded=['R0','mu','xi','xi_d','N','x_func','s_func'])



#attempt to define number of mutants per epidemic

def N_mut_at_peak(R0,mu, N):
    return N*mu* z_peak(R0)*p_est_neutral(R0)

##substitute Npop with N in the next expression: (-1 + (1 + Npop*(1 - 1/R0)*Log(R0))*(1 + (-1 + R0)*Log(1 + Npop*(1 - 1/R0)*Log(R0)) + (-1 + R0)*Log((-1 + R0)/(-1 + R0*(1 + Npop*(1 - 1/R0)*Log(R0))))))/(-1 + R0)
Nmut_single_epidemic = lambda R0,mu, N: -(((R0 + N*(-1 + R0)*np.log(R0))*(np.log(1/(1 + N*np.log(R0))) + np.log(1 + (N*(-1 + R0)*np.log(R0))/R0)))/(R0))*mu
Nmut_single_epidemic_zero_order = lambda R0,mu, N: 0.5* mu*N*np.log(R0)*np.log(R0/(R0-1))*(R0-1)/R0
old_Nmut_single_epidemic = lambda R0,mu,N: N*mu* z_peak(R0)*p_est_neutral(R0)
###############################################
# luria dellbruck functions

def x_min_luria_dellbruck(R0,N):
    return 1/(N*y_peak(R0))

def x_max_luria_dellbruck(d, mu, R0, N, p_alpha):
    x_min = x_min_luria_dellbruck(R0,N)
    Nmut = N_mut_at_peak(R0,mu, N)

    return x_min* (1+ Nmut* rho_d(d,p_alpha))/(1+ 10*Nmut* rho_d(d,p_alpha)*x_min)

def x_max_of_Nmut(d,Nmut, R0, N, p_alpha):
    x_min = x_min_luria_dellbruck(R0,N)
    return x_min* (1+ Nmut* rho_d(d,p_alpha))/(1+ 10*Nmut* rho_d(d,p_alpha)*x_min)

def x_max_of_Nmut_alt(d,Nmut, R0, N, p_alpha):
    
    return ( Nmut* rho_d(d,p_alpha))/(1+ 10*Nmut* rho_d(d,p_alpha))

def p_x_given_d(x, d, R0, mu, N, p_alpha):
    Zd = Z_d(d, R0, mu, p_alpha,N)
    xmax =  x_max_luria_dellbruck(d, mu, R0, N, p_alpha)
    return np.power(x, -2)/Zd * (x<xmax)
#vectorized version of p_x_given_d with respect to first two 
p_x_given_d_vec = np.vectorize(p_x_given_d, excluded = [2,3,4,5])

##function that returns the average frequency of the antigenic effct by integrating x^-1 between xmin and xmax above defined
def avg_freq_func(d, R0,  mu, p_alpha, N):
    return (1/xi_d(p_alpha) + np.log(N* mu* np.log(R0)/R0 * (R0-1)/R0) + np.log(N) - np.log(R0) - np.log(2*np.e**(1/xi_d(p_alpha))*N* mu* np.log(R0)/R0 * (R0-1)/R0 + np.e**(d/xi_d(p_alpha))*xi_d(p_alpha)) + np.log(-1 + R0 - np.log(R0)))/(-2 - (np.e**((-1 + d)/xi_d(p_alpha))*xi_d(p_alpha))/N* mu* np.log(R0)/R0 * (R0-1)/R0 + (N*(-1 + R0 - np.log(R0)))/R0)

##functions obtained from the naive optimization, by using x(d) = mu rho(d)/(R0)
ld_func = lambda R0, mu, xi, xi_d, I0: (1 - xi - xi_d*np.log(((R0-1)*xi_d)/(mu*I0)) - xi*np.real(sps.lambertw(-(np.exp(-1 + 1/xi)/(((R0-1)*xi_d)/(mu*I0))**(xi_d/xi)),-1)))
ld_func_2 = lambda R0, mu, xi, xi_d, N: 1 - xi + xi_d*np.log((2*t_peak(R0,N)*mu)/xi_d) - xi*np.real(sps.lambertw((-(2**(xi_d/xi)*np.exp(-1 + 1/xi)*((t_peak(R0,N)*mu)/xi_d)**(xi_d/xi))),-1))

##
#partition functions
def Z_d (d, R0, mu, p_alpha, N):
    return 1/x_min_luria_dellbruck(R0,N) - 1/x_max_luria_dellbruck(d, mu, R0, N, p_alpha)

#define Z_d_beta as (s[d] (-(1 - xmin)^(1 - \[Beta]/s[d]) xmin^(-1 + \[Beta]/s[d]) + (1 - xmax[d])^(1 - \[Beta]/s[d])xmax[d]^(-1 + \[Beta]/s[d])))/(\[Beta] - s[d])
def Z_d_beta(d, R0, mu, p_alpha,xi, N, beta):
    x_min = x_min_luria_dellbruck(R0,N)
    x_max = x_max_luria_dellbruck(d, mu, R0, N, p_alpha)
    s_d = s_d_infinity(R0, d, xi)
    return np.maximum((s_d*(-(1-x_min)**(1-beta/s_d)*x_min**(-1+beta/s_d) + (1-x_max)**(1-beta/s_d)*x_max**(-1+beta/s_d)))/(beta-s_d),0)

def marginal_p_d(R0, mu, p_alpha,xi, N, beta,ds  = None):
    if ds is None:
        d_max = 20/p_alpha
        ds = np.arange(1,d_max)

    p_d_betas = np.array([rho_d(d,p_alpha)* Z_d_beta(d, R0, mu, p_alpha,xi, N, beta) for d in ds])
    return p_d_betas/np.sum(p_d_betas)

def gamma_func(p_alpha,beta,ds  = None):
    if ds is None:
        d_max = 20/p_alpha
        ds = np.arange(1,d_max)
    p_d_betas = np.array([rho_d(d,p_alpha)* np.power(d,beta) for d in ds])
    return p_d_betas/np.sum(p_d_betas)
    
def optimal_beta(b, kde_h, R0, mu, p_alpha,xi, N):
    d_kl_beta = lambda beta: scipy.stats.entropy(marginal_p_d(R0,mu,p_alpha,xi,N,beta= beta,ds=b),kde_h)
    beta_sols = np.logspace(-3,2,1000)
    # minimize(d_kl_beta,1)
    d_kls =np.array([d_kl_beta(beta) for beta in beta_sols])
    d_kls[np.isnan(d_kls)]=np.inf
    betamin = beta_sols[np.argmin(d_kls)]
    d_kl_min = np.min(d_kls)
    return betamin, d_kl_min

def optimal_beta_gamma(b, kde_h, p_alpha):
    d_kl_beta = lambda beta: scipy.stats.entropy(gamma_func(p_alpha, beta= beta,ds=b),kde_h)
    beta_sols = np.logspace(-3,2,1000)
    # minimize(d_kl_beta,1)
    d_kls =np.array([d_kl_beta(beta) for beta in beta_sols])
    d_kls[np.isnan(d_kls)]=np.inf
    betamin = beta_sols[np.argmin(d_kls)]
    d_kl_min = np.min(d_kls)
    return betamin, d_kl_min            

## geometric - family of distributions

def p_d_geometric_arb(R0, mu, p_alpha, xi, N, ds  = None, N_mutant_func = None, cutoff = None, soft_cutoff = None, bias = None):
    if N_mutant_func is None:
        N_mutant_func = Nmut_single_epidemic_zero_order
    else:
        #check that N_mutant_func is a function, with the 3 arguments R0, mu, N
        assert callable(N_mutant_func)
        assert len(inspect.signature(N_mutant_func).parameters) == 3

    psum = lambda d: -(((1 - p_alpha)**d*p_alpha*((1 - (1 - p_alpha)**d)**N_mutant_func(R0,mu, N) - ((1 - p_alpha)**(-1 + d)*p_alpha)**N_mutant_func(R0,mu, N)))/(-1 + (1 - p_alpha)**d + p_alpha))

    if cutoff is None:
        dmin =0
    else:
        dmin = xi * np.log((1-np.exp(-R0))*R0/(R0-1))
    
    if bias is None:
        bias = lambda d: 1

    if soft_cutoff is None:
        ps = np.array( [psum(d)*bias(d) if d>dmin else 0 for d in ds ])
    else:
        soft_cutoff = lambda d: np.exp((d-dmin)/xi)/(1+np.exp((d-dmin)/xi))
        ps = np.array( [psum(d)*soft_cutoff(d)*bias(d) for d in ds ])
    return ps/np.sum(ps)

def avg_d_geom_arb(filtered_df, R0,mu,xi, p_alpha,N, N_mutant_func = None, cutoff = None, soft_cutoff = None, bias = None):
    qstring= f'xi=={xi} and mutation_rate=={mu} and infection_rate=={R0}'
    dM = int(filtered_df.query(qstring)['first_d'].max())
    ds = np.arange(1,dM)
    h,b = np.histogram(filtered_df.query(qstring)['first_d'], bins=ds,density=True)
    p_geom= p_d_geometric_arb(R0,mu,p_alpha,xi,N,ds=b, N_mutant_func=N_mutant_func, cutoff = cutoff, soft_cutoff = soft_cutoff, bias = bias)
    return np.sum(b*p_geom)


def p_d_geometric_est(R0, mu, p_alpha, xi, N, ds  = None):
    psum = lambda d: -(((1 - p_alpha)**d*p_alpha*((1 - (1 - p_alpha)**d)**Nmut_single_epidemic_zero_order(R0,mu, N) - ((1 - p_alpha)**(-1 + d)*p_alpha)**Nmut_single_epidemic_zero_order(R0,mu, N)))/(-1 + (1 - p_alpha)**d + p_alpha))
    pest = lambda d: 1 - 1/(1+s_d_infinity(R0, d, xi))

    ps = np.array( [psum(d)*pest(d) for d in ds])
    #correct version of last line
    return ps/np.sum(ps)

def simple_p_fix(R0, mu, p_alpha,xi, N, beta,ds  = None):
    if ds is None:
        d_max = 20/p_alpha
        ds = np.arange(1,d_max)

    #implement Ztot = sum rho(d) Z_d_beta(d, R0, mu, p_alpha,xi, N, beta) * s(d)
    p_d_fix = np.array([rho_d(d,p_alpha)* Z_d_beta(d, R0, mu, p_alpha,xi, N, beta) * s_d_infinity(R0, d, xi) for d in ds])
    return p_d_fix/np.sum(p_d_fix)
#Ztot = sum rho(d) Z_d_beta(d, R0, mu, p_alpha,xi, N, beta) * s(d)

def average_d_fix(R0, mu, p_alpha,xi, N, beta,ds  = None):
    if ds is None:
        d_max = 20/p_alpha
        ds = np.arange(1,d_max)
    p_d_fix = simple_p_fix(R0, mu, p_alpha,xi, N, beta,ds)
    return np.sum(p_d_fix*ds)



#useful for plotting
color_dict =lambda x, cm: dict(zip(x, cm(np.linspace(0,1,len(x)))))

###############################################
#ODEs and associated

def model(X, t, R0, kappa):
    x = X[0]
    y = X[1]
    z = X[2]
    dxdt = - R0/(1+kappa * y) *x * y
    dydt =  R0/(1+kappa* y) * x * y - y
    dzdt = y
    return [dxdt, dydt, dzdt]

fit_d= lambda y,z,R0,kappa, d,xi : R0/(1+kappa*y)*(1-y-c_d(d,xi)*z)-1
derfit_d = lambda x,y,z,R0,kappa,d,xi :(R0*y*((1 + kappa)*(1 - R0*x + kappa*y) + c_d(d,xi)*(-(1 + kappa*y)**2 + kappa*(-1 + R0*x - kappa*y)*z)))/(1 + kappa*y)**3

###############################################
#two strain models
def two_strain_model(X,t, c, R0, kappa):
    x = X[0]
    y = X[1]
    z = X[2]

    x_child = X[3]
    y_child = X[4]
    z_child = X[5]

    dxdt_self_parent = - R0/(1+kappa * (y_child+y)) *x * y
    dxdt_self_child = - R0/(1+kappa * (y_child+y)) *x_child * y_child

    dxdt = dxdt_self_parent - c* y_child - c*dxdt_self_child
    dydt =  R0/(1+kappa* (y_child+y)) * x * y - y
    dzdt = y

    dxdt_child = dxdt +(1-c)*y  +dxdt_self_child
    dydt_child = R0/(1+kappa* (y_child+y)) * x_child * y_child - y_child
    dzdt_child = y_child

    return [dxdt, dydt, dzdt, dxdt_child, dydt_child, dzdt_child]

def two_strain_integration(c, t0,R0,kappa,N, I0=10):
    X0 = (1-I0/N, I0/N, 0, 1-I0/N,0,0)
    T = T_ext_deterministic(R0,kappa,N)
    ts= np.linspace(0,T,int(1e3))
    X = odeint(two_strain_model, X0, ts[ts<=t0], args= (c,R0,kappa))
    
    X0_new = X[-1] + np.array((0,-1/N,0, 0, 1/N,0))

    Xn = odeint(two_strain_model, X0_new, ts[ts>t0], args= (c,R0,kappa))

    return ts, np.concatenate((X,Xn))


def find_ind_star (d, x, y, z, ts, R0, kappa,xi, t_lim, dt):
    fitd = fit_d(y,z,R0,kappa, d,xi)
    p_surv_d = p_surv(fitd,ts, t_lim,dt)
    fit0 = fit_d(x,y,R0,kappa, 0 ,xi)
    ind_star= int(find_ind_intersection(p_surv_d,((fitd-fit0)/(1+fitd)),verbose=False))
    return ind_star

def saddle_solution(d, x, y, z, ts, R0, kappa,xi, t_lim, dt):
    fitd = fit_d(y,z,R0,kappa, d,xi)
    p_surv_d = p_surv(fitd,ts, t_lim,dt)
    fit0 = fit_d(x,y,R0,kappa,0,xi)
    ind_star= int(find_ind_intersection(p_surv_d,((fitd-fit0)/(1+fitd)),verbose=False))
    return y[ind_star]*p_surv_d[ind_star]

def full_solution(d, x, y, z, ts, R0, kappa,xi, t_lim, dt):
    fitd = fit_d(y,z,R0,kappa, d,xi)
    p_surv_d = p_surv(fitd,ts, t_lim,dt)
    fit0 = fit_d(x,y,R0,kappa,0,xi)
    #ind_star= int(find_ind_intersection(p_surv_d,((fitd-fit0)/(1+fitd)),verbose=False))
    return dt*np.sum(y*p_surv_d)

def get_saddle_distribution(R0, kappa, xi, p_alpha,cutoff =1e-5):
    T = np.log(N)*(1/(R0-1) + 1/(1-R0*x_inf(R0,kappa)))
    d_max = 1 + np.log(cutoff)/np.log(1-p_alpha)

    ds = np.arange(1,d_max)
    bigT  = 3*T
    ts = np.linspace(0,bigT,int(1e3))
    dt = ts[1]-ts[0]
    X0 = [1-1/N, 1/N, 0]
    x,y,z = odeint( model, t= ts, args= (R0,kappa), y0= X0).T

    t_lim = bigT
    
    return np.array((ds,np.maximum(np.array([saddle_solution(d, x, y, z, ts, R0, kappa,xi, t_lim, dt) for d in ds]),0)))

def get_full_distribution(R0, kappa, xi, p_alpha, N,cutoff =1e-5, hor= 3):
    T = np.log(N)*(1/(R0-1) + 1/(1-R0*x_inf(R0,kappa)))
    d_max = 1 + np.log(cutoff)/np.log(1-p_alpha)

    ds = np.arange(1,d_max)
    bigT  = 3*T
    ts = np.linspace(0,bigT,int(1e3))
    dt = ts[1]-ts[0]
    X0 = [1-1/N, 1/N, 0]
    x,y,z = odeint( model, t= ts, args= (R0,kappa), y0= X0).T

    t_lim = T*hor
    
    return np.array((ds,np.maximum(np.array([full_solution(d, x, y, z, ts, R0, kappa,xi, t_lim, dt) for d in ds]),0)))

def get_d_star_saddle(R0,kappa,xi,p_alpha,cutoff =1e-6):
    ds, saddle = get_saddle_distribution(R0,kappa,xi,p_alpha)
    p_tot = rho_d(ds,p_alpha)*saddle
    return ds[np.argmax(p_tot)]

def get_d_star_full(R0,kappa,xi,p_alpha, N,cutoff =1e-6):
    ds, saddle = get_full_distribution(R0,kappa,xi,p_alpha, N)
    p_tot = rho_d(ds,p_alpha)*saddle
    return ds[np.argmax(p_tot)]

def get_avg_d_saddle(R0,kappa,xi,p_alpha,cutoff =1e-6):
    ds, saddle = get_saddle_distribution(R0,kappa,xi,p_alpha)
    p_tot = rho_d(ds,p_alpha)*saddle
    p_tot /= np.sum(p_tot)
    return np.sum(ds*p_tot)

def get_avg_d_full(R0,kappa,xi,p_alpha,cutoff =1e-6):
    p_tot = rho_d(ds,p_alpha)*saddle
    return ds[np.argmax(p_tot)]

def get_d_star_full(R0,kappa,xi,p_alpha,N,cutoff =1e-6, ):
    ds, saddle = get_full_distribution(R0,kappa,xi,p_alpha,N)
    p_tot = rho_d(ds,p_alpha)*saddle
    return ds[np.argmax(p_tot)]

def get_avg_d_saddle(R0,kappa,xi,p_alpha,cutoff =1e-6, ):
    ds, saddle = get_saddle_distribution(R0,kappa,xi,p_alpha)
    p_tot = rho_d(ds,p_alpha)*saddle
    p_tot /= np.sum(p_tot)
    return np.sum(ds*p_tot)

def get_avg_d_full(R0,kappa,xi,p_alpha,N,cutoff =1e-6, ):
    ds, saddle = get_full_distribution(R0,kappa,xi,p_alpha,N)
    p_tot = rho_d(ds,p_alpha)*saddle
    p_tot /= np.sum(p_tot)
    return np.sum(ds*p_tot)

def get_avg_num_est_full(R0,kappa,xi,p_alpha, mu,N,cutoff =1e-6, hor=3):
    ds, saddle = get_full_distribution(R0,kappa,xi,p_alpha,N,cutoff =cutoff ,hor=hor)
    p_tot = rho_d(ds,p_alpha)*saddle

    return mu*N* np.sum(p_tot)


###############################################
#probability of establishment

## extinction integral
@jit(nopython=True)
def I_t_numba(fit, ts,  t_lim, dt):
    L = len(fit[ts<t_lim])
    I_t = np.zeros_like(ts)    
    for i in range(L):
        Q=0
        t1 = dt*i
        rho_t = dt*(np.cumsum(-fit[i:L+1])- (-fit[i]) )
        for j in range(i,L):
            S = dt*np.exp(rho_t[j-i]) 
            Q+=S
        I_t[i]=Q
    return I_t

## survival probability
def p_surv(fit, ts,t_lim,dt):
    I_t =np.array(I_t_numba(fit,ts,t_lim,dt))
    return 1- I_t/(1+I_t)
            
def product_sum_probas(x,y,z , t, t_lim, R0, kappa ,xi, p_alpha,ds, f_d = None):
    if f_d is None:
        f_d = fit_d

    dt= t[1]-t[0]
    product_sum = 0
    
    L = len(ds)
    for i in range(L):
        d= ds[i]
        fit = fit_d(y,z,R0,kappa,d,xi)
        p_surv_t = p_surv(fit,t,t_lim, dt)
        product_sum +=rho_d(d,p_alpha)*p_surv_t
    
    return product_sum

###############################################
#children trajectories
@jit(nopython=True)
def get_child_trajectories_d_tb_t(expcumfitarray,ds, N,ts,T):
    L = len(ts[ts<T])

    ndtbt = np.zeros((len(ds),L,len(ts))) #(d, tb, t)
    for ind_birth in np.arange(L):
        t_birth = ts[ind_birth]
        for i,d in enumerate(ds):
            
            ndtbt[i,ind_birth,:] = np.minimum(1/expcumfitarray[i,ind_birth]*expcumfitarray[i]/N,1)*(ts>t_birth)
    return ndtbt

@jit(nopython=True)
def get_inds_intersections_d_tb(ndtbt,y,ds,ts,T):
    L = len(ts[ts<T])
    intersections_d_tb = np.zeros((len(ds),L))
    for ind_birth in np.arange(L):
        t_birth = ts[ind_birth]
        for i,d in enumerate(ds):
            intersections_d_tb[i,ind_birth] = int(find_ind_intersection(ndtbt[i,ind_birth,:],y))
    return intersections_d_tb

@jit(nopython=True)
def get_final_landscape(intersections_d_tb, proba_landscape,cumulative_landscape, mu,N, ts,T,ds):
    L = len(ts[ts<T])
    final_landscape = np.zeros_like(proba_landscape)
    for i,d in enumerate(ds):
        for ind_birth in range(L):
            weighting= np.ones(L)
            for ind_birth in np.arange(L):
                for j in range(i+1,len(ds)):
                    weighting[ind_birth]*= np.exp(-mu*N* cumulative_landscape[j,intersections_d_tb[i,ind_birth]])                
                final_landscape[i,:L]= proba_landscape[i,:L]*weighting
    return final_landscape

def calculate_final_landscape(R0, kappa, xi, p_alpha, mu, N, horizon =None ,cutoff =1e-5):
    T = np.log(N)*(1/(R0-1) + 1/(1-R0*x_inf(R0,kappa)))

    if horizon is None:
        horizon = 1
    d_max = 1 + np.log(cutoff)/np.log(1-p_alpha)

    ds = np.arange(1,30)
    bigT  = 3*T
    ts = np.linspace(0,bigT,int(1000))
    dt = ts[1]-ts[0]
    X0 = [1-1/N, 1/N, 0]
    x,y,z = odeint( model, t= ts, args= (R0,kappa), y0= X0).T
    proba_landscape = np.zeros((len(ds),len(ts)))
    expcumfitarray = np.zeros((len(ds),len(ts)))
    # fd = fit_d(y,z,R0,kappa,d,xi)
    # Fd= dt *np.cumsum(fd)
    
    for i,d in enumerate(ds):
        fd = fit_d(y,z,R0,kappa,d,xi)
        Fd= dt *np.cumsum(fd)

        expcumfitd =np.exp(Fd)
        expcumfitarray[i,:]=expcumfitd
        proba_landscape[i,:]= rho_d(d,p_alpha)*y*p_surv(fd,ts,t_lim = horizon*T, dt= dt)
    ind_lessthanT = np.arange(len(ts))[ts<T]
    ndtbt = get_child_trajectories_d_tb_t(expcumfitarray,ds, N,ts,T)
    intersections_d_tb = get_inds_intersections_d_tb(ndtbt,y,ds,ts,T).astype(int)
    cumulative_landscape = dt*np.cumsum(proba_landscape,axis=1)
    final_landscape = get_final_landscape(intersections_d_tb, proba_landscape,cumulative_landscape, mu, N, ts, T,ds)
    return ds, final_landscape

def get_avg_d_final(R0,kappa,xi,p_alpha,mu,N, horizon,cutoff =1e-6):
    ds, saddle = calculate_final_landscape(R0, kappa, xi, p_alpha, mu,N, horizon = horizon,cutoff=cutoff)
    p_tot = saddle.T
    p_tot /= np.sum(p_tot)
    return np.sum(ds*p_tot)

###############################################
#minimization targets
to_minimize_func = lambda d,xi_d, xi, R0, mu, N: (-((np.exp((1 - d)/xi_d + (np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d))*N*mu)/((1 - np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d)))*(-1 + np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d)))*xi_d*(xi_d - R0*xi_d))) - (np.exp((1 - d)/xi_d + (np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d))*N*mu*(1 + np.log(1 - np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d)))))/((-1 + np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d)))**2*xi_d*(xi_d - R0*xi_d)))/((1 - np.exp(-R0))*(1 - np.exp(-d/xi))*R0*(1/(-1 + np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d))) + (N*(-1 + R0 - np.log(R0)))/R0)) - (np.exp((1 - d)/xi_d + (np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d))*N*mu*(-((1 + np.log(1 - np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d))))/(-1 + np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d)))) - (N*(-1 + R0 - np.log(R0))*(1 + np.log(R0/(N*(-1 + R0 - np.log(R0))))))/R0))/((1 - np.exp(-R0))*(1 - np.exp(-d/xi))*(-1 + np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d)))**2*R0*xi_d*(xi_d - R0*xi_d)*(1/(-1 + np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d))) + (N*(-1 + R0 - np.log(R0)))/R0)**2) - (-((1 + np.log(1 - np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d))))/(-1 + np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d)))) - (N*(-1 + R0 - np.log(R0))*(1 + np.log(R0/(N*(-1 + R0 - np.log(R0))))))/R0)/(np.exp(d/xi)*(1 - np.exp(-R0))*(1 - np.exp(-d/xi))**2*R0*xi*(1/(-1 + np.exp((np.exp((1 - d)/xi_d)*N*mu)/(xi_d - R0*xi_d))) + (N*(-1 + R0 - np.log(R0)))/R0))
horrible_func = lambda d,xi_d, xi, R0, mu, N: (np.e**(R0 + d/xi - 1/xi_d)*R0*(np.e**(d/xi_d) - (np.e**((-1 + d)/xi_d)*R0**2*(np.e**(d/xi_d)*xi_d + (np.e**(1/xi_d)*N*(-1 + R0)*mu*np.log(R0))/R0**2))/ (N*(-1 + R0)*mu*(1 + (np.e**((-1 + d)/xi_d)*R0**2*xi_d)/(N*(-1 + R0)*mu*np.log(R0)))*np.log(R0)) -  np.e**(d/xi_d)*np.log(1 + (np.e**((-1 + d)/xi_d)*R0**2*xi_d)/(N*(-1 + R0)*mu*np.log(R0)))))/((-1 + np.e**R0)*(-1 + np.e**(d/xi))*N*(-1 + R0)*mu*(-2 + (N*(-1 + R0 - np.log(R0)))/R0 - (np.e**((-1 + d)/xi_d)*R0**2*xi_d)/(N*(-1 + R0)*mu*np.log(R0)))*np.log(R0)) -  (np.e**(R0 + (2*d)/xi)*((R0**2*(np.e**(d/xi_d)*xi_d + (2*np.e**(1/xi_d)*N*(-1 + R0)*mu*np.log(R0))/R0**2 -  (np.e**(d/xi_d)*xi_d + (np.e**(1/xi_d)*N*(-1 + R0)*mu*np.log(R0))/R0**2)*np.log(1 + (np.e**((-1 + d)/xi_d)*R0**2*xi_d)/(N*(-1 + R0)*mu*np.log(R0)))))/ (np.e**(1/xi_d)*N*(-1 + R0)*mu*np.log(R0)) - (N*(-1 + R0) + N*np.log(R0)*(-1 + np.log(-((N + R0 - N*R0 + N*np.log(R0))/R0))) +  (N + R0 - N*R0)*np.log(-((N + R0 - N*R0 + N*np.log(R0))/R0)))/R0))/((-1 + np.e**R0)*(-1 + np.e**(d/xi))**2*R0*xi*(-2 + (N*(-1 + R0 - np.log(R0)))/R0 - (np.e**((-1 + d)/xi_d)*R0**2*xi_d)/(N*(-1 + R0)*mu*np.log(R0)))) +  (np.e**(R0 + d/xi)*((R0**2*(np.e**(d/xi_d)*xi_d + (2*np.e**(1/xi_d)*N*(-1 + R0)*mu*np.log(R0))/R0**2 - (np.e**(d/xi_d)*xi_d + (np.e**(1/xi_d)*N*(-1 + R0)*mu*np.log(R0))/R0**2)*np.log(1 + (np.e**((-1 + d)/xi_d)*R0**2*xi_d)/(N*(-1 + R0)*mu*np.log(R0)))))/(np.e**(1/xi_d)*N*(-1 + R0)*mu*np.log(R0)) - (N*(-1 + R0) + N*np.log(R0)*(-1 + np.log(-((N + R0 - N*R0 + N*np.log(R0))/R0))) + (N + R0 - N*R0)*np.log(-((N + R0 - N*R0 + N*np.log(R0))/R0)))/R0))/((-1 + np.e**R0)*(-1 + np.e**(d/xi))*R0*xi*(-2 + (N*(-1 + R0 - np.log(R0)))/R0 - (np.e**((-1 + d)/xi_d)*R0**2*xi_d)/(N*(-1 + R0)*mu*np.log(R0)))) + (np.e**(R0 + d/xi + (-1 + d)/xi_d)*R0*((R0**2*(np.e**(d/xi_d)*xi_d + (2*np.e**(1/xi_d)*N*(-1 + R0)*mu*np.log(R0))/R0**2 - (np.e**(d/xi_d)*xi_d + (np.e**(1/xi_d)*N*(-1 + R0)*mu*np.log(R0))/R0**2)*np.log(1 + (np.e**((-1 + d)/xi_d)*R0**2*xi_d)/(N*(-1 + R0)*mu*np.log(R0)))))/(np.e**(1/xi_d)*N*(-1 + R0)*mu*np.log(R0)) - (N*(-1 + R0) + N*np.log(R0)*(-1 + np.log(-((N + R0 - N*R0 + N*np.log(R0))/R0))) + (N + R0 - N*R0)*np.log(-((N + R0 - N*R0 + N*np.log(R0))/R0)))/R0))/((-1 + np.e**R0)*(-1 + np.e**(d/xi))*N*(-1 + R0)*mu*(-2 + (N*(-1 + R0 - np.log(R0)))/R0 - (np.e**((-1 + d)/xi_d)*R0**2*xi_d)/(N*(-1 + R0)*mu*np.log(R0)))**2*np.log(R0))

#(np.e**(R0 + d/xi)*(-(np.e**(d/xi_d)/(np.e**(1/xi_d)*Nmut + np.e**(d/xi_d)*xi_d)) + (np.e**(d/xi_d)*(-1 + N*yP))/(np.e**(1/xi_d)*Nmut + np.e**(d/xi_d)*(-1 + N*yP)*xi_d)))/((-1 + np.e**R0)*(-1 + np.e**(d/xi))*R0) - (np.e**(R0 + (2*d)/xi)*(-np.log(np.e**(1/xi_d)*Nmut + np.e**(d/xi_d)*xi_d) + np.log(np.e**(1/xi_d)*Nmut + np.e**(d/xi_d)*(-1 + N*yP)*xi_d)))/((-1 + np.e**R0)*(-1 + np.e**(d/xi))**2*R0*xi) + (np.e**(R0 + d/xi)*(-np.log(np.e**(1/xi_d)*Nmut + np.e**(d/xi_d)*xi_d) + np.log(np.e**(1/xi_d)*Nmut + np.e**(d/xi_d)*(-1 + N*yP)*xi_d)))/((-1 + np.e**R0)*(-1 + np.e**(d/xi))*R0*xi)

def get_double_cumulative(frequency_df, beta, mu, p_alpha):
    ds= np.array(list(set(frequency_df.query('d>0')['d'])))
    bins = np.logspace(np.log10(frequency_df.min()['chi']), np.log10(.5),1000).flatten()
    qstring=f"beta == {beta} & mu=={mu} & d>0"

    double_cumulative = np.zeros((len(ds),len(bins)-1))
    for i,d in enumerate(ds):
        d_df = frequency_df.query(qstring).query(f'd>={d}')
        h_d,b_d = np.histogram(d_df['chi'],bins=bins,density=True)
        dx = np.diff(b_d)

        f_d = np.cumsum(dx*h_d)
        double_cumulative[i,:] = (1-p_alpha)**(d-1) *(1-f_d)
    return double_cumulative, bins[1:], ds

def get_xbar(frequency_df,beta,mu,p_alpha,N):
    double_cumulative, bins, ds = get_double_cumulative(frequency_df, beta, mu, p_alpha)
    return ds, bins[np.argmin(np.abs(double_cumulative-1/(Nmut_single_epidemic(beta,mu,N))),axis=1)]

###############################################
#utility functions
@jit(nopython=True)    
def find_t_intersection(f,g, t):
    #print(np.sign(f - g))
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    #print(idx)
    if len(idx)>0:
        return t[idx[0]]
    else:
        return np.inf

@jit(nopython=True)    
def find_ind_intersection(f,g, verbose=False):
    #print(np.sign(f - g))
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    if verbose: print(idx)
    if len(idx)>0:
        return int(idx[0])
    else:
        return -1

fmt = lambda x, pos: '{:.1f}'.format(x)

#find the endpoint of a substring inside a string
def find_end_substring(string,substring):
    return string.find(substring)+len(substring)