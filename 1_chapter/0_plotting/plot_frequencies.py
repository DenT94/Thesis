import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter

from numba import jit
#import mathematical functions from analysis/mathematical_functions/funcs.py
from analysis.mathematical_functions.funcs import *

#matplotlib.use('TkAgg')
from matplotlib.lines import Line2D
plt.rc('mathtext', default='regular')

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

plt.rc('lines', linewidth=3.0)
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
plt.rcParams['axes.labelsize']=35
plt.rcParams['legend.fontsize']= 25
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['image.cmap']='coolwarm'
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['figure.titlesize'] = 40


fmt = lambda x, pos: '{:.1f}'.format(x)

def get_counts_df(df,Nsim):
    counts_df  =df.groupby(['mu','beta', 'N', 'p_alpha','d']).size().to_frame('counts').reset_index()
    counts_df[r'$\mu N$']=np.round(counts_df['mu']*counts_df['N'])
    counts_df[r'$R_0$']=counts_df['beta']
    counts_df[r'$d/\langle d \rangle $']=counts_df['d']*counts_df['p_alpha']
    counts_df['fracs']=counts_df['counts']/Nsim
    return counts_df

def plot_counts_by_antigenic_effect(frequency_df,ax, Nsim):
    counts_df = get_counts_df(frequency_df, Nsim)
    sns.lineplot(counts_df.query("d>0"),x=r'$d/\langle d \rangle $', y='counts', hue=r'$\mu N$',style=r'$R_0$',ax=ax)
    ax.set_yscale('log')
    # ax.set_xlabel(r'$d/\langle d \rangle $')
    # ax.set_ylabel(r'$N_ \mathrm{sim} \,N_\mathrm{m}\, \rho(d)$, counts')
    # ax.set_title(fr'Antigenic effect counts')
    ax.set_xlabel('')
    ax.set_ylabel('')
    return ax

#make a function that plots the ecdf of the frequency of the antigenic effect, with the appropriate colorbar, using the code above from ax[1]
# def plot_ecdf_by_antigenic_effect(frequency_df,ax, fig,beta,mu,N,p_alpha, N_sim):
#     counts_df = get_counts_df(frequency_df, N_sim)
#     bins = np.logspace(np.log10(frequency_df.min()), np.log10(.5),1000).flatten()

#     qstring=f"beta == {beta} & mu=={mu} & d>0"
#     sns.ecdfplot(frequency_df.query(qstring),x='chi',complementary=True,stat='proportion',hue='d',legend=None,ax=ax,palette = "turbo")
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     ax.set_ylim(top=1,bottom =1/counts_df['counts'].max())
#     ax.set_xlim(left= .9* x_min_luria_dellbruck(beta,N),  right=.5)
#     ds= np.array(list(set(frequency_df['d'])))
#     cmap = plt.cm.turbo  # define the colormap
#     cmaplist = [cmap(i) for i in range(cmap.N)]
#     cmaplist[0] = (.5, .5, .5, 1.0)
#     cmap = mpl.colors.LinearSegmentedColormap.from_list(
#         'Custom cmap', cmaplist, cmap.N)
#     bounds = ds*p_alpha
#     norm = mpl.colors.BoundaryNorm(ds*p_alpha, cmap.N)
#     ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
#     fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
#                     cax=ax2, orientation='vertical', label=r'$d/\langle d \rangle$', ticks=bounds[::14], boundaries=bounds[::14],format=FuncFormatter(fmt))#, format='%1f')
    
#     ax.plot(bins,bins[0]/bins,color='orange',linestyle=':')
#     print(bins[0])
#     ax.annotate(r'$x^{-1}$',(1e-2,1e3),color='orange',fontsize=40)
#     ax.set_xlabel(r'$x$, frequency')
#     ax.set_ylabel(r'$N_\mathrm{sim} \,N_\mathrm{m}\, P_d(X>x)$, counts')
#     ax.set_title(fr'Frequency counts, $R_0={beta}\; \mu N= {mu*N:.0f}$')
#     return ax

def my_ecdf_plot(p_alpha, qstring,frequency_df, ax, cm):
    ds= np.array(list(set(frequency_df.query('d>0')['d'])))
    bins = np.logspace(np.log10(frequency_df.min()['chi']), np.log10(.5),1000).flatten()
    colors_d = color_dict(ds,cm)
    for d in ds:
        d_df = frequency_df.query(qstring).query(f'd=={d}')
        h_d,b_d = np.histogram(d_df['chi'],bins=bins,density=True)
        dx = np.diff(b_d)
        f_d = np.cumsum(dx*h_d)
        ax.plot(b_d[1:],rho_d(d,p_alpha)*(1-f_d),label=f'd={d}',color=colors_d[d])
    
    ax.set_xscale('log')
    ax.plot(bins,bins[0]*p_alpha/bins,color='orange',linestyle='--')
    ax.set_yscale('log')

def my_double_ecdf_plot(p_alpha, qstring,frequency_df, ax, cm):
    ds= np.array(list(set(frequency_df.query('d>0')['d'])))
    bins = np.logspace(np.log10(frequency_df.min()['chi']), np.log10(.5),1000).flatten()
    colors_d = color_dict(ds,cm)
    for d in ds[::3]:
        d_df = frequency_df.query(qstring).query(f'd>={d}')
        h_d,b_d = np.histogram(d_df['chi'],bins=bins,density=True)
        dx = np.diff(b_d)
        f_d = np.cumsum(dx*h_d)
        ax.plot(b_d[1:],(1-p_alpha)**(d)*(1-f_d),label=f'd={d}',color=colors_d[d])
    
    ax.set_xscale('log')
    ax.plot(bins,2*bins[0]/bins,color='orange',linestyle='--')
    ax.set_yscale('log')

def plot_ecdf_by_antigenic_effect(frequency_df,ax, fig,beta,mu,N,p_alpha, N_sim):
    counts_df = get_counts_df(frequency_df, N_sim)
    bins = np.logspace(np.log10(frequency_df.min()['chi']), np.log10(.5),1000).flatten()

    qstring=f"beta == {beta} & mu=={mu} & d>0"
    my_ecdf_plot(p_alpha, qstring,frequency_df, ax, plt.cm.turbo)
    
    ax.axhline(1/(Nmut_single_epidemic(beta,mu,N)),color='black',linestyle='--')                 
    ax.set_xlim(left= .9* x_min_luria_dellbruck(beta,N),  right=.5)
    ax.set_ylim(bottom=1/(Nmut_single_epidemic(beta,mu,N)*N_sim))

    ds= np.array(list(set(frequency_df['d'])))
    cmap = plt.cm.turbo  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.5, .5, .5, 1.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = ds*p_alpha
    norm = mpl.colors.BoundaryNorm(ds*p_alpha, cmap.N)
    
    #make place for a colorbar and add it
    # cax = 
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=ax2, orientation='vertical', label=r'$d/\langle d \rangle$', ticks=bounds[::14], boundaries=bounds[::14],format=FuncFormatter(fmt))#, format='%1f')
    
    ax.annotate(r'$x^{-1}$',(1e-2,1e3),color='orange',fontsize=40)
    ax.set_xlabel(r'$x$, frequency')
    ax.set_ylabel(r'$ P_d(X>x)$, counts')
    ax.set_title(fr'Frequency counts, $R_0={beta}\; \mu N= {mu*N:.0f}$')
    return ax

def plot_double_ecdf_by_antigenic_effect(frequency_df,ax, fig,beta,mu,N,p_alpha, N_sim):
    counts_df = get_counts_df(frequency_df, N_sim)
    bins = np.logspace(np.log10(frequency_df.min()['chi']), np.log10(.5),1000).flatten()

    qstring=f"beta == {beta} & mu=={mu} & d>0"
    my_double_ecdf_plot(p_alpha, qstring,frequency_df, ax, plt.cm.turbo)
    
    ax.axhline(1/(Nmut_single_epidemic(beta,mu,N)),color='black',linestyle='--')                 
    ax.set_xlim(left= .9* x_min_luria_dellbruck(beta,N),  right=.5)
    ax.set_ylim(bottom=1/(Nmut_single_epidemic(beta,mu,N)*N_sim))

    ds= np.array(list(set(frequency_df['d'])))
    cmap = plt.cm.turbo  # define the colormap
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmaplist[0] = (.5, .5, .5, 1.0)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    bounds = ds*p_alpha
    norm = mpl.colors.BoundaryNorm(ds*p_alpha, cmap.N)
    ax2 = fig.add_axes([1., 0.1, 0.03, 0.8])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=ax2, orientation='vertical', label='', ticks=bounds[::14], boundaries=bounds[::14],format=FuncFormatter(fmt))#, format='%1f')
    
    ax.annotate(r'$\chi^{-1}$',(5e-3,.3),color='orange',fontsize=40)
    ax.annotate(r'$\frac{1}{N_{\mu,p}}$', (1e-2,2e-1),color='black',fontsize=40)
    # ax.set_xlabel(r'$x$, frequency')
    ax.set_xlabel('')
    # ax.set_ylabel(r'$ P_d(X>x ,\; D\geq d )$')
    ax.set_ylabel('')
    # ax.set_title(fr'Frequency spectrum, $R_0={beta}\; \mu N= {mu*N:.0f}$')
    return ax

    