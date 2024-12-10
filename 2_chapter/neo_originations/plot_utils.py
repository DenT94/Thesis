import numpy.random as rnd
# import networkx as nx
# from anytree import Node
from tqdm import tqdm
import numpy as np
import os
import sys
import statsmodels.api as sm
if(sys.version_info[1]<= 7):
    import pickle5 as pickle
else:
    import pickle

import scipy.stats as spst

import pandas as pd
import scipy.interpolate as spi
import scipy.optimize as so
from scipy.integrate import cumulative_trapezoid

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import neo_originations.utils as ut
from matplotlib.ticker import FuncFormatter

fmt = lambda x, pos: '{:.1f}'.format(x)
from numba import jit

#matplotlib.use('TkAgg')
from matplotlib.lines import Line2D
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
sns.set_theme(context='poster')
sns.set_style("white")
plt.rc('lines', linewidth=3.0)
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['axes.labelsize']=25
plt.rcParams['legend.fontsize']= 15
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['image.cmap']='coolwarm'
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['figure.titlesize'] = 30

sys.path.append('..')
# from src.evo_eq_model import *



home = os.path.expanduser("~")
project_path =  os.path.relpath("..")
if project_path not in sys.path:
    sys.path.append(project_path)

data_folder = project_path+'/data/external/'
os.makedirs(data_folder, exist_ok = True)

saves_folder = project_path+'/data/interim/'
os.makedirs(saves_folder, exist_ok = True)

flupredict_to_pango = {"1":"WT",
            "1A":"A",
            "1B":"B",
            "1C":"B.1",
            "1C.2A":"B.1",
            "1C.2A.3A":"B.1",
            "1C.2A.3A.4A":"B.1.427/429",
            "1C.2A.3A.4B":"B.1.351",
            '1C.2A.3A.4C':"B.1.526",
            "1C.2B":"B1.1",
            "1C.2B.3D":"B.1.1.7",
            "1C.2B.3G":"P.1",
            "1C.2B.3J":"BA",
            "1C.2B.3J.4D":"BA.1",
            "1C.2B.3J.4D.5A":"BA.1.1",
            "1C.2B.3J.4E":"BA.2",
            "1C.2B.3J.4E.5B": "BA.2.12.1",
            "1C.2B.3J.4E.5C":"BA.2.75",
            "1C.2B.3J.4E.5C.6A":"BA.2.75.2",
            "1C.2B.3J.4E.5C.6E":"BM.1.1",
            "1C.2B.3J.4E.5C.6I.7C":"CH.1.1",
            "1C.2B.3J.4E.5C.6F":"BN.1",
            "1C.2B.3J.4E.5O":"BA.2.86",
            "1C.2B.3J.4E.5O.6L":"JN.1",
            "1C.2B.3J.4E.5O.6L.7G":"JN.1.1",
            "1C.2B.3J.4E.5O.6L.7H":"JN.1.4",
            "1C.2B.3J.4E.5O.6L.7I":"JN.1.11",
            "1C.2B.3J.4F":"BA.4",
            "1C.2B.3J.4F.5D":"BA.4.6",
            "1C.2B.3J.4G":"BA.5",
            "1C.2B.3J.4G.5E":"BF.7",
            "1C.2B.3J.4G.5F":"BQ.1",
            "1C.2B.3J.4G.5F.6B":"BQ.1.1",
            "1C.2C":"B.1.177",
            "1C.2D":"B.1.1",
            "1C.2D.3E":"B.1.617.1",
            "1C.2D.3F":"B.1.617.2",
            "1C.2A.3I":"B.1.621",
            "1C.2B.3J.4E.5N":"XBB",
            "1C.2B.3J.4E.5N.6J":"XBB.1.5",
            "1C.2B.3J.4E.5N.6J.7F":"JD.1.1",
            "1C.2B.3J.4E.5N.6J.7D":"XBB.1.9",
            "1C.2B.3J.4E.5N.6J.7D.8A":"EG.5.1",
            "1C.2B.3J.4E.5N.6J.7D.8A.9A":"HK.3",
            "1C.2B.3J.4E.5N.6J.7D.8A.9B":"HV.1",
            "1C.2B.3J.4E.5N.6J.7E":"XBB.1.16",
            "1C.2B.3J.4E.5N.6K":"XBB.2.3"}

# sys.path.insert(1, project_path)
# output_folder= project_path+'/outputs/'

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
def find_ind_intersection(f,g):
    #print(np.sign(f - g))
    idx = np.argwhere(np.diff(np.sign(f - g))).flatten()
    #print(idx)
    if len(idx)>0:
        return idx[0]
    else:
        return np.inf

#replicate last cell, for a single chunk represented in a column
def plot_column_chunk(data_chunk, axes_chunk, facecolor_chunk, plot_counts = True):
    logbins = np.logspace(-6,0,100)
    
    if plot_counts:
        ax_chunk_counts = axes_chunk[-1]
    ax_chunk_density_syn = axes_chunk[1]
    ax_chunk_g_factor = axes_chunk[0]

    ##################### Plot densities
    h_syn_chunk, b_syn_chunk, p_syn_chunk = ax_chunk_density_syn.hist(
        data_chunk.query('nonsyn_weight == 0')['Sublineage_Freq'],
        weights=data_chunk.query('nonsyn_weight == 0')['syn_weight'],
        bins = logbins,
        label='chunk',
        density=True, 
        histtype='step',
        color='steelblue',
        lw= 3,
        cumulative= -1)

    h_nsyn_chunk, b_nsyn_chunk, p_nsyn_chunk = ax_chunk_density_syn.hist(
        data_chunk['Sublineage_Freq'],
        weights=data_chunk['nonsyn_weight'],
        bins = logbins,
        label='chunk',
        density=True,
        histtype='step',
        color='orange',
        lw= 3,
        cumulative= -1)

    h_nsyn_RBD_chunk, b_nsyn_RBD_chunk, p_nsyn_RBD_chunk = ax_chunk_density_syn.hist(
        data_chunk['Sublineage_Freq'],
        weights=data_chunk['S_RBD'],
        bins = logbins,
        label='chunk',
        density=True,
        histtype='step',
        color='darkred',
        lw= 3,
        cumulative= -1)

    h_nsyn_S_NonRBD_chunk, b_nsyn_S_NonRBD_chunk, p_nsyn_S_NonRBD_chunk = ax_chunk_density_syn.hist(
        data_chunk.query('S_RBD==0')['Sublineage_Freq'],
        weights=data_chunk.query('S_RBD==0')['S'],
        bins = logbins,
        label='chunk',
        density=True,
        histtype='step',
        color='green',
        lw= 3,
        cumulative= -1)

    h_nsyn_nonS_chunk, b_nsyn_nonS_chunk, p_nsyn_nonS_chunk = ax_chunk_density_syn.hist(
        data_chunk.query('S == 0')['Sublineage_Freq'],
        weights=data_chunk.query('S == 0')['nonsyn_weight'],
        bins = logbins,
        label='chunk',
        density=True,
        histtype='step',
        color='purple',
        lw= 3,
        cumulative= -1)

    ax_chunk_density_syn.set_xscale('log')
    ax_chunk_density_syn.set_yscale('log')

    ##################### Plot total counts
    if plot_counts:
        h_syn_chunk_counts, b_syn_chunk_counts, p_syn_chunk_counts = ax_chunk_counts.hist(
            data_chunk['Sublineage_Freq'],
            weights=data_chunk['syn_weight'],
            bins = logbins,
            label='chunk',
            density=False,
            histtype='step',
            color='steelblue',
            lw= 3,
            cumulative=-1)

        h_nsyn_chunk_counts, b_nsyn_chunk_counts, p_nsyn_chunk_counts = ax_chunk_counts.hist(
            data_chunk['Sublineage_Freq'],
            weights=data_chunk['nonsyn_weight'],
            bins = logbins,
            label='chunk',
            density=False,
            histtype='step',
            color='orange',
            lw= 3,
            cumulative=-1)

        h_nsyn_RBD_chunk_counts, b_nsyn_RBD_chunk_counts, p_nsyn_RBD_chunk_counts = ax_chunk_counts.hist(
            data_chunk['Sublineage_Freq'],
            weights=data_chunk['S_RBD'],
            bins = logbins,
            label='chunk',
            density=False,
            histtype='step',
            color='darkred',
            lw= 3,
            cumulative=-1)

        h_nsyn_S_NonRBD_chunk_counts, b_nsyn_S_NonRBD_chunk_counts, p_nsyn_S_NonRBD_chunk_counts = ax_chunk_counts.hist(
            data_chunk.query('S_RBD==0')['Sublineage_Freq'],
            weights=data_chunk.query('S_RBD==0')['S'],
            bins = logbins,
            label='chunk',
            density=False,
            histtype='step',
            color='green',
            lw= 3,
            cumulative=-1)

        h_nsyn_nonS_chunk_counts, b_nsyn_nonS_chunk_counts, p_nsyn_nonS_chunk_counts = ax_chunk_counts.hist(
            data_chunk.query('S == 0')['Sublineage_Freq'],
            weights=data_chunk.query('S == 0')['nonsyn_weight'],
            bins = logbins,
            label='chunk',
            density=False,
            histtype='step',
            color='purple',
            lw= 3,
            cumulative=-1)

        ax_chunk_counts.axhline(h_nsyn_RBD_chunk_counts[0], color='darkred', lw=3, linestyle='--')
        ax_chunk_counts.annotate(f'{h_nsyn_RBD_chunk_counts[0]:.1e}', xy=(1e-2,h_nsyn_RBD_chunk_counts[0]), xytext=(1e-2,h_nsyn_RBD_chunk_counts[0]), color='darkred', fontsize=35)
        ax_chunk_counts.axhline(h_nsyn_chunk_counts[0], color='orange', lw=3, linestyle='--')
        ax_chunk_counts.annotate(f'{h_nsyn_chunk_counts[0]:.1e}', xy=(1e-6,h_nsyn_chunk_counts[0]), xytext=(1e-6,1.5*h_nsyn_chunk_counts[0]), color='orange', fontsize=35)
        ax_chunk_counts.axhline(h_syn_chunk_counts[0], color='steelblue', lw=3, linestyle='--')
        ax_chunk_counts.annotate(f'{h_syn_chunk_counts[0]:.1e}', xy=(1e-2,h_syn_chunk_counts[0]), xytext=(1e-2,1.5*h_syn_chunk_counts[0]), color='steelblue', fontsize=35)

        ax_chunk_counts.set_xscale('log')
        ax_chunk_counts.set_yscale('log')

    else: #get counts via numpy histogram
        h_syn_chunk_counts, b_syn_chunk_counts = np.histogram(data_chunk['Sublineage_Freq'], bins = logbins, weights=data_chunk['syn_weight'])
        h_nsyn_chunk_counts, b_nsyn_chunk_counts = np.histogram(data_chunk['Sublineage_Freq'], bins = logbins, weights=data_chunk['nonsyn_weight'])
        h_nsyn_RBD_chunk_counts, b_nsyn_RBD_chunk_counts = np.histogram(data_chunk['Sublineage_Freq'], bins = logbins, weights=data_chunk['S_RBD'])
        h_nsyn_S_NonRBD_chunk_counts, b_nsyn_S_NonRBD_chunk_counts = np.histogram(data_chunk['Sublineage_Freq'], bins = logbins, weights=data_chunk['S_NonRBD'])
        h_nsyn_nonS_chunk_counts, b_nsyn_nonS_chunk_counts = np.histogram(data_chunk.query('S == 0')['Sublineage_Freq'], bins = logbins, weights=data_chunk.query('S == 0')['nonsyn_weight'])

    ##################### Plot g factors
    g_factor_nsyn = h_nsyn_chunk/h_syn_chunk
    g_factor_nsyn_RBD = h_nsyn_RBD_chunk/h_syn_chunk
    g_factor_nsyn_S_NonRBD = h_nsyn_S_NonRBD_chunk/h_syn_chunk
    g_factor_nsyn_N = h_nsyn_nonS_chunk/h_syn_chunk

    error_g_factor_nsyn_RBD = g_factor_nsyn_RBD*np.sqrt(1/h_nsyn_RBD_chunk_counts + 1/h_nsyn_RBD_chunk_counts[0] + 1/h_syn_chunk_counts + 1/h_syn_chunk_counts[0])
    error_g_factor_nsyn_S_NonRBD = g_factor_nsyn_S_NonRBD*np.sqrt(1/h_nsyn_S_NonRBD_chunk_counts + 1/h_nsyn_S_NonRBD_chunk_counts[0] + 1/h_syn_chunk_counts + 1/h_syn_chunk_counts[0])
    error_g_factor_nsyn_N = g_factor_nsyn_N*np.sqrt(1/h_nsyn_nonS_chunk_counts + 1/h_nsyn_nonS_chunk_counts[0] + 1/h_syn_chunk_counts + 1/h_syn_chunk_counts[0])

    ax_chunk_g_factor.plot(b_syn_chunk[:-1], g_factor_nsyn, label='nonsyn', color='orange', lw=3)
    ax_chunk_g_factor.plot(b_nsyn_chunk[:-1], g_factor_nsyn_RBD, label='nonsyn', color='darkred', lw=3)
    ax_chunk_g_factor.fill_between(b_nsyn_chunk[:-1], np.maximum(g_factor_nsyn_RBD-error_g_factor_nsyn_RBD,0), g_factor_nsyn_RBD+error_g_factor_nsyn_RBD, color='darkred', alpha=0.25)

    ax_chunk_g_factor.plot(b_nsyn_chunk[:-1], g_factor_nsyn_S_NonRBD, label='$\mathrm{S}\backslash\mathrm{RBD}$', color='green', lw=3)
    ax_chunk_g_factor.fill_between(b_nsyn_chunk[:-1], np.maximum(g_factor_nsyn_S_NonRBD-error_g_factor_nsyn_S_NonRBD,0), g_factor_nsyn_S_NonRBD+error_g_factor_nsyn_S_NonRBD, color='green', alpha=0.25)

    ax_chunk_g_factor.plot(b_nsyn_chunk[:-1], g_factor_nsyn_N, label='nonsyn', color='purple', lw=3)
    ax_chunk_g_factor.fill_between(b_nsyn_chunk[:-1], np.maximum(g_factor_nsyn_N-error_g_factor_nsyn_N,0), g_factor_nsyn_N+error_g_factor_nsyn_N, color='purple', alpha=0.25)


    #fit g factor for RBD in the range of 1e-3 to 1e-1 to linear function
    idx = int(find_ind_intersection(b_syn_chunk, 1e-3))
    idx2 = int(find_ind_intersection(b_syn_chunk, 1e-1))
    print(idx, idx2)
    fitfunc = lambda x, a, b: a*x+b

    try:
        fit, cov = so.curve_fit(fitfunc, np.log(b_syn_chunk[idx:idx2]), np.log(g_factor_nsyn_RBD[idx:idx2]))
        fitresult = np.exp(fit[1])*b_syn_chunk[1:]**fit[0]
        ax_chunk_g_factor.plot(b_syn_chunk[1:][np.logical_and(fitresult< g_factor_nsyn_RBD.max(),fitresult>=1)], fitresult[np.logical_and(fitresult< g_factor_nsyn_RBD.max(),fitresult>=1)], label='fit', color='darkred', lw=3,ls='--')
        a,b = fit
        print(f' {b:.2f} x^{a:.2f} fit for RBD g factor')
    except ValueError:
        print('Fitting unsuccessful')

    ax_chunk_g_factor.set_xscale('log')

    ax_chunk_density_syn.plot(logbins, np.maximum(np.minimum(1,50*logbins[0]/logbins),np.min(h_syn_chunk)), label='1', color='black', lw=3, linestyle='--')
    ax_chunk_density_syn.set_ylim(1e-5,2)

    # ax_chunk_counts.plot(logbins, np.maximum(np.minimum(1,20*logbins[0]/logbins), np.min(h_syn_chunk)), label='1', color='black', lw=3, linestyle='--')
    
    for a in axes_chunk:
        a.set_facecolor(facecolor_chunk)
        a.patch.set_alpha(0.25)

    return np.array([b_nsyn_chunk[:-1], g_factor_nsyn_RBD, error_g_factor_nsyn_RBD])

def get_mosaic(n_chunks):

    mosaic = np.zeros(( 9,2*n_chunks + 7), dtype=object)
    mosaic[0,:-7] = 'Z'
    mosaic[1,:-7] = 'Z'

    c = 0
    for col in range(0,2*n_chunks,2):
        for i in range(3):
            mosaic[2 + 2*i+1: 2*(i+2)+1, col] = f'{c}'
            mosaic[2 + 2*i+1: 2*(i+2)+1, col+1] = f'{c}'
            c+=1
            

    mosaic[4:8, -4:] = 'W'

    return mosaic

def plot_chunks_mosaic(summary_chuncks, global_time_series_df, t_chunks= None, variant_mode = False, g_log_scale = False):
    n_chunks = len(summary_chuncks)
    mosaic = get_mosaic(n_chunks)

    fig,ax= plt.subplot_mosaic(mosaic, figsize=(10*len(summary_chuncks),30), empty_sentinel=0)

    axes_chunks = np.array([[ax[f'{3*i}'],ax[f'{3*i+1}'],ax[f'{3*i+2}']] for i in range(n_chunks)], dtype=object)

    internal_axes = axes_chunks[1:,:-1].flatten()
    lower_axes = axes_chunks[:,-1]
    left_axes = axes_chunks[0,:]

    first_row_axes = axes_chunks[:,0]
    second_row_axes = axes_chunks[:,1]
    third_row_axes = axes_chunks[:,2]

    plt.subplots_adjust(wspace=0, hspace=0.1)

    sns.lineplot(data= global_time_series_df.query('day_diff < 1084'), x='day_diff', y='Freq', ax=ax['Z'], hue= 'Pango_Variant')

    #get the line color of every variant with more than 1e4 clades
    if variant_mode:
        lines_dict= {line.get_label():line.get_color() for line in ax['Z'].get_lines() if "_child" not in line.get_label()}
        facecolor_chunks = [lines_dict[summary_chuncks[i].Pango_Variant.unique()[0]] for i in range(len(summary_chuncks)) if summary_chuncks[i].Pango_Variant.unique()[0] in lines_dict]
    else:
        #generate the facecolors for each chunk from tab20
        tab20 = plt.cm.get_cmap('tab20', 20)
        facecolor_chunks = [tab20(i) for i in range(len(summary_chuncks))]

    g_factors_chunks = []
    for i in range(len(summary_chuncks)):
        try:
            pv = summary_chuncks[i].Pango_Variant.unique()[0]
            g_factors_chunks.append(plot_column_chunk(summary_chuncks[i], axes_chunks[i], facecolor_chunks[i]))
            if variant_mode:
                plotlabel = f'{pv}'
                lw = global_time_series_df.query('Pango_Variant == @pv').Freq.max()*5
            elif t_chunks is not None:
                plotlabel = f'${t_chunks[i-1]:.0f} < t < {t_chunks[i]:.0f}$' if i>0 else f'$t < {t_chunks[i]:.0f}$'
                lw = 5
            ax['W'].plot(g_factors_chunks[i][0], g_factors_chunks[i][1], color=facecolor_chunks[i],lw=lw, label = plotlabel)
        except IndexError:
            pass
        # ax['W'].fill_between(g_factors_chunks[i][0], g_factors_chunks[i][1] - g_factors_chunks[i][2], g_factors_chunks[i][1] + g_factors_chunks[i][2], color=facecolor_chunks[i], alpha=0.2)
    

    ylim_first_row = (np.min([a.get_ylim() for a in first_row_axes]), np.max([a.get_ylim() for a in first_row_axes]))
    ylim_second_row = (np.min([a.get_ylim() for a in second_row_axes]), np.max([a.get_ylim() for a in second_row_axes]))    
    ylim_third_row = (np.min([a.get_ylim() for a in third_row_axes]), np.max([a.get_ylim() for a in third_row_axes]))

    for a in first_row_axes:
        if g_log_scale:
            a.set_yscale('log')
            a.axhline(1,color='black',linestyle='--')   
            a.set_ylim(5e-1,1e2)
        else:
            a.set_ylim(0,20)
            a.axhline(2,color='black',linestyle='--')
    for a in second_row_axes:
        a.set_ylim((1e-5,2))
    for a in third_row_axes:
        a.set_ylim((1,ylim_third_row[1]*3))
    
    handles = [Line2D([0], [0], color='steelblue', lw=3, label='synonymous'),
              Line2D([0], [0], color='orange', lw=3, label='nonsynonymous'),
              Line2D([0], [0], color='darkred', lw=3, label='RBD, nonsynonymous'),
              Line2D([0], [0], color='green', lw=3, linestyle='-', label=r'$\mathrm{S}\backslash\mathrm{RBD}$, nonsynonymous'),
              Line2D([0], [0], color='purple', lw=3, linestyle='-', label=r'$\overline{\mathrm{S}}$, nonsynonymous'),]
    axes_chunks[-1,1].legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=30)

    ax['Z'].legend(bbox_to_anchor=(1.05, 1), loc='upper left',fontsize=30)
    ax['Z'].set_xlim(left= 65, right=1090)
    ax['Z'].set_title('Frequency trajectories')
    ax['Z'].set_xlabel('Time (days from 2020-01-01)')

    ax['W'].axhline(1,color='black',linestyle='--')
    ax['W'].legend(loc='best',fontsize=30)
    ax['W'].set_title('RBD G factor', pad= 50)
    ax['W'].set_xlabel('Frequency')
    ax['W'].set_ylabel('G factor')
    ax['W'].set_xscale('log')

    axes_chunks[0,0].set_ylabel('G factor',fontsize=25)
    axes_chunks[0,1].set_ylabel('inv. cumulative density',fontsize=25)
    axes_chunks[0,2].set_ylabel('inv. cumulative counts',fontsize=25)
    for a in lower_axes:
        a.set_xlabel('Frequency',fontsize=25)
    
    for a in internal_axes:
        a.set_xticklabels([])
        a.set_yticklabels([])
    for a in lower_axes[1:]:
        a.set_yticklabels([])
    for a in left_axes[:-1]:
        a.set_xticklabels([])

    for a in axes_chunks.flatten():
        a.set_xticks([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1])

#annotate each ax, in the upper right corner    
    for a in ax:
        if a not in ['W','Z']:
            ax[a].annotate(int(a)+1, (0.95, 0.95), xycoords='axes fraction', va='top', ha='right', fontsize=40)
        elif a == 'W':
            ax[a].annotate(a, (0.15, 0.85), xycoords='axes fraction', va='bottom', ha='right', fontsize=40)
        else:
            ax[a].annotate(a, (0.99, 0.95), xycoords='axes fraction', va='top', ha='right', fontsize=40)
        ax[a].tick_params(axis='both', which='major', labelsize=20)

    if t_chunks is not None:
        for i,t in enumerate(t_chunks):
            ax['Z'].axvline(t, color='black', linestyle='--')
            if i>0:
                ax['Z'].fill_betweenx((0,1), t_chunks[i-1], t_chunks[i], color=facecolor_chunks[i], alpha=0.25)
                first_row_axes[i].set_title(f'${t_chunks[i-1]:.0f} < t < {t_chunks[i]:.0f}$')
            else:
                print(i, facecolor_chunks[i])
                ax['Z'].fill_betweenx((0,1), 0, t_chunks[i], color=facecolor_chunks[i], alpha=0.25)
                first_row_axes[i].set_title(f'$t < {t_chunks[i]:.0f}$')

    return fig, ax

def plot_variant_statistics(obs, clade_statistics_df, variant_freqs_df, fit_and_div_df, variant_color_dict, S_RBD_palette, crossover_times,bin_size, x_thresh = 5e-3, max_times = False, figsize = None):
    if figsize is None:
        figsize = (50,50)
    fig,ax = plt.subplots(6,4,figsize=figsize,sharex=True)

    day_diff_vals = np.sort(fit_and_div_df['day_diff'].unique())

    variant_axes = [ax[0,0],ax[0,1], ax[0,2], ax[0,3], ax[3,0], ax[3,1], ax[3,2], ax[3,3]]
    counts_axes = [ax[1,0],ax[1,1], ax[1,2], ax[1,3], ax[4,0], ax[4,1], ax[4,2], ax[4,3]]
    cumulative_axes = [ax[2,0],ax[2,1], ax[2,2], ax[2,3], ax[5,0], ax[5,1], ax[5,2], ax[5,3]]

    plt.subplots_adjust(wspace = 0.5)
    for iv,variant_name in enumerate(['WT','B.1.351', 'B.1.617.2','BA.1', 'BA.2', 'BA.5','XBB.1.5','XBB.1.16']):
    
        tax1,tax2 = plot_single_variant_statistics(obs=obs, variant_name=variant_name, x_thresh = x_thresh,
                                                variant_axis = variant_axes[iv], counts_axis = counts_axes[iv], cumulative_axis = cumulative_axes[iv],
                                                clade_statistics_df = clade_statistics_df, variant_freqs_df = variant_freqs_df, fit_and_div_df = fit_and_div_df,
                                                variant_color_dict = variant_color_dict, S_RBD_palette = S_RBD_palette,
                                                crossover_times = crossover_times, bin_size = bin_size, max_times = max_times)

    #make sure that zero is aligned between the two y-axes
        ylim1 = counts_axes[iv].get_ylim()
        ylim2 = cumulative_axes[iv].get_ylim()

        ylim_tax1 = tax1.get_ylim()
        ylim_tax2 = tax2.get_ylim()

        tax1.set_ylim(np.max((ylim_tax1[0]/(ylim1[1]/ylim_tax1[1]),0)),ylim_tax1[1])

        tax2.set_ylim(np.max((ylim_tax2[0]/(ylim2[1]/ylim_tax2[1]),0)),ylim_tax2[1])
        # print(np.max((ylim_tax1[0]/(ylim1[1]/ylim_tax1[1]),0)),np.max((ylim_tax2[0]/(ylim2[1]/ylim_tax2[1]),0)))
        tax1.set_ylabel(obs, fontsize = 25)

        variant_axes[iv].set_ylabel('Frequency', fontsize = 45, labelpad = 20)
        counts_axes[iv].set_ylabel('Origination \n density', fontsize = 45, labelpad = 20)
        cumulative_axes[iv].set_xlabel('Days since 2020-01-01', fontsize = 45, labelpad = 20)
        cumulative_axes[iv].set_ylabel('Cumulative \n origination density', fontsize = 45, labelpad = 20)
        tax1.set_ylabel(r'$s_\alpha\pi^\alpha$ [a.u.]' , fontsize = 45, labelpad = 15)
        tax2.set_ylabel(r'$\int s_\alpha\pi^\alpha$ [a.u.]', fontsize = 45, labelpad = 15)

        for a in [variant_axes[iv], counts_axes[iv], cumulative_axes[iv]]:
            a.tick_params(axis='both', which='major', labelsize=35)


    return fig, ax, [tax1,tax2] 

def plot_single_variant_statistics(obs, variant_name, x_thresh,
                                    variant_axis, counts_axis, cumulative_axis,
                                    clade_statistics_df, variant_freqs_df, fit_and_div_df,
                                    variant_color_dict, S_RBD_palette,
                                    crossover_times, bin_size, max_times = False
    ):
    def plot_variant_frequency():
        sns.lineplot(
            data=variant_freqs_df.query(f'Pango_Variant == "{variant_name}"'),
            x='day_diff', y='Freq',
            hue='Pango_Variant', ax=variant_axis,
            palette=variant_color_dict, legend=False, lw=5
        )

    def plot_clade_statistics():
        clades_above_thresh = clade_statistics_df.query(
            f'Pango_Variant == "{variant_name}" & Sublineage_Freq >= {x_thresh}'
        )['Clade'].unique()

        subsub_df = clade_statistics_df[
            (clade_statistics_df['Pango_Variant'] == variant_name) &
            (clade_statistics_df['Clade'].isin(clades_above_thresh))
        ]

        sns.lineplot(data=subsub_df.query(f'nonsyn_weight == 0'),
            x='day_diff', y='Sublineage_Freq',
            hue='Clade', linestyle='-', lw=1, alpha=.25,
            legend=False, ax=variant_axis, palette=['grey']
        )

        sns.lineplot(data=subsub_df.query(f'S_RBD > 0'),
            x='day_diff', y='Sublineage_Freq',
            style='Clade', hue='S_RBD',
            linestyle='-', lw=2, alpha=1,
            legend=False, ax=variant_axis, palette=S_RBD_palette
        )

    def plot_observables():
        tax1 = counts_axis.twinx()
        if not(max_times):
            sns.lineplot(data=fit_and_div_df.query(f'Pango_Variant == "{variant_name}"'),
                x='day_diff', y=obs,
                hue='Pango_Variant', ax=tax1,
                palette=[variant_color_dict[variant_name]]
            )
        else:
            sns.lineplot(data=fit_and_div_df.query(f'Pango_Variant == "{variant_name}" & day_diff < max_times'),
                x='day_diff', y=obs,
                hue='Pango_Variant', ax=tax1,
                palette=[variant_color_dict[variant_name]]
            )
        return tax1

    def plot_histograms_and_cumulative():
        filtered = clade_statistics_df[
            (clade_statistics_df['Sublineage_Freq'] > x_thresh) &
            (clade_statistics_df['Pango_Variant'] == variant_name)
        ]
        min_day_diff = filtered.groupby('Clade')['day_diff'].min().reset_index()
        result = filtered.merge(
            min_day_diff, on=['Clade', 'day_diff']
        )[['Clade', 'day_diff', 'S_RBD', 'nonsyn_weight', 'syn_weight']]

        try:
            bins = np.sort(variant_freqs_df.day_diff.unique())[::bin_size]
            for ax, cumulative in zip([counts_axis, cumulative_axis], [False, True]):
                ax.hist(
                    data=result.query('S_RBD > 0'), x='day_diff',
                    bins=bins, color='darkred', alpha=1, fill=False,
                    density=not cumulative, histtype='step',
                    lw=3, cumulative=cumulative, label='RBD mutations'
                )
        except ValueError:
            pass

    def plot_cumulative_normalized():
        tax2 = cumulative_axis.twinx()

        if not(max_times):
            cumulative_normalized_obs = ut.nancumsum(
                fit_and_div_df.query(f'Pango_Variant == "{variant_name}"')[obs].values
            )
            tax2.plot(
                fit_and_div_df.query(f'Pango_Variant == "{variant_name}"')['day_diff'],
                cumulative_normalized_obs,
                color=variant_color_dict[variant_name], lw=3
            )
        else:
            cumulative_normalized_obs = ut.nancumsum(
                fit_and_div_df.query(f'Pango_Variant == "{variant_name}" & day_diff < max_times')[obs].values
            )
            tax2.plot(
                fit_and_div_df.query(f'Pango_Variant == "{variant_name}" & day_diff < max_times')['day_diff'],
                cumulative_normalized_obs,
                color=variant_color_dict[variant_name], lw=3
            )
        return tax2

    def annotate_plot():
        variant_axis.axhline(x_thresh, color='black', linestyle='--', lw=3)
        variant_axis.set_ylabel('Frequency', fontsize=20)
        variant_axis.set_yscale('logit')
        variant_axis.set_ylim(1e-5, 1 - 1e-2)

        handles = [
            Line2D([0], [0], color=S_RBD_palette[k], lw=3, label=k)
            for k in S_RBD_palette.keys()
        ]
        variant_axis.legend(
            handles=handles, loc='best', fontsize=5,
            title='RBD mutations', title_fontsize=20
        )

        for pv in [
            'WT', 'B.1.1.7', 'B.1.351', 'B.1.617.2',
            'BA.1', 'BA.2', 'BA.5', 'XBB', 'XBB.1.5', 'XBB.1.16'
        ]:
            ylim1 = variant_axis.get_ylim()
            ypos = np.array(ylim1[1]) * np.random.uniform(0.9, 0.65)
            xpos = (
                variant_freqs_df.query(f'Pango_Variant == "{pv}" & Freq > 0.1').day_diff.mean()
                if pv != 'BA.2' else 840
            )
            variant_axis.annotate(pv, xy=(xpos, ypos), fontsize=24)

        for et in crossover_times:
            for ax in [variant_axis, counts_axis]:
                ax.axvline(et, color='red', linestyle='--', alpha=.25)

    # call sub-functions
    plot_variant_frequency()
    plot_clade_statistics()
    tax1 = plot_observables()
    plot_histograms_and_cumulative()
    tax2 = plot_cumulative_normalized()
    annotate_plot()

    variant_axis.set_title(variant_name, y=0.925)
    return tax1,tax2


def plot_single_variant_statistics_full(obs, variant_name, x_thresh,
                                        variant_axis, 
                                        diversity_axis, selection_axis,
                                        counts_axis, cumulative_axis,
                                        clade_statistics_df, variant_freqs_df, fit_and_div_df,
                                        variant_color_dict, S_RBD_palette,
                                        crossover_times, bin_size, max_times = False,
                                        sel_name = None, pot_sel_name = None,
                                        normalize_originations = False,
                                        freq_axis_scale = 'logit',
                                        ):
    def plot_variant_frequency():
        sns.lineplot(
            data=variant_freqs_df.query(f'Pango_Variant == "{variant_name}"'),
            x='day_diff', y='Freq',
            hue='Pango_Variant', ax=variant_axis,
            palette=variant_color_dict, legend=False, lw=5
        )
        # variant_axis.axhline(0.5, color='black', linestyle='--', lw=3)
    
    def plot_diversity():
        sns.lineplot(
            data=fit_and_div_df.query(f'Pango_Variant == "{variant_name}" & day_diff < max_times'),
            x='day_diff', 
            y='diversity',
            hue='Pango_Variant', ax=diversity_axis,
            palette=variant_color_dict, 
            legend=False, 
            lw=3
        )

    def plot_selection(sel_name, pot_sel_name):
        if pot_sel_name is None:
            pot_sel_name = 's_logit_pot_week'
        sns.lineplot(data = fit_and_div_df.query(f'Pango_Variant == "{variant_name}" & day_diff < max_times & Freq> {1e-3}'),
            x = 'day_diff',
            y = pot_sel_name,
            hue = 'Pango_Variant',
            palette= variant_color_dict,
            lw = 3, 
            ax = selection_axis, 
            legend = False)
        if sel_name is not None:
            sns.lineplot(data = fit_and_div_df.query(f'Pango_Variant == "{variant_name}"  & day_diff < max_times & Freq> {1e-3}'),
                x = 'day_diff',
                y = sel_name,
                hue = 'Pango_Variant',
                palette= variant_color_dict,
                lw = 3, 
                ax = selection_axis, 
                legend = False, linestyle = '--')
        if 'week' in pot_sel_name:
            selection_axis.set_ylim(-.2*7, .3*7)
            selection_axis.set_ylabel('Selection, [week]$^{-1}$', fontsize = 20)
        else:
            selection_axis.set_ylim(-.2, .3)
        selection_axis.axhline(0, color = 'black', linestyle = '--', lw = 3)

    def plot_clade_statistics():
        clades_above_thresh = clade_statistics_df.query(
            f'Pango_Variant == "{variant_name}" & Sublineage_Freq >= {x_thresh}'
        )['Clade'].unique()

        subsub_df = clade_statistics_df[
            (clade_statistics_df['Pango_Variant'] == variant_name) &
            (clade_statistics_df['Clade'].isin(clades_above_thresh))
        ]

        sns.lineplot(data=subsub_df.query(f'nonsyn_weight == 0'),
            x='day_diff', y='Sublineage_Freq',
            hue='Clade', linestyle='-', lw=1, alpha=.25,
            legend=False, ax=variant_axis, palette=['grey']
        )

        sns.lineplot(data=subsub_df.query(f'S_RBD > 0'),
            x='day_diff', y='Sublineage_Freq',
            style='Clade', hue='S_RBD',
            linestyle='-', lw=2, alpha=1,
            legend=False, ax=variant_axis, palette=S_RBD_palette
        )

    def plot_observables():
        tax1 = counts_axis.twinx()
        if not(max_times):
            sns.lineplot(data=fit_and_div_df.query(f'Pango_Variant == "{variant_name}"'),
                x='day_diff', y=obs,
                hue='Pango_Variant', ax=tax1,
                palette=[variant_color_dict[variant_name]],
                legend=False
            )
        else:
            sns.lineplot(data=fit_and_div_df.query(f'Pango_Variant == "{variant_name}" & day_diff < max_times'),
                x='day_diff', y=obs,
                hue='Pango_Variant', ax=tax1,
                palette=[variant_color_dict[variant_name]],
                legend=False
            )
        return tax1

    def plot_histograms_and_cumulative():
        filtered = clade_statistics_df[
            (clade_statistics_df['Sublineage_Freq'] > x_thresh) &
            (clade_statistics_df['Pango_Variant'] == variant_name)
        ]
        min_day_diff = filtered.groupby('Clade')['day_diff'].min().reset_index()
        result = filtered.merge(
            min_day_diff, on=['Clade', 'day_diff']
        )[['Clade', 'day_diff', 'S_RBD', 'nonsyn_weight', 'syn_weight']]

        try:
            bins = np.sort(variant_freqs_df.day_diff.unique())[::bin_size]
            for ax, cumulative in zip([counts_axis, cumulative_axis], [False, True]):
                ax.hist(
                    data=result.query('S_RBD > 0'), x='day_diff',
                    bins=bins, color='darkred', alpha=1, fill=False,
                    density= normalize_originations, histtype='step',
                    lw=3, cumulative=cumulative, label='RBD mutations'
                )
        except ValueError:
            pass

    def plot_cumulative_normalized():
        tax2 = cumulative_axis.twinx()

        if not(max_times):
            cumulative_normalized_obs = ut.nancumsum(
                fit_and_div_df.query(f'Pango_Variant == "{variant_name}"')[obs].values
            )
            tax2.plot(
                fit_and_div_df.query(f'Pango_Variant == "{variant_name}"')['day_diff'],
                cumulative_normalized_obs,
                color=variant_color_dict[variant_name], lw=3
            )
        else:
            cumulative_normalized_obs = ut.nancumsum(
                fit_and_div_df.query(f'Pango_Variant == "{variant_name}" & day_diff < max_times')[obs].values
            )
            tax2.plot(
                fit_and_div_df.query(f'Pango_Variant == "{variant_name}" & day_diff < max_times')['day_diff'],
                cumulative_normalized_obs,
                color=variant_color_dict[variant_name], lw=3
            )
        return tax2

    def annotate_plot(ax):
        ax.axhline(x_thresh, color='black', linestyle='--', lw=3)
        ax.set_ylabel('Frequency', fontsize=20)
        ax.set_yscale(freq_axis_scale)
        if freq_axis_scale == 'logit':
            ax.set_ylim(1e-5, 1 - 1e-2)
        elif freq_axis_scale == 'log':
            ax.set_ylim(1e-5, 2)

        # handles = [
        #     Line2D([0], [0], color=S_RBD_palette[k], lw=3, label=k)
        #     for k in S_RBD_palette.keys()
        # ]
        # ax.legend(
        #     handles=handles, loc='best', fontsize=5,
        #     title='RBD mutations', title_fontsize=20
        # )

        for pv in [
            'WT', 'B.1.1.7', 'B.1.351', 'B.1.617.2',
            'BA.1', 'BA.2', 'BA.5', 'XBB', 'XBB.1.5', 'XBB.1.16'
        ]:
            ylim1 = variant_axis.get_ylim()
            ypos = np.array(ylim1[1]) * np.random.uniform(0.01, 0.95)
            xpos = (
                variant_freqs_df.query(f'Pango_Variant == "{pv}" & Freq > 0.1').day_diff.mean()
                if pv != 'BA.2' else 840
            )
            if pv == 'XBB.1.16':
                xpos+=70
            elif pv == 'BA.2':
                xpos+=20
            # ax.annotate(pv, xy=(xpos, ypos), fontsize=24)
            ax.annotate(
                pv,
                xy=(xpos, 1e-4),  # x=50% in axis coords, y=0 (bottom of axis)
                xycoords='data',
                textcoords='offset points',  # offset just the y position
                xytext=(0, -40),  # move only downward by 20 points
                ha='center', va='top',  # horizontal and vertical alignment,
                fontsize=24 
            )

        for et in crossover_times:
            for a in [variant_axis, counts_axis]:
                a.axvline(et, color='red', linestyle='--', alpha=.5)

    # call sub-functions
    plot_variant_frequency()
    plot_diversity()
    plot_selection(sel_name, pot_sel_name)
    plot_clade_statistics()
    tax1 = plot_observables()
    plot_histograms_and_cumulative()
    tax2 = plot_cumulative_normalized()
    annotate_plot(variant_axis)

    variant_axis.set_title(variant_name, y=0.985)
    #make sure all axes have same x-limits
    xlim_left = np.min([a.get_xlim()[0] for a in [variant_axis, diversity_axis, selection_axis, counts_axis, cumulative_axis]])
    xlim_right = np.max([a.get_xlim()[1] for a in [variant_axis, diversity_axis, selection_axis, counts_axis, cumulative_axis]])
    for a in [variant_axis, diversity_axis, selection_axis, counts_axis, cumulative_axis]:
        a.set_xlim(xlim_left, xlim_right)

    
    
    return tax1,tax2


def plot_only_variant_and_subvariant_freqs(variant_name, variant_axis, clade_statistics_df, variant_freqs_df, variant_color_dict, S_RBD_palette, crossover_times, freq_axis_scale = 'logit', x_thresh = 5e-3, yoffset=-40):
                                                                                   
    def plot_variant_frequency():
        sns.lineplot(
            data=variant_freqs_df.query(f'Pango_Variant == "{variant_name}"'),
            x='day_diff', y='Freq',
            hue='Pango_Variant', ax=variant_axis,
            palette=variant_color_dict, legend=False, lw=5
        )

    def plot_clade_statistics():
        clades_above_thresh = clade_statistics_df.query(
            f'Pango_Variant == "{variant_name}" & Sublineage_Freq >= {x_thresh}'
        )['Clade'].unique()

        subsub_df = clade_statistics_df[
            (clade_statistics_df['Pango_Variant'] == variant_name) &
            (clade_statistics_df['Clade'].isin(clades_above_thresh))
        ]

        sns.lineplot(data=subsub_df.query(f'nonsyn_weight == 0'),
            x='day_diff', y='Sublineage_Freq',
            hue='Clade', linestyle='-', lw=1, alpha=.25,
            legend=False, ax=variant_axis, palette=['grey']
        )

        sns.lineplot(data=subsub_df.query(f'S_RBD > 0'),
            x='day_diff', y='Sublineage_Freq',
            style='Clade', hue='S_RBD',
            linestyle='-', lw=2, alpha=1,
            legend=False, ax=variant_axis, palette=S_RBD_palette
        )
        
    def annotate_plot(ax, yoffset):
        ax.axhline(x_thresh, color='black', linestyle='--', lw=3)
        ax.set_ylabel('Frequency', fontsize=20)
        ax.set_yscale(freq_axis_scale)
        if freq_axis_scale == 'logit':
            ax.set_ylim(1e-5, 1 - 1e-2)
        elif freq_axis_scale == 'log':
            ax.set_ylim(1e-5, 2)

        # handles = [
        #     Line2D([0], [0], color=S_RBD_palette[k], lw=3, label=k)
        #     for k in S_RBD_palette.keys()
        # ]
        # ax.legend(
        #     handles=handles, loc='best', fontsize=5,
        #     title='RBD mutations', title_fontsize=20
        # )

        for pv in [
            'WT', 'B.1.1.7', 'B.1.351', 'B.1.617.2',
            'BA.1', 'BA.2', 'BA.5', 'XBB', 'XBB.1.5', 'XBB.1.16'
        ]:
            ylim1 = variant_axis.get_ylim()
            ypos = np.array(ylim1[1]) * np.random.uniform(0.01, 0.95)
            xpos = (
                variant_freqs_df.query(f'Pango_Variant == "{pv}" & Freq > 0.1').day_diff.mean()
                if pv != 'BA.2' else 840
            )
            if pv == 'XBB.1.16':
                xpos+=70
            elif pv == 'BA.2':
                xpos+=20
            # ax.annotate(pv, xy=(xpos, ypos), fontsize=24)
            ax.annotate(
                pv,
                xy=(xpos, 1e-4),  # x=50% in axis coords, y=0 (bottom of axis)
                xycoords='data',
                textcoords='offset points',  # offset just the y position
                xytext=(0, yoffset),  # move only downward by 20 points
                ha='center', va='top',  # horizontal and vertical alignment,
                fontsize=24 
            )

        for et in crossover_times:
            for a in [variant_axis]:
                a.axvline(et, color='red', linestyle='--', alpha=.5)

    # call sub-functions
    plot_variant_frequency()
    plot_clade_statistics()
    annotate_plot(variant_axis, yoffset)











