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

from matplotlib.ticker import FuncFormatter

fmt = lambda x, pos: '{:.1f}'.format(x)
from numba import jit

#matplotlib.use('TkAgg')
from matplotlib.lines import Line2D
plt.rc('mathtext', default='regular')
plt.rc('lines', linewidth=3.0)
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
plt.rcParams['axes.labelsize']=35
plt.rcParams['legend.fontsize']= 35
plt.rcParams['figure.figsize'] = (20, 10)
plt.rcParams['image.cmap']='coolwarm'
plt.rcParams['axes.titlesize'] = 40
plt.rcParams['figure.titlesize'] = 40
#set legend titlesize to 40
plt.rcParams['legend.title_fontsize'] = 40

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

def model(X, t, R0, kappa):
    x = X[0]
    y = X[1]
    z = X[2]
    dxdt = - R0/(1+kappa * y) *x * y
    dydt =  R0/(1+kappa* y) * x * y - y
    dzdt = y
    return [dxdt, dydt, dzdt]

def model_w_params(R0, N, k): 
    def model(y,t):
        S = y[0]
        I = y[1]
        R0_eff = R0/(1+k*I/(N))
        dSdt = - R0_eff * S*I/N 
        dIdt =  R0_eff * S*I/N -I
        return [dSdt,dIdt]
    return model

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

@jit(nopython=True)
def get_p_surv_inv_int(fit, dt,reg= 1e-1):
    L = len(fit)
    pi_ext_ode_hand_inv_t = np.zeros(L)
    pi_ext_ode_hand_inv_t[-1] = np.minimum(1/(1+fit[-1]),1-reg)
    for i in range(L-1):
        pi_ext= pi_ext_ode_hand_inv_t[L-1-i]
        pi_ext_ode_hand_inv_t[L-2-i] = pi_ext -dt * (1-pi_ext)*(-1+pi_ext*(1+fit[L-2-i]))
    p_surv = 1-pi_ext_ode_hand_inv_t
    return p_surv

def find_x_inf(R0,k):
    if k>0:
        c = (R0-k+R0*k)/(k*(R0-k))
        rho = 1/(R0/k)
        y_x = lambda x: c*x**rho + x/(rho-1) - 1/k

        roots = so.fsolve(y_x, 0)
        return roots[0]
    else:
        return np.real(-1/R0*sps.lambertw(-R0*np.exp(-R0)))
    
def integrate_trajectory(R0,kappa,N, I0=10,Nsteps = 10000):
    S0 = N-I0
    y0 = [S0,I0]
    x_inf = find_x_inf(R0,kappa)
    
    t_end = 1.5*np.log(N)* (1/(R0-1) + 1/(1-R0*x_inf))
    ts = np.linspace(0, t_end,Nsteps)
    dt = ts[1]-ts[0]
    solution  = odeint(model_w_params(R0,N,kappa),y0,ts).T
    x,y= solution
    tp = ts[np.argmax(y)]


    return ts, solution


def get_clade_stats_dataframe(folder_name, reference_date, x_thresh_vals = [1e-5,1e-4,1e-3,1e-2,1e-1]):
    thresh_string = f'{x_thresh_vals[0]:.0e}_{x_thresh_vals[-1]:.0e}_{len(x_thresh_vals)}'
    if os.path.exists(saves_folder+folder_name+thresh_string+'.feather'):
        clade_stats = pd.read_feather(saves_folder+folder_name+thresh_string+'.feather')
        
    else:
        clade_folder = data_folder + '/' +  folder_name + '/'
        assert os.path.exists(clade_folder)

        clade_statistics = pd.read_csv(clade_folder + 'clade_statistics.tsv', sep = '\t')
        clade_stats = pd.DataFrame(columns = ['Clade','Max_Freq','Orig_Time'])
        
        clade_stats['Clade'] = clade_statistics['Clade']
        clade_stats['Max_Freq'] = clade_statistics.groupby('Clade')['Sublineage_Freq'].transform('max')
        clade_stats['Orig_Time'] = clade_statistics.groupby('Clade')['Time'].transform('min')
        clade_stats = clade_stats.drop_duplicates()
        for x_th in tqdm(x_thresh_vals):
            clade_stats[f'Time_x_bgr_{x_th}'] = clade_statistics[clade_statistics['Sublineage_Freq']>x_th].groupby('Clade')['Time'].transform('min')
            clade_stats[f'day_diff_x_bgr_{x_th}'] = (pd.to_datetime(clade_stats[f'Time_x_bgr_{x_th}']) - reference_date).dt.days
        clade_stats['day_diff'] = (pd.to_datetime(clade_stats['Orig_Time']) - reference_date).dt.days
        clade_stats.to_feather(saves_folder+folder_name+thresh_string+'.feather')
        print(f'Saved clade_stats to {saves_folder+folder_name+thresh_string}.feather')

    return clade_stats

def get_global_variants_df(in_folder_name, out_folder_name,
                            reference_date, recompute = False):
    # print(saves_folder , folder_name, 'global_variants_df.feather')
    if os.path.exists(saves_folder + out_folder_name + 'global_variants_df.feather') and not recompute:
        global_time_series_df = pd.read_feather(saves_folder + out_folder_name + 'global_variants_df.feather')

        return global_time_series_df
    
    else:
        clade_folder = in_folder_name + '/'
        assert os.path.exists(clade_folder)

        clade_statistics = pd.read_csv(clade_folder + 'clade_statistics.tsv', sep = '\t')
        clade_statistics['day_diff'] = (pd.to_datetime(clade_statistics['Time']) - reference_date).dt.days
        driver_mutation_statistics = pd.read_csv(clade_folder + 'driver_mutation_statistics.tsv', sep = '\t')

        global_merged_df = pd.merge(clade_statistics, driver_mutation_statistics[['Clade', 'Variant']], on='Clade')

        global_time_series_df = global_merged_df.groupby(['Variant', 'day_diff','Time']).agg({
            'Clade_Freq': 'sum'
        }).reset_index()

        # Rename columns to match the desired output
        global_time_series_df.columns = ['Variant', 'day_diff','Time', 'Freq']
        global_time_series_df['Pango_Variant'] = global_time_series_df['Variant'].map(flupredict_to_pango)
        global_time_series_df.to_feather(saves_folder + out_folder_name + 'global_variants_df.feather')
        print(f'Saved global_time_series_df to {saves_folder + out_folder_name + "global_variants_df.feather"}')
        return global_time_series_df

def get_kde(data, bw_adjust = .15, bw_method = 'scott', ax = None, grid_min = None, grid_max = None, grid_size = 1000, weights = None):
    #if ax is None create a fake figure
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit(bw=bw_adjust, fft = True)
    # Define the grid range if not specified
    if grid_min is None:
        grid_min = np.min(data)
    if grid_max is None:
        grid_max = np.max(data)
    
    # Create a fixed grid
    grid = np.linspace(grid_min, grid_max, grid_size)
    
    # Evaluate the KDE on the fixed grid
    kde_values = kde.evaluate(grid)
    
    return grid, kde_values

def get_new_kde(data, bw_adjust = .15, bw_method = 'scott', ax = None, grid_min = None, grid_max = None, grid_size = 1000, weights = None):
    #if ax is None create a fake figure
    kde = sm.nonparametric.KDEUnivariate(data)
    kde.fit(bw= bw_method, adjust=bw_adjust, fft = False, weights = weights)
    # Define the grid range if not specified
    if grid_min is None:
        grid_min = np.min(data)
    if grid_max is None:
        grid_max = np.max(data)
    
    # Create a fixed grid
    grid = np.linspace(grid_min, grid_max, grid_size)
    
    # Evaluate the KDE on the fixed grid
    kde_values = kde.evaluate(grid)
    
    return grid, kde_values

def get_kde_scipy(data, grid, weights, bw = 'scott', norm = 'counts'):
    # print(data.shape, weights.shape)
    h,b= np.histogram(data, bins = grid, density = False, weights = weights)
    sumweights = np.trapz(h,b[:-1])

    kde = spst.gaussian_kde(data, weights = weights/sumweights,)
    kde.set_bandwidth(bw_method = bw)
    if norm == 'counts':
        kde_values = sumweights*kde.evaluate(grid)/np.trapz(kde.evaluate(grid),grid)
    else:
        kde_values = kde.evaluate(grid)/np.trapz(kde.evaluate(grid),grid)
    return grid, kde_values


def get_ratio_of_ratios(data1_x, data1_ref, data2_x, data2_ref, bw_adjust = .15, bw_method = 'scott', ax = None, grid_min = None, grid_max = None, grid_size = 1000, weights = None, reg= 1e-5):
    
    time, smoothed_data1_x = get_kde(data1_x, bw_adjust = bw_adjust, bw_method = bw_method, ax = ax, grid_min = grid_min, grid_max = grid_max, grid_size = grid_size, weights = weights)

    time, smoothed_data1_ref = get_kde(data1_ref, bw_adjust = bw_adjust, bw_method = bw_method, ax = ax, grid_min = grid_min, grid_max = grid_max, grid_size = grid_size, weights = weights)

    time, smoothed_data2_x = get_kde(data2_x, bw_adjust = bw_adjust, bw_method = bw_method, ax = ax, grid_min = grid_min, grid_max = grid_max, grid_size = grid_size, weights = weights)

    time, smoothed_data2_ref = get_kde(data2_ref, bw_adjust = bw_adjust, bw_method = bw_method, ax = ax, grid_min = grid_min, grid_max = grid_max, grid_size = grid_size, weights = weights)
    
    bins = np.linspace(grid_min, grid_max, grid_size)

    h1_x, b1_x = np.histogram(data1_x, bins = bins, density= False)
    h1_ref, b1_ref = np.histogram(data1_ref, bins = bins, density= False)

    h2_x, b2_x = np.histogram(data2_x, bins = bins, density= False)
    h2_ref, b2_ref = np.histogram(data2_ref, bins = bins, density= False)

    counts1_x = np.trapz(h1_x, bins[:-1])
    counts1_ref = np.trapz(h1_ref, bins[:-1])

    counts2_x = np.trapz(h2_x, bins[:-1])
    counts2_ref = np.trapz(h2_ref, bins[:-1])


    ratio_1 = (counts1_x*smoothed_data1_x)/(counts1_ref*smoothed_data1_ref+reg)
    ratio_2 = (counts2_x*smoothed_data2_x)/(counts2_ref*smoothed_data2_ref+reg)

    return time, ratio_1/ratio_2

def get_ratio_of_ratios_new(data1_x, data1_ref, data2_x, data2_ref, bw_adjust = .15, bw_method = 'scott', ax = None, grid_min = None, grid_max = None, grid_size = 1000, weights = None, reg= 1e-5):

    time, smoothed_data1_x = get_new_kde(data1_x, bw_adjust = bw_adjust, bw_method = bw_method, ax = ax, grid_min = grid_min, grid_max = grid_max, grid_size = grid_size, weights = weights)

    time, smoothed_data1_ref = get_new_kde(data1_ref, bw_adjust = bw_adjust, bw_method = bw_method, ax = ax, grid_min = grid_min, grid_max = grid_max, grid_size = grid_size, weights = weights)

    time, smoothed_data2_x = get_new_kde(data2_x, bw_adjust = bw_adjust, bw_method = bw_method, ax = ax, grid_min = grid_min, grid_max = grid_max, grid_size = grid_size, weights = weights)

    time, smoothed_data2_ref = get_new_kde(data2_ref, bw_adjust = bw_adjust, bw_method = bw_method, ax = ax, grid_min = grid_min, grid_max = grid_max, grid_size = grid_size, weights = weights)
    
    bins = np.linspace(grid_min, grid_max, grid_size)

    h1_x, b1_x = np.histogram(data1_x, bins = bins, density= False)
    h1_ref, b1_ref = np.histogram(data1_ref, bins = bins, density= False)

    h2_x, b2_x = np.histogram(data2_x, bins = bins, density= False)
    h2_ref, b2_ref = np.histogram(data2_ref, bins = bins, density= False)

    counts1_x = np.trapz(h1_x, bins[:-1])
    counts1_ref = np.trapz(h1_ref, bins[:-1])

    counts2_x = np.trapz(h2_x, bins[:-1])
    counts2_ref = np.trapz(h2_ref, bins[:-1])


    ratio_1 = (counts1_x*smoothed_data1_x)/(counts1_ref*smoothed_data1_ref+reg)
    ratio_2 = (counts2_x*smoothed_data2_x)/(counts2_ref*smoothed_data2_ref+reg)

    return time, ratio_1/ratio_2

def get_ratio_of_ratios_scipy(data1_x, data1_ref, data2_x, data2_ref, grid, bw = 'scott',
                              weights_1_x = None, weights_1_ref = None, weights_2_x = None, weights_2_ref = None,
                                reg= 1e-5):
    
    g1_x, kde1_x = get_kde_scipy(data1_x, grid, weights_1_x, bw)
    g1_ref, kde1_ref = get_kde_scipy(data1_ref, grid, weights_1_ref, bw)

    g2_x, kde2_x = get_kde_scipy(data2_x, grid, weights_2_x, bw)
    g2_ref, kde2_ref = get_kde_scipy(data2_ref, grid, weights_2_ref, bw)

    ratio_1 = (kde1_x)/(kde1_ref+reg)
    ratio_2 = (kde2_x)/(kde2_ref+reg)

    return grid, ratio_1/ratio_2
    

def get_covid_data_World(reference_date):
    covid_data = pd.read_csv(data_folder + '/covid_data.csv')
    covid_data_World  = covid_data[covid_data['location'] == 'World']#.query(f'date < "{last_date}"')

    covid_data_World['Time_datetime'] = pd.to_datetime(covid_data_World['date'])
    covid_data_World['day_diff'] = covid_data_World['Time_datetime']-reference_date

    covid_data_World['day_diff'] = covid_data_World['day_diff'].apply(lambda x: x.days)
    covid_data_World['weekly_new_cases_smoothed'] = covid_data_World['new_cases_smoothed']
    return covid_data_World

def get_df_reworked(fname = 'fitness_france_newest.txt',reference_date = pd.to_datetime('2021-01-01')):
    fitfrance_data_folder = data_folder + '/2024-03-04-data/'

    fitness_france = pd.read_csv(fitfrance_data_folder+ '/'+ fname,sep = '\t')
    clade_list = list(set([c.split('_')[0] for c in fitness_france.columns]).difference(set([ 'Unnamed: 0','av','cases','date'])))
    
    fitness_france['day_diff'] = (pd.to_datetime(fitness_france['date']) - reference_date).dt.days
    print(fitness_france.columns)
    
    # avg_fit = np.sum([fitness_france[c+'_freq']*(fitness_france[c+'_fitness_inf']+fitness_france[c+'_fitness_vac'])\
    #                 for c in clade_list],axis=0)

    df_reworked = pd.DataFrame(columns = ['clade','day_diff','freq'])

    for c in clade_list:
        vals  =fitness_france[[c+'_freq', c+'_fitness_vac', c+'_fitness_inf', c + '_F_pot_inf', c + '_F_pot_vac' , 'day_diff']]
        df_vals = pd.DataFrame(vals.values,columns=['freq','fit_vac','fit_inf',  'f_pot_inf','f_pot_vac','day_diff'])
        # df_vals['avg_fit'] = avg_fit
        df_vals['cases'] = fitness_france['cases']
        extra_col = [c for i in range(len(vals))]

        df_reworked = pd.concat([df_reworked,pd.concat( [pd.DataFrame(extra_col,columns=['clade']),df_vals],axis=1)],axis=0)
    df_reworked['fit'] = df_reworked['fit_vac'] + df_reworked['fit_inf']
    df_reworked['f_pot'] = df_reworked['f_pot_inf'] + df_reworked['f_pot_vac']

    df_reworked['selection'] = (df_reworked['fit'] )*(df_reworked['freq']>0)
    df_reworked['pot_selection'] = (df_reworked['f_pot'] )*(df_reworked['freq']>0)
    df_reworked['y_t'] = df_reworked['freq']*df_reworked['cases']
    df_reworked['s_times_y_t'] = df_reworked['y_t']*df_reworked['selection']
    df_reworked['pot_s_times_y_t'] = df_reworked['y_t']*df_reworked['pot_selection']

    return df_reworked


def find_dominant_allele(freq_arr_site, threshold, T):
    result_ind = np.full(T, None)
    all_crossings = [0]

    for j in range(freq_arr_site.shape[0]):  # for each allele
        crossings = np.where((freq_arr_site[j][:-1] < threshold) & (freq_arr_site[j][1:] >= threshold))[0] + 1
        all_crossings += list(crossings)
    all_crossings = np.sort(all_crossings)
    result_ind[:all_crossings[0]-1] = np.argmax(freq_arr_site[:,0])
    for i in range(1,len(all_crossings)):
        result_ind[all_crossings[i-1]:all_crossings[i]] = np.argmax(freq_arr_site[:,all_crossings[i-1]])
    result_ind[all_crossings[-1]:] = np.argmax(freq_arr_site[:,all_crossings[-1]])

    return result_ind

def get_polarized_freq_arr(nfba, time_dimension = 2, threshold = 0.95):
    T = nfba.shape[time_dimension]
    polarized_freq_arr = np.copy(nfba)
    # threshold = 0.95

    if time_dimension == 2:
        site_dimension = 1
        for i in tqdm(range(nfba.shape[1])):
            freq_arr_site = nfba[:,i,:]
            dominant_allele_ind = find_dominant_allele(freq_arr_site, threshold,T)
            polarized_freq_arr_site = np.copy(freq_arr_site)
            polarized_freq_arr_site[dominant_allele_ind.astype(int), np.arange(T)] = 0
            polarized_freq_arr[:,i,:] = polarized_freq_arr_site
    elif time_dimension==1:
        site_dimension = 2
        for i in tqdm(range(nfba.shape[2])):
            freq_arr_site = nfba[:,:,i]
            dominant_allele_ind = find_dominant_allele(freq_arr_site, threshold,T)
            polarized_freq_arr_site = np.copy(freq_arr_site)
            polarized_freq_arr_site[dominant_allele_ind.astype(int),np.arange(T)] = 0
            polarized_freq_arr[:,:,i] = polarized_freq_arr_site
    return polarized_freq_arr

def cut_freq_arr(nfba, seq_arr):
    nfba_cut = nfba *np.heaviside(nfba -  1/seq_arr.values.reshape(1,-1,1),0)
    nfba_cut/=np.sum(nfba_cut, axis = 0)
    nfba_cut[np.isnan(nfba_cut)] = 0
    return nfba_cut

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

def find_percentile(data,weights, percentiles,bins = 100):
    h,b = np.histogram(data, bins=bins, weights=weights, density=True)
    cdf = cumulative_trapezoid(h, b[:-1])
    cdf/=cdf[-1]
    interp = spi.interp1d(1-cdf, b[:-2], fill_value='extrapolate')
    print(percentiles/100)
    return interp(percentiles/100)

def logit_space(start, end, num_points):
    def logit(x):
        return np.log(x / (1 - x))
    
    def inv_logit(x):
        return 1 / (1 + np.exp(-x))
    
    linear_space = np.linspace(0.01, 0.99, num_points)  # avoid 0 and 1 to prevent infinity in logit
    logit_space_values = logit(linear_space)
    scaled_values = np.linspace(logit(start), logit(end), num_points)
    result = inv_logit(scaled_values)
    
    return result

def integral_x_1_minus_x(H):
    h,b= H
    x = b[:-1]+0.5*np.diff(b)

    mask = True
    x = x[mask]
    h = h[mask]
    db = np.diff(b)[mask]
    return np.sum(h*x*(1-x))

def integral_x(H):
    h,b = H
    x = b[:-1]+0.5*np.diff(b)
    db = np.diff(b)
    mask = True
    x = x[mask]
    h = h[mask]
    db = db[mask]
    return np.sum(h*x)


def diversity_without_most_diverse_sites(variant_freq_arr, n_sites = 30, t_dimension = 1, most_diverse_sites = None):
    L_dimension = 2 if t_dimension == 1 else 1

    if most_diverse_sites is None:
        most_diverse_sites = get_most_diverse_sites(variant_freq_arr, n_sites, t_dimension)
    print(most_diverse_sites)
    
    full_index = np.arange(0, variant_freq_arr.shape[L_dimension])
    index_without_most_diverse = np.setdiff1d(full_index, most_diverse_sites)
    
    if t_dimension ==2:
        variant_div_without_most_diverse = np.sum(np.sum(variant_freq_arr[:,index_without_most_diverse,:]*(1-variant_freq_arr[:,index_without_most_diverse,:]), axis = 0), axis = L_dimension-1)/len(full_index)
        variant_div_most_diverse = np.sum(np.sum(variant_freq_arr[:,most_diverse_sites,:]*(1-variant_freq_arr[:,most_diverse_sites,:]), axis = 0), axis = L_dimension-1)/len(full_index)
    else:
        variant_div_without_most_diverse = np.sum(np.sum(
            variant_freq_arr[:,:,index_without_most_diverse]*(1-variant_freq_arr[:,:,index_without_most_diverse])
            , axis = 0)
            , axis = L_dimension-1
            )/len(full_index)
        variant_div_most_diverse = np.sum(np.sum(
            variant_freq_arr[:,:,most_diverse_sites]*(1-variant_freq_arr[:,:,most_diverse_sites])
            , axis = 0)
            , axis = L_dimension-1
            )/len(full_index)

    return variant_div_without_most_diverse, variant_div_most_diverse

def get_most_diverse_sites(variant_freq_arr, n_sites, t_dimension):
    return np.argsort(np.average(np.sum(variant_freq_arr*(1-variant_freq_arr), axis = 0), axis = t_dimension-1))[-n_sites:]

def inverse_cumulative_histogram(data, bins=10, range=None, density=False):
    """
    creates an inverse cumulative histogram.
    
    Parameters:
        data (array-like): the data to histogram.
        bins (int or sequence): number of bins or bin edges.
        range (tuple): lower and upper range of the bins.

    Returns:
        bin_edges (ndarray): edges of the bins.
        inverse_cdf (ndarray): inverse cumulative counts.
    """
    hist, bin_edges = np.histogram(data, bins=bins, range=range, density=density)
    inverse_cdf = np.cumsum(hist[::-1])[::-1]  # reverse cumsum and reverse back
    return inverse_cdf, bin_edges

def nancumsum(arr):
    return np.cumsum(np.nan_to_num(arr, nan=0))

def get_selection_diversity_df(
        div_var_dict,
        least_diverse_div_dict,
        variant_freqs_df,
        logder_freq_df,
        to_clean = None,      
):
    
    diversity_df = pd.DataFrame()
    for var,div in div_var_dict.items():
        t0 = variant_freqs_df.query(f'Pango_Variant == "{var}"')['day_diff'].min()
        ts_var = np.arange(t0, t0 + len(div))
        if var in to_clean:
            div = least_diverse_div_dict[var]
        else:
            div = div
        diversity_df = pd.concat([diversity_df, pd.DataFrame({'day_diff':ts_var, 'Pango_Variant':var, 'diversity':div})])
    
    selection_diversity_df = pd.merge(diversity_df, logder_freq_df[['day_diff','Pango_Variant',  
                                                                    's_logit_week', 's_log_week',
                                                                    's_pot_week','s_logit_pot_week','Freq']], on = ['day_diff','Pango_Variant'])        
    
    selection_diversity_df['s_pot_week_times_diversity'] = selection_diversity_df['s_pot_week']\
                                            *selection_diversity_df['diversity']*np.heaviside(selection_diversity_df['s_pot_week'],0)
    selection_diversity_df['s_logit_pot_week_times_diversity'] = selection_diversity_df['s_logit_pot_week']\
                                            *selection_diversity_df['diversity']*np.heaviside(selection_diversity_df['s_logit_pot_week'],0)
    
    selection_diversity_df['smoothed_s_logit_week'] = selection_diversity_df.groupby('Pango_Variant')['s_logit_week'].transform(lambda x: x.rolling(14, min_periods = 1).mean())
    selection_diversity_df['smoothed_s_logit_pot_week'] = selection_diversity_df.groupby('Pango_Variant')['s_logit_pot_week'].transform(lambda x: x.rolling(14, min_periods = 1).mean())
    
    selection_diversity_df['smoothed_s_logit_pot_week_times_diversity'] = selection_diversity_df['smoothed_s_logit_pot_week']\
                                            *selection_diversity_df['diversity']*np.heaviside(selection_diversity_df['smoothed_s_logit_pot_week'],0)
    
    max_time_WT = 600
    max_time_B_1_1_7 = 600
    max_time_B_1_351 = 600
    max_time_B_1_617_2 = 900
    max_time_P_1 = 680
    max_times_BA1 = 900
    max_times_BA = 900
    max_times_BA2 = 1200
    max_times_BA5 = 1130
    max_time_others = np.inf

    selection_diversity_df['max_times'] = np.where(selection_diversity_df['Pango_Variant'] == 'WT', max_time_WT,
        np.where(selection_diversity_df['Pango_Variant'] == 'B.1.1.7', max_time_B_1_1_7,
        np.where(selection_diversity_df['Pango_Variant'] == 'B.1.351', max_time_B_1_351,
        np.where(selection_diversity_df['Pango_Variant'] == 'B.1.617.2', max_time_B_1_617_2,
        np.where(selection_diversity_df['Pango_Variant'] == 'P.1', max_time_P_1,
        np.where(selection_diversity_df['Pango_Variant'] == 'BA.1', max_times_BA1,
        np.where(selection_diversity_df['Pango_Variant'] == 'BA', max_times_BA,
        np.where(selection_diversity_df['Pango_Variant'] == 'BA.2', max_times_BA2,    
        np.where(selection_diversity_df['Pango_Variant'] == 'BA.5', max_times_BA5,
        max_time_others)))))))))
    return selection_diversity_df
    



