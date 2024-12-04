from evo_eq_model import *
from my_batch_runner import *

import random
import numpy.random as rnd
# import networkx as nx
from anytree import Node
from tqdm import tqdm
import numpy as np
import os
import sys
if(sys.version_info[1]<= 7):
    import pickle5 as pickle
else:
    import pickle
from datetime import datetime
import time
import resource 
import json

#pull this 
#how about this

now = datetime.now()

dt_string = now.strftime("%S-%M-%H--%d-%m-%Y")
print(dt_string)

project_name = sys.argv[1]
n_cores = int(sys.argv[2])
iterations = int(sys.argv[3])

N = int(1e6)
# beta_s = 
beta_s = list(np.logspace(0.01,np.log10(5),10))
#kappa_s = [0]
kappa_s = (np.logspace(0,3, 11)-1)
p_alpha = .1
p_alphas= [p_alpha]

# mu_s = np.logspace(1,3,5)/N
mu = 100/N
mu_s = np.array([mu])

x_values = np.logspace(-2,np.log10(2), 12)
# xi_s = np.array([2/p_alpha, 5/p_alpha])
xi_s= (1/p_alpha) *np.log(mu*N)/x_values
I0 = [10]



params = {"N": N , 'infection_rate':beta_s, 'kappa':kappa_s,
           'p_alpha': p_alphas, 'mutation_rate':mu_s, 'xi' : xi_s,
             'recovery_rate':1,  'initial_infected':I0, 'dt':None,'collect_freqs': False}#, 'runmode':'peak','collect_freqs':True}
variable_params = [p for p in params  if isinstance(params[p],np.ndarray) and len(params[p])>0]

time_start = time.perf_counter()

results = batch_run(
    eqModel,
    parameters=params,
    iterations=iterations,
    number_processes=n_cores,
    data_collection_period=-1,
    display_progress=True,
)
      
df = pd.DataFrame(results)
savedir = '/data01/dtrimcev/data/'+project_name+'/'
os.makedirs(savedir,exist_ok=True)

if params['collect_freqs']==False:
    df.to_feather(savedir+dt_string+'abm_result_feather')
else:
    
    freq_df = get_frequency_distribution_df(df,variable_params)
    print(freq_df)
    freq_df.to_feather(savedir+dt_string+'freq_distribs.feather')
    #save params to json in savedir, with dtstring_params.json as name
    with open(savedir+dt_string+'_params.json', 'w') as fp:
        json.dump(params, fp)
        
time_elapsed = (time.perf_counter() - time_start)/3600
memMb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
print ("%5.1f hours %5.1f MByte" % (time_elapsed,memMb))

