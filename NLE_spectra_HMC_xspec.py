import torch
import numpy as np
import pandas
import pickle
import time
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import sys
import os

from NLE_utils import HMC, TruncatedNormal, create_truncated_prior, create_prior, obs_model_multi, para_transform, para_inverse_transform

INPUT_PATH = "./data/"

NUM_WORKERS = 16 #16
CHAIN_LENGTH = 2000 #2000
ADAPT_STEPS = 300 #300
#OBS_IDX = int(input("Enter idx of EoS parameters (has to be between 0 and 148): "))
OBS_IDX = int(sys.argv[1])
#NUM_DENSITY_ESTIMATORS = int(input("Enter number of density estimators (has to be between 1 and 10): "))
NUM_DENSITY_ESTIMATORS = int(sys.argv[2])
#SCENARIO = 'tight'
SCENARIO = 'loose'
OUTPUT_PATH = "./data/{}_".format(SCENARIO)


print ("OBS_IDX ",OBS_IDX, "NUM_DENSITY_ESTIMATORS", NUM_DENSITY_ESTIMATORS, "SCENARIO", SCENARIO )


# read in simulated XSPEC data
spectra_noisy = np.load('./data/spectra_noisy.npy', allow_pickle=True)
theta_spectra = np.load('./data/theta_spectra.npy', allow_pickle=True)

theta_spectra = torch.tensor(theta_spectra, dtype=torch.float32)
spectra_noisy = torch.tensor(spectra_noisy, dtype=torch.float32)


# load beforehand trained density estimators
density_estimators = []
for f in range(1, NUM_DENSITY_ESTIMATORS+1):
    filename = INPUT_PATH + "density_estimator_top" + str(f) + ".pkl"
    with open(filename, "rb") as g:
        density_estimators.append(pickle.load(g))
    
    
start_time = time.time()   

# randomly sample 10 observation stars they have to have the same EoS parameters 
all_obs_indices = np.load('./data/all_obs_indices.npy', allow_pickle=True)
obs_indices = all_obs_indices[OBS_IDX]
obs_idx = obs_indices[0]
# create total observation of all stars
o = [spectra_noisy[i] for i in obs_indices]
 
del spectra_noisy
del theta_spectra  
 
# create total prior for 2 EoS parameters and (3 * number of observations) nuisance parameters
# here: 32 parameters in total
if SCENARIO == 'tight':
    std = [0.3, 0.05, 0.1]
elif SCENARIO == 'loose':
    std = [0.5, 0.2, 0.2]
    
prior_trunc, t_full = create_truncated_prior(obs_indices, std)
prior = create_prior(obs_indices, t_full, std)


# importance sampling of the starting values of the HMC chains
# sample huge number of parameters from truncated prior
start = prior_trunc.sample((500000,)) #500000

# fit standard scaler to transform parameters to std 1 and mean 0
scaler = StandardScaler()
scaler.fit(start.numpy())

# create observation model to compute log probability and gradient of log probability
test_observation = obs_model_multi(o, prior, scaler, density_estimators)



# compute log probability of above prior samples
w = test_observation.log_probability(para_transform(start, scaler))
# use (number of chains) most likely parameter values as starts for the HMC chains
qstart = start[np.argsort(w)[-NUM_WORKERS:]].numpy()
step1 = time.time()
print('FINISHED: importance sampling in', round((step1 - start_time)/60, 2), 'minutes.')
print('Running', NUM_WORKERS, 'HMC chains of length', str(CHAIN_LENGTH) + '.', '\nThe stepsize is adapted for', ADAPT_STEPS, 'steps.')



# set inverse mass matrix for the HMC sampler, this matrix was empirically found to lead to alright results
inv_matrix = np.concatenate([[3,3], np.ones(len(qstart[0]) - 2)]) * np.eye(len(qstart[0]))
# set off-diagonal elements for the EoS parameters 
inv_matrix[0,1] = -0.5
inv_matrix[1,0] = -0.5

# create sampler with above defined inverse mass matrix
sampler = HMC(test_observation.log_probability, grad_log_prob=test_observation.grad_log_probability, invmetric_diag=inv_matrix, obs_idx=OBS_IDX, scaler=scaler, output_path=OUTPUT_PATH)

# function to create samples for one idx which corresponds to the respective chain 
def obtain_samples(i):
    # here have to set hyperparameters: number of leap frog steps, length of chains, starting step size, burn-in, how many steps are used to adapt the step size
    sampler = HMC(test_observation.log_probability, grad_log_prob=test_observation.grad_log_probability, invmetric_diag=inv_matrix, obs_idx=str(OBS_IDX)+'_'+str(i), scaler=scaler, output_path=OUTPUT_PATH)
    hmc_samples = sampler.sample(para_transform(np.array([qstart[i]]), scaler), nleap=40, nsamples=CHAIN_LENGTH, step_size=0.02, skip=0, burnin=0, epsadapt=ADAPT_STEPS).samples
    # result has to be inverse transformed and reshaped
    return para_inverse_transform(hmc_samples.reshape((len(hmc_samples), len(qstart[0]))), scaler)

np.save(OUTPUT_PATH+'obs_indices_'+str(OBS_IDX)+'.npy', obs_indices)
np.save(OUTPUT_PATH+'priors_'+str(OBS_IDX)+'.npy', t_full)

# run HMC in parallel  
hmc_samples_chains = np.array(Parallel(n_jobs=-1, verbose=0)(delayed(obtain_samples)(i) for i in range(len(qstart))))

end_time = time.time()
print('FINISHED: HMC in', round((end_time - step1)/60,2), 'minutes.')

timestr = time.strftime("%Y%m%d%H%M%S")

# save results
np.save(OUTPUT_PATH+'hmc_chains_'+str(OBS_IDX)+'.npy', hmc_samples_chains)

# delete interim files 
for i in range(len(qstart)):
    myfile = OUTPUT_PATH+'stepsize_'+str(OBS_IDX)+'_'+str(i)+'.npy'
    # If file exists, delete it.
    if os.path.isfile(myfile):
        os.remove(myfile)
        
    myfile = OUTPUT_PATH+'interim_hmc_chains_'+str(OBS_IDX)+'_'+str(i)+'.npy'
    # If file exists, delete it.
    if os.path.isfile(myfile):
        os.remove(myfile)