'''
Code to run Hamiltonian Monte Carlo to compute posterior samples for the EoS and nuisance parameters directly based on telescope spectra 
'''

# import necessary packages, make sure to install requirements
import torch
import numpy as np
import pandas
import pickle
import time
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import sys
import os
import warnings

from NLE_utils import HMC, TruncatedNormal, create_truncated_prior, create_prior, obs_model_multi, para_transform, para_inverse_transform

# path containing the telescope spectra as well as the neutron star parameters
INPUT_PATH = "./data/"

# code runs in parallel using joblib, number of workers also specifies the number of chains
NUM_WORKERS = 16 
# reduce length of the chain and stepsize adaptation to test the code
CHAIN_LENGTH = 2000 
ADAPT_STEPS = 300 

# observation index which specifies the telescope spectra used as observations, has to be an integer between 0 and 148 for the test set
OBS_IDX = int(sys.argv[1])
if OBS_IDX > 148 or OBS_IDX < 0: 
     warnings.warn("The specified OBS_IDX is outside the test data.", UserWarning)

# number of used density estimators
NUM_DENSITY_ESTIMATORS = int(sys.argv[2])
if NUM_DENSITY_ESTIMATORS > 5 or NUM_DENSITY_ESTIMATORS < 0: 
     warnings.warn("The specified NUM_DENSITY_ESTIMATORS is outside 1 - 5.", UserWarning)

# specifiy the nuisance parameter scenario, i.e., the prior uncertainties of the nuisance parameters
#SCENARIO = 'tight'
SCENARIO = 'loose'
OUTPUT_PATH = "./data/{}_".format(SCENARIO)

print ("STARTING HMC for the following hyperparameters:")
print ("OBS_IDX ",OBS_IDX, "NUM_DENSITY_ESTIMATORS", NUM_DENSITY_ESTIMATORS, "SCENARIO", SCENARIO)
print ("NUM_WORKERS", NUM_WORKERS ,"CHAIN_LENGTH ", CHAIN_LENGTH, "ADAPT_STEPS", ADAPT_STEPS)


########### Preparation ###########

# read in telescope spectra simulated with XSPEC including Poisson noise 
# as well as the corresponding EoS and nuisance parameters 
spectra_noisy = np.load('./data/spectra_noisy.npy', allow_pickle=True)
theta_spectra = np.load('./data/theta_spectra.npy', allow_pickle=True)

# convert to torch tensors
theta_spectra = torch.tensor(theta_spectra, dtype=torch.float32)
spectra_noisy = torch.tensor(spectra_noisy, dtype=torch.float32)

# load beforehand trained density estimators 
density_estimators = []
for f in range(1, NUM_DENSITY_ESTIMATORS+1):
    filename = INPUT_PATH + "density_estimator_top" + str(f) + ".pkl"
    with open(filename, "rb") as g:
        density_estimators.append(pickle.load(g))
        
start_time = time.time()   

# specifiy the indices of the 10 spectra used as observations  
all_obs_indices = np.load('./data/all_obs_indices.npy', allow_pickle=True)
obs_indices = all_obs_indices[OBS_IDX]
obs_idx = obs_indices[0]
# create total observation of all stars
o = [spectra_noisy[i] for i in obs_indices]
 
# delete tensors to free memory if running in parallel
del spectra_noisy
del theta_spectra  



########### create prior and starting values ###########

# create total prior for 2 EoS parameters and (3 * number of observations) nuisance parameters
# for the 10 spectra used here there are 32 parameters in total
# first specifiy the uncertainties given by the chosen nuisance parameter scenario
if SCENARIO == 'tight':
    std = [0.3, 0.05, 0.1]
elif SCENARIO == 'loose':
    std = [0.5, 0.2, 0.2]
    
# create a truncated version of the prior to set the starting values
# and a non-truncated version used in HMC
prior_trunc, t_full = create_truncated_prior(obs_indices, std)
prior = create_prior(obs_indices, t_full, std)

# importance sampling of the starting values of the HMC chains
# sample large number of parameter sets from truncated prior
start = prior_trunc.sample((500000,))

# fit standard scaler to transform parameters to std 1 and mean 0
scaler = StandardScaler()
scaler.fit(start.numpy())

# create observation model to compute log probability and gradient of log probability using the density estimators
test_observation = obs_model_multi(o, prior, scaler, density_estimators)

# compute log probability of above prior samples
w = test_observation.log_probability(para_transform(start, scaler))

# use (number of chains) most likely parameter values as starting values for the HMC chains
qstart = start[np.argsort(w)[-NUM_WORKERS:]].numpy()

step1 = time.time()
print('FINISHED importance sampling in', round((step1 - start_time)/60, 2), 'minutes.')
print('Running', NUM_WORKERS, 'HMC chains of length', str(CHAIN_LENGTH) + '.', '\nThe stepsize is adapted for', ADAPT_STEPS, 'steps.')



########### Run HMC ###########

# set inverse mass matrix for the HMC sampler, this matrix was empirically found
# could in principle be determined from an estimate of covariance of the posterior
inv_matrix = np.concatenate([[3,3], np.ones(len(qstart[0]) - 2)]) * np.eye(len(qstart[0]))

# set off-diagonal elements for the EoS parameters 
inv_matrix[0,1] = -0.5
inv_matrix[1,0] = -0.5

# create sampler object with above defined inverse mass matrix
sampler = HMC(test_observation.log_probability, grad_log_prob=test_observation.grad_log_probability, invmetric_diag=inv_matrix, obs_idx=OBS_IDX, scaler=scaler, output_path=OUTPUT_PATH)

def obtain_samples(i):
    '''
    run one HMC chain to determine posterior samples
    
    :i:         number of the chain, determines e.g. the starting value
    :return:    values of the HMC chain
    '''
    
    # set hyperparameters: number of leap frog steps, length of chains, starting step size, burn-in, how many steps are used to adapt the step size
    sampler = HMC(test_observation.log_probability, grad_log_prob=test_observation.grad_log_probability, invmetric_diag=inv_matrix, obs_idx=str(OBS_IDX)+'_'+str(i), scaler=scaler, output_path=OUTPUT_PATH)
    hmc_samples = sampler.sample(para_transform(np.array([qstart[i]]), scaler), nleap=40, nsamples=CHAIN_LENGTH, step_size=0.02, skip=0, burnin=0, epsadapt=ADAPT_STEPS).samples
    
    # result has to be inverse transformed and reshaped
    return para_inverse_transform(hmc_samples.reshape((len(hmc_samples), len(qstart[0]))), scaler)

# save observation indices and prior information
np.save(OUTPUT_PATH+'obs_indices_'+str(OBS_IDX)+'.npy', obs_indices)
np.save(OUTPUT_PATH+'priors_'+str(OBS_IDX)+'.npy', t_full)

# run multiple HMC chains in parallel using joblib 
hmc_samples_chains = np.array(Parallel(n_jobs=-1, verbose=0)(delayed(obtain_samples)(i) for i in range(len(qstart))))

end_time = time.time()
print('FINISHED: HMC in', round((end_time - step1)/60,2), 'minutes.')

timestr = time.strftime("%Y%m%d%H%M%S")

# save results
np.save(OUTPUT_PATH+'hmc_chains_'+str(OBS_IDX)+'.npy', hmc_samples_chains)

# delete interim files used to save intermediate results
for i in range(len(qstart)):
    myfile = OUTPUT_PATH+'stepsize_'+str(OBS_IDX)+'_'+str(i)+'.npy'
    # If file exists, delete it.
    if os.path.isfile(myfile):
        os.remove(myfile)
        
    myfile = OUTPUT_PATH+'interim_hmc_chains_'+str(OBS_IDX)+'_'+str(i)+'.npy'
    # If file exists, delete it.
    if os.path.isfile(myfile):
        os.remove(myfile)
