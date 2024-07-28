# Neural Simulation-Based Inference of the Neutron Star Equation of State directly from Telescope Spectra

Neutron stars provide a unique window into the unknown physics of extremely dense matter. Their internal structure is described by the relationship between pressure and energy density, $P(\varepsilon)$, commonly referred to as the **equation of state** (EoS). Many efforts have been made to constrain the EoS based on astrophysical observations of neutron stars using Bayesian inference methods. Since the likelihood for the astrophysical detector data is not analytically available, the conventional inference is carried out in two steps. We have implemented a novel method using recently developed simulation-based inference methods to infer the parameters $[\lambda_1, \lambda_2]$ of an parametrization of the EoS directly from telescope spectra of low-mass X-ray binaries in quiescence. These spectra depend on the mass $M$ and radius $R$ of the star, as well as on additional nuisance parameters $[N_H, d, \log(T_\mathrm{eff})]$. 

In an approach called **neural likelihood estimation**, normalizing flows are trained on simulated data to approximate the analytically unavailable likelihood. Because normalizing flows are by design differentiable, this method allows the use of improved methods for sampling the posterior, such as **Hamiltonian Monte Carlo**, which scale much better to higher-dimensional parameter spaces. The NLE + HMC approach outperforms previous methods and scales better to the growing number of observations expected in the coming years compared to the conventional two-step methods. More details can be found in our paper (https://arxiv.org/pdf/2403.00287). This repository contains the code used in our analysis to allow future studies to build on our progress.

![alt text](https://github.com/lenjonah/neutron_star_inference/blob/main/illustration.png)

## Organization

This repository is organized as follows: 

- `requirements.txt` contains the packages beyond standard ones necessary to run the code 
- `NLE_spectra_HMC_xspec.py` python script to run multiple HMC chains in parallel to sample the posterior for a given observation of telescope spectra
- `NLE_utils.py` utility functions necessary to run HMC
- `working_examples.ipynb` jupyter notebook illustrating how to train the normalizing flows based on the simulated data and how to analyze the HMC chains
- `\data` subdirectory contains pretrained normalizing flows, one example output of an HMC run and the parameters of the simulated spectra

## Execution

In order to run the code provided in this repository follow the following steps: 

- install `requirements.txt`
- download the telescope spectra `spectra_noisy.npy` (which are to large for github) from this link (https://tumde-my.sharepoint.com/:f:/g/personal/len_brandes_tum_de/EtNEvqZjljBApBDcoBC8C7EBgEnnlG-Y4SQX0X7j7XZDMg?e=tTRbo7)
- optional: take a look at `working_examples.ipynb` to understand the spectral data and the training of the normalizing flows
- run `py NLE_spectra_HMC_xspec_mass.py OBS_IDX NUM_DENSITY_ESTIMATORS` to sample the posterior for a provided number of normalizing flows `NUM_DENSITY_ESTIMATORS` between 1 - 5 and an observation index `OBS_IDX` that specifies the spectra used as observations (between 0 and 148 for the test set)

## Citation

If you use the code or the data provided in this repository, please cite: 

````
@article{Brandes2024,
    author = "Brandes, Len and Modi, Chirag and Ghosh, Aishik and Farrell, Delaney and Lindblom, Lee and Heinrich, Lukas and Steiner, Andrew W. and Weber, Fridolin and Whiteson, Daniel",
    title = "{Neural Simulation-Based Inference of the Neutron Star Equation of State directly from Telescope Spectra}",
    eprint = "2403.00287",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "3",
    year = "2024"
}
````
