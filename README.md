# Neural Simulation-Based Inference of the Neutron Star Equation of State directly from Telescope Spectra

Neutron stars provide a unique window into the unknown physics of extremely dense matter. Their internal structure is described by the relationship between pressure and energy density, commonly reffered to as the equation of state (EoS). Many efforts are put to constrain this equation of state based on astrophysical observations of neutron stars using Bayesian inference methods. However, since the likelihood for the neutron star detector data is analytically unavailable, the conventional inference is carried out in two steps. We utilized a novel method using recently developed simulation-based inference methods to infer the equation of state directly from the telescope spectra of low-mass X-ray binaries in quiescence. In an approach called neural likelihood estimation, normalizing flows are trained on simulated data to approximate the analytically unavailable likelihood. Because normalizing flows are differentiable, this methods allows the usage of improved sampling methods such as Hamiltonian Monte Carlo sampling that scale much better to higher dimensional parameter spaces. The NLE + HMC approach outperforms previous methods and better scales to the growing number of observations expected in the coming years compared to the conventional two-step methods. For more details view our paper (https://arxiv.org/pdf/2403.00287). This repository contains the code used in our analysis to allow future studies to build on our progress.

## Organization

This repository is organized as follows: 

- 

If you use the code or the data provided in this repository, please cite: 

@article{Brandes2024,
    author = "Brandes, Len and Modi, Chirag and Ghosh, Aishik and Farrell, Delaney and Lindblom, Lee and Heinrich, Lukas and Steiner, Andrew W. and Weber, Fridolin and Whiteson, Daniel",
    title = "{Neural Simulation-Based Inference of the Neutron Star Equation of State directly from Telescope Spectra}",
    eprint = "2403.00287",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    month = "3",
    year = "2024"
}

