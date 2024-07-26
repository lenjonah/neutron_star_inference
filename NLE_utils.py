import math
from numbers import Number
import numpy as np
from sbi.utils import process_prior
from sbi import utils as utils
import pandas
import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        return super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale



# HMC sampler from Chirag Modi
class Sampler():

    def __init__(self):
        self.samples = []
        self.accepts = []
        self.Hs = []
        self.counts = []
        self.i = 0

    def to_array(self):
        for key in self.__dict__:
            if type(self.__dict__[key]) == list:
                self.__dict__[key] = np.array(self.__dict__[key])            

    def to_list(self):
        for key in self.__dict__:
            if type(self.__dict__[key]) == np.ndarray:
                self.__dict__[key] = list(self.__dict__[key])

    def appends(self, q, acc, Hs, count):
        self.i += 1
        self.accepts.append(acc)
        self.samples.append(q)
        self.Hs.append(Hs)
        self.counts.append(count)
        
    def save(self, path):
        pass


class HMC():

    def __init__(self, log_prob, grad_log_prob=None, log_prob_and_grad=None, invmetric_diag=None, obs_idx=None, scaler=None, output_path="./data/"):

        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.log_prob_and_grad = log_prob_and_grad
        if invmetric_diag is None: self.invmetric_diag = 1.
        else: self.invmetric_diag = invmetric_diag
        #self.metricstd = self.invmetric_diag**-0.5
        self.invinvmetric = np.linalg.inv(self.invmetric_diag)
        self.obs_idx = obs_idx
        self.scaler = scaler
        self.output_path = output_path
        
        assert not((self.grad_log_prob == None) & (self.log_prob_and_grad == None))
        #
        self.V = lambda x : self.log_prob(x)*-1.
        self.KE = lambda p: 0.5*(p[0].T * np.dot(self.invinvmetric, p[0])).sum()
        self.KE_g = lambda p: np.dot(self.invinvmetric, p[0]).reshape(p.shape) #self.invmetric_diag * p
        #
        self.leapcount = 0 
        self.Vgcount = 0 
        self.Hcount = 0
 
   
    def V_g(self, x):
        self.Vgcount += 1
        if self.grad_log_prob is not None:
            v_g = self.grad_log_prob(x)
        elif self.log_prob_and_grad is not None:
            v, v_g = self.log_prob_and_grad(x)
        return v_g *-1.

        
    def V_vandg(self, x):
        if self.log_prob_and_grad is not None:
            self.Vgcount += 1
            v, v_g = self.log_prob_and_grad(x)
            return v*-1., v_g*-1
        else:
            raise NotImplementedError
        

    def unit_norm_KE(self, p):
        return 0.5 * (p**2).sum()


    def unit_norm_KE_g(self, p):
        return p


    def H(self, q, p, Vq=None):
        if Vq is None: 
            self.Hcount += 1
            Vq = self.V(q)
        return Vq + self.KE(p)


    def leapfrog(self, q, p, N, step_size):
        self.leapcount += 1 
        q0, p0 = q, p
        try:
            p = p - 0.5*step_size * self.V_g(q) 
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q) 
            q = q + step_size * self.KE_g(p)
            p = p - 0.5*step_size * self.V_g(q) 
            return q, p

        except Exception as e:
            print("exception : ", e)
            return q0, p0


    def leapfrog_Vgq(self, q, p, N, step_size, V_q=None, V_gq=None):
        self.leapcount += 1 
        q0, p0, V_q0, V_gq0 = q, p, V_q, V_gq
        try:
            if V_gq is None:
                p = p - 0.5*step_size * self.V_g(q) 
            else:
                p = p - 0.5*step_size * V_gq
            for i in range(N-1):
                q = q + step_size * self.KE_g(p)
                p = p - step_size * self.V_g(q) 

            q = q + step_size * self.KE_g(p)
            if self.log_prob_and_grad is not None:
                V_q1, V_gq1 = self.V_vandg(q) 
            else:
                V_q1, V_gq1 = None, self.V_g(q) 
            p = p - 0.5*step_size * V_gq1
            return q, p, V_q1, V_gq1

        except Exception as e:
            print("exception : ", e)
            return q0, p0, V_q0, V_gq0


    def metropolis(self, qp0, qp1, V_q0=None, V_q1=None, u=None):
        
        q0, p0 = qp0
        q1, p1 = qp1
        H0 = self.H(q0, p0, V_q0)
        H1 = self.H(q1, p1, V_q1)
        prob = np.exp(H0 - H1)
        #prob = min(1., np.exp(H0 - H1))
        
        if u is None: u =  np.random.uniform(0., 1., size=1)
        if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0: 
            return q0, p0, -1, [H0, H1]
        elif  u > min(1., prob):
            return q0, p0, 0., [H0, H1]
        else: return q1, p1, 1., [H0, H1]


    def step(self, q, nleap, step_size, **kwargs):

        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        p = np.random.multivariate_normal(np.zeros(q.size), self.invmetric_diag).reshape(q.shape) #np.random.normal(size=q.size).reshape(q.shape) * self.metricstd
        q1, p1 = self.leapfrog(q, p, nleap, step_size)
        q, p, accepted, Hs = self.metropolis([q, p], [q1, p1])
        return q, p, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount]


    def _parse_kwargs_sample(self, **kwargs):

        self.nsamples = kwargs['nsamples']
        self.burnin = kwargs['burnin']
        self.step_size = kwargs['step_size']
        self.nleap = kwargs['nleap']


    def adapt_stepsize(self, q, epsadapt, **kwargs):
        #print("Adapting step size for %d iterations"%epsadapt)
        step_size = self.step_size
        epsadapt_kernel = DualAveragingStepSize(step_size)
        self._parse_kwargs_sample(**kwargs)
        
        for i in range(epsadapt+1):
            q, p, acc, Hs, count = self.step(q, self.nleap, step_size)
            prob = np.exp(Hs[0] - Hs[1])
            if i < epsadapt:
                if np.isnan(prob): prob = 0.
                if prob > 1: prob = 1.
                step_size, avgstepsize = epsadapt_kernel.update(prob)
                #print(step_size, avgstepsize)
            elif i == epsadapt:
                _, step_size = epsadapt_kernel.update(prob)
                #print("Step size fixed to : ", step_size)
                self.step_size = step_size
                np.save(self.output_path+'stepsize_'+str(self.obs_idx)+'.npy', self.step_size)
                
        return q

    
    def sample(self, q, p=None, callback=None, skipburn=True, epsadapt=0, **kwargs):

        kw = kwargs
        self._parse_kwargs_sample(**kwargs)
        
        state = Sampler()

        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, **kwargs)
            
        for i in range(self.nsamples + self.burnin):
            q, p, acc, Hs, count = self.step(q, self.nleap, self.step_size)
            state.i += 1
            state.accepts.append(acc)
            if skipburn & (i > self.burnin):
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                if callback is not None: callback(state)
                
            if len(state.samples) > 500 and (len(state.samples) % 50) == 0:
                hmc_samples = np.array(state.samples)
                #print(hmc_samples, len(hmc_samples), len(q[0]))
                hmc_samples = para_inverse_transform(hmc_samples.reshape((len(hmc_samples), len(q[0]))), self.scaler)
                np.save(self.output_path+'interim_hmc_chains_'+str(self.obs_idx)+'.npy', hmc_samples)
            
        state.to_array()
        return state
    
# dual averaging stepsize for dynamic stepsize determination
class DualAveragingStepSize():
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75, nadapt=0):
        self.initial_step_size = initial_step_size 
        self.mu = np.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0
        self.nadapt = nadapt
        
    def update(self, p_accept):

        if np.isnan(p_accept) : p_accept = 0.
        if p_accept > 1: p_accept = 1. 
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept
        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa
        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return np.exp(log_step), np.exp(self.log_averaged_step)

    
    def __call__(self, i, p_accept):
        if i == 0:
            return self.initial_step_size 
        elif i < self.nadapt:
            step_size, avgstepsize = self.update(p_accept)
        elif i == self.nadapt:
            _, step_size = self.update(p_accept)
            #print("\nStep size fixed to : %0.3e\n"%step_size)
        else:
            step_size = np.exp(self.log_averaged_step)
        return step_size
     
     
# read in simulated XSPEC data
theta_spectra = np.load('./data/theta_spectra.npy', allow_pickle=True)
theta_spectra = torch.tensor(theta_spectra, dtype=torch.float32)
     
# write a function to determine a prior with a gaussian uncertainties for the nuisance parameters 
# each observation is characterized by one index to determine the true value
# the number of observations is defined via the length of indices

def create_truncated_prior(indices, std=[0.3, 0.05, 0.1]):
    # standard deviations are already set to the values in Farrell et al.
    
    # uniform interval for 2 EoS parameters
    dist = [torch.distributions.Uniform(torch.tensor([4.7546]), torch.tensor([5.2544])),
        torch.distributions.Uniform(torch.tensor([-2.0476]), torch.tensor([-1.8531]))]
    
    # normal distributions for 3 nuisance parameters for each observation
    t_full = []
    for idx in indices: 
        t = torch.tensor([TruncatedNormal(torch.tensor([theta_spectra[idx][2]]), std[0] * torch.tensor([theta_spectra[idx][2]]), 0.01, 3.16).sample()[0],
            TruncatedNormal(torch.tensor([theta_spectra[idx][3]]), std[1] * torch.tensor([theta_spectra[idx][3]]), 2.3, 12.3).sample()[0],
            TruncatedNormal(torch.tensor([theta_spectra[idx][4]]), torch.tensor([std[2]]), 6, 6.3).sample()[0]
            ])
        [t_full.append(s) for s in t]
        
        dist.append(TruncatedNormal(torch.tensor([t[0]]), std[0] * torch.tensor([theta_spectra[idx][2]]), 0.01, 3.16))
        dist.append(TruncatedNormal(torch.tensor([t[1]]), std[1] * torch.tensor([theta_spectra[idx][3]]), 2.3, 12.3))
        dist.append(TruncatedNormal(torch.tensor([t[2]]), torch.tensor([std[2]]), 6, 6.3))

    # create prior using sbi procedure
    prior, *_ = process_prior(dist)
    return prior, torch.tensor(t_full)
    
    
    
# write a function to determine a prior with a gaussian uncertainties for the nuisance parameters 
# each observation is characterized by one index to determine the true value
# the number of observations is defined via the length of indices

def create_prior(indices, t_full, std=[0.3, 0.05, 0.1]):
    # standard deviations are already set to the values in Farrell et al.
    
    # uniform interval for 2 EoS parameters
    dist = [torch.distributions.Uniform(torch.tensor([4.7546]), torch.tensor([5.2544])),
        torch.distributions.Uniform(torch.tensor([-2.0476]), torch.tensor([-1.8531]))]
    
    # normal distributions for 3 nuisance parameters for each observation
    for j, idx in enumerate(indices): 
        dist.append(torch.distributions.Normal(torch.tensor([t_full[3*j]]), std[0] * torch.tensor([theta_spectra[idx][2]])))
        dist.append(torch.distributions.Normal(torch.tensor([t_full[3*j+1]]), std[1] * torch.tensor([theta_spectra[idx][3]])))
        dist.append(torch.distributions.Normal(torch.tensor([t_full[3*j+2]]), torch.tensor([std[2]])))

    # create prior using sbi procedure
    prior, *_ = process_prior(dist)
    return prior
    
    

# parameter transform functions to scale all parameters to mean 0 and standard deviation 1
def para_transform(theta, scaler): 
    return scaler.transform(theta)
    
def para_inverse_transform(theta, scaler): 
    return scaler.inverse_transform(theta)

# variance is needed because it corresponds to the Jacobian which is needed if the gradient is transformed
def para_var(grad, scaler):
    return scaler.var_
        

    
# define a set of multiple observations of several stars 
# these all have the same EoS parameters but different nuisance parameters

class obs_model_multi():

    def __init__(self, observation, prior, scaler, density_estimators):

        self.observation = observation 
        self.prior = prior 
        self.scaler = scaler
        self.density_estimators = density_estimators
        
        low_b = [4.7, -2.05] + len(self.observation) * [0.01, 2.3, 6]
        high_b = [5.3, -1.85] + len(self.observation) * [3.16, 12.3, 6.3]
        self.boundaries = utils.BoxUniform(low=low_b, high=high_b)
        
    # log probability 
    def log_probability(self, theta): 
        # the input theta is scaled to std 1 and mean 0
        # but the density estimator is trained on unscaled data
        # hence need first to inverse transform the input theta
        theta1 = torch.tensor(para_inverse_transform(theta, self.scaler), dtype=torch.float32)
        theta1[:,0] = 5.0
        theta1[:,1] = -1.95
        
        # log probability of total prior for all observations for given theta
        log_p = self.prior.log_prob(theta1).detach() * len(self.density_estimators)

        theta = torch.tensor(para_inverse_transform(theta, self.scaler), dtype=torch.float32)
        # log likelihood for given theta and multiple observations
        for i, o in enumerate(self.observation):
            # create correct five dimensional parameter value corresponding for one particular observation
            t = torch.cat([theta[:,:2], theta[:,i*3+2:i*3+5]], 1)
            # log likelihood for this particular observation 'o' computed using the density estimator
            for d in self.density_estimators:
                log_p +=  d.log_prob(o * torch.ones((len(t),250)), t).detach()
            
        log_p = log_p/len(self.density_estimators)        
        
        log_p = torch.where(self.prior.support.check(theta), log_p, -torch.inf)
        
        return log_p.numpy() 

    # gradient of log probability
    def grad_log_probability(self, theta):  
        
        # see above, have to inverse transform input theta
        # change to torch tensor with gradient
        theta_test = para_inverse_transform(theta, self.scaler)
        theta_test[:,0] = 5.0
        theta_test[:,1] = -1.95
        theta2 = torch.tensor(theta_test, dtype=torch.float32, requires_grad=True)
        self.prior.log_prob(theta2).backward()
        # take gradient of the total prior
        grad = theta2.grad * len(self.density_estimators)
        
        # gradient of the log likelihood for given theta and multiple observations
        # see above, have to inverse transform input theta
        theta = torch.tensor(para_inverse_transform(theta, self.scaler))
        for i, o in enumerate(self.observation):
            # create correct five dimensional parameter value corresponding for one particular observation
            t = torch.cat([theta[:,:2], theta[:,i*3+2:i*3+5]], 1).numpy()
            for d in self.density_estimators: 
                # change to torch tensor with gradient
                theta1 = torch.tensor(t, dtype=torch.float32, requires_grad=True)
                # log likelihood for this particular observation computed using the density estimator
                d.log_prob(o * torch.ones((len(theta1),250)), theta1).backward()
                
                # gradient of log likelihood for EoS parameters
                grad[:,:2] += theta1.grad[:,:2]
                # gradient of log likelihood for nuisance parameters
                grad[:,i*3+2:i*3+5] += theta1.grad[:,2:]

        grad = grad/len(self.density_estimators)
        # multiply gradient by Jacobian of the transformation
        return (grad * para_var(grad, self.scaler)**(1/2)).numpy()