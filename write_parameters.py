###########################################
##### Write out
#####  (i) results from original HS run (posteriors, data seen etc.)
#####  (ii) baseline parameters for bootstrapping
#####  (iii) residual matrix
###########################################
# %%
# %%
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from collections import OrderedDict
import scipy.stats as stats
import scipy.linalg as linalg
import argparse
import os
import pickle as pkl

PKL_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/all91.pkl"
PRIOR_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/bandit-prior.RData"
NDAYS = 90
NUSERS = 91
NTIMES = 5

LAMBDA = 0.95

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation", "temperature", "logpresteps", "sqrt_totalsteps"]
G_LEN = len(G_KEYS)

E0 = 0.2
E1 = 0.1

true_mu= np.random.uniform(-1,1,(18,))
true_sigma= np.random.uniform(0,.1,(18,18))
true_sigma=np.matmul(np.transpose(true_sigma), true_sigma)

# %%
# Load data
def load_data():
    with open(PKL_DATA_PATH, "rb") as f:
        data = pkl.load(f)
    return data

# %%
# Load initial run result
def load_initial_run(residual_path, baseline_thetas_path, baseline):
    with open(residual_path, "rb") as f:
        residual_matrix = pkl.load(f)
    with open(baseline_thetas_path, "rb") as f:
        baseline_pickle = pkl.load(f)
    if baseline == "Prior":
        baseline_thetas = baseline_pickle["prior"]
    elif baseline == "Posterior":
        baseline_thetas = baseline_pickle["posterior"]
    elif baseline == "Zero":
        baseline_thetas = baseline_pickle["all0TxEffect"]
    else:
        raise ValueError("Invalid baseline")

    return residual_matrix, baseline_thetas

# %%
def determine_user_state(data, dosage):
    '''Determine the state of each user at each time point'''
    availability = data[2]

    features = {}

    features["engagement"] = data[7]
    features["other_location"] = data[8]
    # features["work_location"] = data[9]
    features["variation"] = data[10]
    features["temperature"] = data[11]
    features["logpresteps"] = data[12]
    features["sqrt_totalsteps"] = data[13]
    features["prior_anti"] = data[14]
    features["dosage"] = LAMBDA * dosage + features["prior_anti"]

    features["intercept"] = 1

    fs = np.array([features[k] for k in F_KEYS])
    gs = np.array([features[k] for k in G_KEYS])

    prob=data[3]
    action=data[4]
    reward = data[5]
    
    return availability, fs, gs, features["dosage"], reward, action, prob

# %%
def load_priors():
    '''Load priors from RData file'''
    robjects.r['load'](PRIOR_DATA_PATH)
    priors = robjects.r['bandit.prior']
    alpha_pmean = np.array(priors.rx2("mu1"))
    alpha_psd = np.array(priors.rx2("Sigma1"))
    beta_pmean = np.array(priors.rx2("mu2"))
    beta_psd = np.array(priors.rx2("Sigma2"))
    sigma = float(priors.rx2("sigma")[0])

    prior_sigma = linalg.block_diag(alpha_psd, beta_psd, beta_psd)
    prior_mu = np.concatenate([alpha_pmean, beta_pmean, beta_pmean])

    return prior_sigma, prior_mu, sigma

# %%
def get_priors_alpha_beta(post_mu, post_sigma):
    '''Get alpha and beta priors from mu and sigma'''
    alpha_pmean = post_mu[:G_LEN].flatten()
    alpha_psd = post_sigma[:G_LEN, :G_LEN]
    beta_pmean = post_mu[-F_LEN:].flatten()
    beta_psd = post_sigma[-F_LEN:, -F_LEN:]

    return alpha_pmean, alpha_psd, beta_pmean, beta_psd

# %%
def sample_lr_params(alpha_pmean, alpha_psd, beta_pmean, beta_psd, sigma):
    '''Sample alpha, beta and noise from priors for BLR'''

    alpha0 = np.random.multivariate_normal(alpha_pmean, alpha_psd)
    alpha1 = np.random.multivariate_normal(beta_pmean, beta_psd)
    beta = np.random.multivariate_normal(beta_pmean, beta_psd)
    et = np.random.normal(0, np.sqrt(sigma**2))

    return alpha0, alpha1, beta, et

# %%
def clip(x, eta = .2):
    '''Clipping function'''
    return min(1 - E0, max(x, E1))

# %%
def calculate_post_prob(fs, post_mu, post_sigma, eta = 0):
    '''Calculate the posterior probability of Pr(fs * b > eta)'''

    # Get beta's posterior mean and covariance
    _, _, beta_pmean, beta_psd = get_priors_alpha_beta(post_mu, post_sigma)

    # Calculate the mean of the fs*beta distribution
    fs_beta_mean = fs.T.dot(beta_pmean)

    # Calculate the variance of the fs*beta distribution
    fs_beta_cov = fs.T @ beta_psd @ fs

    # Calculate the probability of Pr(fs * b > eta) using cdf
    post_prob = 1 - stats.norm.cdf(eta, fs_beta_mean, np.sqrt(fs_beta_cov))

    # Clip the probability
    phi_prob = clip(post_prob)
    
    return phi_prob

# %%
def calculate_phi(prob_matrix, action_matrix, fs_matrix, gs_matrix):
    '''Calculate phi for each user at each time point'''
    Phi = np.expand_dims(np.hstack((gs_matrix, fs_matrix * prob_matrix.reshape(-1, 1), \
                (fs_matrix * (action_matrix - prob_matrix).reshape(-1, 1)))), axis=2)
    return Phi

# %%
def calculate_post_sigma(prior_sigma, sigma, availability_matrix, Phi):
    '''Calculate the posterior sigma'''

    # Phi squared
    Phi_square = np.multiply(Phi, Phi.transpose(0, 2, 1))

    # Sum of availability times Phi squared
    avail_phi_squared_sum = np.sum(np.multiply(availability_matrix.reshape(-1, 1, 1), Phi_square), axis=0) / (sigma**2)

    # Posterior sigma
    post_sigma = np.linalg.inv(np.linalg.inv(prior_sigma) + avail_phi_squared_sum)

    return post_sigma

# %%
def calculate_post_mu(prior_sigma, prior_mu, sigma, availability_matrix, reward_matrix, Phi, post_sigma):
    '''Calculate the posterior mu'''

    # Product of prior sigma inverse and prior mu
    sig_mu = (np.linalg.inv(prior_sigma) @ prior_mu.T).reshape(-1, 1)
    
    # Product of Phi and reward
    Phi_reward = np.multiply(Phi, reward_matrix.reshape(-1, 1, 1))

    # Sum of availability times Phi and reward
    avail_phi_reward_sum = np.sum(np.multiply(availability_matrix.reshape(-1, 1, 1), Phi_reward), axis=0)

    # Posterior mu
    post_mu = (post_sigma @ (sig_mu + avail_phi_reward_sum)) / (sigma ** 2)

    return post_mu

# %%
def calculate_posterior(prior_sigma, prior_mu, sigma, availability_matrix, prob_matrix, reward_matrix, action_matrix, fs_matrix, gs_matrix):
    '''Calculate the posterior distribution'''
    
    # Calculate phi(s, a)
    Phi = calculate_phi(prob_matrix, action_matrix, fs_matrix, gs_matrix)

    # Calculate posterior sigma
    post_sigma = calculate_post_sigma(prior_sigma, sigma, availability_matrix, Phi)

    # Calculate posterior mu
    post_mu = calculate_post_mu(prior_sigma, prior_mu, sigma, availability_matrix, reward_matrix, Phi, post_sigma)
    
    return post_mu, post_sigma

# %%
def select_action(p):
    '''Select action from bernoulli distribution with probability p'''
    return stats.bernoulli.rvs(p)

# o.w. 
def calculate_reward(mu, sigma, fs, gs, prob,action):
    alpha0 =  mu[:len(G_KEYS)].flatten()
    alpha1 =  mu[-len(F_KEYS):].flatten()
    beta=mu[-len(F_KEYS):].flatten()   
        
    reward = gs @ alpha0 + (prob * (fs @ alpha1)) + (action - prob) * (fs @ beta)

    return reward

# %%
def run_algorithm(data, user):
    '''Run the algorithm for each user and each bootstrap'''

    rewards=data[:,5]
    rewards=list(rewards[~np.isnan(rewards)])
    imputeRewardValue=sum(rewards)/len(rewards)
    
    # Load priors
    prior_sigma, prior_mu, sigma = load_priors()

    # Initializing dosage to first dosage value (can be non-zero if user was already in the trial)
    dosage = data[0][6]

    # Posterior initialized using priors
    post_sigma, post_mu = np.copy(prior_sigma), np.copy(prior_mu)
    post_mu=np.reshape(post_mu, (post_mu.shape[0], 1))
    
    # DS to store availability, probabilities, features, actions, posteriors and rewards
    availability_matrix = np.zeros((NDAYS * NTIMES))
    prob_matrix = np.zeros((NDAYS * NTIMES))
    reward_matrix = np.zeros((NDAYS * NTIMES))
    action_matrix = np.zeros((NDAYS * NTIMES))
    fs_matrix = np.zeros((NDAYS * NTIMES, F_LEN))
    gs_matrix = np.zeros((NDAYS * NTIMES, G_LEN))
    post_mu_matrix = np.zeros((NDAYS * NTIMES, G_LEN + 2 * F_LEN))
    post_sigma_matrix = np.zeros((NDAYS * NTIMES, G_LEN + 2 * F_LEN, G_LEN + 2 * F_LEN))
    
    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):

            # Get the current timeslot
            ts = (day) * 5 + time
            
            # State of the user at time ts
            availability, fs, gs, dosage, reward, action, prob_fsb = determine_user_state(data[ts], dosage)

            if np.isnan(reward):
                reward=imputeRewardValue

            # Save user's availability
            availability_matrix[ts] = availability
            
            # If user is available
            if availability == 1:
                # Calculate probability of (fs x beta) > n
                #prob_fsb = calculate_post_prob(fs, post_mu, post_sigma)

                # Sample action with probability prob_fsb from bernoulli distribution
                #action = select_action(prob_fsb)
                
                # Save probability, features, action and reward
                prob_matrix[ts] = prob_fsb
                action_matrix[ts] = action
                reward_matrix[ts] = reward#calculate_reward(true_mu, true_sigma,fs, gs, prob_fsb, action)

            # Save features and state
            fs_matrix[ts] = fs
            gs_matrix[ts] = gs
                        
            post_mu_matrix[ts] = post_mu[:,0]
            post_sigma_matrix[ts] = post_sigma
                            
        # Update posterior
        post_mu, post_sigma = calculate_posterior(prior_sigma, prior_mu, sigma, availability_matrix[:ts + 1], 
                                                    prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                    action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1])

    result = {"availability": availability_matrix, "prob": prob_matrix, "action": action_matrix, "reward": reward_matrix,
            "fs": fs_matrix, "gs": gs_matrix, "post_mu": post_mu_matrix, "post_sigma": post_sigma_matrix}
    return result
                                                    

def initial_run():
    data = load_data()
    allResults=[]
    for i in range(NUSERS):
        result_i=run_algorithm(data[i],i)
        allResults.append(result_i)
    return allResults,data

def get_residual_pairs(results, baseline="Prior"):
    prior_sigma, prior_mu, sigma = load_priors()
    residual_matrix = np.zeros((NUSERS, NDAYS * NTIMES))
    baseline_thetas={}
    for user in range(NUSERS):
        posterior_user_T=results[user]['post_mu'][NDAYS-1]
        for day in range(NDAYS):
            for time in range(NTIMES):
                ts = (day) * 5 + time
                gs=results[user]['gs'][ts]
                fs=results[user]['fs'][ts]
                prob=results[user]['prob'][ts]
                action=results[user]['action'][ts]
                reward=results[user]['reward'][ts]

                # get estimated reward,
                alpha = posterior_user_T[:G_LEN].flatten()
                alpha2=posterior_user_T[G_LEN:G_LEN+F_LEN].flatten()
                beta = posterior_user_T[-F_LEN:].flatten()
    
                baseline_reward= gs[1] * alpha[1] #dosage*alpha_1
                estimated_reward = baseline_reward + prob * (fs @ alpha2) + (action - prob) * (fs @ beta)
                residual_matrix[user, ts]=reward-estimated_reward

        baseline_theta=posterior_user_T
        if baseline=="Prior":
            baseline_theta[-len(F_KEYS):]=prior_mu[-len(F_KEYS):]
        elif baseline=="ZeroAtAll": # need to add in additional for target states 0
            baseline_theta[-len(F_KEYS):]=np.zeros(prior_mu[-len(F_KEYS):].shape)
        baseline_thetas[user]=baseline_theta
    
    return residual_matrix, baseline_thetas

np.random.seed(0)

result,data=initial_run()

res_matrix, baseline_prior = get_residual_pairs(result, "Prior")
res_matrix, baseline_Posterior=get_residual_pairs(result, "Posterior")
res_matrix, baseline_0Tx=get_residual_pairs(result, "ZeroAtAll")

baselines={"prior": baseline_prior, "posterior": baseline_Posterior, "all0TxEffect": baseline_0Tx}

with open('/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/baseline_parameters.pkl', 'wb') as handle:
    pkl.dump(baselines, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
with open('/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/residual_matrix.pkl', 'wb') as handle:
    pkl.dump(res_matrix, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
with open('/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/original_result_91.pkl', 'wb') as handle:
    pkl.dump(result, handle, protocol=pkl.HIGHEST_PROTOCOL)
