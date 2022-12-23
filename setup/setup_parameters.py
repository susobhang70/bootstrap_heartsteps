############################################################################################################################################################################
##### This file writes out the parameter files necessary for conducting bootstrap simulations. In particular, it includes the following:
#####  (i) results from original run (posteriors and data). 
#####           It is saved as an array allResults, with each index holding a result dictionary corresponding to each user
#####  (ii) baseline parameters for bootstrapping: what parameters should be used in reward calculation during the parametric bootstrap? 
#####           This will tell you under "posterior", "0TxEffect", "prior", and "0TxEffect_beta_i" which has a list with indices in it. 
#####           The 0TxEffect_beta_i was added to 0 out coef for var index i, according to F_KEYS. This was added for the 0'ing out in interestingness bootstrap!
#####           It is stored as a 2d array where first dimension is on the user index, and the second holds the params in an np array
#####  (iii) residual matrix: the residuals for each user based on the posterior fit. 
#####           It is a matrix of NUSERS x T
#####  Notes: residual matrix and rest are calculated only at available and non nan reward times.
############################################################################################################################################################################

# %%
import pandas as pd
import pickle as pkl
import numpy as np
import rpy2.robjects as robjects
from collections import OrderedDict
import scipy.stats as stats
import scipy.linalg as linalg
import argparse
import os
import copy
import skfda.representation.basis as basis
import itertools
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
rpackages.importr('fda')
import random

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.7g" % x))

# %%
PKL_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/all91.pkl"
PRIOR_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/bandit-prior.RData"
PRIOR_NEW_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/bandit-spec-new.RData"
NDAYS = 90
NUSERS = 91
NTIMES = 5

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "temperature", "logpresteps", "sqrt_totalsteps", "dosage", "engagement", "other_location", "variation"]
G_LEN = len(G_KEYS)

robjects.r['load'](PRIOR_NEW_DATA_PATH)
banditSpec=robjects.r['bandit.spec'] 
PSED=banditSpec.rx2("p.sed")[0]
W=banditSpec.rx2("weight.est")[0]
GAMMA=banditSpec.rx2("gamma")[0]
etaInit=banditSpec.rx2("eta.init")
alpha0_pmean = np.array(banditSpec.rx2("mu0"))
alpha0_psd = np.array(banditSpec.rx2("Sigma0"))

# %%
# Load data
def load_data():
    with open(PKL_DATA_PATH, "rb") as f:
        data = pkl.load(f)
    return data

# %%
def determine_user_state(data):
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

    # calculating dosage
    features["dosage"] = data[6] # already standardized

    features["intercept"] = 1

    fs = np.array([features[k] for k in F_KEYS])
    gs = np.array([features[k] for k in G_KEYS])

    reward = data[5]
    prob = data[3]
    action = data[4]

    return availability, fs, gs, reward, prob, action

# %%
def load_priors():
    '''Load priors from RData file'''
    robjects.r['load'](PRIOR_DATA_PATH)
    priors = robjects.r['bandit.prior']
    alpha0_pmean = np.array(banditSpec.rx2("mu0"))
    alpha0_psd = np.array(banditSpec.rx2("Sigma0"))
    alpha_pmean = np.array(priors.rx2("mu1"))
    alpha_psd = np.array(priors.rx2("Sigma1"))
    beta_pmean = np.array(priors.rx2("mu2"))
    beta_psd = np.array(priors.rx2("Sigma2"))
    sigma = float(priors.rx2("sigma")[0])

    prior_sigma = linalg.block_diag(alpha_psd, beta_psd, beta_psd)
    prior_mu = np.concatenate([alpha_pmean, beta_pmean, beta_pmean])

    return alpha0_pmean, alpha0_psd, alpha_pmean, alpha_psd, beta_pmean, beta_psd, sigma, prior_sigma, prior_mu

# %%
def get_priors_alpha_beta(post_mu, post_sigma):
    '''Get alpha and beta priors from mu and sigma'''
    alpha_pmean = post_mu[:G_LEN].flatten()
    alpha_psd = post_sigma[:G_LEN, :G_LEN]
    beta_pmean = post_mu[-F_LEN:].flatten()
    beta_psd = post_sigma[-F_LEN:, -F_LEN:]

    return alpha_pmean, alpha_psd, beta_pmean, beta_psd

# %%
def calculate_posteriors(X, Y, prior_mu, prior_sigma, sigma):
    '''Calculate the posterior mu and sigma'''

    # Calculate posterior sigma
    post_sigma = (sigma**2) * np.linalg.inv(X.T @ X + (sigma**2) * np.linalg.inv(prior_sigma))

    # Calculate posterior mu
    post_mu = (post_sigma @ ((X.T @ Y)/(sigma**2) + np.linalg.inv(prior_sigma) @ prior_mu) )

    return post_mu, post_sigma

# %%
def calculate_posterior_l2Reg(alpha_sigma, alpha_mu, beta_sigma, beta_mu, sigma, availability_matrix, prob_matrix, reward_matrix, action_matrix, fs_matrix, gs_matrix, day):
    "calc post of l2 model at end of study"
    # Get indices with non nan rewards
    avail_idx = np.logical_and(~np.isnan(reward_matrix), ~np.isnan(reward_matrix))

    R = reward_matrix[avail_idx]
    A = action_matrix[avail_idx].reshape(-1, 1)
    P = prob_matrix[avail_idx].reshape(-1, 1)
    F = fs_matrix[avail_idx]
    G = gs_matrix[avail_idx]

    # Calculate prior mu and sigma
    prior_mu = np.hstack((alpha_mu, beta_mu))
    prior_sigma = linalg.block_diag(alpha_sigma, beta_sigma)

    # If there are no available datapoints, return the prior
    if(len(R) == 0):
        return prior_mu, prior_sigma#beta_mu, beta_sigma

    # Calculate X and Y
    X = np.hstack((G, A * F))
    Y = R

    # Calculate posterior mu and sigma
    post_mu, post_sigma = calculate_posteriors(X, Y, prior_mu, prior_sigma, sigma)
    return post_mu, post_sigma

# %%
def calculate_posterior_avail(alpha_sigma, alpha_mu, beta_sigma, beta_mu, sigma, availability_matrix, prob_matrix, reward_matrix, action_matrix, fs_matrix, gs_matrix, day):
    '''Calculate the posterior distribution when user is available'''

    # Get indices with non nan rewards, and where availability is 1
    #avail_idx = np.logical_and(~np.isnan(reward_matrix), availability_matrix == 1)
    avail_idx = np.logical_and(~np.isnan(reward_matrix), availability_matrix == 1)

    R = reward_matrix[avail_idx]
    A = action_matrix[avail_idx].reshape(-1, 1)
    P = prob_matrix[avail_idx].reshape(-1, 1)
    F = fs_matrix[avail_idx]
    G = gs_matrix[avail_idx]

    # Calculate prior mu and sigma
    prior_mu = np.hstack((alpha_mu, beta_mu, beta_mu))
    prior_sigma = linalg.block_diag(alpha_sigma, beta_sigma, beta_sigma)

    # If there are no available datapoints, return the prior
    if(len(R) == 0):
        return prior_mu, prior_sigma#beta_mu, beta_sigma

    # Calculate X and Y
    X = np.hstack((G, P * F, (A - P) * F))
    Y = R

    # Calculate posterior mu and sigma
    post_mu, post_sigma = calculate_posteriors(X, Y, prior_mu, prior_sigma, sigma)
    return post_mu, post_sigma

# %%
def calculate_posterior_unavail(prior_alpha0_sigma, prior_alpha0_mu, sigma, availability_matrix, reward_matrix, gs_matrix, day):
    '''Calculate the posterior distribution for the case when there are no available timesloday'''

    # Get the index of unavailable timeslots and non nan rewards
    unavail_idx = np.logical_and(~np.isnan(reward_matrix), availability_matrix == 0)

    # the feature matrix, and reward matrix
    X = gs_matrix[unavail_idx]
    Y = reward_matrix[unavail_idx]

    # If there are no unavailable datapoints, return the prior
    if len(Y) == 0:
        return prior_alpha0_mu, prior_alpha0_sigma

    # Calculate posterior mu and sigma
    post_alpha0_mu, post_alpha0_sigma = calculate_posteriors(X, Y, prior_alpha0_mu, prior_alpha0_sigma, sigma)

    return post_alpha0_mu, post_alpha0_sigma

# %%
def calculate_posterior_maineffect(prior_alpha1_sigma, prior_alpha1_mu, sigma, availability_matrix, reward_matrix, action_matrix, gs_matrix, day):
    '''Calculate the posterior distribution for the case when user is available but we don't take action (action = 0)'''

    # Get the index of available timeslots with action = 0, and non nan rewards
    maineff_idx = np.logical_and.reduce((availability_matrix == 1, action_matrix == 0, ~np.isnan(reward_matrix)))

    # the feature matrix, and reward matrix
    X = gs_matrix[maineff_idx]
    Y = reward_matrix[maineff_idx]

    # If there are no unavailable datapoints, return the prior
    if len(Y) == 0:
        return prior_alpha1_mu, prior_alpha1_sigma

    # Calculate posterior mu and sigma
    post_alpha1_mu, post_alpha1_sigma = calculate_posteriors(X, Y, prior_alpha1_mu, prior_alpha1_sigma, sigma)

    return post_alpha1_mu, post_alpha1_sigma

# %%
def run_algorithm(data):
    '''Run the algorithm for each user and each bootstrap'''
    rewards=data[:,5]
    rewards=list(rewards[~np.isnan(rewards)])
    imputeRewardValue=sum(rewards)/len(rewards)

    # Load priors
    alpha0_pmean, alpha0_psd, alpha1_pmean, alpha1_psd, beta_pmean, beta_psd, sigma, prior_sigma, prior_mu = load_priors()

    # Posterior initialized using priors
    post_alpha0_mu, post_alpha0_sigma = np.copy(alpha0_pmean), np.copy(alpha0_psd)
    post_alpha1_mu, post_alpha1_sigma = np.copy(alpha1_pmean), np.copy(alpha1_psd)
    post_actionCenter_mu, post_actionCenter_sigma = np.copy(prior_mu), np.copy(prior_sigma)    

    # DS to store availability, probabilities, features, actions, posteriors and rewards
    availability_matrix = np.zeros((NDAYS * NTIMES))
    prob_matrix = np.zeros((NDAYS * NTIMES))
    reward_matrix = np.zeros((NDAYS * NTIMES))
    action_matrix = np.zeros((NDAYS * NTIMES))
    fs_matrix = np.zeros((NDAYS * NTIMES, F_LEN))
    gs_matrix = np.zeros((NDAYS * NTIMES, G_LEN))

    # Posterior matrices
    # alpha0
    post_alpha0_mu_matrix = np.zeros((NDAYS * NTIMES, G_LEN))
    post_alpha0_sigma_matrix = np.zeros((NDAYS * NTIMES, G_LEN, G_LEN))

    # alpha1
    post_alpha1_mu_matrix = np.zeros((NDAYS * NTIMES, G_LEN))
    post_alpha1_sigma_matrix = np.zeros((NDAYS * NTIMES, G_LEN , G_LEN))

    # beta/action centered
    post_actionCenter_mu_matrix = np.zeros((NDAYS * NTIMES, 2*F_LEN +G_LEN))
    post_actionCenter_sigma_matrix = np.zeros((NDAYS * NTIMES, 2*F_LEN +G_LEN ,  2*F_LEN +G_LEN ))

    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):

            # Get the current timeslot
            ts = (day) * 5 + time
            
            # State of the user at time ts
            availability, fs, gs, reward, prob_fsb, action = determine_user_state(data[ts])

            # Save user's availability
            availability_matrix[ts] = availability

            # Save probability, features, action and reward
            prob_matrix[ts] = prob_fsb
            action_matrix[ts] = action

            # Save features and state
            #if np.isnan(reward) and availability==1:
            #    reward=imputeRewardValue
            reward_matrix[ts] = reward

            fs_matrix[ts] = fs
            gs_matrix[ts] = gs

            post_alpha0_mu_matrix[ts] = post_alpha0_mu
            post_alpha0_sigma_matrix[ts] = post_alpha0_sigma

            post_alpha1_mu_matrix[ts] = post_alpha1_mu
            post_alpha1_sigma_matrix[ts] = post_alpha1_sigma

            post_actionCenter_mu_matrix[ts] = post_actionCenter_mu
            post_actionCenter_sigma_matrix[ts] = post_actionCenter_sigma

        # Update posteriors at the end of the day
        post_actionCenter_mu, post_actionCenter_sigma = calculate_posterior_avail(alpha1_psd, alpha1_pmean, beta_psd, beta_pmean, sigma, 
                                                                  availability_matrix[:ts + 1], prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                                  action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1], day)
    
        post_alpha0_mu, post_alpha0_sigma = calculate_posterior_unavail(alpha0_psd, alpha0_pmean, sigma, availability_matrix[:ts + 1], 
                                                                            reward_matrix[:ts + 1], gs_matrix[:ts + 1], day)

        post_alpha1_mu, post_alpha0_sigma = calculate_posterior_maineffect(alpha1_psd, alpha1_pmean, sigma, availability_matrix[:ts + 1], 
                                                                                        reward_matrix[:ts + 1], action_matrix[:ts + 1], gs_matrix[:ts + 1], day)


    l2Reg_mu, l2Reg_sigma = calculate_posterior_l2Reg(alpha1_psd, alpha1_pmean, beta_psd, beta_pmean, sigma, 
                                                                  availability_matrix[:ts + 1], prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                                  action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1], day)

    result = {"availability": availability_matrix, "prob": prob_matrix, "action": action_matrix, "reward": reward_matrix,
            "post_alpha0_mu": post_alpha0_mu_matrix, "post_alpha0_sigma": post_alpha0_sigma_matrix,
            "post_alpha1_mu": post_alpha1_mu_matrix, "post_alpha1_sigma": post_alpha1_sigma_matrix,
            "post_beta_mu": post_actionCenter_mu_matrix, "post_beta_sigma": post_actionCenter_sigma_matrix,
            "l2Reg_mu": l2Reg_mu, "l2Reg_sigma": l2Reg_sigma,
            "fs": fs_matrix, "gs": gs_matrix}
    
    # Save results
    return result

def initial_run():
    data = load_data()
    allResults=[]
    for i in range(NUSERS):
        allResults.append(run_algorithm(data[i]))
    return allResults,data

def get_residual_pairs(results, baseline="Prior"):
    alpha0_pmean, alpha0_psd, alpha1_pmean, alpha1_psd, beta_pmean, beta_psd, sigma, prior_sigma, prior_mu = load_priors()
    residual_matrix = np.zeros((NUSERS, NDAYS * NTIMES))
    baseline_thetas=[]

    for user in range(NUSERS):
        #lastTime=np.where(data[:,1]==0)[0] # if decision.time = 0, then they would not be available after right?
        posterior_user_T=results[user]['post_beta_mu'][NDAYS*NTIMES-1]
        for day in range(NDAYS):
            for time in range(NTIMES):
                ts = (day) * 5 + time
                gs=results[user]['gs'][ts]
                fs=results[user]['fs'][ts]
                prob=results[user]['prob'][ts]
                action=results[user]['action'][ts]
                reward=results[user]['reward'][ts]
                available=results[user]['availability'][ts]

                # get estimated reward,
                alpha = posterior_user_T[:G_LEN].flatten()
                alpha2=posterior_user_T[G_LEN:G_LEN+F_LEN].flatten()
                beta = posterior_user_T[-F_LEN:].flatten()
    
                estimated_reward = gs @ alpha + prob * (fs @ alpha2) + (action - prob) * (fs @ beta)
                if available and not np.isnan(reward):
                        residual_matrix[user, ts]=reward-estimated_reward

        baseline_theta=np.zeros(F_LEN+G_LEN)
        # set beta
        if baseline=="Prior":
            baseline_theta[-F_LEN:]=prior_mu[-F_LEN:]
        elif baseline=="ZeroAtAll":
            baseline_theta[-F_LEN:]=np.zeros(prior_mu[-F_LEN:].shape)
        elif baseline=="Posterior":
            baseline_theta[-F_LEN:]=posterior_user_T[-F_LEN:].flatten()
        # 0 out coef at baseline coef
        elif baseline in F_KEYS:
            baseline_theta[-F_LEN:]=posterior_user_T[-F_LEN:].flatten()
            index=F_KEYS.index(baseline)
            baseline_theta[G_LEN+index]=0.0

        # set alpha
        baseline_theta[:G_LEN]=results[user]['l2Reg_mu'][:G_LEN].flatten()

        # remove middle alpha
        baseline_thetas.append(baseline_theta)
    return residual_matrix, baseline_thetas

################################################################################################################

np.random.seed(0)
random.seed(0)

result,data=initial_run()

res_matrix, baseline_prior = get_residual_pairs(result, "Prior")
res_matrix, baseline_Posterior=get_residual_pairs(result, "Posterior")
res_matrix, baseline_0Tx=get_residual_pairs(result, "ZeroAtAll")
baseline_txEffectForInteresting=[]
for i in range(1,5):
    res_matrix, baseline_i=get_residual_pairs(result, F_KEYS[i])
    baseline_txEffectForInteresting.append(baseline_i)

baselines={"prior": baseline_prior, "posterior": baseline_Posterior, "all0TxEffect": baseline_0Tx, "0TxEffect_beta_i": baseline_txEffectForInteresting}

# write result!
with open('/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/baseline_parameters.pkl', 'wb') as handle:
    pkl.dump(baselines, handle, protocol=pkl.HIGHEST_PROTOCOL)
        
with open('/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/residual_matrix.pkl', 'wb') as handle:
    pkl.dump(res_matrix, handle, protocol=pkl.HIGHEST_PROTOCOL)
        
with open('/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/original_result_91.pkl', 'wb') as handle:
    pkl.dump(result, handle, protocol=pkl.HIGHEST_PROTOCOL)
