# %%
import pandas as pd
import pickle5 as pkl
import numpy as np
import rpy2.robjects as robjects
from collections import OrderedDict
import scipy.stats as stats
import scipy.linalg as linalg
import argparse
import os
import copy

# %%
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

dosageGrid=[]
for i in np.arange(0,20,.1):
    dosageGrid.append(round(i,1))

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

    # calculating dosage
    newdosage = LAMBDA * dosage + features["prior_anti"]

    # standardizing the dosage
    features["dosage"] = newdosage / 20.0

    features["intercept"] = 1

    fs = np.array([features[k] for k in F_KEYS])
    gs = np.array([features[k] for k in G_KEYS])

    reward = data[5]

    return availability, fs, gs, newdosage, reward

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
def clip(x, E0=E0, E1=E1):
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
def calculate_reward(ts, fs, gs, action, prob, baseline_theta, residual_matrix):
    '''Calculate the reward for a given action'''

    # Get alpha and betas from the baseline
    alpha0 = baseline_theta[:G_LEN].flatten()
    beta   = baseline_theta[-F_LEN:].flatten()

    # Calculate reward
    estimated_reward = gs[1] * alpha0[1] + action* (fs @ beta) #for dosage as baseline
    reward = residual_matrix[ts] + estimated_reward # this residual matrix will either by the one from original data or a resampled with replacemnet version if user-specific

    return reward

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

def getFrequencies(fs, gs, days):
    stateProbabilityDict={}
    for t in range(fs.shape[0]):
        key=np.concatenate([fs[t], gs[t]])
        key=str(key)
        if key in stateProbabilityDict:
            stateProbabilityDict[key]=stateProbabilityDict[key]+1
        else:
            stateProbabilityDict[key]=1
    for key in list(stateProbabilityDict.keys()):
        stateProbabilityDict[key]=float(stateProbabilityDict[key])/float(days*NTIMES)
    return stateProbabilityDict

def getEmpiricalRewardEstimate(available, treated, availability, action, fs, gs, probsZ, post_mu, day):
    indicesAvailableA_TreatT=np.intersect1d(np.where(availability==available), np.where(action==treated))
    rewardEst=0
    for t in indicesAvailableA_TreatT:
        key=np.concatenate([fs[t],gs[t]])
        key=str(key)

        alpha = post_mu[:G_LEN].flatten()
        beta = post_mu[-F_LEN:].flatten()
        fittedReward=gs[t] @ alpha + action[t]*fs[t] @ beta

        rewardEst=rewardEst+fittedReward*probsZ[key]
    return rewardEst

def dosageTransitions(xPrime, x, tx, lamb=.95,psed=.2):
    indicator=xPrime==lamb*x+1
    if tx==1:
        return indicator
    else:
        return psed*indicator+(1-psed)*(xPrime==lamb*x)

def pavailableDensity(avail, pavail):
    return pavail**avail + (1-pavail)**(1-avail)

def getValueSummand(dosage, a, pavail, oldV, psed=.2, lamb=LAMBDA):
    summand=0
    offset=len(dosageGrid)
    if a==0:
        # case: x'=\lambda*dosage+1,i'=1
        key=dosageGrid.index(lamb*dosage+1)
        summand=(psed)*pavail*oldV[key+offset] 

        # case: x'=\lambda*dosage+1,i'=0. index into oldV with or without offset depending on availability
        summand=summand+(psed)*(1-pavail)*oldV[key]
        
        # case: x'=\lambda*dosage,i'=1
        key=dosageGrid.index(lamb*dosage)
        summand=summand+(1-psed)*pavail*oldV[key+offset] 

        # case: x'=\lambda*dosage,i'=0. index into oldV with or without offset depending on availability
        summand=summand+(1-psed)*(1-pavail)*oldV[key]
    if a==1:
        # case: x'=\lambda*dosage+1,i'=1
        key=dosageGrid.index(lamb*dosage+1)
        summand=pavail*oldV[key+offset] 

        # case: x'=\lambda*dosage+1,i'=0. index into oldV with or without offset depending on availability
        summand=summand+(1-pavail)*oldV[key]
    return summand

import itertools
def getValueEstimate(dosage, availability, action, fs, gs, probsZ, post_mu, day, avail_i, pavailHat, oldV, gamma=1): #read in gamma properly
    # calculate the V(X,i) - only thing that changes in valis0 vs valis1 is tau(x'|x,a) and r_1(x,a). So compute the pavail*oldV
    if avail_i==0:
        rAvailable0_Treat0=getEmpiricalRewardEstimate(0, 0, availability, action, fs, gs, probsZ, post_mu, day)
        v=rAvailable0_Treat0+getValueSummand(dosage, 0, pavailHat, oldV)
    else:
        rAvailable1_Treat0=getEmpiricalRewardEstimate(1, 0, availability, action, fs, gs, probsZ, post_mu, day)
        v0=rAvailable1_Treat0+getValueSummand(dosage,0, pavailHat, oldV)

        rAvailable1_Treat1=getEmpiricalRewardEstimate(1, 1, availability, action, fs, gs, probsZ, post_mu, day)
        v1=rAvailable1_Treat1+getValueSummand(dosage,1, pavailHat, oldV)

        v=max(v1,v0)
        #return max

    offset=len(dosageGrid)
    key=dosageGrid.index(dosage)
    V=copy.deepcopy(oldV)
    V[key+int(offset*avail_i)]=v
    return V

def calculate_eta(prior_sigma, prior_mu, sigma, availability_matrix, prob_matrix, reward_matrix, action_matrix, fs_matrix, gs_matrix, post_mu, day, dosage, w=1, H1=1):#read in w and H1 from prior spec rda
    # estimate ECDF of Z
    probsZ=getFrequencies(fs_matrix, gs_matrix, day)
    pavailHat=sum(availability_matrix)/(NTIMES*day)

    # get dosage transition function
    ### 'project into the basis'

    # get initial value estimates for V(dosage, i)
    V=[0]*(len(dosageGrid)*2) #init to 0's! have ordering be dosageGrid(0), dosageGrid(1). dosageGrid(0) means dosagegrid x availability=i

    epsilon=.00000001
    delta=10
    while delta>epsilon:
        oldV=V

        #update value function
        curAvail=availability_matrix[availability_matrix.shape[0]-1]
        V=getValueEstimate(dosage, availability_matrix, action_matrix, fs_matrix, gs_matrix, probsZ, post_mu, day, curAvail, pavailHat, oldV)
        delta=np.linalg.norm(np.array(V) - np.array(oldV))

    # get H
    Hstar1=getValueSummand(dosage, 1, pavailHat, V)
    Hstar0=getValueSummand(dosage, 0, pavailHat, V)
    H_d_plus_1=(1-w)*H1+w*Hstar1 #read in H1 from it
    H_d_plus_0=(1-w)*H1+w*Hstar0 #read in H1 from it
    eta=H_d_plus_1-H_d_plus_0
    #need to get for each dosage we see! modifyyyyy
    return eta

# %%
def run_algorithm(data, user, boot_num, user_specific, residual_matrix, baseline_theta, output_dir, log_dir):
    '''Run the algorithm for each user and each bootstrap'''

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

    eta=0

    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):

            # Get the current timeslot
            ts = (day) * 5 + time
            
            # State of the user at time ts
            availability, fs, gs, dosage, reward = determine_user_state(data[ts], dosage)

            # Save user's availability
            availability_matrix[ts] = availability
            
            # If user is available
            if availability == 1:

                # Calculate probability of (fs x beta) > n
                prob_fsb = calculate_post_prob(fs, post_mu, post_sigma, eta)

                # Sample action with probability prob_fsb from bernoulli distribution
                action = select_action(prob_fsb)

                # Bayesian LR to estimate reward
                reward = calculate_reward(ts, fs, gs, action, prob_fsb, baseline_theta, residual_matrix, user_specific)

                # Save probability, features, action and reward
                prob_matrix[ts] = prob_fsb
                action_matrix[ts] = action
                reward_matrix[ts] = reward

            # Save features and state
            fs_matrix[ts] = fs
            gs_matrix[ts] = gs
            #post_mu=np.reshape(post_mu.shape[0], 1)
            post_mu_matrix[ts] = post_mu[:,0]
            post_sigma_matrix[ts] = post_sigma
            
        # Update posterior
        post_mu, post_sigma = calculate_posterior(prior_sigma, prior_mu, sigma, availability_matrix[:ts + 1], 
                                                    prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                    action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1])
        # update Eta
        eta = calculate_eta(prior_sigma, prior_mu, sigma, availability_matrix[:ts + 1], 
                                                    prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                    action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1], post_mu, day+1, dosage)

    result = {"availability": availability_matrix, "prob": prob_matrix, "action": action_matrix, "reward": reward_matrix,
            "fs": fs_matrix, "gs": gs_matrix, "post_mu": post_mu_matrix, "post_sigma": post_sigma_matrix}
    
    # Save results
    if not user_specific:
        with open(output_dir + f"/results_{user}_{boot_num}.pkl", "wb") as f:
            pkl.dump(result, f)
    else:
        pass

def resample_user_residuals(residual_matrix, user):
    T= NDAYS * NTIMES
    resampled_indices = np.random.choice(range(T), T)
    residual_matrix=residual_matrix[resampled_indices]
    return residual_matrix

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=int, required=True, help="User number")
    parser.add_argument("-b", "--bootstrap", type=int, required=True, help="Bootstrap number")
    parser.add_argument("-s", "--seed", type=int, required=True, help="Random seed")
    parser.add_argument("-us", "--user_specific", default=False, type=bool, required=False, help="User specific experiment")
    parser.add_argument("-o", "--output", type=str, default="./output", required=False, help="Output file directory name")
    parser.add_argument("-l", "--log", type=str, default="./log", required=False, help="Log file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=False, help="Baseline of interest")
    parser.add_argument("-rm", "--residual_matrix", type=str, default="./init/residual_matrix.pkl", required=False, help="Pickle file for residual matrix")
    parser.add_argument("-bp", "--baseline_theta", type=str, default="./init/baseline_parameters.pkl", required=False, help="Pickle file for baseline parameters")
    args = parser.parse_args()

    # Set random seed to the bootstrap number
    np.random.seed(args.seed)

    # Load initial run data
    residual_matrix, baseline_thetas = load_initial_run(args.residual_matrix, args.baseline_theta, args.baseline)

    # Load data
    data = load_data()

    # Prepare directory for output and logging
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(args.log):
        os.makedirs(args.log)
    output_dir = os.path.join(args.output, "bootstrap_" + str(args.bootstrap), "user_" + str(args.seed))
    log_dir = os.path.join(args.log, "bootstrap_" + str(args.bootstrap), "user_" + str(args.seed))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    args.user_specific=False
    print(args.user_specific)
    residual_matrix=residual_matrix[args.user]
    # need to make it s.t. we are doing different seq of seeds in user specific case.
    if args.user_specific:
        residual_matrix=resample_user_residuals(residual_matrix, args.user)

    # Run algorithm
    run_algorithm(data[args.user], args.user, args.bootstrap, args.user_specific, residual_matrix, baseline_thetas[args.user], output_dir, log_dir)

# %%

if __name__ == "__main__":
    main()
