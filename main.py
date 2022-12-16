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
import skfda.representation.basis as basis
import itertools

# %%
PKL_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/all91.pkl"
PRIOR_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/bandit-prior.RData"
PRIOR_NEW_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/bandit-spec-new.RData"
NDAYS = 90
NUSERS = 91
NTIMES = 5

LAMBDA = 0.95
MAX_ITERS=100

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation", "temperature", "logpresteps", "sqrt_totalsteps"]
G_LEN = len(G_KEYS)

E0 = 0.2
E1 = 0.1

MIN_DOSAGE = 0
MAX_DOSAGE = 20

NBASIS=50

# %%
# create the dosage basis
dosage_basis = basis.BSpline((MIN_DOSAGE, MAX_DOSAGE), n_basis=NBASIS, order=4)

# create the dosage grid
dosage_grid = np.arange(MIN_DOSAGE, MAX_DOSAGE + .1, .1)

# evaluate the dosage values using the basis
dosage_eval = dosage_basis.evaluate(dosage_grid)
next_dosage_eval0 = dosage_basis.evaluate(dosage_grid * 0.95).squeeze().T
next_dosage_eval1 = dosage_basis.evaluate(dosage_grid * 0.95 + 1).squeeze().T
dosage_eval=dosage_eval[:,:,0]

dosage_matrix=[]
for dosage in dosage_grid:
    dosageI=np.repeat(dosage, NTIMES*NDAYS)
    dosage_matrix.append(dosageI)
dosage_matrix=np.matrix(dosage_matrix)
dosage_matrix=dosage_matrix.T
    
# %%
pd.DataFrame(next_dosage_eval0)

dosage_OLS_soln=np.linalg.inv(np.matmul(dosage_eval.T,dosage_eval))#(X'X)^{-1}#201 x 201
dosage_OLS_soln=np.matmul(dosage_OLS_soln, dosage_eval.T)#(X'X)^{-1}X'# 201 x 50

# %%
robjects.r['load'](PRIOR_NEW_DATA_PATH)
banditSpec=robjects.r['bandit.spec'] 
PSED=banditSpec.rx2("p.sed")[0]
W=banditSpec.rx2("weight.est")[0]
GAMMA=banditSpec.rx2("gamma")[0]

etaInit=banditSpec.rx2("eta.init")
#beta_psd = np.array(priors.rx2("Sigma2"))
#sigma = float(priors.rx2("sigma")[0])

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

# %%
# later update on the fly with state_prob dict
def get_state_probabilities(fs, gs, ts):
    '''Compute the probability of occurence for each state, given the history until timeslot ts'''

    # Dict to first store occurence counts
    state_prob = {}

    # Remove the dosage from the state
    fm = np.delete(fs, 1, axis=1)
    gm = np.delete(gs, 1, axis=1)

    # Count occurences
    for i in range(ts):
        # Remove the dosage from the state
        key = str(np.concatenate([fm[i], gm[i]]))
        if key in state_prob:
            state_prob[key] += 1
        else:
            state_prob[key]  = 1

    # Normalize to get probabilities
    for key in state_prob.keys():
        state_prob[key] /= ts

    pZ=[]
    for i in range(ts):
        key = str(np.concatenate([fm[i], gm[i]]))
        pZ.append(state_prob[key])
    return np.array(pZ)

# %%
def get_empirical_rewards_estimate(target_availability, target_action, fs, gs, pZ, post_mu, p_avail_avg, ts):
    '''Calculate the empirical reward estimate'''
    #valid_indices = np.intersect1d(
    #    np.where(availability == target_avail), np.where(action == target_action))
    rewardEstimates=[]

    # Remove the dosage from the state for generating the key for Z_dist
    fm = np.delete(fs, 1, axis=1)
    gm = np.delete(gs, 1, axis=1)

    # Extract alpha0 and beta from posterior mu
    alpha = post_mu[:G_LEN].flatten()
    beta = post_mu[-F_LEN:].flatten()

    # Compute r(x, a) i.e. r(target_avail, target_action)
    for i in range(len(dosage_grid)):
        fs[:,1]=np.repeat(dosage_grid[i], ts)#dosage_matrix[:fs.shape[0], i]
        gs[:,1]=np.repeat(dosage_grid[i], ts)#dosage_matrix[:fs.shape[0], i]

        # using target action instead of action[t] for speedup
        fittedReward = gs @ alpha * pZ[:ts]  + target_availability * target_action * (fs @ beta * pZ[:ts])
        rewardEstimates.append(np.sum(fittedReward))
    return rewardEstimates

# %%
def pavailableDensity(avail, pavail):
    return pavail**avail + (1-pavail)**(1-avail)

# %%
def get_value_summand(dosage, availability, pavail, theta0, theta1, psed=.2, lamb=LAMBDA):
    summand = 0
    #offset = len(dosage_grid)
    basis_representation0=dosage_basis.evaluate(dosage*lamb)[:,0,0]
    basis_representation1=dosage_basis.evaluate(dosage*lamb+1)[:,0,0]
    if availability == 0:
        # case: x'=\lambda*dosage+1,i'=1
        #key = dosage_grid.index(lamb*dosage+1)
        #key=np.where(dosage_grid==lamb*dosage+1)[0][0]
        V_1_1=basis_representation1 @ theta1
        summand = (psed)*pavail*V_1_1 #V[key+offset] 

        # case: x'=\lambda*dosage+1,i'=0. index into V_old with or without offset depending on availability
        V_1_0=basis_representation1 @ theta0
        summand = summand+(psed)*(1-pavail)*V_1_0 #V[key]
        
        # case: x'=\lambda*dosage,i'=1
        V_0_1=basis_representation0 @ theta1
        #key=np.where(dosage_grid==lamb*dosage)[0][0]
        summand = summand+(1-psed)*pavail* V_0_1#V[key+offset] 

        # case: x'=\lambda*dosage,i'=0. index into V_old with or without offset depending on availability
        V_0_0=basis_representation0 @ theta0
        summand = summand+(1-psed)*(1-pavail)*V_0_0#V[key]
    if availability == 1:
        # case: x'=\lambda*dosage+1,i'=1
        #key=np.where(dosage_grid==lamb*dosage+1)[0][0]
        #key = dosage_grid.index(lamb*dosage+1)
        V_1_1=basis_representation1 @ theta1
        summand = pavail*V_1_1#V[key+offset] 

        # case: x'=\lambda*dosage+1,i'=0. index into V_old with or without offset depending on availability
        V_1_0=basis_representation1 @ theta0
        summand = summand+(1-pavail)*V_1_0#V[key]
    return summand

# %%
def bellman_backup(availability_matrix, action_matrix, fs_matrix, gs_matrix, post_mu, p_avail_avg, theta0, theta1, reward_available0_action0, reward_available1_action0, reward_available1_action1, gamma=GAMMA):
    V = [0]*(2*len(dosage_grid))
    V[0:len(dosage_grid)]= next_dosage_eval0 @ theta0
    V[len(dosage_grid):(2*len(dosage_grid))]= next_dosage_eval1 @ theta1
    for i in range(len(dosage_grid)):
        # calculate the V(X,i) - only thing that changes in valis0 vs valis1 is tau(x'|x,a) and r_1(x,a). So compute the (p_avail_avg * V_old) 
        dosage=dosage_grid[i]
        key=np.where(dosage_grid==dosage)[0][0]
        #key = dosage_grid.index(dosage)

        #get reward update when available=0
        r00 = reward_available0_action0[i]
        V[key] = r00 + get_value_summand(dosage, 0, p_avail_avg, theta0, theta1)

        #get reward update when available=1
        r10 = reward_available1_action0[i]
        v0 = r10 + get_value_summand(dosage, 0, p_avail_avg, theta0,theta1)

        r11 = reward_available1_action1[i]
        v1  = r11 + get_value_summand(dosage, 1, p_avail_avg, theta0, theta1)
        v = max(v1, v0)
        V[key + len(dosage_grid)] = v
    return V

# get the value for H
def get_value_summand_H(dosage, action, pavail, V, psed=.2, lamb=LAMBDA):
    summand = 0
    offset = len(dosage_grid)

    if action==1:
        key=np.where(dosage_grid==lamb*dosage+1)[0][0]
        #key = dosage_grid.index(lamb*dosage+1)
        summand = pavail*V[key+offset] + (1-pavail)*V[key]
    else:
        # case: x'=\lambda*dosage,i'=0
        key=np.where(dosage_grid==lamb*dosage)[0][0]
        #key = dosage_grid.index(lamb*dosage)
        summand = (1-psed)*(1-pavail)*V[key]
        
        # case: x'=\lambda*dosage,i'=1
        summand = summand+(1-psed)*pavail*V[key+offset] 

        # case: x'=\lambda*dosage+1,i'=0.
        key=np.where(dosage_grid==lamb*dosage+1)[0][0]
        #key = dosage_grid.index(lamb*dosage+1)
        summand = summand+(psed)*(1-pavail)*V[key]

        # case: x'=\lambda*dosage+1,i'=1.
        summand = summand+(psed)*(pavail)*V[key+offset]
    return summand

# %%
def calculate_value_functions(prior_sigma, prior_mu, sigma, availability_matrix, prob_matrix, reward_matrix, action_matrix, fs_matrix, gs_matrix, post_mu, ts, dosage, w=1, H1=1):#read in w and H1 from prior spec rda
    '''Calculate eta for a given dosage'''
    # estimate ECDF of Z
    pZ = get_state_probabilities(fs_matrix, gs_matrix, ts)

    # calculate mean p(availability)
    p_avail_avg = np.mean(availability_matrix)
    
    # get rewards matrices for each case: r_i(x,a)
    #r_0(x,0)
    reward_available0_action0 = get_empirical_rewards_estimate(0, 0, fs_matrix, gs_matrix, pZ, post_mu, p_avail_avg, ts)
    #r_1(x,0)
    reward_available1_action0 = get_empirical_rewards_estimate(1, 0, fs_matrix, gs_matrix, pZ, post_mu, p_avail_avg, ts) 
    #r_1(x,1)
    reward_available1_action1 = get_empirical_rewards_estimate(1, 1, fs_matrix, gs_matrix, pZ, post_mu, p_avail_avg, ts) 

    # get initial value estimates for V(dosage, i)
    V = [0] * (len(dosage_grid)*2) #init to 0's! have ordering be dosage_grid(0), dosage_grid(1). dosage_grid(i) means dosagegrid x availability=i
    theta0=np.zeros(NBASIS)
    theta1=np.zeros(NBASIS)

    epsilon = 1e-8
    delta = 10
    iters=0
    while delta > epsilon and iters < MAX_ITERS:
        V_old = V

        # get OLS Estimate
        theta0=np.matmul(V[0:len(dosage_grid)], dosage_OLS_soln)
        theta1=np.matmul(V[len(dosage_grid):(2*len(dosage_grid))], dosage_OLS_soln)

        #print(str(theta0)+" ; "+str(theta1))

        # update value function
        V = bellman_backup(availability_matrix, action_matrix, fs_matrix, gs_matrix, post_mu, p_avail_avg, theta0, theta1, reward_available0_action0, reward_available1_action0, reward_available1_action1)
        delta = np.linalg.norm(np.array(V) - np.array(V_old))
        iters=iters+1
    return theta0, theta1

def calculate_eta(theta0, theta1, dosage, availability, psed=.2, w=W, gamma=GAMMA):
    cur_dosage_eval0 = dosage_basis.evaluate(dosage*LAMBDA)
    cur_dosage_eval1 = dosage_basis.evaluate(dosage*LAMBDA+1)

    p_avail_avg = np.mean(availability)
    thetabar=theta0*(1-p_avail_avg)+theta1*(p_avail_avg)
    val=np.sum(thetabar * (cur_dosage_eval0 - cur_dosage_eval1))
    etaHat=val*(1-psed)*(1-gamma) 

    eta=w*etaHat+(1-w)*1#etaInit(dosage)
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

    theta0, theta1=np.zeros(50),np.zeros(50)

    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):

            # Get the current timeslot
            ts = (day) * 5 + time
            
            # State of the user at time ts
            availability, fs, gs, dosage, reward = determine_user_state(data[ts], dosage)

            # Save user's availability
            availability_matrix[ts] = availability

            print("T is "+str(ts))
            
            # If user is available
            if availability == 1:
                # Calculate probability of (fs x beta) > n
                eta=calculate_eta(theta0, theta1, dosage, availability_matrix)
                print("ETA is "+str(eta))
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
            post_mu_matrix[ts] = post_mu[:,0]
            post_sigma_matrix[ts] = post_sigma
            
        # Update posterior
        post_mu, post_sigma = calculate_posterior(prior_sigma, prior_mu, sigma, availability_matrix[:ts + 1], 
                                                    prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                    action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1])
        # update Eta
        theta0,theta1 = calculate_value_functions(prior_sigma, prior_mu, sigma, availability_matrix[:ts + 1], 
                                                    prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                    action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1], post_mu, ts + 1, dosage)

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
