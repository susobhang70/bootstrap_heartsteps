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
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
rpackages.importr('fda')
import matplotlib.pyplot as plt
from matplotlib import rc
from main import load_data, calculate_eta, calculate_post_prob, calculate_posteriors, calculate_posterior_maineffect, calculate_posterior_unavail, calculate_value_functions,select_action,load_priors

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.7g" % x))

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
G_KEYS = ["intercept", "temperature", "logpresteps", "sqrt_totalsteps", "dosage", "engagement", "other_location", "variation"]
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
DOSAGE_GRID_LEN = len(dosage_grid)

# evaluate the dosage values using the basis
dosage_eval = dosage_basis.evaluate(dosage_grid)
next_dosage_eval0 = dosage_basis.evaluate(dosage_grid * 0.95).squeeze().T
next_dosage_eval1 = dosage_basis.evaluate(dosage_grid * 0.95 + 1).squeeze().T
dosage_eval = dosage_eval[:, :, 0]

# setup dosage matrix instead of np.repeat in calculating marginal rewards
dosage_matrix = []
for dosage in dosage_grid:
    dosageI = np.repeat(dosage/20.0, NTIMES*NDAYS)
    dosage_matrix.append(dosageI)
dosage_matrix = np.matrix(dosage_matrix)
    
# %%
# partial dosage ols solutions used in eta proxy update (in particular, where value updates are done via function approximation)
dosage_OLS_soln=np.linalg.inv(np.matmul(dosage_eval,dosage_eval.T))#(X'X)^{-1}#50 x 50
dosage_OLS_soln=np.matmul(dosage_OLS_soln, dosage_eval)#(X'X)^{-1}X'# 50 x 201

# %%
# load in prior data results like H1 (eta.init), w, and gamma tuned by peng
robjects.r['load'](PRIOR_NEW_DATA_PATH)
banditSpec=robjects.r['bandit.spec'] 
PSED=banditSpec.rx2("p.sed")[0]
W=banditSpec.rx2("weight.est")[0]
GAMMA=banditSpec.rx2("gamma")[0]
etaInit=banditSpec.rx2("eta.init")

# %%
def determine_user_state(data, dosage, last_action, useSimData=True):
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
    newdosage = LAMBDA * dosage + (1 if (features["prior_anti"] == 1 or last_action == 1) else 0)

    # standardizing the dosage
    features["dosage"] = newdosage / 20.0
    features["intercept"] = 1

    fs = np.array([features[k] for k in F_KEYS])
    gs = np.array([features[k] for k in G_KEYS])

    if useSimData:
        for i in range(len(fs)):
            fs[i]=np.random.normal(0, 1)
        for i in range(len(gs)):
            gs[i]=np.random.normal(0, 1)
    reward = data[5]

    return availability, fs, gs, newdosage, reward

# %%
def calculate_reward(ts, fs, gs, action, trueTheta, trueThetaSigma):
    '''Calculate the reward for a given action'''

    # Get alpha and betas from the baseline
    alpha0 = trueTheta[:G_LEN].flatten()
    beta   = trueTheta[-F_LEN:].flatten()

    alpha_psd=trueThetaSigma[:G_LEN, :G_LEN]
    beta_psd=trueThetaSigma[-F_LEN:, -F_LEN:]

    alpha0 = np.random.multivariate_normal(alpha0, alpha_psd)
    beta = np.random.multivariate_normal(beta, beta_psd)

    # Calculate reward
    estimated_reward = (gs @ alpha0) + action * (fs @ beta) #for dosage as baseline
    reward = np.random.randn() + estimated_reward # this residual matrix will either by the one from original data or a resampled with replacemnet version if user-specific

    return reward, fs @ beta

# %%
def calculate_posterior_avail_err(alpha_sigma, alpha_mu, beta_sigma, beta_mu, sigma, availability_matrix, prob_matrix, reward_matrix, action_matrix, fs_matrix, gs_matrix, beta_sigmaInv, trueTheta):
    '''Calculate the posterior distribution when user is available'''

    # Get indices with non nan rewards, and where availability is 1
    avail_idx = np.logical_and(~np.isnan(reward_matrix), availability_matrix == 1)
    R = reward_matrix[avail_idx]
    A = action_matrix[avail_idx].reshape(-1, 1)
    P = prob_matrix[avail_idx].reshape(-1, 1)
    F = fs_matrix[avail_idx]
    G = gs_matrix[avail_idx]

    # If there are no available datapoints, return the prior
    if(len(R) == 0):
        return beta_mu, beta_sigma, 0

    # Calculate prior mu and sigma
    prior_mu = np.hstack((alpha_mu, beta_mu, beta_mu))
    prior_sigma = linalg.block_diag(alpha_sigma, beta_sigma, beta_sigma)

    # Calculate X and Y
    X = np.hstack((G, P * F, (A - P) * F))
    Y = R

    # Calculate posterior mu and sigma
    post_mu, post_sigma = calculate_posteriors(X, Y, prior_mu, beta_sigmaInv, sigma)

    # Get the posterior beta mu and sigma
    post_beta_mu, post_beta_sigma = post_mu[-F_LEN:], post_sigma[-F_LEN:, -F_LEN:]
    post_alpha_mu, post_alpha_sigma = post_mu[:G_LEN], post_sigma[:G_LEN, :G_LEN]
    err=np.linalg.norm(np.concatenate([post_alpha_mu,post_beta_mu])-trueTheta)

    return post_beta_mu, post_beta_sigma, err

# %%
def run_algorithm(data, user, endogenous, trueTheta, trueThetaSigma):
    '''Run the algorithm for each user and each bootstrap'''

    # Load priors
    alpha0_pmean, alpha0_psd, alpha1_pmean, alpha1_psd, beta_pmean, beta_psd, sigma, prior_sigma,prior_mu = load_priors()

    # Initializing dosage to first dosage value (can be non-zero if user was already in the trial)
    dosage = data[0][6]

    # Posterior initialized using priors
    post_alpha0_mu, post_alpha0_sigma = np.copy(alpha0_pmean), np.copy(alpha0_psd)
    post_alpha1_mu, post_alpha1_sigma = np.copy(alpha1_pmean), np.copy(alpha1_psd)
    post_beta_mu, post_beta_sigma = np.copy(beta_pmean), np.copy(beta_psd)

    # get inverses
    alpha0_sigmaInv=np.linalg.inv(alpha0_psd)
    alpha1_sigmaInv=np.linalg.inv(alpha1_psd)
    beta_sigmaInv=np.linalg.inv(prior_sigma)

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

    # beta
    post_beta_mu_matrix = np.zeros((NDAYS * NTIMES, F_LEN ))
    post_beta_sigma_matrix = np.zeros((NDAYS * NTIMES, F_LEN , F_LEN ))

    eta = 0
    p_avail_avg = 0
    theta0, theta1 = np.zeros(NBASIS), np.zeros(NBASIS)

    truePosteriors=[]
    errs=[]

    info="we do not use sim data" if endogenous else "we use sim data"
    print("INFO: "+info)

    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):

            # Get the current timeslot
            ts = (day) * 5 + time
            
            # State of the user at time ts
            availability, fs, gs, dosage, reward = determine_user_state(data[ts], dosage, action_matrix[ts-1], useSimData=(not endogenous))# if endogenous, use sim data is false.

            # Save user's availability
            availability_matrix[ts] = availability

            # If user is available
            action, prob_fsb = 0, 0
            availability=1

            # Calculate probability of (fs x beta) > n
            eta = calculate_eta(theta0, theta1, dosage, p_avail_avg, ts)

            #print("\tAvailable: ETA is " + str(eta) + " . Dosage: " + str(dosage))
            prob_fsb = calculate_post_prob(day, data, fs, post_beta_mu, post_beta_sigma, eta)

            # Sample action with probability prob_fsb from bernoulli distribution
            action = select_action(prob_fsb)

            # Bayesian LR to estimate reward
            reward, truePosterior = calculate_reward(ts, fs, gs, action, trueTheta,trueThetaSigma)
            truePosteriors.append(truePosterior)

            # Save probability, features, action and reward
            prob_matrix[ts] = prob_fsb
            action_matrix[ts] = action

            # Save features and state
            reward_matrix[ts] = reward

            fs_matrix[ts] = fs
            gs_matrix[ts] = gs

            post_alpha0_mu_matrix[ts] = post_alpha0_mu
            post_alpha0_sigma_matrix[ts] = post_alpha0_sigma

            post_alpha1_mu_matrix[ts] = post_alpha1_mu
            post_alpha1_sigma_matrix[ts] = post_alpha1_sigma

            post_beta_mu_matrix[ts] = post_beta_mu
            post_beta_sigma_matrix[ts] = post_beta_sigma

        # Update posteriors at the end of the day
        post_beta_mu, post_beta_sigma, err = calculate_posterior_avail_err(alpha1_psd, alpha1_pmean, beta_psd, beta_pmean, sigma, 
                                                                  availability_matrix[:ts + 1], prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                                  action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1], beta_sigmaInv, trueTheta)

        post_alpha0_mu, post_alpha0_sigma = calculate_posterior_unavail(alpha0_psd, alpha0_pmean, sigma, availability_matrix[:ts + 1], 
                                                                            reward_matrix[:ts + 1], gs_matrix[:ts + 1], alpha0_sigmaInv)

        post_alpha1_mu, post_alpha0_sigma = calculate_posterior_maineffect(alpha1_psd, alpha1_pmean, sigma, availability_matrix[:ts + 1], 
                                                                                        reward_matrix[:ts + 1], action_matrix[:ts + 1], gs_matrix[:ts + 1], alpha1_sigmaInv)

        # update value functions
        theta0, theta1, p_avail_avg = calculate_value_functions(availability_matrix[:ts + 1], action_matrix[:ts + 1], 
                                                    fs_matrix[:ts + 1], gs_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                    post_beta_mu, post_alpha0_mu, post_alpha1_mu, ts)

        for jjj in range(NTIMES):
            errs.append(err)

    result = {"availability": availability_matrix, "prob": prob_matrix, "action": action_matrix, "reward": reward_matrix,
            "post_alpha0_mu": post_alpha0_mu_matrix, "post_alpha0_sigma": post_alpha0_sigma_matrix,
            "post_alpha1_mu": post_alpha1_mu_matrix, "post_alpha1_sigma": post_alpha1_sigma_matrix,
            "post_beta_mu": post_beta_mu_matrix, "post_beta_sigma": post_beta_sigma_matrix,
            "fs": fs_matrix, "gs": gs_matrix}

    from eta_checks import getTxEffect
    learnedPosteriors=[]
    T=len(result['availability'])
    posteriorErrors=[]

    for t in range(T):
        txEffect=getTxEffect(result,t,standardize=False)
        learnedPosteriors.append(txEffect)
        posteriorErrors.append(abs(txEffect-truePosteriors[t]))

    result['l2errors']=errs
    result['posteriorErrors']=posteriorErrors

    return result,truePosteriors,learnedPosteriors

def plotResult(true, learned, l2errs, postErrs, user, image_path):
    plt.clf()
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)

    T=len(true)

    plt.plot(range(T), true, color='b', label="True Posterior Mean", linewidth=2, alpha=1)
    plt.plot(range(T), learned, color='r', label="Learned Posterior Mean", linewidth=2, alpha=1)
    plt.legend(loc="upper right")

    plt.title('Posterior Mean Tx Effect vs. Time in the Study for user '+str(user))
    plt.xlabel('Time in the Study')
    plt.ylabel('Posterior Mean')
    plt.savefig(image_path+'.png')
    plt.clf()

    # plot posteriors and errors
    rc('mathtext', default='regular')
    fig = plt.figure(figsize=(6.4+10, 4.8+5))
    ax = fig.add_subplot(111)

    lns1 = ax.plot(range(T), l2errs, '-g', label = 'Theta errors')
    #lns2 = ax.plot(range(T), postErrs, '-o', label = 'Posterior errors')
    lns2 = ax.plot(range(T), postErrs, '-', color='orange', label = 'Posterior errors')
    plt.legend(loc="upper right")

    # added these lines
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower left', bbox_to_anchor=(0,-.15), fancybox = True, shadow = True)

    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel("l2 Errors")

    plt.title('Errors vs. Time for user '+str(user))
    plt.savefig(image_path+'_errors'+'.png')
    plt.clf()

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=int, required=True, help="User number")
    #parser.add_argument("-endo", "--endogenous", type=bool, required=True, help="endogenous or exogenous")
    args = parser.parse_args()
    endogenous=True

    print("\nLearning Checks\n")
    # Set random seed to the bootstrap number
    np.random.seed(0)

    # Load data
    data = load_data()

    output_dir = os.path.join("./checks", "user-"+str(args.user), "learning_algorithm_check")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir=output_dir+"/"

    # Run experiment 1
    trueTheta=np.zeros(F_LEN+G_LEN)
    trueThetaSigma=np.identity(F_LEN+G_LEN)
    print(trueTheta)
    image_path=os.path.join(output_dir, "posterior_check_0tx_endogenous-"+str(endogenous))
    result0, true0, learned0=run_algorithm(data[args.user], args.user, endogenous, trueTheta, trueThetaSigma)
    plotResult(true0, learned0, result0['l2errors'],result0['posteriorErrors'],args.user,image_path)

    # Run experiment 2
    trueTheta=np.random.normal(1, 1, F_LEN+G_LEN)
    replace=np.random.normal(-1,1,F_LEN+G_LEN)
    for i in range(0, F_LEN+G_LEN, 2):
            trueTheta[i]=replace[i]

    trueThetaSigma=np.identity(F_LEN+G_LEN)
    print(trueTheta)
    image_path=os.path.join(output_dir, "posterior_check_moderateTx_endogenous-"+str(endogenous))
    resultM, trueM, learnedM=run_algorithm(data[args.user], args.user, endogenous,trueTheta, trueThetaSigma)
    plotResult(trueM, learnedM, resultM['l2errors'], resultM['posteriorErrors'],args.user, image_path)

    # Run experiment 3
    trueTheta=np.random.normal(2, 1, F_LEN+G_LEN)
    replace=np.random.normal(-2,1,F_LEN+G_LEN)
    for i in range(0, F_LEN+G_LEN, 2):
            trueTheta[i]=replace[i]
    trueThetaSigma=np.identity(F_LEN+G_LEN)
    print(trueTheta)
    image_path=os.path.join(output_dir, "posterior_check_largeTx_endogenous-"+str(endogenous))
    resultL, trueL, learnedL=run_algorithm(data[args.user], args.user, endogenous,trueTheta, trueThetaSigma)
    plotResult(trueL, learnedL, resultL['l2errors'],resultL['posteriorErrors'],args.user,image_path)

# %%

if __name__ == "__main__":
    main()
