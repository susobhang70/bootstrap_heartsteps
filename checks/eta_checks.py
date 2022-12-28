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
from main import calculate_post_prob, select_action, calculate_reward, calculate_posterior_avail, calculate_posterior_maineffect, calculate_posterior_unavail, calculate_value_functions, load_data, load_initial_run, resample_user_residuals, load_priors

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
def determine_user_state(data, dosage, last_action):
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

    reward = data[5]

    return availability, fs, gs, newdosage, reward
# %%
def calculate_eta(theta0, theta1, dosage, p_avail, ts, psed=PSED, w=W, gamma=GAMMA, lamb=LAMBDA):

    # If less than 10 time steps, use etaInit from HeartStepsV1
    if ts < 10:
        return etaInit(float(dosage))[0],etaInit(float(dosage))[0],etaInit(float(dosage))[0]

    cur_dosage_eval0 = dosage_basis.evaluate(dosage * lamb)
    cur_dosage_eval1 = dosage_basis.evaluate(dosage * lamb + 1)

    # Calculate etaHat using peng's
    thetabar = theta0 * (1 - p_avail) + theta1 * (p_avail)
    val = np.sum(thetabar * (cur_dosage_eval0 - cur_dosage_eval1).squeeze().T)

    # Peng most likely got this wrong (He used 1-gamma instead of gamma)
    etaHat = val * (1-psed) * (gamma)
    eta = w * etaHat + (1-w) * etaInit(float(dosage))[0]
    return eta, etaHat, etaInit(float(dosage))[0]

def calculate_eta_correct(theta0, theta1, dosage, p_avail, ts, psed=PSED, w=W, gamma=GAMMA, lamb=LAMBDA):

    # If less than 10 time steps, use etaInit from HeartStepsV1
    if ts < 10:
        return etaInit(float(dosage))[0],etaInit(float(dosage))[0],etaInit(float(dosage))[0]

    cur_dosage_eval0 = dosage_basis.evaluate(dosage * lamb)
    cur_dosage_eval1 = dosage_basis.evaluate(dosage * lamb + 1)

    # Calculate etaHat using peng's
    thetabar = theta0 * (1 - p_avail) + theta1 * (p_avail)
    val = np.sum(thetabar * (cur_dosage_eval0 - cur_dosage_eval1).squeeze().T)

    # Peng most likely got this wrong (He used 1-gamma instead of gamma)
    eta1=etaInit(float(dosage))[0]/(1-gamma)*gamma #since original code multiplies by 1-gamma
    etaHat = val * (1-psed) * (gamma)
    eta = w * etaHat + (1-w) * eta1
    return eta, etaHat,eta1

# first check is to compare etaStar and eta1 to see if they are of the same scale. while it is weighted more heavily (.75 on eta1, .25 on etaStar), we want to see how much the output is affected by each value
# what term dominates? 
def check1(etaStars, eta1s, etaStarToEta1, availability, user, theta0, theta1, p_avail, gamma=GAMMA):
    output_dir = os.path.join("./checks", "user-"+str(user),"etaCheck")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir=output_dir+"/"

    indices=np.where(availability==1.0)
    if len(indices[0])>0:
        indices=indices[0]
    else:
        print("NO AVAILABLE TIMES")
        return
    xs=indices

    # plot etaStar and eta1 across time
    rc('mathtext', default='regular')
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lns1 = ax.plot(xs, eta1s, '-', label = 'η1')
    lns2 = ax.plot(xs, etaStars, '-', label = 'η*')
    ax2 = ax.twinx()
    lns3 = ax2.plot(xs, etaStarToEta1, '-r', label = 'η*/η1')

    # added these three lines
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax.grid()
    ax.set_xlabel('(Available) Time in the Study')
    ax.set_ylabel("Proxy (η)")
    ax2.set_ylabel("Proxy Ratio (η*/η1)")
    #plt.legend(loc="upper right")
    plt.title('η*, η1, and η*/η1 ratio for user '+str(user))
    plt.savefig(output_dir+'etaStarVsEta1_acrossAvailability_user-'+str(user) +'.png')

    # plot etaStar and eta1 by end across dosage grid.
    etaStarsGrid=[]
    eta1sGrid=[]
    etaStarToEta1Grid=[]
    dosage_grid_x=dosage_grid[1:len(dosage_grid)-1]
    for dosage in dosage_grid_x:
        if dosage!=0 and dosage!=20.0:
            cur_dosage_eval0 = dosage_basis.evaluate(dosage * LAMBDA)
            cur_dosage_eval1 = dosage_basis.evaluate(dosage * LAMBDA + 1)

            thetabar = theta0 * (1 - p_avail) + theta1 * (p_avail)
            val = np.sum(thetabar * (cur_dosage_eval0 - cur_dosage_eval1).squeeze().T)

            # Peng most likely got this wrong (He used 1-gamma instead of gamma)
            etaHat = val * (1-PSED) * (gamma)
            eta1=etaInit(float(dosage))[0]

            etaStarsGrid.append(etaHat)
            eta1sGrid.append(eta1)
            etaStarToEta1Grid.append(etaHat/eta1)
    plt.clf()

    rc('mathtext', default='regular')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lns1 = ax.plot(dosage_grid_x, eta1sGrid, '-', label = 'η1')
    lns2 = ax.plot(dosage_grid_x, etaStarsGrid, '-', label = 'η*')
    ax2 = ax.twinx()
    lns3 = ax2.plot(dosage_grid_x, etaStarToEta1Grid, '-r', label = 'η*/η1')

    # added these three lines
    lns = lns1+lns2+lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax.grid()
    ax.set_xlabel('Dosage')
    ax.set_ylabel("Proxy (η)")
    ax2.set_ylabel("Proxy Ratio (η*/η1)")
    #plt.legend(loc="upper right")
    plt.title('η*, η1, and η*/η1 ratio for user '+str(user))
    plt.savefig(output_dir+'etaStarVsEta1_acrossDosageGrid_user-'+str(user) +'.png')
    plt.clf()


# second check is to check how much action selection probs are affected by changing scale of gamma for etaHat computation. eta1 remains the same.
def check2(actionProbs, availability, user):
    output_dir = os.path.join("./checks", "user-"+str(user),"etaCheck")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir=output_dir+"/"

    indices=np.where(availability==1.0)
    if len(indices[0])>0:
        indices=indices[0]
    else:
        print("NO AVAILABLE TIMES")
        return
    xs=indices

    # plot etaStar as a function of gamma. full range
    rc('mathtext', default='regular')
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for xe, ye in zip(xs, actionProbs):
        newX=[xe]*len(ye)
        newX=newX+np.random.normal(.5, 1, len(ye))
        ax.scatter(newX, ye, s=2)

    plt.ylim(0, 1)
    plt.xlabel('(Available) Time in the Study')
    plt.ylabel('Action Probability across Gamma')
    plt.title('Action selection probabilities for user '+str(user))
    plt.savefig(output_dir+'full_actionProbAcrossTimeVaryingGamma_user-'+str(user) +'.png')
    plt.clf()

    # plot etaStar as a function of gamma. lower half
    rc('mathtext', default='regular')
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for xe, ye in zip(xs, actionProbs):
        newX=[xe]*len(ye)
        newX=newX+np.random.normal(.5, np.var(ye), len(ye))
        plt.scatter(newX, ye, s=2)

    plt.ylim(0,1)
    plt.xlim(0,100)
    plt.xlabel('(Available) Time in the Study')
    plt.ylabel('Action Probability across Gamma')
    plt.title('Action selection probabilities for user '+str(user))
    plt.savefig(output_dir+'lower_actionProbAcrossTimeVaryingGamma_user-'+str(user) +'.png')
    plt.clf()

    # plot etaStar as a function of gamma. upper half
    rc('mathtext', default='regular')
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for xe, ye in zip(xs, actionProbs):
        newX=[xe]*len(ye)
        newX=newX+np.random.normal(.5, np.var(ye), len(ye))
        plt.scatter(newX, ye, s=2)

    plt.ylim(0,1)
    plt.xlim(300,450)
    plt.xlabel('(Available) Time in the Study')
    plt.ylabel('Action Probability across Gamma')
    plt.title('Action selection probabilities for user '+str(user))
    plt.savefig(output_dir+'upper_actionProbAcrossTimeVaryingGamma_user-'+str(user) +'.png')
    plt.clf()

    rc('mathtext', default='regular')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for xe, ye in zip(xs, actionProbs):
        newX=[xe]*len(ye)
        newX=newX+np.random.normal(.5, np.var(ye), len(ye))
        ax.scatter(newX, ye, s=2)
    ax2 = ax.twinx()
    vars=[]
    for aList in actionProbs:
        vars.append(np.var(aList))
    ax2.plot(xs, vars, label = 'Variance across gamma')

    ax.grid()
    ax.set_xlabel('Dosage')
    ax.set_ylabel("Action Probability")
    ax2.set_ylabel("Variance")
    plt.legend(loc="lower right")
    plt.title('Action Probability at each time (varying γ) with variance for user '+str(user))
    plt.savefig(output_dir+'actionProbAcrossTimeVaryingGammaWithVariances_user-'+str(user) +'.png')
    plt.clf()

    print("Variance for action selection probabilities, varying gamma, is "+str(vars))
    print(vars)

# %%
def run_algorithm(data, user, user_specific, residual_matrix, baseline_theta):
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

    # for check 1
    etaStars=[]
    eta1s=[]
    etaStarToEta1=[]

    # for check 2
    actionProbs=[]
    gammaGrid=[]
    for i in np.arange(0,1,.05):
        gammaGrid.append(i)

    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):

            # Get the current timeslot
            ts = (day) * 5 + time
            
            # State of the user at time ts
            availability, fs, gs, dosage, reward = determine_user_state(data[ts], dosage, action_matrix[ts-1])

            # Save user's availability
            availability_matrix[ts] = availability

            # If user is available
            action, prob_fsb = 0, 0
            if availability == 1:
                # for check 2
                actionProbList_t=[]
                for gamma_i in gammaGrid:
                    eta, etaStar, eta1 = calculate_eta(theta0, theta1, dosage, p_avail_avg, ts, gamma=gamma_i)
                    prob_fsb = calculate_post_prob(day, data, fs, post_beta_mu, post_beta_sigma, eta)
                    actionProbList_t.append(prob_fsb)
                actionProbs.append(actionProbList_t)

                # Calculate probability of (fs x beta) > n
                eta, etaStar, eta1 = calculate_eta(theta0, theta1, dosage, p_avail_avg, ts)

                # for check 1
                etaStars.append(etaStar)
                eta1s.append(eta1)
                etaStarToEta1.append(etaStar/eta1)

                #print("\tAvailable: ETA is " + str(eta) + " . Dosage: " + str(dosage))
                prob_fsb = calculate_post_prob(day, data, fs, post_beta_mu, post_beta_sigma, eta)

                # Sample action with probability prob_fsb from bernoulli distribution
                action = select_action(prob_fsb)

            # Bayesian LR to estimate reward
            reward = calculate_reward(ts, fs, gs, action, baseline_theta, residual_matrix)

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
        post_beta_mu, post_beta_sigma = calculate_posterior_avail(alpha1_psd, alpha1_pmean, beta_psd, beta_pmean, sigma, 
                                                                  availability_matrix[:ts + 1], prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                                  action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1], beta_sigmaInv)

        post_alpha0_mu, post_alpha0_sigma = calculate_posterior_unavail(alpha0_psd, alpha0_pmean, sigma, availability_matrix[:ts + 1], 
                                                                            reward_matrix[:ts + 1], gs_matrix[:ts + 1], alpha0_sigmaInv)

        post_alpha1_mu, post_alpha0_sigma = calculate_posterior_maineffect(alpha1_psd, alpha1_pmean, sigma, availability_matrix[:ts + 1], 
                                                                                        reward_matrix[:ts + 1], action_matrix[:ts + 1], gs_matrix[:ts + 1], alpha1_sigmaInv)

        # update value functions
        theta0, theta1, p_avail_avg = calculate_value_functions(availability_matrix[:ts + 1], action_matrix[:ts + 1], 
                                                    fs_matrix[:ts + 1], gs_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                    post_beta_mu, post_alpha0_mu, post_alpha1_mu, ts)

    result = {"availability": availability_matrix, "prob": prob_matrix, "action": action_matrix, "reward": reward_matrix,
            "post_alpha0_mu": post_alpha0_mu_matrix, "post_alpha0_sigma": post_alpha0_sigma_matrix,
            "post_alpha1_mu": post_alpha1_mu_matrix, "post_alpha1_sigma": post_alpha1_sigma_matrix,
            "post_beta_mu": post_beta_mu_matrix, "post_beta_sigma": post_beta_sigma_matrix,
            "fs": fs_matrix, "gs": gs_matrix}

    # check 1 on etas
    check1(eta1s, etaStars, etaStarToEta1, availability_matrix, user, theta0, theta1, p_avail_avg)

    # check 2 
    check2(actionProbs, availability_matrix, user)

    return result

# %%
# for check 3
def run_algorithm_correctEta1(data, user, user_specific, residual_matrix, baseline_theta):
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

    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):

            # Get the current timeslot
            ts = (day) * 5 + time
            
            # State of the user at time ts
            availability, fs, gs, dosage, reward = determine_user_state(data[ts], dosage, action_matrix[ts-1])

            # Save user's availability
            availability_matrix[ts] = availability

            # If user is available
            action, prob_fsb = 0, 0
            if availability == 1:
                # Calculate probability of (fs x beta) > n
                eta, etaStar, eta1 = calculate_eta_correct(theta0, theta1, dosage, p_avail_avg, ts)

                #print("\tAvailable: ETA is " + str(eta) + " . Dosage: " + str(dosage))
                prob_fsb = calculate_post_prob(day, data, fs, post_beta_mu, post_beta_sigma, eta)

                # Sample action with probability prob_fsb from bernoulli distribution
                action = select_action(prob_fsb)

            # Bayesian LR to estimate reward
            reward = calculate_reward(ts, fs, gs, action, baseline_theta, residual_matrix)

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
        post_beta_mu, post_beta_sigma = calculate_posterior_avail(alpha1_psd, alpha1_pmean, beta_psd, beta_pmean, sigma, 
                                                                  availability_matrix[:ts + 1], prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                                  action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1], beta_sigmaInv)

        post_alpha0_mu, post_alpha0_sigma = calculate_posterior_unavail(alpha0_psd, alpha0_pmean, sigma, availability_matrix[:ts + 1], 
                                                                            reward_matrix[:ts + 1], gs_matrix[:ts + 1], alpha0_sigmaInv)

        post_alpha1_mu, post_alpha0_sigma = calculate_posterior_maineffect(alpha1_psd, alpha1_pmean, sigma, availability_matrix[:ts + 1], 
                                                                                        reward_matrix[:ts + 1], action_matrix[:ts + 1], gs_matrix[:ts + 1], alpha1_sigmaInv)

        # update value functions
        theta0, theta1, p_avail_avg = calculate_value_functions(availability_matrix[:ts + 1], action_matrix[:ts + 1], 
                                                    fs_matrix[:ts + 1], gs_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                    post_beta_mu, post_alpha0_mu, post_alpha1_mu, ts)

    result = {"availability": availability_matrix, "prob": prob_matrix, "action": action_matrix, "reward": reward_matrix,
            "post_alpha0_mu": post_alpha0_mu_matrix, "post_alpha0_sigma": post_alpha0_sigma_matrix,
            "post_alpha1_mu": post_alpha1_mu_matrix, "post_alpha1_sigma": post_alpha1_sigma_matrix,
            "post_beta_mu": post_beta_mu_matrix, "post_beta_sigma": post_beta_sigma_matrix,
            "fs": fs_matrix, "gs": gs_matrix}

    return result

def getTxEffect(result, t,standardize=True):
    beta=result['post_beta_mu'][t]
    fs=result['fs'][t]
    mean=beta @ fs
    std=1
    if standardize:
        sigma=result['post_beta_sigma'][t]
        std=(fs @ sigma.T) @ fs
    return mean/std

def check3(resultWrong, resultCorrect,user):
    output_dir = os.path.join("./checks", "user-"+str(user),"etaCheck")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir=output_dir+"/"

    posteriorsWrong, posteriorsCorrect=[],[]

    probsWrong, probsCorrect=[],[]

    availTimes=[]
    T=len(resultWrong['availability'])

    for t in range(T):
        if resultWrong['availability'][t]==1.:
            availTimes.append(t)
            probsWrong.append(resultWrong['prob'][t])
            probsCorrect.append(resultCorrect['prob'][t])
        txEffectWrong=getTxEffect(resultWrong,t,standardize=True)
        posteriorsWrong.append(txEffectWrong)

        txEffectCorrect=getTxEffect(resultCorrect,t,standardize=True)
        posteriorsCorrect.append(txEffectCorrect)

    # plot posteriors
    rc('mathtext', default='regular')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lns1 = ax.plot(range(T), posteriorsWrong, '-', label = 'Correct η')
    lns2 = ax.plot(range(T), posteriorsCorrect, '-', label = 'Incorrect η')

    # added these three lines
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax.grid()
    ax.set_xlabel('Time')
    ax.set_ylabel("Standardized Posterior Treatment Effect")
    plt.title('Standardized Posterior Treatment Effect for user '+str(user))
    plt.savefig(output_dir+'txEffects_wrongVsCorrectDosage_user-'+str(user) +'.png')
    plt.clf()

    # plot action probs
    rc('mathtext', default='regular')
    fig = plt.figure()
    ax = fig.add_subplot(111)

    lns1 = ax.plot(availTimes, probsCorrect, '-', label = 'Correct η')
    lns2 = ax.plot(availTimes, probsWrong, '-', label = 'Incorrect η')

    # added these three lines
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    ax.grid()
    ax.set_xlabel('(Available) Time')
    ax.set_ylabel("Action Selection Probability")
    plt.title('Action Selection Probability for user '+str(user))
    plt.savefig(output_dir+'probs_wrongVsCorrectDosage_user-'+str(user) +'.png')
    plt.clf()

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=int, required=True, help="User number")
    #parser.add_argument("-s", "--seed", type=int, required=True, help="Random seed")
    parser.add_argument("-s", "--seed", default=0, type=int, required=False, help="seed")
    parser.add_argument("-us", "--user_specific", default=False, type=bool, required=False, help="User specific experiment")
    parser.add_argument("-o", "--output", type=str, default="./output", required=False, help="Output file directory name")
    parser.add_argument("-l", "--log", type=str, default="./log", required=False, help="Log file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=False, help="Baseline of interest. If you want to run interestingness analysis, put in the F_KEY as a string to get thetas for that tx Effect coef 0ed out.")
    parser.add_argument("-rm", "--residual_matrix", type=str, default="./init/residual_matrix.pkl", required=False, help="Pickle file for residual matrix")
    parser.add_argument("-bp", "--baseline_theta", type=str, default="./init/baseline_parameters.pkl", required=False, help="Pickle file for baseline parameters")
    args = parser.parse_args()

    print("\nEta Checks\n")

    # Set random seed to the bootstrap number
    np.random.seed(args.seed)

    # Load initial run data
    residual_matrix, baseline_thetas = load_initial_run(args.residual_matrix, args.baseline_theta, args.baseline)

    # Load data
    data = load_data()

    # Prepare directory for output and logging
    args.user_specific=False
    print(args.user_specific)

    residual_matrix=residual_matrix[args.user]
    # need to make it s.t. we are doing different seq of seeds in user specific case.
    if args.user_specific:
        residual_matrix = resample_user_residuals(residual_matrix, args.user)

    # Run algorithm
    resultWrong=run_algorithm(data[args.user], args.user, args.user_specific, residual_matrix, baseline_thetas[args.user])
    resultCorrect=run_algorithm_correctEta1(data[args.user], args.user, args.user_specific, residual_matrix, baseline_thetas[args.user])

    # check 3: compare action selection probabilities and the posterior tx effects at each time.
    check3(resultWrong, resultCorrect, args.user)
# %%

if __name__ == "__main__":
    main()
