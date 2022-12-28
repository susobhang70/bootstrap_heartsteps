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
from main import calculate_posterior_avail, calculate_posterior_maineffect, calculate_posterior_unavail, load_initial_run,load_data, load_priors, calculate_post_prob,calculate_value_functions

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
def determine_user_state(data, dosage, last_action, useOldDosage=False):
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

    reward = data[5]
    prob = data[3]
    action = data[4]

    if not useOldDosage:
        # calculating dosage
        newdosage = LAMBDA * dosage + (1 if (features["prior_anti"] == 1 or last_action == 1) else 0)
        # standardizing the dosage
        features["dosage"] = newdosage / 20.0
    else: # if use old dosage
        features["dosage"]=data[6]/20.0
        newdosage=data[6]

    features["intercept"] = 1

    fs = np.array([features[k] for k in F_KEYS])
    gs = np.array([features[k] for k in G_KEYS])

    return availability, fs, gs, newdosage, reward, prob, action

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

# %%
def run_algorithm(data, user, user_specific, residual_matrix, baseline_theta, useOldDose):
    '''Run the algorithm for each user and each bootstrap'''

    # Load priors
    alpha0_pmean, alpha0_psd, alpha1_pmean, alpha1_psd, beta_pmean, beta_psd, sigma, prior_sigma, prior_mu = load_priors()

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
            availability, fs, gs, dosage, reward, og_prob, action = determine_user_state(data[ts], dosage, action_matrix[ts-1], useOldDosage=useOldDose)

            # Save user's availability
            availability_matrix[ts] = availability

            # If user is available
            prob_fsb = 0
            if availability == 1:
                # Calculate probability of (fs x beta) > n
                eta, etaStar, eta1 = calculate_eta(theta0, theta1, dosage, p_avail_avg, ts)
                #print("\tAvailable: ETA is " + str(eta) + " . Dosage: " + str(dosage))
                prob_fsb = calculate_post_prob(day, data, fs, post_beta_mu, post_beta_sigma, eta)

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

# %%
def run_og_algorithm(data, user, user_specific, residual_matrix, baseline_theta):
    '''Run the algorithm for each user and each bootstrap'''

    # Load priors
    alpha0_pmean, alpha0_psd, alpha1_pmean, alpha1_psd, beta_pmean, beta_psd, sigma, prior_sigma, prior_mu = load_priors()

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

    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):

            # Get the current timeslot
            ts = (day) * 5 + time
            
            # State of the user at time ts
            availability, fs, gs, dosage, reward, prob_fsb, action = determine_user_state(data[ts], dosage, action_matrix[ts-1], useOldDosage=True)

            # Save user's availability
            availability_matrix[ts] = availability

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

    result = {"availability": availability_matrix, "prob": prob_matrix, "action": action_matrix, "reward": reward_matrix,
            "post_alpha0_mu": post_alpha0_mu_matrix, "post_alpha0_sigma": post_alpha0_sigma_matrix,
            "post_alpha1_mu": post_alpha1_mu_matrix, "post_alpha1_sigma": post_alpha1_sigma_matrix,
            "post_beta_mu": post_beta_mu_matrix, "post_beta_sigma": post_beta_sigma_matrix,
            "fs": fs_matrix, "gs": gs_matrix}

    return result

def check(resultOld, resultNew, resultOriginal, user):
    from eta_checks import getTxEffect
    output_dir = os.path.join("./checks", "user-"+str(user), "dosageCheck")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir=output_dir+"/"

    posteriorsOld, posteriorsNew, posteriorsOriginal=[],[],[]
    probsOld, probsNew, probsOriginal=[],[],[]

    availTimes=[]
    T=len(resultOld['availability'])

    for t in range(T):
        if resultOld['availability'][t]==1.:
            availTimes.append(t)
            probsOld.append(resultOld['prob'][t])
            probsNew.append(resultNew['prob'][t])
            probsOriginal.append(resultOriginal['prob'][t])
        txEffectOld=getTxEffect(resultOld,t,standardize=True)
        posteriorsOld.append(txEffectOld)

        txEffectNew=getTxEffect(resultNew,t,standardize=True)
        posteriorsNew.append(txEffectNew)

        txEffectOriginal=getTxEffect(resultOriginal,t,standardize=True)
        posteriorsOriginal.append(txEffectOriginal)

    # plot posteriors
    rc('mathtext', default='regular')
    fig = plt.figure(figsize=(6.4+10, 4.8+5))
    ax = fig.add_subplot(111)

    lns1 = ax.plot(availTimes, probsOld, '-', label = 'Calculated using Dosage formula')
    lns2 = ax.plot(availTimes, probsNew, '-', label = 'Calculated using Realized Dosage')
    lns3 = ax.plot(availTimes, probsOriginal, '-', label = 'Realized Probabilities')

    #ax2 = ax.twinx()
    #lns4 = ax2.plot(availTimes, np.array(posteriorsOriginal)[availTimes], '-r', label = 'Standardized Posterior Treatment Effect')

    # added these three lines
    lns = lns1+lns2+lns3#+lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower left', bbox_to_anchor=(0,-.15), fancybox = True, shadow = True)
    #lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox = True, shadow = True)

    # added these three lines
    ax.grid()
    ax.set_xlabel('Available Time')
    ax.set_ylabel("Action Selection Probability")
    #ax2.set_ylabel("Standardized Posterior Treatment Effect")

    plt.title('Probabilities for user '+str(user))
    plt.savefig(output_dir+'probs_dosageCheck_user-'+str(user) +'.png')
    plt.clf()

    # with posterior
    rc('mathtext', default='regular')
    fig = plt.figure(figsize=(6.4+10, 4.8+7.5))
    ax = fig.add_subplot(111)

    lns1 = ax.plot(availTimes, probsOld, '-', label = 'Calculated using Dosage formula')
    lns2 = ax.plot(availTimes, probsNew, '-', label = 'Calculated using Realized Dosage')
    lns3 = ax.plot(availTimes, probsOriginal, '-', label = 'Realized Probabilities')

    ax2 = ax.twinx()
    lns4 = ax2.plot(availTimes, np.array(posteriorsOriginal)[availTimes], '-r', label = 'Standardized Posterior Treatment Effect')

    # added these three lines
    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower left', bbox_to_anchor=(0,-.13), fancybox = True, shadow = True)
    #lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox = True, shadow = True)

    # added these three lines
    ax.grid()
    ax.set_xlabel('Available Time')
    ax.set_ylabel("Action Selection Probability")
    ax2.set_ylabel("Standardized Posterior Treatment Effect")

    plt.title('Standardized Posterior Treatment Effect and Probabilities for user '+str(user))
    plt.savefig(output_dir+'probsAndTxEffect_dosageCheck_user-'+str(user) +'.png')
    plt.clf()

    print("prob values from varying dosage calculation method")
    print("using realized dosage")
    print(probsOld)
    print("using proper or 'as intended' formula")
    print(probsNew)
    print("original probabilities in data")
    print(probsOriginal)

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=int, required=True, help="User number")
    parser.add_argument("-us", "--user_specific", default=False, type=bool, required=False, help="User specific experiment")
    parser.add_argument("-s", "--seed", default=0, type=int, required=False, help="seed")
    parser.add_argument("-o", "--output", type=str, default="./output", required=False, help="Output file directory name")
    parser.add_argument("-l", "--log", type=str, default="./log", required=False, help="Log file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=False, help="Baseline of interest. If you want to run interestingness analysis, put in the F_KEY as a string to get thetas for that tx Effect coef 0ed out.")
    parser.add_argument("-rm", "--residual_matrix", type=str, default="./init/residual_matrix.pkl", required=False, help="Pickle file for residual matrix")
    parser.add_argument("-bp", "--baseline_theta", type=str, default="./init/baseline_parameters.pkl", required=False, help="Pickle file for baseline parameters")
    args = parser.parse_args()

    print("\nDosage Checks\n")

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
    resultOld=run_algorithm(data[args.user], args.user, args.user_specific, residual_matrix, baseline_thetas[args.user], useOldDose=True)#use old dose
    resultNew=run_algorithm(data[args.user], args.user, args.user_specific, residual_matrix, baseline_thetas[args.user], useOldDose=False)#don't use old dose, calculate using formula
    resultOriginal=run_og_algorithm(data[args.user], args.user, args.user_specific, residual_matrix, baseline_thetas[args.user])#don't use old dose, just pull from data

    check(resultOld, resultNew, resultOriginal,args.user)
    # goal of this is to see how wrong dosage calculation affects algorithm outcomes, where outcomes are measure in action selection prob.
    # plot prob old, prob new, prob og
    # plot post og (maybe new and old too to see how post changes with dosages?)    

# %%

if __name__ == "__main__":
    main()
