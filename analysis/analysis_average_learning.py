
# %%
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import argparse
import os
import matplotlib.pyplot as plt

# %%
PKL_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/all91.pkl"
PRIOR_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/bandit-prior.RData"
NDAYS = 90
NUSERS = 91
NTIMES = 5

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation", "temperature", "logpresteps", "sqrt_totalsteps"]
G_LEN = len(G_KEYS)

# aggregate results
def computeAverageMetric(result, state):
    #result = {"availability": availability_matrix, "prob": prob_matrix, "action": action_matrix, "reward": reward_matrix,
    #        "fs": fs_matrix, "gs": gs_matrix, "post_mu": post_mu_matrix, "post_sigma": post_sigma_matrix}

    averageEffect=[0]*(NDAYS*NTIMES)
    for user in range(NUSERS):
        tIndex=0
        for day in range(NDAYS):
            for time in range(NTIMES):
                ts = (day) * 5 + (NTIMES-1)
                val=0
                beta=result['post_mu'][ts][-len(F_KEYS):]
                val =state @ beta

                averageEffect[tIndex]=averageEffect[tIndex]+val
                tIndex=tIndex+1

    averageEffect = [t/NUSERS for t in averageEffect]    
    return averageEffect

def getPlotData(original_result, bootstrapped_results, state):
    xAxis=[]
    for day in range(NDAYS):
        for time in range(NTIMES):
            xAxis.append((day) * 5 + time)
    
    allYs={}
    allYs['original']=computeAverageMetric(original_result, state)    

    for b in range(len(bootstrapped_results)):
        allYs[b]=computeAverageMetric(bootstrapped_results[b], state)

    return xAxis, allYs

def plotResult_AverageLearning(original_result, bootstrapped_results, state, image_path, baseline="Prior"):
    xs, allYs=getPlotData(original_result, bootstrapped_results, state)

    bsThickness=1.5
    opacity=1
    plt.clf()
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)

    B=len(bootstrapped_results)
    for b in range(B-1):
        plt.plot(xs,allYs[b], color='k', linewidth=bsThickness, alpha=opacity)
    plt.plot(xs,allYs[B-1], color='k',label="Bootstrapped Average Posterior Means", linewidth=bsThickness, alpha=opacity)

    plt.plot(xs,allYs['original'], color='b', label="Observed Average Posterior Means", linewidth=2, alpha=1)
    plt.legend(loc="upper right")
    plt.title('Observed Average Posterior Mean vs. Decision Time: '+baseline+' as the Baseline')
    plt.xlabel('Decision Time')
    plt.ylabel('Posterior Mean')
    plt.show()
    plt.savefig(image_path+'.png')

# %%
def load_original_run(output_path):
    with open(output_path, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return original_result

# %%
def load_bootstrap_metadata():
    pass

# %%
def load_bootstrapped_run(bootstrap_directory, bootstrap_metadata):
    bs_results=[]
    bootstrap_number=bootstrap_metadata["bootstrap_number"]
    for b in range(bootstrap_number):
        bootstrap_directory_i=os.path.join(bootstrap_directory, +f"results_{user}_{b}.pkl") 
        # think it should be results_user_b, need the metainfo for bootstrap runs to collect on
        with open(bootstrap_directory, 'rb') as handle:
            result_b=pkl.load(handle)
        bs_results[b]=result_b
    return bs_results

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bootstrap", type=int, required=True, help="Bootstrap number")
    parser.add_argument("-o", "--output", type=str, default="./output/average_learning", required=False, help="Output file directory name")
    parser.add_argument("-l", "--log", type=str, default="./log", required=False, help="Log file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=False, help="Baseline of interest")
    parser.add_argument("-or", "--original_result", type=str, default="./init/original_result.pkl", required=False, help="Pickle file for original results")
    args = parser.parse_args()

    # Prepare directory for output and logging
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    args.output=args.output+"_"+args.baseline #make it outputdir_average_learning_<Baseline>

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)
    bootstrap_metadata=load_bootstrap_metadata()
    bootstrapped_results=load_bootstrapped_run(args.bootstrap_directory, bootstrap_metadata)

    # Run algorithm
    target_state=np.array([1, 3.86, 1, 1, 0]) # following the target state in JMIR Draft 04-01-21
    image_path=os.path.join(args.output, "average_learning_plot_"+args.baseline)
    plotResult_AverageLearning(original_result, bootstrapped_results, target_state, args.baseline, image_path)

# %%

if __name__ == "__main__":
    main()
