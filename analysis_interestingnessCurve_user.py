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

def computeMetricEngagementCurve(result):
    statistic={'txEffect0':[], 'txEffect1':[], 'differences':[]}
    for t in range(NDAYS*NTIMES):
        fs_engageIsX=result['fs'][t]
        fs_engageIsX[2]=0
        posterior_t=result['post_mu'][t][-len(F_KEYS):]
        txEffect0=posterior_t @ fs_engageIsX

        fs_engageIsX[2]=1
        txEffect1=posterior_t @ fs_engageIsX

        statistic['txEffect0'].append(txEffect0)
        statistic['txEffect1'].append(txEffect1)
        statistic['differences'].append(txEffect1-txEffect0)

    statistic['differences']=np.array(statistic['differences'])
    result['interestingnessStatistic-engagementCurve']=statistic
    return result

# get this or sliding window or...
def computeMetricAllEngagementCurve(result,bootstrapped_results):
    allXs=[]
    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):
            # Get the current timeslot
            allXs.append((day) * 5 + time)

    T=len(result['post_mu'])

    ## in case want a population bootstrap later?
    #metric=np.array([0 for t in range(T)])
    #for i in range(len(result)):
    #    result[i]=computeMetricEngagementCurve(result[i])
    #    metric += result[i]['interestingnessStatistic-engagementCurve']['differences']
    
    allYs={}
    result=computeMetricEngagementCurve(result)
    allYs['original']=result['interestingnessStatistic-engagementCurve']['differences']
    for b in range(len(bootstrapped_results)):
        result_b=np.array([0 for t in range(T)])#computeMetricEngagementCurve(result_b)
        #allYs[b]=result_b['interestingnessStatistic-engagementCurve']['differences']
        allYs[b]=result_b
    return allXs,allYs

def plotResult_AverageInterestingness(original_result, bootstrapped_results, image_path, baseline="Prior"):
    xs,allYs=computeMetricAllEngagementCurve(original_result, bootstrapped_results)

    bsThickness=1.5
    opacity=1
    plt.clf()
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)

    B=len(bootstrapped_results)
    for b in range(B-1):
        plt.plot(xs,allYs[b], color='k', linewidth=bsThickness, alpha=opacity)
    plt.plot(xs,allYs[B-1], color='k',label="Bootstrapped Engagement Difference Curves", linewidth=bsThickness, alpha=opacity)

    plt.plot(xs,allYs['original'], color='b', label="Observed Engagement Difference Curve", linewidth=2, alpha=1)
    plt.legend(loc="upper right")
    plt.title('Engagement Difference vs. Decision Time: '+baseline+' as the Baseline')
    plt.xlabel('Decision Time')
    plt.ylabel('Engagement Difference')
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
###################################################### 
##### input in user and b.s. version of the user #####
######################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bootstrap", type=int, required=True, help="Bootstrap number")
    parser.add_argument("-o", "--output", type=str, default="./output/engagementDifference", required=False, help="Output file directory name")
    parser.add_argument("-l", "--log", type=str, default="./log", required=False, help="Log file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=False, help="Baseline of interest")
    parser.add_argument("-u", "--user", type=int, default=80, required=False, help="User to bootstrap")
    parser.add_argument("-or", "--original_result", type=str, default="./init/original_result_91.pkl", required=False, help="Pickle file for original results")
    args = parser.parse_args()

    # Prepare directory for output and logging
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    args.output=args.output+"_"+args.baseline #make it outputdir_average_learning_<Baseline>

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)
    #bootstrap_metadata=load_bootstrap_metadata()
    #bootstrapped_results=load_bootstrapped_run(args.bootstrap_directory, bootstrap_metadata)
    bootstrapped_results=range(10)

    # Run algorithm
    image_path=os.path.join(args.output, "user_"+args.baseline)
    plotResult_AverageInterestingness(original_result[args.user], bootstrapped_results, args.baseline, image_path)

# %%
if __name__ == "__main__":
    main()
