
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
def computeMetricSlidingWindow(result, x=0, y=35, delta1=0.463, delta2=1):
    #first get the standardized effects
    standardizedEffectsUser=[]
    engagedUser=[]
    for day in range(NDAYS):
        for time in range(NTIMES):
            ts = (day) * 5 + (NTIMES-1)

            beta=result['post_mu'][ts][-len(F_KEYS):]
            mean =result['fs'][ts] @ beta

            sigma=result['post_sigma'][ts][-len(F_KEYS):, -len(F_KEYS):]
            std=(result['fs'][ts] @ sigma.T) @ result['fs'][ts]
                
            standardizedEffectsUser.append(mean/std)
            engagedUser.append(result['fs'][ts][2]) 
    result['standardizedEffects']=np.array(standardizedEffectsUser)
    result['engaged']=np.array(engagedUser)

    # iterate through sliding windows
    n=(y-1)//2
    statistic={}
    tIndex=nUndeterminedSlidingWindows=nInterestingDeterminedWindows=nSlidingWindows=nDeterminedWindows=0
    determinedTimes=unDeterminedTimes=[]
    for day in range(NDAYS):
        for time in range(NTIMES):
            ts = (day) * 5 + (NTIMES-1)
            # if available then we calculate a stat
            if result['availability'][ts]: 
                nSlidingWindows=nSlidingWindows+1
                effects=engages=[]

                #See JMIR doc for details on this
                if ts <= n: 
                    effects=result['standardizedEffects'][0:ts+n]
                    engages=result['engaged'][1:ts+n]
                elif ts > n and ts <= NDAYS*NTIMES-1-n:
                    effects=result['standardizedEffects'][(ts-n):(ts+n)]
                    engages=result['engaged'][(ts-n):(ts+n)]
                else:
                    effects=result['standardizedEffects'][(ts-n):(NDAYS*NTIMES-1)]
                    engages=result['engaged'][(ts-n):(NDAYS*NTIMES-1)]

                isUnderdetermined = (sum(engages) >=x) and (len(engages)-sum(engages) >= x)

                if not isUnderdetermined:
                    nDeterminedWindows=nDeterminedWindows+1
                    determinedTimes.append(ts)

                    # claculate effects
                    engageIndices=np.where(engages==1)[0]
                    nonEngageIndices=np.where(engages==0)[0]
                    avgEngage=np.mean(effects[engageIndices])
                    avgNonEngage=np.mean(effects[nonEngageIndices])
                    if avgEngage > avgNonEngage:
                        nInterestingDeterminedWindows=nInterestingDeterminedWindows+1
                else:
                    nUndeterminedSlidingWindows=nUndeterminedSlidingWindows+1
                    unDeterminedTimes.append(ts)
    if nSlidingWindows >0 and nDeterminedWindows >0:
        statistic["r1"]=nUndeterminedSlidingWindows/nSlidingWindows
        statistic["r2"]=nInterestingDeterminedWindows/nDeterminedWindows
        statistic["isInteresting"]=(statistic["r1"]<delta1) and (abs(statistic["r2"]-.5)>delta2)
    else: 
        statistic["r1"]=statistic["r2"]=statistic["isInteresting"]=None
        statistic["isInteresting"]=False
    statistic["determinedTimes"]=determinedTimes
    statistic["unDeterminedTimes"]=unDeterminedTimes

    statistic["engaged"]=result['engaged']
    statistic["standardizedEffects"]=result['standardizedEffects']
    result['interestingnessStatistic-SlidingWindow']=statistic
    return result

# ignore for now...
def aggComputeMetricSlidingWindow(result):
    #may want to loop through result and get the r1 stats 
    #################### for population! ###########################
    metric=0
    for i in range(len(result)):
        result[i]=computeMetricSlidingWindow(result[i])
        metric += result[i]['interestingnessStatistic-SlidingWindow']['isInteresting']
    
    return metric

def getTableInterestingness(original_result, bootstrapped_results):
    allYs={}
    allYs['original']=aggComputeMetricSlidingWindow(original_result)
    for b in range(len(bootstrapped_results)):
        allYs[b]=aggComputeMetricSlidingWindow(bootstrapped_results[b])
    return allYs

def getResult(original_result, bootstrapped_results, baseline="Prior"):
    allYs=getTableInterestingness(original_result, bootstrapped_results)
    return allYs

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
##### input in population and b.s. version of the population #####
######################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--bootstrap", type=int, required=True, help="Bootstrap number")
    parser.add_argument("-o", "--output", type=str, default="./output/slidingWindow", required=False, help="Output file directory name")
    parser.add_argument("-l", "--log", type=str, default="./log", required=False, help="Log file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=False, help="Baseline of interest")
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
    bootstrapped_results=[]

    # Run algorithm
    # image_path=os.path.join(args.output, "user_"+args.baseline)

    result=getResult(original_result, bootstrapped_results, args.baseline)
    print(result)
    
# %%
if __name__ == "__main__":
    main()
