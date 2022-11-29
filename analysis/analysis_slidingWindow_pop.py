# %%
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import argparse
import os
import matplotlib.pyplot as plt

# %%
PKL_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/original_result_91.pkl"
bootstrapped_results_path = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Susobhan/output"
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

def getTableInterestingness(original_result, bootstrapped_results, baseline="Prior"):
    allYs=[]
    allYs.append(aggComputeMetricSlidingWindow(original_result))
    for b in range(len(bootstrapped_results)):
        allYs.append(aggComputeMetricSlidingWindow(bootstrapped_results[b]))
    allX=['original']
    allX.extend(range(len(bootstrapped_results)))
    df=pd.DataFrame({'count': allYs, 'run': allX})
    return df

# %%
def load_original_run(output_path):
    with open(output_path, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return original_result

# %%
def load_bootstrap_metadata(bootstrap_loc):
    bsInstances=[]
    for i in range(10):#108):
        bootstrap_dir=bootstrap_loc+"/bootstrap_"+str(i)
        users=os.listdir(bootstrap_dir)
        bootstrap_i_paths=[]
        for user_dir in users:
            filepath=os.listdir(os.path.join(bootstrap_dir, user_dir))[0]
            filepath=os.path.join(bootstrap_dir, user_dir, filepath)
            bootstrap_i_paths.append(filepath)
        bsInstances.append(bootstrap_i_paths)
    return bsInstances

# %%
def load_bootstrapped_run(bootstrap_metadata):
    bs_results=[]
    for b in range(len(bootstrap_metadata)):
        result_b=[]
        for userpath in bootstrap_metadata[b]:
            with open(userpath, 'rb') as handle:
                result_user_b=pkl.load(handle)
                result_b.append(result_user_b)
        bs_results.append(result_b)
    return bs_results

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="./output/plots", required=False, help="Output file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=False, help="Baseline of interest")
    parser.add_argument("-or", "--original_result", type=str, default=PKL_DATA_PATH, required=False, help="Pickle file for original results")
    parser.add_argument("-br", "--bootstrapped_result", type=str, default=bootstrapped_results_path, required=False, help="Pickle file for original results")
    args = parser.parse_args()

    # Prepare directory for output and logging
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    #args.output=os.path.join(args.output,"slidingWindow_"+args.baseline)

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)
    bootstrap_metadata=load_bootstrap_metadata(args.bootstrapped_result)
    bootstrapped_results=load_bootstrapped_run(bootstrap_metadata)

    # Run algorithm
    #target_state=np.array([1, 3.86, 1, 1, 0]) # following the target state in JMIR Draft 04-01-21
    #image_path=args.output+".png"

    result= getTableInterestingness(original_result, bootstrapped_results, args.baseline)#, target_state, image_path, args.baseline)
    print(result)
    tab = result.groupby(['count']).size()
    print(tab)

# %%

if __name__ == "__main__":
    main()
