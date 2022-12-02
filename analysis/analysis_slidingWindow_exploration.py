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
def computeMetricSlidingWindow(result, x=9, y=35, delta1=0.463, delta2=1):
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
    res=computeMetricSlidingWindow(result)
    metric = res['interestingnessStatistic-SlidingWindow']
    return metric

def getTableInterestingness(original_result, baseline="Prior"):
    return aggComputeMetricSlidingWindow(original_result)

# %%
def load_original_run(output_path):
    with open(output_path, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return original_result

# %%
def load_bootstrap_metadata(bootstrap_loc, user):
    bsInstances=[]
    for i in range(10):#108):
        bootstrap_dir=bootstrap_loc+"/bootstrap_"+str(i)
        users=os.listdir(bootstrap_dir)
        users=[f for f in users if not f.startswith('.')]#ignore hidden files
        for user_dir in users:
            filepath=os.listdir(os.path.join(bootstrap_dir, user_dir))[0]
            filepath=os.path.join(bootstrap_dir, user_dir, filepath)
            split_path=filepath.split("_")
            if int(split_path[3])==user: #by our encoding, it is the 3rd index
                bsInstances.append(filepath)
    print("bs instance len found for the user "+str(user)+" : "+str(len(bsInstances)))
    print(bsInstances)
    return bsInstances

# %%
def load_bootstrapped_run(bootstrap_metadata):
    bs_results=[]
    for b in range(len(bootstrap_metadata)):
        userpath=bootstrap_metadata[b]
        with open(userpath, 'rb') as handle:
            result_user_b=pkl.load(handle)
            bs_results.append(result_user_b)
    return bs_results

def replaceNones(l):
    for i in range(len(l)):
        if l[i]==None:
            l[i]=-1
    return l

def getCounts(l):
    counts={}
    for val in l:
        if val not in counts:
            counts[val]=1
        else:
            counts[val]=counts[val]+1
    return counts

def presentData(l, key):
    print("Exploration for "+key)
    print(np.percentile(l, [25, 50, 75]))

    counts=getCounts(l)
    allValues=list(counts.keys())
    allValues.sort()
    outputStr=""
    for value in allValues:
        outputStr=outputStr+str(round(value,2))+" : "+str(counts[value])+"\t"
    print(outputStr)
    print("\n")

def seeR1Count(l, delta1):
    count=0
    for val in l:
        if val <= delta1:
            count=count+1
    return count

def seeR2Count(l, delta2):
    count=0
    for val in l:
        if abs(val-.5) >= delta2:
            count=count+1
    return count

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="./output/plots", required=False, help="Output file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=False, help="Baseline of interest")
    parser.add_argument("-or", "--original_result", type=str, default=PKL_DATA_PATH, required=False, help="Pickle file for original results")
    args = parser.parse_args()

    # Prepare directory for output and logging
    #if not os.path.exists(args.output):
    #    os.makedirs(args.output)
    #args.output=args.output+"/slidingWindowUserIndex_"+str(args.user)+"_"+args.baseline #make it outputdir_average_learning_<Baseline>

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)

    # Run algorithm
    # image_path=os.path.join(args.output, "user_"+args.baseline)
    results=[]
    r1s=[]
    r2s=[]
    engagedS=[]
    for user in range(NUSERS):
        result=getTableInterestingness(original_result[user])        
        results.append(result)
        r1s.append(result['r1'])
        r2s.append(result['r2'])
        engagedS.append(sum(result['engaged']))

    r1s=replaceNones(r1s)
    r2s=replaceNones(r2s)
    engagedS=replaceNones(engagedS)
    
    presentData(r1s, "r1")
    print(seeR1Count(r1s, .463))
    presentData(r2s, "r2")
    print(seeR2Count(r2s, .5))
    presentData(engagedS, "engages")
    #tab = result.groupby(['count']).size()
    #print(tab)

    # Run algorithm
    #image_path=args.output+".png"
    #plotResult(result, image_path)


# %% NOTE DO NOT USE THIS AS we need to bootstrap on user -specific AND plot it properly (data should be inputted a little differently perhaps from average_ or this)

if __name__ == "__main__":
    main()
