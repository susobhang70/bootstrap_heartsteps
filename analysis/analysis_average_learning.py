# %%
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import argparse
import os
import matplotlib.pyplot as plt

# %%
PKL_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/original_result_91_priorPaper.pkl"
bootstrapped_results_path = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Susobhan/output"
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
                beta=result[user]['post_mu'][ts][-len(F_KEYS):]
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

def isBSAboveOGProportionAmountOfTimes(allYs, b, prop):
    counter=0
    T=len(allYs[b])
    for t in range(T):
        if allYs['original'][t] <= allYs[b][t]: # black above blue
            counter=counter+1
    return 1 if (float(counter)/float(T) >= prop) else 0

def getStatisticsBSCurves(allYs,B,proportion):
    counter=0
    for b in range(B):
        counter=counter+int(isBSAboveOGProportionAmountOfTimes(allYs,b, proportion))
    return float(counter)/float(B)*100

def getStringStatistics(stats):
    space="    "
    display=space+"Percent of Bootstrapped Curves above the Observed Curve 80 Percent of the Time: "+str(stats['Geq_80'])
    display=display+"\n"+space+"Percent of Bootstrapped Curves above the Observed Curve 100 Percent of the Time: "+str(stats['Geq_100'])
    return display

def plotResult_AverageLearning(original_result, bootstrapped_results, state, image_path, baseline="Prior"):
    xs, allYs=getPlotData(original_result, bootstrapped_results, state)

    #compute other statistics, like the proportion of black curves above the blue prop amount of the times
    proportionStats={}
    proportionStats['Geq_80']=round(getStatisticsBSCurves(allYs, len(bootstrapped_results), .8),2)
    proportionStats['Geq_100']=round(getStatisticsBSCurves(allYs, len(bootstrapped_results), 1),2)
    statisticsLine=getStringStatistics(proportionStats)

    bsThickness=.75
    opacity=.5
    plt.clf()
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(8)

    B=len(bootstrapped_results)
    from scipy.interpolate import make_interp_spline, BSpline
    newX=np.linspace(min(xs), max(xs), 450*100)
    for b in range(B-1):
        #spl=make_interp_spline(xs, allYs[b], k=3)
        #smooth_b=spl(newX)
        #plt.plot(newX,smooth_b, color='k', linewidth=bsThickness, alpha=opacity)
        plt.plot(xs,allYs[b], color='k', linewidth=bsThickness, alpha=opacity)
    #spl=make_interp_spline(xs, allYs[B-1], k=3)
    #smooth_b=spl(newX)
    #plt.plot(newX,smooth_b, color='k',label="Bootstrapped Average Posterior Means", linewidth=bsThickness, alpha=opacity)
    plt.plot(xs,allYs[B-1], color='k',label="Bootstrapped Average Posterior Means", linewidth=bsThickness, alpha=opacity)

    #spl=make_interp_spline(xs, allYs['original'], k=3)
    #smooth_o=spl(newX)
    #plt.plot(newX,smooth_o, color='b', label="Observed Average Posterior Means", linewidth=2, alpha=1)
    plt.plot(xs,allYs['original'], color='b', label="Observed Average Posterior Means", linewidth=2, alpha=1)
    
    plt.legend(loc="upper right")
    plt.title('Observed Average Posterior Mean vs. Decision Time: '+baseline+' as the Baseline')
    plt.annotate(statisticsLine,(0,0), (5, -35), xycoords='axes fraction', textcoords='offset points', va='top')
    plt.xlabel('Decision Time')
    plt.ylabel('Posterior Mean')
    plt.savefig(image_path, format="png")
    #plt.show()
    plt.close()    

# %%
def load_original_run(output_path):
    with open(output_path, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return original_result

# %%
def load_bootstrap_metadata(bootstrap_loc):
    bsInstances=[]
    for i in range(108):#108):
        bootstrap_dir=bootstrap_loc+"/bootstrap_"+str(i)
        users=os.listdir(bootstrap_dir)
        bootstrap_i_paths=[]
        users=[f for f in users if not f.startswith('.')]#ignore hidden files
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
    args.output=os.path.join(args.output,"Average_Learning_Curve_"+args.baseline)

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)
    bootstrap_metadata=load_bootstrap_metadata(args.bootstrapped_result)
    bootstrapped_results=load_bootstrapped_run(bootstrap_metadata)

    # Run algorithm
    target_state=np.array([1, 3.86, 1, 1, 0]) # following the target state in JMIR Draft 04-01-21
    image_path=args.output+".png"
    plotResult_AverageLearning(original_result, bootstrapped_results, target_state, image_path, args.baseline)

# %%

if __name__ == "__main__":
    main()
