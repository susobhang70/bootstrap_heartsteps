# %%
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import argparse
import os
import matplotlib.pyplot as plt

# %%
PKL_PRIOR_PAPER_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/original_result_91_priorPaper.pkl"
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

def getPlotData(original_result, result_priorPaper, state):
    xAxis=[]
    for day in range(NDAYS):
        for time in range(NTIMES):
            xAxis.append((day) * 5 + time)
    
    allYs={}
    allYs['original']=computeAverageMetric(original_result, state)    
    allYs['originalPriorPaper']=computeAverageMetric(result_priorPaper, state)    

    return xAxis, allYs

def plotResult_AverageLearning(original_result, priorPaper_result, state, image_path, baseline="Prior"):
    xs, allYs=getPlotData(original_result, priorPaper_result, state)

    #compute other statistics, like the proportion of black curves above the blue prop amount of the times
    statisticsLine=""

    bsThickness=.75
    opacity=.5
    plt.clf()
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(8)

    #spl=make_interp_spline(xs, allYs['original'], k=3)
    #smooth_o=spl(newX)
    #plt.plot(newX,smooth_o, color='b', label="Observed Average Posterior Means", linewidth=2, alpha=1)
    plt.plot(xs,allYs['original'], color='b', label="Observed Average Posterior Means from Peng", linewidth=2, alpha=1)
    plt.plot(xs,allYs['originalPriorPaper'], color='g', label="Observed Average Posterior Means from Prior Paper", linewidth=2, alpha=1)
    plt.axhline(y = 0, color = 'r', linestyle = '-')
    
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
    args.output=os.path.join(args.output,"Average_Learning_Curve_PriorPaper_vsPengsPrior"+args.baseline)

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)
    original_result_priorPaper=load_original_run(PKL_PRIOR_PAPER_DATA_PATH)

    # Run algorithm
    target_state=np.array([1, 3.86, 1, 1, 0]) # following the target state in JMIR Draft 04-01-21
    image_path=args.output+".png"
    plotResult_AverageLearning(original_result, original_result_priorPaper, target_state, image_path, args.baseline)

# %%

if __name__ == "__main__":
    main()
