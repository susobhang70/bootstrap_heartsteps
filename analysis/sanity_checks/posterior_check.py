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

# aggregate results
def computeAverageMetric(result, state):
    averageEffect=[0]*(NDAYS)
    for day in range(NDAYS):
        val=0
        for user in range(NUSERS):
            ts = (day) * 5
            beta=result[user]['post_beta_mu'][ts][-F_LEN:]
            val = val + (state @ beta)
        averageEffect[day]=val/float(NUSERS)
    print(averageEffect) 
    return averageEffect

def getPlotData(original_result, state):
    xAxis=[]
    for day in range(NDAYS):
        xAxis.append(day)
    
    allYs={}
    allYs['original']=computeAverageMetric(original_result, state)    
    return xAxis, allYs

def plotResult_AverageLearning(original_result, state, image_path):
    xs, allYs=getPlotData(original_result, state)

    bsThickness=1.5
    opacity=1
    plt.clf()
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)

    plt.plot(xs,allYs['original'], color='b', label="Observed Average Posterior Means", linewidth=2, alpha=1)
    plt.legend(loc="upper right")
    plt.title('Observed Average Posterior Mean vs. Day in the Study')
    plt.xlabel('Day in the Study')
    plt.ylabel('Posterior Mean')
    #plt.show()
    plt.savefig(image_path+'.png')

# %%
def load_original_run(output_path):
    with open(output_path, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return original_result

def checkActionSelectionProbs(result,s,image_path):
    vs=[]
    probs=[]
    for i in range(NUSERS):
        v=s @ result[i]['post_beta_mu'][445][-F_LEN:]
        prob=result[i]['prob'][445]
        #if s@beta > 0, espect prob to be at clipping prob
        # vice versa for lower clipping prob
        vs.append(v)
        probs.append(prob)
    
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)

    xs=range(NUSERS)
    plt.plot(xs, vs, color='b', label="Observed Posterior Mean at time T", linewidth=2, alpha=1)
    plt.plot(xs, probs, color='r', label="Observed Probs at time T", linewidth=2, alpha=1)
    plt.axhline(y = 0.8, color = 'k', linestyle = '-')
    plt.axhline(y = 0.1, color = 'k', linestyle = '-')

    print(vs)
    print(probs)
    plt.legend(loc="upper right")
    plt.title('Observed Posterior Mean and Prob at time T vs. User')
    plt.xlabel('User')
    plt.ylabel('')
    #plt.show()
    plt.savefig(image_path+'.png')

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="./output", required=False, help="Output file directory name")
    parser.add_argument("-l", "--log", type=str, default="./log", required=False, help="Log file directory name")
    parser.add_argument("-or", "--original_result", type=str, default="./init/original_result_91.pkl", required=False, help="Pickle file for original results")
    args = parser.parse_args()

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)

    # Run algorithm
    #target_state=np.array([1, 3.86, 1, 1, 0]) # following the target state in JMIR Draft 04-01-21
    target_state=np.array([1, 1.98, 1, 1, 0]) # following the target state in JMIR Draft 04-01-21
    # derived from avg dosage in data: np.sum(data[:,:,6])/(91*1350)

    image_path=os.path.join(args.output, "average_observed_posteriors_check_plot")
    plotResult_AverageLearning(original_result, target_state, image_path)

    image_path=os.path.join(args.output, "posterior_prob_check_plot")
    checkActionSelectionProbs(original_result,target_state,image_path)
# %%

if __name__ == "__main__":
    main()
