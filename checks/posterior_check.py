import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import argparse
import os
import matplotlib.pyplot as plt
from scipy.stats import sem

# %%
PKL_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/all91.pkl"
PRIOR_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/bandit-prior.RData"
NDAYS = 270
NUSERS = 91
NTIMES = 5

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)

# aggregate results
def computeAverageMetric(result, state,avail):
    averageEffect=[0]*(NDAYS)
    errors=[0]*(NDAYS)
    for day in range(NDAYS):
        valueList=[]
        val=0
        for user in range(NUSERS):
            ts = (day) * 5
            if avail: #only compute for avail users
                if result[user]['availability'][ts]:
                    beta=result[user]['post_beta_mu'][ts][-F_LEN:]
                    val = val + (state @ beta)
                    valueList.append(state @ beta)
            else: #compute for all users, regardless of avail
                beta=result[user]['post_beta_mu'][ts][-F_LEN:]
                val = val + (state @ beta)
                valueList.append(state @ beta)
        errors[day]=sem(valueList)
        averageEffect[day]=val/len(valueList)

    print(averageEffect) 
    print(errors)
    return averageEffect,errors

def getPlotData(original_result, state, avail):
    xAxis=[]
    for day in range(NDAYS):
        xAxis.append(day)
    
    allYs={}
    allYs['original'], allYs['original-error']=computeAverageMetric(original_result, state, avail)    
    return xAxis, allYs

def plotResult_AverageLearning(original_result, state, image_path, avail=False):
    xs, allYs=getPlotData(original_result, state, avail)

    bsThickness=1.5
    opacity=1
    plt.clf()
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)

    plt.plot(xs,allYs['original'], color='b', label="Observed Average Posterior Means", linewidth=2, alpha=1)
    plt.fill_between(xs, np.array(allYs['original'])-np.array(allYs['original-error']),np.array(allYs['original'])+np.array(allYs['original-error']),alpha=0.2, edgecolor='b', facecolor='b')#edgecolor='#CC4F1B', facecolor='#FF9848')
    #plt.legend(loc="upper right")

    if avail:
        plt.title('(Available) Observed Average Posterior Mean vs. Day in the Study')
    else: 
        plt.title('Observed Average Posterior Mean vs. Day in the Study')

    plt.xlabel('Day in the Study')
    plt.ylabel('Posterior Mean')
    plt.savefig(image_path+'.png')
    plt.clf()

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

    newDose=False #use newly calculated doses in data instead of those realized in dataset.
    if newDose:
        parser.add_argument("-or", "--original_result", type=str, default="./init/original_result_newDose_91.pkl", required=False, help="Pickle file for original results")
    else:
        parser.add_argument("-or", "--original_result", type=str, default="./init/original_result_91.pkl", required=False, help="Pickle file for original results")
    args = parser.parse_args()

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)

    # Run algorithm
    target_state=np.array([1, 3.86, 1, 0, 0]) # following the target state in JMIR Draft 04-01-21
    # engaged, home/work location, variation is 0
    target_state=np.array([1, 1.98, 1, 0, 0]) # following the target state in JMIR Draft 04-01-21

    # derived from avg dosage in data: np.sum(data[:,:,6])/(91*1350)
    # derived from avg dosage in data: 
    # - np.sum(np.isnan(data[:,:,6])) = 0
    # - data[:,:,6].shape -> 91 x 1350
    # - np.sum(data[:,:,6])/(91*1350) = 1.9805890349783688

    output_dir = os.path.join("./checks", "averageLearningCheck")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_dir=output_dir+"/"

    # avail indicates whether or not to claculate average effects sensitive to availability. here we do not.
    image_path=os.path.join(output_dir, "average_observed_posteriors_check_avgDose="+str(target_state[1])+"plot")
    plotResult_AverageLearning(original_result, target_state, image_path, avail=False)

    # avail indicates whether or not to claculate average effects sensitive to availability. here we do worry about availabliyt (so only available users are used in computations)
    image_path=os.path.join(output_dir, "avail_Average_observed_posteriors_check_avgDose="+str(target_state[1])+"_plot")
    plotResult_AverageLearning(original_result, target_state, image_path, avail=True)

    #image_path=os.path.join(args.output, "posterior_action_selection_prob_check_plot")
    #checkActionSelectionProbs(original_result,target_state,image_path)
# %%

if __name__ == "__main__":
    main()