# %%
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import argparse
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from interesting_ogData import load_original_run, getInterestingness

# %%
PKL_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/all91.pkl"
PRIOR_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/bandit-prior.RData"
NDAYS = 90
NUSERS = 91
NTIMES = 5

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "temperature", "logpresteps", "sqrt_totalsteps", "dosage", "engagement", "other_location", "variation"]
G_LEN = len(G_KEYS)

def load_bootstrapped_run(baseline, output, B=50):
    bootstrapped_results=[]
    for b in range(B):
        resultb=[]
        bootstrap_dir="Bootstrap-"+ str(b) +"_Baseline-"+ baseline+ "_UserSpecific-False"
        bootstrap_dir=os.path.join(output, bootstrap_dir)
        print("Get files for "+bootstrap_dir)
        if os.path.exists(bootstrap_dir):
            userDirList = [f for f in listdir(bootstrap_dir)]
            for userDir in userDirList:
                bootstrap_subdir=os.path.join(bootstrap_dir, userDir)
                result_u_i = [f for f in listdir(bootstrap_subdir)][0]
                filePath=os.path.join(bootstrap_subdir, result_u_i)
                resulti=pkl.load(open(filePath, 'rb'))
                resultb.append(resulti)
        bootstrapped_results.append(resultb)
    return bootstrapped_results

# %%
###################################################### 
##### input in user and b.s. version of the user #####
######################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", type=int, default=1, required=True, help="Index for which tx effect coef to test interestingness on (betaIndex)")
    parser.add_argument("-o", "--output", type=str, default="./output/interestingness/", required=False, help="Output file directory name")
    parser.add_argument("-b", "--boot_num", type=int, default=50, required=False, help="how many bootstrap sims run")
    parser.add_argument("-r", "--result_dir", type=str, default="./output/", required=False, help="Output file directory name")
    parser.add_argument("-or", "--original_result", type=str, default="./init/original_result_91.pkl", required=False, help="Pickle file for original results")
    args = parser.parse_args()

    var=F_KEYS[args.index+1]
    betaIndex=args.index+1
    print("Interestingness on "+var)
    baseOutput=os.path.join(args.output, var)

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)
    bootstrapped_results=load_bootstrapped_run(var, args.result_dir, args.boot_num)

    var=F_KEYS[args.index+1]
    betaIndex=args.index+1
    print("Interestingness on "+var)
    baseOutput=os.path.join(args.output, var, "bootstrap")

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)

    # Run algorithm
    #getInterestingness(baseOutput, original_result, betaIndex)
    for i in range(len(bootstrapped_results)):
        if len(bootstrapped_results[i]) > 0:
            getInterestingness(baseOutput+str(i), bootstrapped_results[i], betaIndex)
    
# %%
if __name__ == "__main__":
    main()

