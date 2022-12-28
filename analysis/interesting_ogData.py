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
G_KEYS = ["intercept", "temperature", "logpresteps", "sqrt_totalsteps", "dosage", "engagement", "other_location", "variation"]
G_LEN = len(G_KEYS)

def computeMetricEngagementCurve(result, betaIndex):
    statistic={'txEffect0':[], 'txEffect1':[], 'differences':[], 'isGreater':[]}

    #sum of timepoints where they are 1 or 0 according to betaIndex (var)
    sumNotVar=0
    sumVar=0
    times=[]
    for t in range(NDAYS*NTIMES):
        available=result['availability'][t]
        if available:
            times.append(t)

            # track how many timepts are engaged or not engaged
            if result['fs'][t,betaIndex]==1:
                sumVar=sumVar+1
            else:
                sumNotVar=sumNotVar+1

            # get standardized posterior tx effect under engage=1 and engage = 1
            fs_betaIndexIsX=result['fs'][t]
            fs_betaIndexIsX[betaIndex]=0
            posterior_t=result['post_beta_mu'][t][-F_LEN:]
            sigma_t=result['post_beta_sigma'][t][-len(F_KEYS):, -len(F_KEYS):]

            txEffect0=posterior_t @ fs_betaIndexIsX
            std=(fs_betaIndexIsX @ sigma_t.T) @ fs_betaIndexIsX
            txEffect0=txEffect0/std

            if betaIndex==1:#if dosage
                fs_betaIndexIsX[betaIndex]=1.86
            else:
                fs_betaIndexIsX[betaIndex]=1

            txEffect1=posterior_t @ fs_betaIndexIsX
            std=(fs_betaIndexIsX @ sigma_t.T) @ fs_betaIndexIsX
            txEffect1=txEffect1/std

            statistic['txEffect0'].append(txEffect0)
            statistic['txEffect1'].append(txEffect1)
            statistic['differences'].append(txEffect1-txEffect0)
            statistic['isGreater'].append(txEffect1 > txEffect0)

    statistic['differences']=np.array(statistic['differences'])
    statistic['txEffect0']=np.array(statistic['txEffect0'])
    statistic['txEffect1']=np.array(statistic['txEffect1'])

    statistic['proportionInteresting']=sum(statistic['isGreater'])/float(len(statistic['isGreater']))
    statistic['sumNotVar']=sumNotVar
    statistic['sumVar']=sumVar
    statistic['betaVar']=F_KEYS[betaIndex]
    statistic['times']=times
    result['interestingness']=statistic
    return result

# get this or sliding window or...
def computeMetricAllEngagementCurve(result, betaIndex):
    allYs={}
    result=computeMetricEngagementCurve(result, betaIndex)
    allYs['diff']=result['interestingness']['differences']
    allYs['tx1']=result['interestingness']['txEffect1']
    allYs['tx0']=result['interestingness']['txEffect0']

    allXs=result['interestingness']['times']
    return allXs,allYs,result

def plotResult_AverageInterestingness(original_result, image_path, betaIndex):
    xs,allYs,result=computeMetricAllEngagementCurve(original_result, betaIndex)

    bsThickness=1.5
    opacity=1
    plt.clf()
    f = plt.figure()
    f.set_figwidth(20)
    f.set_figheight(10)

    statisticsLine="Proportion of tx larger than not:"+str(result['interestingness']['proportionInteresting'])+"\n"
    statisticsLine=statisticsLine+"sumNotVar:"+str(result['interestingness']['sumNotVar'])+"        sumVar:"+str(result['interestingness']['sumVar'])

    varOfInterest=F_KEYS[betaIndex]
    plt.plot(xs,allYs['tx1'], color='b', label="Observed " + varOfInterest + " Posterior fixing 1", linewidth=2, alpha=1)
    plt.plot(xs,allYs['tx0'], color='r', label="Observed " + varOfInterest + " Posterior fixing 0", linewidth=2, alpha=1)
    plt.legend(loc="upper right")
    plt.title(varOfInterest+'Posterior Tx Effects fixing ' + varOfInterest +'_1_0 vs. Decision Time')
    plt.xlabel('Decision Time')
    plt.ylabel('Posterior tx Effect')
    plt.annotate(statisticsLine,(0,0), (5, -35), xycoords='axes fraction', textcoords='offset points', va='top')

    plt.savefig(image_path+"/"+ varOfInterest+ "_1_0_diffs_curve"+'.png')
    plt.close()
    return result

# %%
def load_original_run(output_path):
    with open(output_path, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return original_result

def getInterestingness(baseOutput, result, betaIndex):
    allRes=[]
    gtr80=[]
    sumNotVars=[]
    sumVars=[]

    nInteresting=0
    percInterestingThres=.8
    percTimesNotAndYes=.2
    nTimesNotAndYes=50

    interestingUsers=[]
    for i in range(len(result)):
        isInteresting=False
        output=baseOutput+"/user_"+str(i)
        # Prepare directory for output and logging
        print("logging in output dir "+output)
        if not os.path.exists(output):
            os.makedirs(output)
        
        res=plotResult_AverageInterestingness(result[i], output, betaIndex)
        allRes.append(res)

        gtr80.append(res['interestingness']['proportionInteresting'])
        sumVars.append(res['interestingness']['sumVar'])
        sumNotVars.append(res['interestingness']['sumNotVar'])
        T=float(len(res['interestingness']['times']))
        #if gtr80[i] > percInterestingThres and sumVars[i]/T > percTimesNotAndYes and sumNotVars[i]/T > percTimesNotAndYes:
        if gtr80[i] > percInterestingThres and sumVars[i] > nTimesNotAndYes and sumNotVars[i] > nTimesNotAndYes:
            isInteresting=True
            nInteresting=nInteresting+1
        if isInteresting:
            interestingUsers.append(i)

    print("RESULTS on n avail and n engaged/n avail")
    percs=[0,25,50,75,100]
    print("\tn props")
    print("\t\t"+str(np.percentile(gtr80, percs)))
    print("\tn var")
    print("\t\t"+str(np.percentile(sumVars, percs)))
    print("\tn not varops")
    print("\t\t"+str(np.percentile(sumNotVars, percs)))
    print("N interesting users is "+str(nInteresting))
    print(interestingUsers)

    a_file = open(baseOutput+"/interestingUsers.csv", "w")
    header="InterestingUser, Proportion of PostTx(1)>PostTx(0), sumNotVar, sumVar \n"
    a_file.write(header)
    for user in interestingUsers:
        statisticsLine=str(user)+","+str(gtr80[user])+ ","+str(sumNotVars[user])+ ","+str(sumVars[user])
        a_file.write(statisticsLine+"\n")
    a_file.close()

# %%
###################################################### 
##### input in user and b.s. version of the user #####
######################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--index", type=int, default=1, required=True, help="Index for which tx effect coef to test interestingness on (betaIndex)")
    parser.add_argument("-o", "--output", type=str, default="./output/interestingness/", required=False, help="Output file directory name")
    parser.add_argument("-or", "--original_result", type=str, default="./init/original_result_91.pkl", required=False, help="Pickle file for original results")
    args = parser.parse_args()

    var=F_KEYS[args.index+1]
    betaIndex=args.index+1
    print("Interestingness on "+var)
    baseOutput=os.path.join(args.output, var, "original")

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)

    # Run algorithm
    getInterestingness(baseOutput, original_result, betaIndex)

    import pdb
    pdb.set_trace()
    
# %%
if __name__ == "__main__":
    main()

