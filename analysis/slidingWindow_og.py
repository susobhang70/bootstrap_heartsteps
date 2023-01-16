
# %%
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import argparse
import os
import matplotlib.pyplot as plt
import math

import matplotlib as mpl
import pylab
from matplotlib import rc

fix_plot_settings = True
if fix_plot_settings:
    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)
    label_size = 10
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = label_size
    mpl.rcParams['figure.titlesize'] = label_size
    mpl.rcParams['lines.markersize'] = label_size
    mpl.rcParams['grid.linewidth'] = 2.5
    mpl.rcParams['legend.fontsize'] = label_size
    pylab.rcParams['xtick.major.pad']=5
    pylab.rcParams['ytick.major.pad']=5

    lss = ['--',  ':', '-.', '-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    mss = ['>', 'o',  's', 'D', '>', 's', 'o', 'D', '>', 's', 'o', 'D']
    ms_size = [25, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    colors = ['#e41a1c', '#0000cd', '#4daf4a',  'black' , 'magenta']
else:
    pass

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.7g" % x))


# %%
PKL_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/all91.pkl"
PRIOR_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/bandit-prior.RData"
NDAYS = 90
NUSERS = 91
NTIMES = 5
T=NDAYS*NTIMES

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "temperature", "logpresteps", "sqrt_totalsteps", "dosage", "engagement", "other_location", "variation"]
G_LEN = len(G_KEYS)
from posterior_check import rindex

def computeMetricSlidingDay(result, indexFS, x=2, delta1=.5, delta2=.5):
    # iterate through sliding windows
    #idx = np.where(data[:T,2]==1)[0]
    ndata=rindex(result['availability'][:T], 1)+1#np.where(data[:T,1]==0)[0] #first time the user is out
    if ndata==0: #if no 1 is found
        print("no availability :(")
        return {'isEqualAvail':False, 'isEqualEngAvail':False, 'r1':None, 'r2':None, 'stdEffectRatio': None}

    last_day=math.floor((ndata-1)/NTIMES) #ndata is the ts index of the last available time, reset the -1, then /NTIMES to get the day of the last available time.

    engaged=[]
    standardizedEffectsUser=[]
    stdEffectRatioEngaged=0
    stdEffectRatioNotEngaged=0
    nEngaged=0
    nNonEngaged=0

    for day in range(last_day+1): #want it to iterate to last day
        for time in range(NTIMES):
            ts = (day) * 5 + time
            if result['availability'][ts]==1.:
                beta=result['post_beta_mu'][day*5][-len(F_KEYS):] # the one at ts is really the posteriors from the last day, or (ts//5)-1 since we update at end of the day. similarly, the one at day 0, or times 0-4, is the prior params
                mean =result['fs'][ts] @ beta

                sigma=result['post_beta_sigma'][day*5][-len(F_KEYS):, -len(F_KEYS):]
                std=math.sqrt((result['fs'][ts] @ sigma.T) @ result['fs'][ts])
                stdEffect=mean/std

                standardizedEffectsUser.append(stdEffect)
                engaged.append(result['fs'][ts, indexFS])
                if result['fs'][ts, indexFS]:
                    stdEffectRatioEngaged=stdEffectRatioEngaged+stdEffect
                    nEngaged=nEngaged+1
                else:
                    stdEffectRatioNotEngaged=stdEffectRatioNotEngaged+stdEffect
                    nNonEngaged=nNonEngaged+1
            else: #if not available.
                standardizedEffectsUser.append(-1)
                engaged.append(-1)
    if nEngaged>0 and nNonEngaged >0:
        stdEffectRatioEngaged=stdEffectRatioEngaged/nEngaged
        stdEffectRatioNotEngaged=stdEffectRatioNotEngaged/nNonEngaged

    standardizedEffectsUser=np.array(standardizedEffectsUser)
    engaged=np.array(engaged)

    # for interestingness
    nInterestingDeterminedWindows=0
    nSlidingWindows=last_day+1
    nDeterminedWindows=0

    avgEngageAll=[]
    avgNonEngageAll=[]
    determinedTimes=[]
    for day in range(last_day+1): #loop through all avail decision times
        if day == 0: #length is 2*NTIMES
            startTime=0
            endTime=NTIMES*2
        elif day >= 1 and day < last_day: #length is 3*NTIMES
            startTime=day*NTIMES-NTIMES
            endTime=day*NTIMES+NTIMES*2
        else: #if last_day-1, length is 2*NTIMES
            startTime=last_day*NTIMES-NTIMES
            endTime=last_day*NTIMES+NTIMES
        #print("day is "+str(day)+". s: "+str(startTime)+ " , e: "+str(endTime))
        engages=engaged[startTime:endTime]
        effects=standardizedEffectsUser[startTime:endTime]

        nBlue=len(np.where(engages==1.)[0])
        nRed=len(np.where(engages==0.)[0])

        isDetermined = (nBlue >=x) and (nRed >= x)

        if isDetermined:
            nDeterminedWindows=nDeterminedWindows+1
            determinedTimes.append(day)

            # calculate effects
            engageIndices=np.where(engages==1)[0]
            nonEngageIndices=np.where(engages==0)[0]
            avgEngage=np.mean(effects[engageIndices])
            avgNonEngage=np.mean(effects[nonEngageIndices])

            avgEngageAll.append(avgEngage)
            avgNonEngageAll.append(avgNonEngage)
            if avgEngage > avgNonEngage:
                nInterestingDeterminedWindows=nInterestingDeterminedWindows+1

    nUndeterminedSlidingWindows=nSlidingWindows-nDeterminedWindows

    # output
    statistic={}
    if nSlidingWindows >0 and nDeterminedWindows >0:
        statistic["r1"]=nUndeterminedSlidingWindows/nSlidingWindows
        statistic["r2"]=abs(nInterestingDeterminedWindows/nDeterminedWindows - 0.5)
        statistic["isInteresting"]=(statistic["r1"]<=delta1) and (statistic["r2"]>=delta2)
    else: 
        statistic["r1"]=None
        statistic["r2"]=None
        statistic["isInteresting"]=False
    
    # to reproduce twin curves of avg of engaged and not engaged effects at dtermiend times
    statistic["determinedTimes"]=determinedTimes
    statistic["avgNonEngageAll"]=avgNonEngageAll
    statistic["avgEngageAll"]=avgEngageAll
    
    statistic['stdEffectRatioEngaged']=stdEffectRatioEngaged
    statistic['stdEffectRatioNotEngaged']=stdEffectRatioNotEngaged
    statistic['stdEffectRatio']=stdEffectRatioEngaged/stdEffectRatioNotEngaged
    #result['nAvailable']=nAvail
    #result['nEngagedAndAvailable']=nEngagedAndAvail

    # to reproduce plot of standardized posterior at engaged and not engaged states
    statistic['engaged']=engaged
    statistic['standardizedEffects']=standardizedEffectsUser
    return statistic

# %%
def load_original_run(output_path):
    with open(output_path, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return original_result

def load_data():
    with open(PKL_DATA_PATH, "rb") as f:
        data= pkl.load(f)
    return data

def plotUserDay(result, resultRun, user, outputPath, stateName):
    # with posterior
    outputPath=os.path.join(outputPath, "user"+str(user))
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    # determined times, standardized effects
    rc('mathtext', default='regular')
    fig = plt.figure(figsize=(6.4+10, 4.8+7.5))
    ax = fig.add_subplot(111)

    lns1 = ax.plot(result['determinedTimes'], result['avgEngageAll'], '-',  marker='o', color='b', label = 'Window Avg Std Posterior Treatment Effect when '+stateName+ ' is 1')
    lns2 = ax.plot(result['determinedTimes'], result['avgNonEngageAll'], '-',  marker='o', color='r', label = 'Window Avg Std Posterior Treatment Effect when '+stateName+ ' is 0')

    # added these three lines
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='lower left', bbox_to_anchor=(0,-.13), fancybox = True, shadow = True)

    # added these three lines
    ax.grid()
    ax.set_xlabel('Determined Day')
    ax.set_ylabel("Average Standardized Posterior Treatment Effect")

    plt.title('Window Average Standardized Posterior Treatment Effect at Determined Days for User '+str(user)+". R1: "+str(result['r1'])+ ", R2: "+str(result['r2']))
    plt.savefig(outputPath+'/blueRedDetermined_user-'+str(user) +'.png')
    plt.clf()

    # standardized effects directly
    rc('mathtext', default='regular')
    fig = plt.figure(figsize=(6.4+10, 4.8+7.5))
    ax = fig.add_subplot(111)

    # idx = np.where(result['engaged']==1.)[0]
    # nonIdx = np.where(result['engaged']==0.)[0]
    xs=np.array(range(len(result['engaged'])))
    availTimes=np.where( resultRun['availability'][:T]==1)[0]

    xs=xs[availTimes]
    mapper=['red','blue']
    colors=[mapper[int(i)] for i in result['engaged'][availTimes]]
    ax.plot(xs, result['standardizedEffects'][xs], marker='o',zorder=1, color='k',linewidth=.5)
    ax.scatter(xs, result['standardizedEffects'][xs], marker='o', color=colors,zorder=2)

    #lns2 = ax.scatter(T[nonIdx], result['standardizedEffects'][nonIdx], marker='o', color='r',)
    labelBlue = 'Avg Std Posterior Treatment Effect when '+stateName+ ' is 1'
    labelRed = 'Avg Std Posterior Treatment Effect when '+stateName+ ' is 0'
    # added these three lines
    #lns = lns1+lns2
    #labs = [l.get_label() for l in lns]
    #ax.legend(lns, labs, loc='lower left', bbox_to_anchor=(0,-.13), fancybox = True, shadow = True)
    from matplotlib.lines import Line2D
    legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label=labelBlue,
                            markerfacecolor='b', markersize=15),
                    Line2D([0], [0], marker='o', color='w', label=labelRed,
                            markerfacecolor='r', markersize=15)
                      ]
    ax.legend(handles=legend_elements, loc="lower left", bbox_to_anchor=(0,-.13), fancybox = True, shadow = True)

    ax.grid()
    ax.set_xlabel('Available Time')
    ax.set_ylabel("Average Standardized Posterior Treatment Effect")

    plt.title('Standardized Posterior Treatment Effect for User '+str(user)+". R1: "+str(result['r1'])+ ", R2: "+str(result['r2']))
    plt.savefig(outputPath+'/blueRedOverAll_user-'+str(user) +'.png')
    plt.clf()

# %%
###################################################### 
##### input in user and b.s. version of the user #####
######################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default="./output/slidingWindow", required=False, help="Output file directory name")
    parser.add_argument("-l", "--log", type=str, default="./log", required=False, help="Log file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=False, help="Baseline of interest")
    parser.add_argument("-u", "--user", type=int, default=80, required=False, help="User to bootstrap")
    parser.add_argument("-or", "--original_result", type=str, default="./init/original_result_91.pkl", required=False, help="Pickle file for original results")
    args = parser.parse_args()

    option=1

    if option==1:
        indexFS=2 #for engagement
        outputPath="./output/interestingness/"+str(F_KEYS[indexFS])
        delta1=.26
        delta2=.47 #user 88
    elif option==2:
        indexFS=3 #for home/work
        outputPath="./output/interestingness/"+str(F_KEYS[indexFS])
        delta1=.49
        delta2=.5 #user 1
    elif option==3:
        indexFS=4 #for var
        outputPath="./output/interestingness/"+str(F_KEYS[indexFS])
        delta1=.1
        delta2=.5 #user 33
    #data=load_data()

    # read in results from original run and bootstrap
    original_result=load_original_run(args.original_result)

    r1s=[]
    r2s=[]
    ids=[]
    effectRatios=[]
    effectIds=[]
    interestingR2s=[]
    interestingR1s=[]

    nNone=0
    for i in range(NUSERS):
        result=computeMetricSlidingDay(original_result[i], indexFS,delta1=delta1, delta2=delta2)
        print("USER "+str(i))
        if result['r1']!=None and result['r2']!=None:
            r1s.append(result['r1'])
            r2s.append(result['r2'])
            effectRatios.append(result['stdEffectRatio'])
            effectIds.append(i)
            print(result['r1'])
            print(result['r2'])
            if result['isInteresting']:
                print("found interesting user")
                ids.append(i)
                interestingR2s.append(result['r2'])
                interestingR1s.append(result['r1'])
                plotUserDay(result, original_result[i], i, outputPath, F_KEYS[indexFS])
        else:
            nNone=nNone+1

    print("see percs")
    percs=[0,25,50,75,100]
    print(np.percentile(r1s, percs))
    print(np.percentile(r2s, percs))
    print(np.percentile(effectRatios, percs))
    print(nNone)
    print(ids)
    r1s=np.array(r1s)
    r2s=np.array(r2s)
    import pdb
    pdb.set_trace()


# %%
if __name__ == "__main__":
    main()
