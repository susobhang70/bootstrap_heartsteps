
# %%
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import argparse
import os
import matplotlib.pyplot as plt
import math
import pdb

import matplotlib as mpl
import pylab
from matplotlib import rc
import matplotlib.dates as mdates
import datetime as dt
import matplotlib.ticker as ticker
from matplotlib.dates import DateFormatter

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
NDAYS = 90
NUSERS = 91
NTIMES = 5
T=NDAYS*NTIMES

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "temperature", "logpresteps", "sqrt_totalsteps", "dosage", "engagement", "other_location", "variation"]
G_LEN = len(G_KEYS)
def rindex(l, value):
    for i in reversed(range(len(l))):
        if l[i]==value:
            return i
    return -1

# %%
def load_original_run(output_path):
    with open(output_path, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return original_result


def computeMetricSlidingDay(result, indexFS, x=2, delta1=.5, delta2=.5):
    # iterate through sliding windows
    #idx = np.where(data[:T,2]==1)[0]
    #ndata=replace with first timeopint where where its all 0
    ndata=rindex(result['availability'][:T], 1)+1 # +1 for actual number of timepoints. #np.where(data[:T,1]==0)[0] #first time the user is out
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

    effLastDay=min(last_day+1+1,90)
    for day in range(effLastDay): #want it to iterate to last day so +1. One more +1 for in case we stop before day 90 and can still forecase next day.
        for time in range(NTIMES):
            ts = (day) * 5 + time
            if result['availability'][ts]==1.: #and not np.isnan(result['reward'][ts]):
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
    nSlidingWindows=NDAYS#last_day+1, so those who drop out after 10 days do not come out as interesting
    nDeterminedWindows=0

    nInterestingDeterminedWindows_intscore1=0
    nSlidingWindows_intscore1=NDAYS#last_day+1, so those who drop out after 10 days do not come out as interesting
    nDeterminedWindows_intscore1=0

    avgEngageAll=[]
    avgNonEngageAll=[]
    determinedTimes=[]
    for day in range(effLastDay): #loop through all avail decision times
        # we check these are at least 1 when filtering on available and non nanreward. 
        # The reason is we check update happeneed day before each of the considered windows
        # indeed, check day-2, day-1, and day.
        avail_idx_pre2=np.array([])
        avail_idx_pre1=np.array([])
        avail_idx_cur=np.array([])
        if day == 0: #length is 2*NTIMES
            startTime=0
            endTime=NTIMES*2

            # check day 0 has any updates
            avail_idx_cur = np.logical_and(~np.isnan(result['reward'][0:NTIMES]), result['availability'][0:NTIMES] == 1)

        elif day >= 1 and day < last_day: #length is 3*NTIMES
            startTime=day*NTIMES-NTIMES
            endTime=day*NTIMES+NTIMES*2

            #if day>=2:
            if day>=2:
                avail_idx_pre2 = np.logical_and(~np.isnan(result['reward'][(day*NTIMES-2*NTIMES):(day*NTIMES-NTIMES)]), result['availability'][(day*NTIMES-2*NTIMES):(day*NTIMES-NTIMES)] == 1)
            avail_idx_pre1 = np.logical_and(~np.isnan(result['reward'][(day*NTIMES-NTIMES):(day*NTIMES)]), result['availability'][(day*NTIMES-NTIMES):(day*NTIMES)] == 1)
            avail_idx_cur = np.logical_and(~np.isnan(result['reward'][(day*NTIMES):(day*NTIMES+NTIMES)]), result['availability'][(day*NTIMES):(day*NTIMES+NTIMES)] == 1)

        else: #if last_day, length is 2*NTIMES
            startTime=day*NTIMES-NTIMES
            endTime=day*NTIMES+NTIMES
            # check day lastday-1 has any updates
            avail_idx_pre2 = np.logical_and(~np.isnan(result['reward'][(day*NTIMES-2*NTIMES):(day*NTIMES-NTIMES)]), result['availability'][(day*NTIMES-2*NTIMES):(day*NTIMES-NTIMES)] == 1)
            avail_idx_pre1 = np.logical_and(~np.isnan(result['reward'][(day*NTIMES-NTIMES):(day*NTIMES)]), result['availability'][(day*NTIMES-NTIMES):(day*NTIMES)] == 1)
            if day < 89:
                endTime=day*NTIMES+2*NTIMES
                avail_idx_cur = np.logical_and(~np.isnan(result['reward'][(day*NTIMES):(day*NTIMES+NTIMES)]), result['availability'][(day*NTIMES):(day*NTIMES+NTIMES)] == 1)

        engages=engaged[startTime:endTime]
        effects=standardizedEffectsUser[startTime:endTime]

        nBlue=len(np.where(engages==1.)[0])
        nRed=len(np.where(engages==0.)[0])

        # check that an update happened before each of the 5 days window. 
        enoughUpdates = (sum(avail_idx_pre2)>0) or (sum(avail_idx_pre1)>0) or (sum(avail_idx_cur)>0) #a function of non observed reward. 
        # One flag will be true if an update happened day before, meaning at least one available and nonNanReward point exists

        isDetermined = (nBlue >=x) and (nRed >= x) and enoughUpdates

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

        #print(avail_idx_pre1)
        #print(effects)
        if sum(avail_idx_pre1) > 0 or day==0:#for int_score1
            effects_intscore1=standardizedEffectsUser[(day*NTIMES):(day*NTIMES+NTIMES)]
            if len(effects_intscore1[effects_intscore1!=-1])>=2:
                nDeterminedWindows_intscore1=nDeterminedWindows_intscore1+1
                if np.mean(effects_intscore1[effects_intscore1!=-1])>0:
                    nInterestingDeterminedWindows_intscore1=nInterestingDeterminedWindows_intscore1+1
    
    nUndeterminedSlidingWindows=nSlidingWindows-nDeterminedWindows
    nUndeterminedSlidingWindows_intscore1=nSlidingWindows_intscore1-nDeterminedWindows_intscore1

    # output
    statistic={}
    if nSlidingWindows >0 and nDeterminedWindows >0:
        statistic["r1"]=nUndeterminedSlidingWindows/nSlidingWindows
        statistic["rawR2"]=nInterestingDeterminedWindows/nDeterminedWindows
        statistic["r2"]=abs(nInterestingDeterminedWindows/nDeterminedWindows - 0.5)
        statistic["isInteresting"]=(statistic["r1"]<=delta1) and (statistic["r2"]>=delta2)
    else: 
        statistic["r1"]=None
        statistic["r2"]=None
        statistic["rawR2"]=None
        statistic["isInteresting"]=None

    if nSlidingWindows_intscore1>0 and nDeterminedWindows_intscore1 > 0:
        statistic["r3"]=nUndeterminedSlidingWindows_intscore1/nSlidingWindows_intscore1#modified to just be for if there are enough updates.
        statistic["rawR4"]=nInterestingDeterminedWindows_intscore1/nDeterminedWindows_intscore1 #
        statistic["r4"]=abs(nInterestingDeterminedWindows_intscore1/nDeterminedWindows_intscore1 - 0.5) #
        statistic["isInteresting"]=(statistic["r3"]<=delta1) and (statistic["rawR4"]>=delta2)
    else:
        statistic["r3"]=None
        statistic["r4"]=None
        statistic["rawR4"]=None
        statistic["isInteresting"]=None
    
    # to reproduce twin curves of avg of engaged and not engaged effects at dtermiend times
    statistic["determinedTimes"]=determinedTimes
    statistic["avgNonEngageAll"]=avgNonEngageAll
    statistic["avgEngageAll"]=avgEngageAll
    
    statistic['stdEffectRatioEngaged']=stdEffectRatioEngaged
    statistic['stdEffectRatioNotEngaged']=stdEffectRatioNotEngaged
    if stdEffectRatioNotEngaged !=0:
        statistic['stdEffectRatio']=stdEffectRatioEngaged/stdEffectRatioNotEngaged
    else:
        statistic['stdEffectRatio']=None
    #result['nAvailable']=nAvail
    #result['nEngagedAndAvailable']=nEngagedAndAvail

    # to reproduce plot of standardized posterior at engaged and not engaged states
    statistic['engaged']=engaged
    statistic['standardizedEffects']=standardizedEffectsUser
    return statistic

def plotUserDayDateAndResims(result, resultRun, user, outputPath, stateName, bs_results, bs_resultsRuns, bs):

    pathPeng="./init/all91_uid.pkl"
    resultPeng=pkl.load(open(pathPeng, "rb"))
    uids=resultPeng[:,0,15]
    otherDF=pd.DataFrame(uids, range(resultPeng.shape[0]), columns=["StudyID"])
    otherDF['indices']=range(len(uids))

    pathBaselineInfo="./init/baseline_info.csv"
    baseline_info=pd.read_csv(pathBaselineInfo)
    baseline_info=baseline_info[['start.date', 'StudyID']] # match by UIDS!

    start_dates=pd.merge(otherDF, baseline_info, on="StudyID")

    # determined times, standardized effects
    rc('mathtext', default='regular')
    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot()#111)#, aspect='equal')
    spacing = .4

    #fig.subplots_adjust(bottom=spacing)
    ndata=rindex(resultRun['availability'][:T], 1)+1 # +1 for actual number of timepoints. #np.where(data[:T,1]==0)[0] #first time the user is out
    last_day=math.floor((ndata-1)/NTIMES) #ndata is the ts index of the last available time, reset the -1, then /NTIMES to get the day of the last available time.
    start = dt.datetime.strptime(start_dates.iloc[user][2]+" 00", '%Y-%m-%d %H')#.date()
    end = start + dt.timedelta(days=last_day)
    plotEnd = start + dt.timedelta(days=last_day+1)

    print(start)

    xs=np.array(range(len(result['engaged'])))
    availTimes = np.logical_and(~np.isnan(resultRun['reward']), resultRun['availability'] == 1)[:len(xs)]
    availTimes=np.where(resultRun['availability'] == 1)[0]
    availTimes=availTimes[availTimes>=35]#get first week
    availsAndNonNanReward=xs[availTimes]
    mapper=['red','blue']

    #markersize=100
    markersize=650
    opacity=.7
    colors=[mapper[int(i)] for i in result['engaged'][availTimes]]
    x = [ ]
    hourInc=[0,5,10,15,20]
    for i in range(NDAYS):
        for j in range(NTIMES):
            x.append((start + dt.timedelta(days=i)+dt.timedelta(hours=hourInc[j]))) 
    x=np.array(x)

    xs=x[availsAndNonNanReward]
    y=result['standardizedEffects'][availsAndNonNanReward]

    epsilon=.1
    #ax.set_ylim([min(y)-epsilon, max(y)+epsilon])
    ax.set_ylim([-2, 2.75])
    vals=[-2,-1,0,1,2, 2.85]
    labs=['-2','-1','0','1','2','']
    ax.set_yticks(vals,labs)
    #ax.plot(xs, y, color='k', linewidth=.25)   #lines or not?
    #, marker='o',zorder=1, color='k',linewidth=.5,c lip_on=False)
    #ax.scatter(xs, y, marker='o', color=colors,zorder=2, alpha=.5)

    ax.axhline(y=0, color='k', linestyle='--', linewidth=6, alpha=.75)
    #ax.scatter(xs[result['engaged'][availTimes] == 1], y[result['engaged'][availTimes] == 1], marker='^', color='red',zorder=2, alpha=opacity,s=markersize)
    #ax.scatter(xs[result['engaged'][availTimes] == 0], y[result['engaged'][availTimes] == 0], marker='o', color='blue',zorder=2, alpha=opacity,s=markersize)
    #ax.plot(xs[result['engaged'][availTimes] == 1], y[result['engaged'][availTimes] == 1], color='red',zorder=2, alpha=opacity,linewidth=4)#markersize=markersize)
    #ax.plot(xs[result['engaged'][availTimes] == 0], y[result['engaged'][availTimes] == 0], color='blue',zorder=2, alpha=opacity,linewidth=4)#markersize=markersize)
    opacityB=.2
    #for b in range(len(bs_results)):
    yb=bs_results[len(bs_results)-1]['standardizedEffects'][availsAndNonNanReward]
    #ax.scatter(xs, yb, marker='s', color='k',zorder=2, alpha=opacity,s=markersize)
    print("BS for this user is "+str(bs)+", "+ str(len(yb)))
    #ax.plot(xs[result['engaged'][availTimes] == 1], yb[result['engaged'][availTimes] == 1], color='red',zorder=2, alpha=opacityB, linewidth=4)#,markersize=markersize)
    #ax.plot(xs[result['engaged'][availTimes] == 0], yb[result['engaged'][availTimes] == 0], color='blue',zorder=2, alpha=opacityB, linewidth=4)#,markersize=markersize)
    ax.scatter(xs[result['engaged'][availTimes] == 1], yb[result['engaged'][availTimes] == 1], marker='^', color='red',zorder=2, alpha=opacity,s=markersize)
    ax.scatter(xs[result['engaged'][availTimes] == 0], yb[result['engaged'][availTimes] == 0], marker='o', color='blue',zorder=2, alpha=opacity,s=markersize)

    #nanRewardTimes=np.isnan(resultRun['reward'])[:len(x)] 
    nanRewardTimes=np.isnan(resultRun['reward'])[35:len(x)] #exclude first week
    nanRewardTimes=np.where(nanRewardTimes==1.)[0]
    nanRewardTimes=x[nanRewardTimes]
    y=np.repeat(min(y)-epsilon, len(nanRewardTimes))
    #ax.scatter(nanRewardTimes, y, marker='|', zorder=3, s=70, clip_on=False) #add in if we want.
    
    colors=[mapper[int(i)] for i in result['engaged'][availTimes]]


    from matplotlib.lines import Line2D#https://stackoverflow.com/questions/47391702/how-to-make-a-colored-markers-legend-from-scratch
    markersize=35
    lSize=40
    #labelBlue=""
    #legend_elements = [
    #                Line2D([0], [0], marker='^', color='w', label=labelBlue,
    #                        markerfacecolor='red', markersize=markersize, alpha=opacity)
    #                  ]
    labelBlue=stateName+" = 0"
    labelRed=stateName+" = 1"
    legend_elements = [
                    Line2D([0], [0], marker='o', color='w', label=labelBlue,
                            markerfacecolor='b', markersize=markersize, alpha=opacity),
                    Line2D([0], [0], marker='^', color='w', label=labelBlue,
                            markerfacecolor='r', markersize=markersize, alpha=opacity)
                      ]
   
    ax.legend(handles=legend_elements, loc="upper left",fancybox = True, shadow = True, fontsize=lSize, handletextpad=-.2)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.4)#https://stackoverflow.com/questions/54342199/configuring-grid-lines-in-matplotlib-plot

    lSize=80
    ax.set_xlabel('Date (MM-DD)', fontsize=lSize)
    ax.set_ylabel("Std. Advantage", fontsize=lSize)

    lSize=75
    newLabels=[i.strftime('%m-%d') for i in xs]
    newXTicks=[]
    newLabels2=[]
    for i in range(len(x)):
        if i%(35*2)==0 and i!=0:
            newXTicks.append(x[i])
            newLabels2.append(x[i].strftime('%m-%d'))
    ax.set_xticks(newXTicks, newLabels2)
    ax.tick_params(axis='both', which='major', labelsize=lSize)
    ax.tick_params(axis='both', which='minor', labelsize=lSize)

    #fig.autofmt_xdate()
    #ax.set_xlim([newLabels2[0], newLabels2[len(newLabels2)-1]])
    #ax.set_xlim([start,end])
    ax.set_xlim([start,plotEnd])
    myFmt = mdates.DateFormatter('%m-%d')
    ax.xaxis.set_major_formatter(myFmt)
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)

    #plt.gcf().autofmt_xdate()
    print(outputPath+'/blueRedOverAll_user-'+str(user)+'-state-'+stateName + "_resim-"+str(bs)+'.pdf')
    plt.tight_layout()
    plt.savefig(outputPath+'/blueRedOverAll_user-'+str(user)+'-state-'+stateName + "_resim-"+str(bs)+'.pdf')
    plt.clf()
    return 


# %%
###################################################### 
##### input in user and b.s. version of the user #####
######################################################
import sys
import random

np.random.seed(0)
random.seed(0)
from os import listdir
from os.path import isfile, join

def main():
    #indexFS=int(sys.argv[1])
    #baseline=F_KEYS[indexFS]
    baseline="Zero"
    user_specific=False

    original_result="./init/original_result_91.pkl"
    delta1=.75
    delta2=.4
    B=500
    subset=2
    output="./output"

    # read in results from original run and bootstrap
    original_result=load_original_run(original_result)

    # users to plot
    users=range(NUSERS)
    users=[77]
    indexFS=4

    from slidingWindow_og import computeMetricSlidingDay
    #iterate thru
    for i in users: #range(NUSERS):
        print("User "+str(i))
        resultSliding=computeMetricSlidingDay(original_result[i], indexFS,delta1=delta1, delta2=delta2)

        ## get bootstrap results
        trajectories=random.sample(range(B), subset)
        bs_results=[]
        bs_results_sliding=[]
        for bootstrap in trajectories:
            print("processing "+str(bootstrap))
            output_dir = os.path.join(output, "Baseline-"+ str(baseline)+"_UserSpecific-"+str(True)+"_User-"+str(i), "Bootstrap-" + str(bootstrap))
            # get pkl file. 
            onlyfiles = [f for f in listdir(output_dir) if isfile(join(output_dir, f))]
            if len(onlyfiles)>0:
                filepath=os.path.join(output_dir, onlyfiles[0])
                bs_res=pkl.load(open(filepath, "rb"))
                bs_results.append(bs_res)
                bs_results_sliding.append(computeMetricSlidingDay(bs_res, indexFS, delta1=delta1, delta2=delta2))

                plotUserDayDateAndResims(resultSliding, original_result[i], i, "./output/redblue", F_KEYS[indexFS], bs_results_sliding, bs_results, bootstrap)

# %%
if __name__ == "__main__":
    main()
