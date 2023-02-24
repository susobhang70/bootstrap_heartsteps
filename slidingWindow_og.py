
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

def computeMetricSlidingDay(result, indexFS, x=2, delta1=.5, delta2=.5):
    # iterate through sliding windows
    #idx = np.where(data[:T,2]==1)[0]
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

        # if day==0 or day==effLastDay-1:
        #     import pdb
        #     pdb.set_trace()

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

    nUndeterminedSlidingWindows=nSlidingWindows-nDeterminedWindows

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
        statistic["isInteresting"]=False
    
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

# %%
def load_original_run(output_path):
    with open(output_path, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)
    return original_result

def plotUserDay(result, resultRun, user, outputPath, stateName, userSpec):
    # with posterior
    # with posterior
    if not userSpec:
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
    availTimes = np.logical_and(~np.isnan(resultRun['reward']), resultRun['availability'] == 1)[:len(xs)]
    availsAndNonNanReward=xs[availTimes]
    mapper=['red','blue']
    colors=[mapper[int(i)] for i in result['engaged'][availTimes]]

    ax.plot(availsAndNonNanReward, result['standardizedEffects'][availsAndNonNanReward], marker='o',zorder=1, color='k',linewidth=.5)
    ax.scatter(availsAndNonNanReward, result['standardizedEffects'][availsAndNonNanReward], marker='o', color=colors,zorder=2)

    nanRewardTimes=xs[np.isnan(resultRun['reward'])[:len(xs)]]
    ax.scatter(nanRewardTimes, np.repeat(0, len(nanRewardTimes)), marker='|',zorder=3)

    #lns2 = ax.scatter(T[nonIdx], result['standardizedEffects'][nonIdx], marker='o', color='r',)
    labelBlue = 'Std Posterior Treatment Effect when '+stateName+ ' is 1'
    labelRed = 'Std Posterior Treatment Effect when '+stateName+ ' is 0'

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
    ax.set_ylabel("Standardized Posterior Treatment Effect")

    plt.title('Standardized Posterior Treatment Effect for User '+str(user)+". R1: "+str(result['r1'])+ ", R2: "+str(result['r2']))
    print(outputPath)
    plt.savefig(outputPath+'/blueRedOverAll_user-'+str(user) +'.png')
    plt.clf()
    return sum(result['engaged'][availTimes]==1), sum(result['engaged'][availTimes]==0)

# %%
###################################################### 
##### input in user and b.s. version of the user #####
######################################################
import sys

def main():
    indexFS=int(sys.argv[1])

    original_result="./init/original_result_91.pkl"
    if indexFS==2:
        delta1=.7147835301340764
        delta2=.25 #user 
    elif indexFS==3:
        outputPath="./output/interestingness/"+str(F_KEYS[indexFS])
        delta1=.75
        delta2=.25 #user 1
    elif indexFS==4:
        outputPath="./output/interestingness/"+str(F_KEYS[indexFS])
        delta1=.6236469600585157 #.5 is even stil pretty interesting imo, as user 85 is .48888 may as well set to .5
        delta2=.25 #user 85!
    # #data=load_data()
    
    delta1=.75
    delta2=.4
    outputPath="./output/interestingness/"+str(F_KEYS[indexFS])

    # read in results from original run and bootstrap
    original_result=load_original_run(original_result)

    totalAllPeeps=[]
    FS_availPeepsYes=[]
    FS_availPeepsNo=[]
    r1s=[]
    r2s=[]
    ids=[]
    effectRatios=[]
    effectIds=[]
    interestingR2s=[]
    interestingR1s=[]

    #compute the probabilities of each and take x=1, do 1-min().
    
    T=NDAYS*NTIMES

    nNone=0
    results=[]
    FS_averages=[0,0]
    FS_notAverages=[0,0]

    FS_avail_averages=[0,0]
    FS_all_averages=[0,0]
    FS_allPeepsYes=[]
    FS_allPeepsNo=[]

    totals=0
    b=0
    r=0
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
                results.append(result)
                print("found interesting user")
                ids.append(i)
                interestingR2s.append(result['r2'])
                interestingR1s.append(result['r1'])
                print(result['rawR2'])
            b,r=plotUserDay(result, original_result[i], i, outputPath, F_KEYS[indexFS])
        else:
            nNone=nNone+1

        availTimes = np.logical_and(~np.isnan(original_result[i]['reward']), original_result[i]['availability'] == 1)[:len(result['engaged'])]
        totals=totals+len(availTimes)
        totalAllPeeps.append(len(availTimes))

        FS_avail_averages[1]=FS_avail_averages[1]+b#sum(original_result[i]['fs'][:len(result['engaged']),indexFS][availTimes]==1)
        FS_avail_averages[0]=FS_avail_averages[0]+r# sum(original_result[i]['fs'][:len(result['engaged']),indexFS][availTimes]==0)

        #FS_all_averages[1]=FS_all_averages[1]+sum(original_result[i]['fs'][:len(result['engaged']),indexFS])
        #FS_all_averages[0]=FS_all_averages[0]+ (len(original_result[i]['fs'][:len(result['engaged']),indexFS])-sum(original_result[i]['fs'][:len(result['engaged']),indexFS]))

        #FS_allPeepsYes.append(sum(original_result[i]['fs'][:len(result['engaged']),indexFS]))
        #FS_allPeepsNo.append( (len(original_result[i]['fs'][:len(result['engaged']),indexFS])-sum(original_result[i]['fs'][:len(result['engaged']),indexFS])))

        FS_availPeepsYes.append(b)#sum(original_result[i]['fs'][:len(result['engaged']),indexFS][availTimes]==1))
        FS_availPeepsNo.append(r)#sum(original_result[i]['fs'][:len(result['engaged']),indexFS][availTimes]==0))

# engagement
# subset avail/nonNan
#[0.12970884503932661]
#[0.19437008417276114]
# among all.
#0.3273504
#0.5575092
#Users yes and no
#[0.01111111 0.1888889 0.3333333 0.4833333 0.6777778]
#[0.04444444 0.4166667 0.5555556 0.7 0.9888889]

# location
# subset avail/nonNan
#[0.12013246860769974]
#[0.20394646060438804]
# among all
#0.3744811
#0.5103785
#USers yes and no percs
#see percs
#[0.6222222 0.7777778 0.8888889 0.9444444 0.9888889]
#[0 0.2964286 0.4272487 0.5 0.5]

#array([0, 0, 0.002222222, 0.004444444, 0.004444444, 0.01777778, 0.02222222, 0.02888889, 0.05555556, 0.07555556, 0.07777778, 0.08, 0.08444444, 0.09555556, 0.09555556, 0.1, 0.1022222, 0.1044444, 0.1044444, 0.1088889, 0.1088889, 0.1133333, 0.1177778, 0.1288889, 0.1311111, 0.1444444, 0.1555556, 0.1577778, 0.1644444, 0.1733333, 0.1777778, 0.18, 0.1955556, 0.1977778, 0.2111111, 0.2155556, 0.2266667, 0.2266667, 0.2333333, 0.2533333, 0.2577778, 0.2577778, 0.26, 0.2644444, 0.2644444, 0.2711111, 0.2911111, 0.3044444, 0.3088889, 0.3133333, 0.3133333, 0.3244444, 0.3266667, 0.3511111, 0.3533333, 0.36, 0.36, 0.3711111, 0.3977778, 0.4377778, 0.4444444, 0.4444444, 0.4444444, 0.4711111, 0.4933333, 0.4955556, 0.4977778, 0.5511111, 0.5955556, 0.6266667, 0.6533333, 0.6688889, 0.7, 0.74, 0.7444444, 0.7666667, 0.7711111, 0.7866667, 0.8111111, 0.8444444, 0.8444444, 0.8777778, 0.9, 0.9333333, 0.9466667, 0.96, 0.9977778, 1, 1, 1, 1])
#array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.002222222, 0.04, 0.05333333, 0.05333333, 0.06666667, 0.07111111, 0.1466667, 0.16, 0.2133333, 0.2177778, 0.2222222, 0.2288889, 0.2488889, 0.26, 0.26, 0.3311111, 0.3466667, 0.4044444, 0.4155556, 0.4288889, 0.4311111, 0.44, 0.4844444, 0.5022222, 0.5044444, 0.5066667, 0.5066667, 0.5177778, 0.5177778, 0.5555556, 0.5555556, 0.5688889, 0.5955556, 0.5977778, 0.6022222, 0.6022222, 0.6288889, 0.6355556, 0.64, 0.6422222, 0.6422222, 0.6466667, 0.6688889, 0.6866667, 0.6866667, 0.6955556, 0.6955556, 0.7088889, 0.7133333, 0.7266667, 0.7288889, 0.7355556, 0.7422222, 0.7666667, 0.7733333, 0.7822222, 0.7888889, 0.8222222, 0.8244444, 0.8355556, 0.8422222, 0.8444444, 0.8666667, 0.8688889, 0.8866667, 0.8888889, 0.8955556, 0.9, 0.9, 0.9044444, 0.9044444, 0.9155556, 0.9777778, 0.9822222, 0.9955556, 0.9955556, 0.9977778, 1])

# variation
# subset avail/nonNan
#[0.15449151372981923], for var=1
#[0.16958741548226852], for var=0

# among all
#0.3685714, for var=1
#0.5162882, for var=0

#users yes and no percs
#Users - Yes and No percs
#[0.02888889 0.2922222 0.3844444 0.4777778 0.6266667]
#[0.02888889 0.4377778 0.5044444 0.5944444 0.9711111]

#array([0.02888889, 0.03777778, 0.05777778, 0.07111111, 0.09111111, 0.1022222, 0.1066667, 0.1333333, 0.1377778, 0.1422222, 0.1444444, 0.16, 0.1644444, 0.2044444, 0.2266667, 0.2266667, 0.2533333, 0.26, 0.2733333, 0.2822222, 0.2822222, 0.2822222, 0.2911111, 0.2933333, 0.3022222, 0.3177778, 0.3222222, 0.3266667, 0.3333333, 0.3333333, 0.3355556, 0.3355556, 0.3466667, 0.3511111, 0.3555556, 0.3644444, 0.3644444, 0.3666667, 0.3666667, 0.3688889, 0.3711111, 0.3755556, 0.3777778, 0.3844444, 0.3844444, 0.3844444, 0.3888889, 0.3955556, 0.4, 0.4, 0.4022222, 0.4044444, 0.4066667, 0.4088889, 0.4111111, 0.4155556, 0.4177778, 0.4177778, 0.4177778, 0.4288889, 0.4355556, 0.4377778, 0.44, 0.4466667, 0.4533333, 0.46, 0.4711111, 0.4777778, 0.4777778, 0.4844444, 0.4888889, 0.4911111, 0.4911111, 0.4955556, 0.4977778, 0.5088889, 0.5111111, 0.5155556, 0.5177778, 0.52, 0.5222222, 0.5311111, 0.5355556, 0.5444444, 0.5488889, 0.5533333, 0.5688889, 0.58, 0.58, 0.5888889, 0.6266667])
#array([0.02888889, 0.09555556, 0.1644444, 0.1844444, 0.2844444, 0.3111111, 0.34, 0.3422222, 0.3733333, 0.3755556, 0.3777778, 0.3888889, 0.4088889, 0.4111111, 0.4133333, 0.4177778, 0.4177778, 0.42, 0.42, 0.4266667, 0.4288889, 0.4311111, 0.4355556, 0.44, 0.4444444, 0.4444444, 0.4466667, 0.4466667, 0.4488889, 0.4511111, 0.4555556, 0.4666667, 0.4688889, 0.4733333, 0.4755556, 0.4777778, 0.48, 0.4822222, 0.4822222, 0.4844444, 0.4888889, 0.4911111, 0.4955556, 0.5022222, 0.5022222, 0.5044444, 0.5044444, 0.5066667, 0.5088889, 0.5111111, 0.5155556, 0.5222222, 0.5222222, 0.5288889, 0.5311111, 0.5333333, 0.5444444, 0.5533333, 0.5644444, 0.5644444, 0.5711111, 0.5755556, 0.5777778, 0.5777778, 0.5822222, 0.5822222, 0.5933333, 0.5933333, 0.5955556, 0.5977778, 0.6, 0.6, 0.6155556, 0.6155556, 0.6244444, 0.6288889, 0.6355556, 0.6488889, 0.6511111, 0.6622222, 0.6666667, 0.6733333, 0.6977778, 0.7044444, 0.7177778, 0.7266667, 0.8355556, 0.8577778, 0.8977778, 0.9422222, 0.9711111])

    FS_avail_averages=np.array(FS_avail_averages)
    FS_all_averages=np.array(FS_all_averages)

    FS_allPeepsYes=np.array(FS_allPeepsYes)
    FS_allPeepsNo=np.array(FS_allPeepsNo)

    FS_availPeepsYes=np.array(FS_availPeepsYes)
    FS_availPeepsNo=np.array(FS_availPeepsNo)

    nt=NUSERS*T

    # print("average acrsos all users and time")
    # FS_all_averages=FS_all_averages/nt
    # print(FS_all_averages)

    # FS_allPeepsYes=FS_allPeepsYes/T
    # FS_allPeepsNo=FS_allPeepsNo/T

    # FS_allPeepsYes.sort()
    # FS_allPeepsNo.sort()

    # percs=[0,25,50,75,100]

    # print("for users")
    # print(np.percentile(FS_allPeepsYes, percs))
    # print(np.mean(FS_allPeepsYes))

    # print(np.percentile(FS_allPeepsNo, percs))
    # print(np.mean(FS_allPeepsNo))

    print("average acrsos all users and AVAIL time")
    FS_avail_averages=FS_avail_averages/nt#(sum(totalAllPeeps)), normalize by global or else sure they'd be on same scale
    print(FS_avail_averages)

    FS_availPeepsYes=FS_availPeepsYes #/totalAllPeeps
    FS_availPeepsNo=FS_availPeepsNo #/totalAllPeeps

    FS_availPeepsYes.sort()
    FS_availPeepsNo.sort()

    percs=[0,25,50,75,100]
    
    print("for users")
    print(np.percentile(FS_availPeepsYes, percs))
    print(np.mean(FS_availPeepsYes))

    print(np.percentile(FS_availPeepsNo, percs))
    print(np.mean(FS_availPeepsNo))

    # print("user spec")
    # print(FS_availPeepsYes)
    # print(FS_availPeepsNo)

    import pdb
    pdb.set_trace()
    print("see percs")
    print(np.percentile(r1s, percs))
    print(np.percentile(r2s, percs))
    print(np.percentile(effectRatios, percs))
    print(nNone)
    print(ids)
    r1s=np.array(r1s)
    r2s=np.array(r2s)

    print("N-Interesting")
    print(len(ids))

    import pdb
    pdb.set_trace()


# %%
if __name__ == "__main__":
    main()
