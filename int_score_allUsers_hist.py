### plot percs/frac of how extreme each user's C2 value is w.r.t. replication distribution, in a histogram. Additionally plot histogram of each user's C2 ###
### NOTE: R3 and R4 are analogues of C1 and C2 for int score 1, while R1 and R2 are analogues of C1 and C2 for int score 2.

import subprocess
import numpy as np
import sys
import os
from datetime import date, datetime
from itertools import product
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import pickle as pkl

import matplotlib as mpl
import pylab
from matplotlib import rc
import seaborn as sns

lsize=80
axSize=60
def setPlotSettings(changeAx=False):
    fix_plot_settings = True
    if fix_plot_settings:
        label_size=lsize
        plt.rc('font', family='serif')
        plt.rc('text', usetex=False)
        mpl.rcParams['axes.labelsize'] = label_size
        mpl.rcParams['axes.titlesize'] = label_size
        if changeAx:
            mpl.rcParams['xtick.labelsize'] = axSize
            mpl.rcParams['ytick.labelsize'] = axSize
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


NUSERS = 91
F_KEYS=["intercept", "dosage", "engagement", "other_location", "variation"]

def main():
    experiment = int(sys.argv[1])
    from slidingWindow_og import computeMetricSlidingDay
    users=range(NUSERS)#[77]
    ogR1=[]
    ogR2=[]
    rawR2s=[]
    baseline="Zero"
    r1Key="r3"
    r2Key="r4"
    r2RawKey="rawR4"
    if experiment!=-1:#if 4, then [4]
        baseline=F_KEYS[experiment]
        r1Key="r1"
        r2Key="r2"
        r2RawKey="rawR2"
    original_result="./init/original_result_91.pkl"
    delta1=.75
    delta2=.4
    allRawR2=[]
    holderR2s=[]
    isInteresting=[]
    i=0
    iids=[]
    with open(original_result, 'rb') as handle:
        original_result=pkl.load(handle)
    for result in original_result:
        result=computeMetricSlidingDay(result, experiment,delta1=delta1, delta2=delta2)
        intVal=False
        if result[r1Key] != None and result[r2Key] != None:
            if result[r1Key] <= delta1 and result[r2Key]>=delta2:
                ogR1.append(float(result[r1Key]))
                ogR2.append(float(result[r2Key]))
                rawR2s.append(result[r2RawKey])
                intVal=True
                iids.append(i)
            holderR2s.append(float(result[r2Key]))
        else:
            holderR2s.append('None')
        allRawR2.append(result[r2RawKey])
        isInteresting.append(intVal)
        i=i+1
    ogR1=np.array(ogR1)
    ogR2=np.array(ogR2)

    # Get the user index
    B=500
    delta1=.75
    delta2=.4
    user_specific=False
    output="./output"
    interesting=[]
    nNone=[]
    observed=np.logical_and(ogR2 >= delta2, ogR1 <= delta1)
    c2Users={}
    c2RawUsers={}
    for i in users:
        c2Users[str(i)]=[]
        c2RawUsers[str(i)]=[]
        output_dir = os.path.join(output, "Baseline-"+ str(baseline)+"_UserSpecific-"+str(True)+"_User-" + str(i))
        statistics=os.path.join(output_dir, "statistics.csv")
        results=os.path.join(output_dir, "results.csv")
        if os.path.exists(results):
            results=pd.read_csv(results)
            ids=results['trueId']
            c2RawUsers[str(i)]=results['rawR4']
    
    #now get percs
    percs=[]
    iids=[]
    allPercs=[]
    for i in users:
        perc=None
        if len(c2Users[str(i)])>0:#only added if its interesting and r2 is good 
            perc=stats.percentileofscore(c2Users[str(i)], float(holderR2s[i]), 'weak')/100
            perc=1-perc
            iids.append(i)
            percs.append(perc)
        allPercs.append(perc)

    #plot histogram too!
    outputPath=os.path.join(output, "Baseline-"+ str(baseline)+"_UserSpecific-"+str(user_specific))
    image=outputPath+'/histogram_percs_'+baseline+'_x=2'+'.pdf'
    plt.clf()
    setPlotSettings(True)
    plt.rcParams['text.usetex']=True
    fig, ax=plt.subplots(figsize=(15,15))
    plt.grid(axis='y', alpha=.5, zorder=0)
    barcol="gray"
    df=pd.DataFrame(percs, columns=['percs'])
    p = sns.histplot(data=df, x='percs', stat='count', ax=ax, color=barcol, cbar_kws={"linewidth":0}, line_kws={"linewidth":0}, linewidth=0)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    ax=plt.gca()
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    plt.xlabel("")
    plt.ylabel("\# Users")

    plt.tight_layout()
    plt.savefig(image, format="pdf")
    print(image)
    plt.clf()

    for i in users:
        if len(c2RawUsers[str(i)])>0:
            image=output+'/histogram_c2s_user_'+str(i)+'_'+baseline+'_x=2'+'.pdf'
            plt.clf()
            c2s=c2RawUsers[str(i)]
            setPlotSettings(True)
            fig, ax=plt.subplots(figsize=(24,16))#, dpi=80)
            plt.grid(axis='y', alpha=.5, zorder=0)
            barcol="gray"
            df=pd.DataFrame(np.array(c2s), columns=['c2s'])
            p = sns.histplot(data=df, x='c2s', stat='probability', ax=ax, color=barcol, cbar_kws={"linewidth":0}, line_kws={"linewidth":0}, linewidth=0)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax=plt.gca()
            plt.xlabel("")
            plt.ylabel("Proportion")
            plt.axvline(allRawR2[i], ls='-.', color='b', zorder=4, lw=6)
            plt.xlim(left=-.05)
            plt.xticks([0, .25,.5,.75,1])
            
            if i==77 and baseline=="Zero": #for cosmetic purposes
                labs=['0', '0.02','0.04','0.06','0.08','0.10','0.12','0.14']
                vals=[float(lab) for lab in labs]
                labs[0]=''
                plt.yticks(vals,labs)
            else:
                yticks = ax.yaxis.get_major_ticks()
                yticks[0].label1.set_visible(False)
            plt.tight_layout()
            plt.savefig(image, format="pdf")
            print(image)
            plt.clf()

if __name__ == "__main__":
    main()
