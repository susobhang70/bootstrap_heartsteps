### Plot histograms of int scores from original data for int score 1 (baseline Zero) and int score 2 ###

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
import seaborn as sns

import matplotlib as mpl
import pylab
from matplotlib import rc

lsize=80
axSize=67

def setPlotSettings(changeAx=False):
    fix_plot_settings = True
    if fix_plot_settings:
        plt.rc('font', family='serif')
        plt.rc('text', usetex=True)
        label_size = lsize
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
    
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.7g" % x))

NUSERS = 91
F_KEYS=["intercept", "dosage", "engagement", "other_location", "variation"]

def main():
    experiment = int(sys.argv[1]) 
    # Get the user index
    if experiment!=-1:
        baseline = F_KEYS[experiment]
    else:
        baseline="Zero"
        experiment=2

    B=500
    user_specific=False
    output="./output"
    interesting=[]
    nNone=[]

    # load original R1/R2
    from slidingWindow_og import computeMetricSlidingDay
    ogR1=[]
    ogR2=[]
    rawR2s=[]

    original_result="./init/original_result_91.pkl"
    delta1=.75
    delta2=.4
    with open(original_result, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)

    allr1=[]
    allr2=[]
    allr2Raw=[]
    iids=[]
    i=0
    for result in original_result:
        result=computeMetricSlidingDay(result, experiment,delta1=delta1, delta2=delta2)
        if baseline=="Zero":
            allr1.append(result['r1'])
            allr2.append(result['r2'])
            allr2Raw.append(result['rawR2'])
        else:
            allr1.append(result['r3'])
            allr2.append(result['r4'])
            allr2Raw.append(result['rawR4'])

        r1=result['r1']
        r2=result['r2']
        rawr2=result['rawR2']
        if baseline=="Zero":
            r1=result['r3']
            r2=result['r4']
            rawr2=result['rawR4']
        if r1 != None and r2 != None:
            if r1 <= delta1:
                ogR1.append(r1)
                ogR2.append(r2)
                rawR2s.append(rawr2)
                if r2>=delta2:
                    iids.append(i)
        i=i+1

    ogR1=np.array(ogR1)
    ogR2=np.array(ogR2)

    # plot C2 histograms
    outputPath=os.path.join(output, "Baseline-"+ str(baseline)+"_UserSpecific-"+str(user_specific))
    image=outputPath+'/histogram_C2_'+baseline+'.pdf'
    setPlotSettings(True)
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(figsize=(15, 15))
    barcol='gray'
    df=pd.DataFrame(rawR2s, columns=['rawR2s'])
    bins="auto"
    p = sns.histplot(data=df, x='rawR2s', bins=bins, stat='count', ax=ax, color=barcol, cbar_kws={"linewidth":0}, line_kws={"linewidth":0}, linewidth=0)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.xlabel("")
    plt.ylabel("\# Users")
    plt.grid(axis='y', alpha=.5, zorder=0) 
    ax=plt.gca()
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False)
    plt.xlim([0,1])
    ax.set_xticklabels([0,.25,.5,.75,1])
    plt.axvline(.5, color='k', ls='--', zorder=4, lw=4)
    plt.tight_layout()
    plt.savefig(image, format="pdf")
    print(image)
    plt.clf()

    plt.clf()
    image=outputPath+'/histogram_C2Adj_'+baseline+'.pdf'
    setPlotSettings(True)
    plt.rcParams['text.usetex'] = True
    fig, ax = plt.subplots(figsize=(15, 15))
    barcol='gray'
    df=pd.DataFrame(ogR2, columns=['C2Adj'])
    bins="auto"
    p = sns.histplot(data=df, x='C2Adj', bins=bins, stat='count', ax=ax, color=barcol, cbar_kws={"linewidth":0}, line_kws={"linewidth":0}, linewidth=0)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    plt.xlabel("")
    plt.ylabel("\# Users")
    plt.xlim([0,.5])
    plt.axvline(.25, color='k', ls='--', zorder=4, lw=4)
    plt.grid(axis='y', alpha=.5, zorder=0) 
    plt.tight_layout()
    plt.savefig(image, format="pdf")
    print(image)
    return

if __name__== "__main__":
    main()

