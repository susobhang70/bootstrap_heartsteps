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

lsize=60
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
    experiment = int(sys.argv[1]) #max array size AND index is 10k. we could split 10k into 4 for 2.5 k jobs each, leading to ~28 bootstrap resamples. this is rather small, so opt to fill in experiment

    # Get the user index
    r1Key='r1'
    r2Key='r2'
    r2RawKey='rawR2'
    if experiment==-1:
        baseline="Zero"
        r1Key='r3'
        r2Key='r4'
        r2RawKey='rawR4'
    else:
        baseline = F_KEYS[experiment]

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
    delta1=.5
    delta2=.2
    with open(original_result, 'rb') as handle:
        original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)
    for result in original_result:
        result=computeMetricSlidingDay(result, experiment,delta1=delta1, delta2=delta2)
        if result[r1Key] != None and result[r2Key] != None:
            ogR1.append(result[r1Key])
            ogR2.append(result[r2Key])
            rawR2s.append(result[r2RawKey])
    ogR1=np.array(ogR1)
    ogR2=np.array(ogR2)
    
    # range of \gamma and \delta to go over
    d1s=[.75, .70,.65]
    d2s=[.35,.40,.45]
    output_dir = os.path.join(output, "Baseline-"+ str(baseline)+"_UserSpecific-"+str(user_specific), "Bootstrap-" + str(0))
    print("Output Dir is "+str(output_dir))

    heatMap=[]
    heatMapR1=[]
    for delta1 in d1s:
        heatMapRow=[]
        heatMapRowR1=[]
        for delta2 in d2s:
            bInteresting=[]
            observed=sum(np.logical_and(ogR1 <= delta1, ogR2 >= delta2))
            t=np.logical_and(ogR1 <= delta1, ogR2 >= delta2)
            indices=[i for i in range(len(t)) if t[i]]
            #print("Interesting users for d1 = "+str(delta1)+ " , d2 = "+str(delta2))
            #print(indices)
            heatMapRowR1.append(sum(ogR1<=delta1))
            for bootstrap in range(B):
                output_dir = os.path.join(output, "Baseline-"+ str(baseline)+"_UserSpecific-"+str(user_specific), "Bootstrap-" + str(bootstrap))
                results=os.path.join(output_dir, "results"+".csv")
                df=pd.read_csv(results)

                #bootstrapR1s=np.array(df[' r1'])
                bootstrapR1s=np.array(df[r1Key])
                if 'None' in bootstrapR1s:
                    bootstrapR1s=bootstrapR1s[bootstrapR1s.astype(str) != 'None']
                bootstrapR1s=bootstrapR1s.astype(float)

                #bootstrapR2s=np.array(df[' r2'])
                bootstrapR2s=np.array(df[r2Key])
                if 'None' in bootstrapR2s:
                    bootstrapR2s=bootstrapR2s[bootstrapR2s.astype(str) != 'None']
                bootstrapR2s=bootstrapR2s.astype(float)

                bootstrapValue=sum(np.logical_and(bootstrapR1s <= delta1, bootstrapR2s >= delta2))
                bInteresting.append(bootstrapValue)

            percs=[0,25,50,75,100]
            perc=stats.percentileofscore(bInteresting, observed, 'weak')/100
            quantiles=np.percentile(bInteresting,percs)
            heatMapRow.append(1-perc)

            #plot histogram too!
            outputPath=os.path.join(output, "Baseline-"+ str(baseline)+"_UserSpecific-"+str(user_specific))
            image=outputPath+'/histogram_Interesting_'+baseline+"_delta1="+str(delta1)+"_delta2="+str(delta2)+"_B="+str(B)+'.pdf'
            plt.clf()
            setPlotSettings(True)
            plt.rcParams['text.usetex'] = True
            fig, ax = plt.subplots(figsize=(15, 15))
            barcol='gray'
            df=pd.DataFrame(bInteresting, columns=['nInt'])
            bins="auto"
            p = sns.histplot(data=df, x='nInt', bins=bins, stat='probability', ax=ax, color=barcol, cbar_kws={"linewidth":0}, line_kws={"linewidth":0}, linewidth=0)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            yticks = ax.yaxis.get_major_ticks() 
            yticks[0].label1.set_visible(False)
            plt.xlabel("")
            plt.ylabel("Proportion")
            plt.grid(axis='y', alpha=.5, zorder=0)
            plt.axvline(observed, color='b', ls='-.', zorder=4, lw=6)
            if experiment==3:
                plt.xticks(range(0, 10, 2))
            plt.xlim(left=0)
            plt.tight_layout()
            plt.savefig(image, format="pdf")
            print(image)
        
        heatMap.append(heatMapRow)
        heatMapR1.append(heatMapRowR1)

    sns.set(rc={'text.usetex': True})
    plt.clf()
    setPlotSettings(True)
    fig, ax=plt.subplots(figsize=(15,15))
    heatMap=pd.DataFrame(heatMap)
    heatMapR1=pd.DataFrame(heatMapR1)
    print(heatMap)
    print(heatMapR1)
    d1s=['0.75', '0.70','0.65']
    d2s=['0.35','0.40','0.45']
    s=sns.heatmap(heatMap,xticklabels=d2s, ax=ax, yticklabels=d1s, cmap="Blues_r", fmt='.2f', annot=True, vmin=0, vmax=1, annot_kws={'fontsize': lsize})
    plt.yticks(rotation=0) 
    s.set(xlabel="$\delta$", ylabel="$\gamma$")
    f=s.get_figure()
    toWrite=baseline
    if baseline=="other_location":
        toWrite="other location"
    image=outputPath+'/heatMap_Interesting_'+toWrite+"_B="+str(B)+'.pdf'
    plt.tight_layout()
    print(image)
    f.savefig(image, format="pdf")
    plt.clf()

if __name__ == "__main__":
    main()
