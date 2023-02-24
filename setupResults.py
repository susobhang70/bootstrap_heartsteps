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

fix_plot_settings = True
if fix_plot_settings:
    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)
    label_size = 12
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = label_size
    mpl.rcParams['figure.titlesize'] = label_size
    mpl.rcParams['lines.markersize'] = label_size
    mpl.rcParams['grid.linewidth'] = 2.5
    mpl.rcParams['legend.fontsize'] = label_size
#    mpl.rcParams['text.usetex'] = True
#    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

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
F_LEN=len(F_KEYS)

def main():
    experiment = int(sys.argv[1]) #max array size AND index is 10k. we could split 10k into 4 for 2.5 k jobs each, leading to ~28 bootstrap resamples. this is rather small, so opt to fill in experiment

    # Get the user index
    baseline = F_KEYS[experiment]
    #baseline="Posterior"
    user_specific=False
    output="./output"

    learnedThetas={}
    results={}
    for i in range(NUSERS):
        learnedThetas[str(i)]=[]
        results[str(i)]=[]

    import re
    #first write zero out effects
    output_dir = os.path.join(output, "Baseline-"+ str(baseline)+"_UserSpecific-"+str(False))
    bootstrapDirs=os.listdir(output_dir)
    bootstrapDirs=[us for us in bootstrapDirs if os.path.isdir(os.path.join(output_dir, us))]
    for bootstrapDir in bootstrapDirs:
    #for bootstrapDir in bootstrapDirs[0:4]:
        print(bootstrapDir)
        bootstrapDirs=os.path.join(output_dir, bootstrapDir)
        userDirs=os.listdir(bootstrapDirs)
        userDirs=[us for us in userDirs if os.path.isdir(os.path.join(bootstrapDirs, us))]
        for candidateUser in userDirs:
            #if str(user) in candidateUser[]:
            userDir=os.path.join(bootstrapDirs, candidateUser)
            userFile=os.listdir(userDir)
            userFile=[f for f in userFile if ".pkl" in f]
            if len(userFile)>0:
                values=re.split("[^0-9]", userFile[0])
                values = [i for i in values if i]
                userFile_path=os.path.join(userDir, userFile[0])
                result=pkl.load(open(userFile_path, "rb"))
                betavals=result['post_beta_mu'][449][-F_LEN:]
                learnedThetas[values[0]].append(betavals)#[experiment])
                results[values[0]].append(result)#[experiment])

    with open(output_dir+'/bootstrapPosteriorCoefs_'+str(baseline)+'.pkl', 'wb') as handle:
        pkl.dump(learnedThetas, handle, protocol=pkl.HIGHEST_PROTOCOL)
    with open(output_dir+'/bootstrapUserSortedResults_'+str(baseline)+'.pkl', 'wb') as handle:
        pkl.dump(results, handle, protocol=pkl.HIGHEST_PROTOCOL)
    print("wrote bootstrapPosterior and results on "+str(baseline))
    return

if __name__ == "__main__":
    main()
