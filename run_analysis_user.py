import subprocess
import numpy as np
import sys
import os
from datetime import date, datetime
from itertools import product

NUSERS = 91
F_KEYS=["intercept", "dosage", "engagement", "other_location", "variation"]

def main():
    user=int(sys.argv[1])
    experiment = int(sys.argv[2])
    #BRUN=1000
    BRUN=500
    #experiment = 2 #max array size AND index is 10k. we could split 10k into 4 for 2.5 k jobs each, leading to ~28 bootstrap resamples. this is rather small, so opt to fill in experiment

    users=[]
    # Bootstrap user indices
    # This will be consistent across runs as we set the seed
    # also remove the user who is 'interesting'
    #if experiment == 2: #engagement
    #    users=[38, 62, 84, 89, 3, 32, 81]
    #if experiment == 3: #other_location
    #    users= [1, 6, 7, 13, 24, 25, 45, 62]
    #if experiment == 4: #variation
    #    #delta1=.5,delta2=.25
    #    #users=[4, 5, 13, 25, 28, 32, 33, 37, 38, 39, 40, 45, 46, 48, 49, 60, 62, 65, 72, 76, 81, 82, 83, 84, 85, 87, 90]
    #    #delta1=.75,delta2=.4
    #    users=[3, 4, 5, 6, 9, 12, 13, 14, 18, 24, 25, 27, 28, 32, 33, 34, 37, 38, 40, 45, 46, 50, 54, 56, 60, 62, 64, 65, 66, 67, 68, 71, 72, 74, 76, 78, 79, 82, 83, 84, 85, 90]

#   # idx=idx % BRUN

    # Get the user index
    #baseline = F_KEYS[experiment]
    baseline="Zero"

    #idx does not matter here
    idx=user
    subprocess.run(f'python slidingWindow_bootstrap.py -b {idx} -bi {baseline} -pec True -u {user}', shell=True) 
    print("User "+str(user)+" baseline is "+baseline)

if __name__ == "__main__":
    main()
