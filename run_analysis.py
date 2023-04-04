import subprocess
import numpy as np
import sys
import os
from datetime import date, datetime
from itertools import product

NUSERS = 91
F_KEYS=["intercept", "dosage", "engagement", "other_location", "variation"]

def main():
    idx = int(sys.argv[1])
    experiment = 4 #max array size AND index is 10k. we could split 10k into 4 for 2.5 k jobs each, leading to ~28 bootstrap resamples. this is rather small, so opt to fill in experiment

    # Get the user index
    baseline = F_KEYS[experiment]
    #baseline="Posterior"
    baseline="Zero"

    subprocess.run(f'python slidingWindow_bootstrap.py -b {idx} -bi {baseline}', shell=True) 

if __name__ == "__main__":
    main()
