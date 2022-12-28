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
    experiment = 2 #max array size AND index is 10k. we could split 10k into 4 for 2.5 k jobs each, leading to ~28 bootstrap resamples. this is rather small, so opt to fill in experiment

    # Define the parameters
    boot_run = idx // NUSERS

    # Set the seed
    np.random.seed(boot_run)

    # Bootstrap user indices
    # This will be consistent across runs as we set the seed
    bootstrapped_users = np.random.choice(range(NUSERS), NUSERS)

    # Get the user index
    useridx = idx % NUSERS
    user = bootstrapped_users[useridx]
    baseline = F_KEYS[experiment]
    subprocess.run(f'python main.py -u {user} -b {boot_run} -s {idx} -bi {baseline}', shell=True) 

if __name__ == "__main__":
    main()
