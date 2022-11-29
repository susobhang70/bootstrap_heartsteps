import subprocess
import numpy as np
import sys
import os
from datetime import date, datetime
from itertools import product

NUSERS = 91
BOOTSTRAPS = 2000

def main():
    idx = int(sys.argv[1])


    if idx < 20000:
        experiment = 1
        idx = idx-0

    # ---------- Experiment 1: Baseline - Prior ---------------------

    if experiment == 1:

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

        subprocess.run(f'python main.py -u {user} -b {boot_run} -s {idx}', shell=True) 

if __name__ == "__main__":
    main()