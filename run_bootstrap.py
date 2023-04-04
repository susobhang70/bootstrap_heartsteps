### feeder script for running population bootstrap ###

import subprocess
from subprocess import Popen
import numpy as np
import sys
import os
from datetime import date, datetime
from itertools import product

NUSERS = 91
F_KEYS=["intercept", "dosage", "engagement", "other_location", "variation"]

def main():
    offset=0
    offset=10000
    offset=20000
    offset=30000
    #offset=40000
    #offset=50000

    idx = int(sys.argv[1])+offset
    #experiment = 4 
    #baseline = F_KEYS[experiment]
    baseline = "Zero"

    boot_run = idx // NUSERS #which bootstrap pop to run in
    # random subset w/ replacement
    np.random.seed(boot_run)
    bootstrapped_users = [i for i in range(NUSERS)]
    bootstrapped_users = np.random.choice(bootstrapped_users, len(bootstrapped_users))

    # Get the user index in the bootstrapped pop, and real user to run
    useridx = idx % (NUSERS) 
    user = bootstrapped_users[useridx]

    subprocess.run(f'python main.py -u {user} -b {boot_run} -s {idx} -userBIdx {useridx} -bi {baseline}', shell=True) 
    print("boot run "+str(boot_run)+" for user "+str(useridx)+ ". idx is "+str(idx)+ ". Baseline is "+baseline)

if __name__ == "__main__":
    main()
