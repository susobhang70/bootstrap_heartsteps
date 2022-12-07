import pandas as pd
import numpy as np
import scipy.stats as st
import pickle as pkl
import rpy2.robjects as robjects
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import math
import argparse


PKL_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/original_result_91_priorPaper.pkl"
PRIOR_DATA_PATH = "/Users/raphaelkim/Dropbox (Harvard University)/HeartStepsV2V3/Raphael/bandit-prior.RData"
F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_KEYS2 = ["Intercept", "Dosage", "Engagement", "Other_Location", "Variation"]

F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation", "temperature", "logpresteps", "sqrt_totalsteps"]
G_LEN = len(G_KEYS)

NDAYS=90
NTIMES=5

def load_priors():
    '''Load priors from RData file'''
    robjects.r['load'](PRIOR_DATA_PATH)
    priors = robjects.r['bandit.prior']
    alpha_pmean = np.array(priors.rx2("mu1"))
    alpha_psd = np.array(priors.rx2("Sigma1"))
    beta_pmean = np.array(priors.rx2("mu2"))
    beta_psd = np.array(priors.rx2("Sigma2"))
    sigma = float(priors.rx2("sigma")[0])

    prior_sigma = linalg.block_diag(alpha_psd, beta_psd, beta_psd)
    prior_mu = np.concatenate([alpha_pmean, beta_pmean, beta_pmean])
    print(prior_mu)
    print(prior_sigma)
    print(ttt)

    return prior_sigma, prior_mu, sigma

def getUpperAndLowers(data, prior_mu, prior_sigma, F_index, z=1.96):
    data_dict={'upper': [], 'lower':[], 'increment':[], 'mean': [], 'labels':[]}
    FinalTime=NDAYS*NTIMES-1
    for i in range(len(data)):
        mean=data[i]['post_mu'][FinalTime][-F_LEN:][F_index]
        var=data[i]['post_sigma'][FinalTime][-F_LEN:, -F_LEN:]
        std=math.sqrt(var[F_index, F_index])

        data_dict['upper'].append(mean+z*std)
        data_dict['lower'].append(mean-z*std)
        data_dict['increment'].append(z*std)
        data_dict['mean'].append(mean)
        data_dict['labels'].append(i)
    
    mean=prior_mu[-F_LEN:][F_index]
    std=math.sqrt(prior_sigma[-F_LEN:, -F_LEN:][F_index,F_index])

    data_dict['upper'].append(mean+z*std)
    data_dict['lower'].append(mean-z*std)
    data_dict['increment'].append(z*std)
    data_dict['mean'].append(mean)
    data_dict['labels'].append("Prior")
    return data_dict

def plot_CIs(data, prior_mu, prior_sigma, F_index):
    uppersAndLowers=getUpperAndLowers(data, prior_mu, prior_sigma, F_index)
    
    plt.clf()
    figure(figsize=(8, 12), dpi=80)

    z=1.96
    color='#2187bb'
    priorColor='#f44336'
    horizontal_line_width=0.25

    df=pd.DataFrame(uppersAndLowers)
    #print(df.head())
    df=df.sort_values(by=['mean'])
    remap={}
    for i in range(df.shape[0]):
        remap[df.index.values[i]]=i
    df=df.rename(index = remap)
    #print(df.head())

    #print(ggg)
    somethingHit=False
    for i in range(df.shape[0]):
        left = i - horizontal_line_width / 2
        top = df['upper'][i]
        right = i + horizontal_line_width / 2
        bottom = df['lower'][i]
        label=df['labels'][i]
        mean=df['mean'][i]
        
        color_i=color
        if label=="Prior":
            color_i=priorColor
            label=i
            plt.plot([top, bottom], [label, label], color=color_i)
            plt.plot([top, top], [left, right], color=color_i)
            plt.plot([bottom, bottom],[left, right], color=color_i)
            plt.plot(mean, i, 'o', color=color_i, label="Prior")

        else:
            label=i
            if somethingHit:
                plt.plot([top, bottom], [label, label], color=color_i)
                plt.plot([top, top], [left, right], color=color_i)
                plt.plot([bottom, bottom],[left, right], color=color_i)
                plt.plot(mean, i, 'o', color=color_i)
            else:
                plt.plot([top, bottom], [label, label], color=color_i)
                plt.plot([top, top], [left, right], color=color_i)
                plt.plot([bottom, bottom],[left, right], color=color_i)
                plt.plot(mean, i, 'o', color=color_i,label="Posterior")
                somethingHit=True

    plt.axvline(x = 0, ymin = 0, ymax = 92, color ='red', linestyle ="--", linewidth=1)
    plt.legend(loc="upper right")
    plt.title('95% Credible intervals at the End of the Study for '+F_KEYS2[F_index])
    plt.xlabel('Mean Coefficient for '+ F_KEYS2[F_index] +' ± Credible Interval')
    plt.ylabel('User')

    plt.savefig("./output/plots/"+"CI_Plots_"+F_KEYS2[F_index]+'.png', format="png")

with open(PKL_DATA_PATH, 'rb') as handle:
    original_result=pkl.load(handle)#, handle, protocol=pkl.HIGHEST_PROTOCOL)

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--F_index", type=int, default=0, required=False, help="Index of F advantage")
args = parser.parse_args()
prior_sigma,prior_mu,sigma=load_priors()
plot_CIs(original_result, prior_mu, prior_sigma, args.F_index)
