'''
Compare results from transformer architecture to deep 
learning approach published by Sarwar et al. 2021.

Note that I ran Sarwar's publicly available code on the 
exact same `hcp.mat` input dataset and provided the output
in `Sarwar2021_hcp_cv10.mat`.

'''

import scipy
from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})  #uniformly increase fontsize of figure


def plotout(df):
    ax = sns.violinplot(data=df, x='subset', y='corr', hue='machine learning method', inner='quart', split=True)
    sns.move_legend(ax, "lower left")
    plt.xlabel('')
    plt.ylabel('$r$(predicted FC $-$ empirical FC)')

    plt.tight_layout()
    plt.show()


def organize_data(num_subs, output_data, pred_data, compare_pred_data):
    outputs = []
    for i in range(num_subs):
        corr, pval = pearsonr(pred_data[i], output_data[i])
        outputs.append([corr, '{0} HCP individuals'.format(num_subs), 'transformer'])

        comp_corr, comp_pval = pearsonr(compare_pred_data[i], output_data[i])
        outputs.append([comp_corr, '{0} HCP individuals'.format(num_subs), 'DNN (Sarwar 2021)'])

    df = pd.DataFrame(outputs, columns = ['corr', 'subset', 'machine learning method'])
    return df


if __name__ == "__main__":

    save_path='hcp_cv10.mat' 
    out_data = scipy.io.loadmat(save_path)
    input_data, output_data, pred_data = out_data['in'], out_data['out'], out_data['predicted'] 

    compare_path = 'Sarwar2021_hcp_cv10.mat' 
    compare_out_data = scipy.io.loadmat(compare_path)
    compare_pred_data = compare_out_data['predicted'] #input_data, output_data are the same
    num_subs = pred_data.shape[0]

    df = organize_data(num_subs, output_data, pred_data, compare_pred_data)
    plotout(df)
