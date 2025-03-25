'''
Reload fitted weights from train.py. Assess model 
performance at predicting functional connectivity by 
correlating prediction vs empirical.

'''


import numpy as np
import tensorflow as tf
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from train import make_model, load_data 


def plotout(pred_data, output_data):
    num_subjects = output_data.shape[0]

    corr_mat = []
    for i in range(num_subjects):
        corr, pval = pearsonr(pred_data[i], output_data[i])
        corr_mat.append(corr)

    plt.hist(corr_mat)
    plt.title('Transformer across {0} HCP individuals'.format(num_subjects))
    plt.xlabel('$r$(predicted FC $-$ empirical FC)')
    plt.ylabel('frequency')
    plt.axvline(np.mean(corr_mat), color='red', linestyle='--')

    print('mean ± std Pearson correlation: {0:.3f} ± {1:.3f}'.format(np.mean(corr_mat), np.std(corr_mat)))
    plt.show()




if __name__ == "__main__":
    
    data_path='hcp.mat' #data processed from the Human Connectome Project Young-Adult dataset
    input_data, output_data = load_data(data_path) 

    model = make_model(input_data.shape[1], input_data.shape[2])
    model.compile(optimizer='adam', loss='mse') 


#path to saved model weights 
    save_dir='final_weights/training_tests/epochs/final100.weights.h5'
    model.load_weights(save_dir)

    pred_data = model.predict(input_data, batch_size=1) #i guess could be larger than batch_size = 1 

    loss_fn = tf.keras.losses.MeanSquaredError()
    loss_value = loss_fn(output_data, pred_data).numpy()
    print('mean-squared error of fitted model is: {0:.3E}'.format(loss_value))

#    plotout(input_data[:,:,0], output_data)    #plot input vs output, last dim of input_data is for embed
    plotout(pred_data, output_data)
