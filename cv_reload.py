'''
Reload fitted weights from cv_train.py. Assess model 
performance at predicting functional connectivity by 
correlating testing dataset prediction vs empirical.

'''

import scipy, os
import numpy as np
from sklearn.model_selection import KFold
import gc   #help free up gpu memory from model

from train import make_model, load_data
from reload import plotout


#path to saved model weights 
save_path='predFC/hcp_cv10.mat' 

num_folds = 10  #total number of folds in cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=22)   #make sure random_state the same in cv_train.py to correctly identify test and training


def make_output(filename):
    input_data, output_data = load_data(data_path) 
    subs = scipy.io.loadmat(data_path)['subjects'][0]

    output = np.zeros(np.shape(output_data))
    input_ = np.zeros(np.shape(input_data))
    estimated = np.zeros(np.shape(output_data))
    subjects = np.zeros(np.shape(input_data)[0])

    cross_splits = list(kf.split(input_data))
    for fold in range(num_folds):

        train_indi, test_indi = cross_splits[fold]
        save_dir='final_weights/fold{0}.weights.h5'.format(fold)

        model = make_model(input_data.shape[1], input_data.shape[2])
        model.compile(optimizer='adam', loss='mse') 
        model.load_weights(save_dir)


        pred_data = model.predict(input_data[test_indi], batch_size=1) 
        for num,i in enumerate(test_indi): 
            input_[i,:] = input_data[i,:,:]
            output[i,:] = output_data[i,:]
            estimated[i,:] = pred_data[num,:] 
            subjects[i] = subs[i]

        gc.collect()
    scipy.io.savemat(filename, {'in': input_, 'out': output, 'predicted': estimated, 'subjects': subjects})




if __name__ == "__main__":

    data_path = 'hcp.mat' #data processed from the Human Connectome Project Young-Adult dataset

    if not os.path.exists(save_path):
        make_output(save_path)

    out_data = scipy.io.loadmat(save_path)
    input_data, output_data, pred_data = out_data['in'], out_data['out'], out_data['predicted'] 

    plotout(pred_data, output_data)

