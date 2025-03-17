'''
Runs cross-validation on train.py. Set the number of folds
in `num_folds`. Only runs one fold at a time, set by the 
`fold` variable.

'''

import time
from sklearn.model_selection import KFold

from train import make_model, load_data


####MANNUALLY ADJUST FOLD TO TRAIN ON HERE*********
fold = 1 

#path to save model weights 
save_dir='final_weights/fold{0}.weights.h5'.format(fold)

num_folds = 10  #total number of folds in cross-validation
kf = KFold(n_splits=num_folds, shuffle=True, random_state=22)   #make sure random_state the same in reload to correctly identify test and training


if __name__ == "__main__":
    start_time = time.perf_counter()

    data_path = 'hcp.mat' #data processed from the Human Connectome Project Young-Adult dataset
    input_data, output_data = load_data(data_path) 

#only include subjects in training subset
    indi = list(kf.split(input_data))[fold][0]
    input_data = input_data[indi]
    output_data = output_data[indi]


    model = make_model(input_data.shape[1], input_data.shape[2])
# Compile the model 
    model.compile(optimizer='adam', loss='mse') 

# Train the model
    epochs = 1000
    model.fit(input_data, output_data, epochs=epochs, batch_size=4)
    model.save_weights(save_dir)

    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time)/ 3600
    print(f"Elapsed time: {elapsed_time:.6f} hours")

