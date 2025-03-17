'''
This code applies the transformer architecture to predict
functional connectivity between two brain regions from 
the entirety of the structural conenctivity matrix for a 
given human subject in the HCP Young Adult dataset.

-Structural connectivity is taken from dMRI data and 
corresponds to the number white matter tracts between
brain regions. Values are resampled to have mean = 0.5 
and std = 0.1.
-Functional connectivity is taken from fMRI data and
corresponds to the Pearson correlation between concatenated
time-series data across 4 seperate runs. Values bounded 
by -1 and 1.


The transformer architecture (MultiHeadSelfAttention, 
TransformerBlock, TransformerEncoder) was taken from a 
Coursera course titled "Deep Learning with Keras and 
Tensorflow" by IBM (Module 3). Parameters were tweaked such 
that memory requirements are under 16GB (GPU limit).

'''

import time
import tensorflow as tf
import scipy
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout 



#path to save model and training results
save_dir='final_weights/final.weights.h5'    #must end in .weights.h5 or else throws an error

#use the mimimum required memory
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def load_data(filename):
    input_data = scipy.io.loadmat(filename)['sc']
    input_data = input_data.reshape(input_data.shape[0], input_data.shape[1], 1) #need the extra 1 dimension for embed dim

    output_data = scipy.io.loadmat(filename)['fc']
    return input_data, output_data


class MultiHeadSelfAttention(Layer): 
    def __init__(self, embed_dim, num_heads=8): 
        super(MultiHeadSelfAttention, self).__init__() 
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.projection_dim = embed_dim // num_heads 
        self.query_dense = Dense(embed_dim) 
        self.key_dense = Dense(embed_dim) 
        self.value_dense = Dense(embed_dim) 
        self.combine_heads = Dense(embed_dim) 
 

    def attention(self, query, key, value): 
        score = tf.matmul(query, key, transpose_b=True) 
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32) 
        scaled_score = score / tf.math.sqrt(dim_key) 
        weights = tf.nn.softmax(scaled_score, axis=-1) 
        output = tf.matmul(weights, value) 
        return output, weights 


    def split_heads(self, x, batch_size): 
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim)) 
        return tf.transpose(x, perm=[0, 2, 1, 3]) 


    def call(self, inputs): 
        batch_size = tf.shape(inputs)[0] 
        query = self.query_dense(inputs) 
        key = self.key_dense(inputs) 
        value = self.value_dense(inputs) 
        query = self.split_heads(query, batch_size) 
        key = self.split_heads(key, batch_size) 
        value = self.split_heads(value, batch_size) 
        attention, _ = self.attention(query, key, value) 
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) 
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim)) 
        output = self.combine_heads(concat_attention) 
        return output 

class TransformerBlock(Layer): 
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1): 
        super(TransformerBlock, self).__init__() 
        self.att = MultiHeadSelfAttention(embed_dim, num_heads) 
        self.ffn = tf.keras.Sequential([ 
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim), 
        ]) 

        self.layernorm1 = LayerNormalization(epsilon=1e-6) 
        self.layernorm2 = LayerNormalization(epsilon=1e-6) 
        self.dropout1 = Dropout(rate) 
        self.dropout2 = Dropout(rate) 
 

    def call(self, inputs, training): 
        attn_output = self.att(inputs) 
        attn_output = self.dropout1(attn_output, training=training) 
        out1 = self.layernorm1(inputs + attn_output) 
        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training) 
        return self.layernorm2(out1 + ffn_output) 

class TransformerEncoder(Layer): 
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1): 
        super(TransformerEncoder, self).__init__() 
        self.num_layers = num_layers 
        self.embed_dim = embed_dim 
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.enc_layers = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)] 
        self.dropout = Dropout(rate) 

    def call(self, inputs, training=False): 
        x = inputs 
        for i in range(self.num_layers): 
            x = self.enc_layers[i](x, training=training) 
        return x 

def default_params():
#embed_dim = 128 #memory issues
    embed_dim = 32

    num_heads = 8 
#num_heads = 4   #no diff to memory 

    ff_dim = 512 
#ff_dim = 256 #no diff to memory

#num_layers = 4     #memory issues
    num_layers = 2 
    return embed_dim, num_heads, ff_dim, num_layers

def make_model(conn_dim, extra_dim):
    embed_dim, num_heads, ff_dim, num_layers = default_params()

    # Define the Transformer Encoder 
    transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim) 

# Build the model 
    inputs = tf.keras.Input(shape=(conn_dim, extra_dim)) 

# Project the inputs to the embed_dim 
    x = tf.keras.layers.Dense(embed_dim)(inputs) 
    encoder_outputs = transformer_encoder(x) 
    flatten = tf.keras.layers.Flatten()(encoder_outputs) 
    outputs = tf.keras.layers.Dense(conn_dim)(flatten) 
    model = tf.keras.Model(inputs, outputs) 

    return model


if __name__ == "__main__":
    start_time = time.perf_counter()

    data_path = 'hcp.mat' #data processed from the Human Connectome Project Young-Adult dataset
    input_data, output_data = load_data(data_path) 

    model = make_model(input_data.shape[1], input_data.shape[2])

# Compile the model 
    model.compile(optimizer='adam', loss='mse') 

# Summary of the model 
#   print(model.summary()) 

# Train the model
    epochs = 1000
    model.fit(input_data, output_data, epochs=epochs, batch_size=4)    #memory exhausted with batch_size = 8
    model.save_weights(save_dir)

    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time)/ 3600
    print(f"Elapsed time: {elapsed_time:.6f} hours")

