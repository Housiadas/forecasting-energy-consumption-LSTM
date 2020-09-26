import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from math import sqrt
from collections import deque

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics                               # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, LSTM, Activation, RepeatVector, CuDNNGRU, LSTMCell
from tensorflow.keras.layers import Flatten, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

######## Hyperparameters and model configuration  ##################################################################################################################

keras.backend.clear_session()

layers = [30, 30]     # Number of hidden neuros in each layer of the encoder and decoder

learning_rate = 0.001
decay = 0             # Learning rate decay
# optimiser = keras.optimizers.Adam(lr=learning_rate, decay=0) 

optimiser = keras.optimizers.Adam() 
num_input_features = 1              # The dimensionality of the input at each time step. In this case a 1D signal.
num_output_features = 1             # The dimensionality of the output at each time step. In this case a 1D signal.

# There is no reason for the input sequence to be of same dimension as the ouput sequence.
# For instance, using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.

loss = "mse"                        # Other loss functions are possible, see Keras documentation.

# Regularisation isn't really needed for this application
lambda_regulariser = 0.000001                   # Will not be used if regulariser is None
regulariser = None                              # Possible regulariser: keras.regularizers.l2(lambda_regulariser)

batch_size = 80
epochs = 50
input_sequence_length = 72                      # Length of the sequence used by the encoder
target_sequence_length = 72                     # Length of the sequence predicted by the decoder
num_steps_to_predict = 12                       # Length to use when testing the model

#################################################################################################################################################################### 

def plot_history(history):
  
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure(figsize=(17, 8))
  plt.xlabel('Epoch')
  plt.ylabel('mean_squared_error')
  
  plt.plot(hist['epoch'], hist['loss'], label='Train Error')
  plt.plot(hist['epoch'], hist['val_loss'], label = 'Val Error')
  plt.ylim([0, 0.05])
  plt.legend()
  plt.show()

def preprocess_df(df, input_sequence_length, target_sequence_length, shuffle=False):
  
    SEQ_LEN = input_sequence_length + target_sequence_length
    
    sequential_data = []                    # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen = SEQ_LEN)     # These will be our actual sequences.
                                            # They are made with deque, which keeps the maximum length
                                            # by popping out older values as new ones come in
    for i in df:                                             # iterate over the values
        prev_days.append(i)                                         
        if len(prev_days) == SEQ_LEN:                               # make sure we have 30 sequences!
            sequential_data.append(np.array(prev_days))            
            
            
    if(shuffle == True):
      random.shuffle(sequential_data)                                 # shuffle for good measure.
    
    X = []
    for seq in sequential_data:  
        X.append(seq)     
    
    X = np.array(X)
    
    encoder_input = X[:, :input_sequence_length, :]
    decoder_output = X[:, input_sequence_length:, :]
            
    # The output must be ([encoder_input, decoder_input], [decoder_output])
    decoder_input = np.zeros((decoder_output.shape[0], decoder_output.shape[1], 1))
        
    return encoder_input, decoder_input, decoder_output

def plot_prediction(x, y_true, y_pred):
    """Plots the predictions.
    
    Arguments
    ---------
    x: Input sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_true: True output sequence of shape (input_sequence_length,
        dimension_of_signal)
    y_pred: Predicted output sequence (input_sequence_length,
        dimension_of_signal)
    """
    plt.figure(figsize=(15, 3))

    output_dim = x.shape[-1]
    for j in range(output_dim):
        past = x[:, j] 
        true = y_true[:, j]
        pred = y_pred[:, j]

        label1 = "Seen (past) values" if j==0 else "_nolegend_"
        label2 = "True future values" if j==0 else "_nolegend_"
        label3 = "Predictions" if j==0 else "_nolegend_"

        plt.plot(range(len(past)), past, "o--b",
                 label=label1)
        plt.plot(range(len(past),
                 len(true)+len(past)), true, "x--b", label=label2)
        plt.plot(range(len(past), len(pred)+len(past)), pred, "o--y",
                 label=label3)
    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()


def train_length(length, batch_size):
  
  length_values = []
  for x in range( int(length)-100, int(length) ):
    modulo = x % batch_size
    if(modulo == 0 ):
      length_values.append(x)
  
  return ( max(length_values) )

def split_data(df, percent):
  
  length = len(df)
  test_index = int(length*percent)   
  train_index = length - test_index
  test_df = df[train_index:]                       
  df = df[:train_index] 
  
  return test_df , df

def predict(x, encoder_predict_model, decoder_predict_model, num_steps_to_predict):
    """Predict time series with encoder-decoder.
    
    Uses the encoder and decoder models previously trained to predict the next
    num_steps_to_predict values of the time series.
    
    Arguments
    ---------
    x: input time series of shape (batch_size, input_sequence_length, input_dimension).
    encoder_predict_model: The Keras encoder model.
    decoder_predict_model: The Keras decoder model.
    num_steps_to_predict: The number of steps in the future to predict
    
    Returns
    -------
    y_predicted: output time series for shape (batch_size, target_sequence_length,
        ouput_dimension)
    """
    y_predicted = []

    # Encode the values as a state vector
    states = encoder_predict_model.predict(x)

    # The states must be a list
    if not isinstance(states, list):
        states = [states]

    # Generate first value of the decoder input sequence
    decoder_input = np.zeros((x.shape[0], 1, 1))

    for _ in range(num_steps_to_predict):
        outputs_and_states = decoder_predict_model.predict([decoder_input] + states, batch_size=batch_size)
        output = outputs_and_states[0]
        states = outputs_and_states[1:]

        # add predicted value
        y_predicted.append(output)

    return np.concatenate(y_predicted, axis=1)

########## Create encoder   #######################################
# Define an input sequence.

encoder_inputs = keras.layers.Input(shape=(None, num_input_features))

# Create a list of RNN Cells, these are then concatenated into a single layer
# with the RNN layer.

encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser
                                             ))

encoder = keras.layers.RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)

# Discard encoder outputs and only keep the states.
# The outputs are of no interest to us, the encoder's
# job is to create a state describing the input sequence.
encoder_states = encoder_outputs_and_states[1:]

########## Create dencoder   ####################################################################################################
# The decoder input will be set to zero (see random_sine function of the utils module).
# Do not worry about the input size being 1, I will explain that in the next cell.

decoder_inputs = keras.layers.Input(shape=(None, 1))
decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(keras.layers.GRUCell(hidden_neurons,
                                              kernel_regularizer=regulariser,
                                              recurrent_regularizer=regulariser,
                                              bias_regularizer=regulariser
                                             ))

decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

# Set the initial state of the decoder to be the ouput state of the encoder.
# This is the fundamental part of the encoder-decoder.
decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

# Only select the output of the decoder (not the states)
decoder_outputs = decoder_outputs_and_states[0]

# Apply a dense layer with linear activation to set output to correct dimension
# and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
decoder_dense = keras.layers.Dense(num_output_features,
#                                    activation='linear',
                                   kernel_regularizer=regulariser,
                                   bias_regularizer=regulariser)

decoder_outputs = decoder_dense(decoder_outputs)
####################################################################################################################################################################        
### load dataset  
  
df = pd.read_csv('kaggle_data_1h.csv', sep=',', infer_datetime_format=True, low_memory=False, index_col='time', encoding='utf-8')


### global_active_power only (data features selection)
df = df.drop(['Voltage', 'Global_intensity', 'Global_reactive_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1)
df = df.round(5)
df.index = pd.to_datetime(df.index)
df.sort_index(inplace=True)
df.dropna(inplace=True)

##### 1sample/1hour gia ta sequences #####################################################################################
# df = df.resample('1h').mean()
# df.dropna(inplace=True)

##### Data Preprocessing #########################################################################################################################################
##### Smothing our data at first........
df = df.ewm(alpha=0.15).mean()
df.dropna(inplace=True)

##### Dropping the outlier rows with standard deviation
factor = 3
upper_lim = df['Global_active_power'].mean () + df['Global_active_power'].std () * factor
lower_lim = df['Global_active_power'].mean () - df['Global_active_power'].std () * factor
df = df[(df['Global_active_power'] < upper_lim) & (df['Global_active_power'] > lower_lim)]

##### fill values summer of 2008........
times = pd.to_datetime(df.index)
times = times.strftime("%Y-%m-%d").tolist()
temp = []
for index,value in enumerate(times):
  if(value > '2007-08-08' and value <'2007-09-01'):
     
    temp.append(float(df['Global_active_power'][index]))
    
temp = pd.Series(temp)
temp = temp.ewm(alpha=0.15).mean()
temp = temp.values

k=0
for index,value in enumerate(times):
  if(value > '2008-08-08' and value <'2008-09-01'):
    df['Global_active_power'].iloc[index] = temp[k]
    k += 1

print(f'df_shape arxika: {df.shape}\n')

#####################################################################################################################################################################


train_len = int(len(df) * 0.8)
test_df = df[train_len:]
df = df[:train_len]
print(f'df_shape{df.shape}')
print(f'test_shape{test_df.shape}')
df.dropna(inplace=True)
test_df.dropna(inplace=True)

###### scale our data............ ###################################################################################################################################

df = df.values
test_df = test_df.values
scaler = MinMaxScaler(feature_range=(0, 1))
df = scaler.fit_transform(np.float64(df))
test_df = scaler.transform(np.float64(test_df))

####################################################################################################################################################################        

encoder_input_data, decoder_input_data, decoder_target_data = preprocess_df(df, input_sequence_length, target_sequence_length, shuffle=True)
print(encoder_input_data.shape)

##### Create a model using the functional API provided by Keras.
##### The functional API is great, it gives an amazing amount of freedom in architecture of your NN.
##### A read worth your time: https://keras.io/getting-started/functional-api-guide/ 
##### This creates the model

model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer=optimiser, loss=loss)
model.summary()

"""  Train model   """ 

history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=0.1,
#                     callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=0, mode='auto')]
                   )

"""  Evaluate model   """ 
#### test data ##################################

test_x, decoder_input_data, test_y = preprocess_df(test_df, input_sequence_length, target_sequence_length, shuffle=False)

score = model.evaluate([test_x, decoder_input_data], test_y, batch_size=batch_size)
print('Test loss evaluation:', score)

#   print('Test MSE:', score[1])
#   print('Test MAE:', score[2])
#   evaluate_loss.append(score[0])
#   evaluate_mse.append(score[1])


######## plot loss & val_loss #######################
plot_history(history) 
    
######################################################################################################################################
### ftiaxnoyme neo encoder montelo gia ta predictions based on the trained model....we used the trained parameters/internal states....
### ayto to kanoyme gia na problepsoyme diaforetika sequences apo oti ekpaideusame to montelo..
###..ekpaideytike gia 2.5 meres gia na problepsei tho epomeno 24hours......

encoder_predict_model = keras.models.Model(encoder_inputs, encoder_states)


decoder_states_inputs = []

# Read layers backwards to fit the format of initial_state
# For some reason, the states of the model are order backwards (state of the first layer at the end of the list)
# If instead of a GRU you were using an LSTM Cell, you would have to append two Input tensors since the LSTM has 2 states.

for hidden_neurons in layers[::-1]:
    # One state for GRU
    decoder_states_inputs.append(keras.layers.Input(shape=(hidden_neurons,)))

decoder_outputs_and_states = decoder(decoder_inputs, initial_state=decoder_states_inputs)

decoder_outputs = decoder_outputs_and_states[0]
decoder_states = decoder_outputs_and_states[1:]
decoder_outputs = decoder_dense(decoder_outputs)

### ftiaxnoyme neo decoder montelo gia ta predictions based on the trained model....we used the trained parameters/internal states....

decoder_predict_model = keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
  
#### predictions ##########################################################################################  
test_y_predicted = predict(test_x, encoder_predict_model, decoder_predict_model, 24)


##### Save the last model to a HDF5 file #########################################################
encoder_predict_model.save('encoder.h5')
decoder_predict_model.save('decoder.h5')

encoder_predict_model.save('encoder.h5')
decoder_predict_model.save('decoder.h5')

