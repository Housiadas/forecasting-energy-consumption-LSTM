import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import random
from math import sqrt
from collections import deque

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics                               
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit


import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, CuDNNLSTM, LSTM, Activation, Bidirectional
from tensorflow.keras.layers import Flatten, TimeDistributed

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping



########### constants prameters  ###################################################################################################

SEQ_LEN = 100                    # how long of a preceeding sequence to collect for RNN (in 1hour minutes)
FUTURE_PERIOD_PREDICT = 6        # how far into the future are we trying to predict? (in minutes)
EPOCHS = 30                      # how many passes through our data
BATCH_SIZE = 64                  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.



############# functions  ###########################################################################################################################

def mean_absolute_percentage_error(y_true, y_pred): 
#     y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def theil_u(yhat, y):
  
  len_yhat = len(yhat)
  sum1 = 0
  sum2 = 0
  
  for i in range(len_yhat-1):
    
    sum1 += ( (yhat[i+1] - y[i+1]) / y[i] )**2
    sum2 += ( (y[i+1] - y[i]) / y[i] )**2
    
  return sqrt(sum1/sum2)


def plot_history(loss, val_loss, epochs):
  
  plt.figure(figsize=[15,8])
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(range(epochs), loss, label='Train Error')
  plt.plot(range(epochs), val_loss, label = 'Val Error')
  plt.ylim([0.05,0.15])
  plt.legend()
  plt.show()


def preprocess_df(df, shuffle=False):
  
    sequential_data = []                    # this is a list that will CONTAIN the sequences
    prev_days = deque(maxlen = SEQ_LEN)     # These will be our actual sequences.
                                            # They are made with deque, which keeps the maximum length
                                            # by popping out older values as new ones come in
    for i in df.values:                                             # iterate over the values
        prev_days.append([n for n in i[:-1]])                       # store all but the label
        if len(prev_days) == SEQ_LEN:                               # make sure we have 30 sequences!
            sequential_data.append([np.array(prev_days), i[-1]])    # append those bad boys!
            
            
    if(shuffle == True):
      random.shuffle(sequential_data)                                 # shuffle for good measure.
    

    X = []
    y = []

    for seq, target in sequential_data:  # going over our new sequential data
        X.append(seq)                    # X is the sequences (features)
        y.append(target)                 # y is the targets/labels

    return np.array(X), np.array(y)  # return X and y...and make X a numpy array!
  
  
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
  
  return test_df, df


def get_model():
  
  model = Sequential()

  model.add(CuDNNLSTM(36, stateful=True, return_sequences=True, 
                 batch_input_shape=(BATCH_SIZE, train_x.shape[1], train_x.shape[2])))
  model.add(Activation('relu'))
  
  model.add(CuDNNLSTM(36, stateful=True, return_sequences=False))
  model.add(Activation('relu'))
  
  model.add(Dense(1))
                  
  """.....compile model....."""
  
  model.compile(loss = 'mae',
              optimizer = 'adam',
              metrics = ['mse', 'mae'])
  
  return model

####################################################################################################################################################################        
#### load dataset 
  
  
df = pd.read_csv('kaggle_data_1h.csv', sep=',', infer_datetime_format=True, low_memory=False, index_col='time', encoding='utf-8')

# for c in df.columns:
#   print(c)

df = df.drop(['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], axis=1)


df.index = pd.to_datetime(df.index)
df.sort_index(inplace = True)
df.dropna(inplace=True)

#### 1sample/1hour gia ta sequences #############
df = df.resample('1h').mean()
df.dropna(inplace=True)


print(df.head())


#### Data Preprocessing
##############################################################################################################################################################
#### Moving Average filtering... smoothing our data ##########################################################################################################

######## Smothing our data at first........
df = df.ewm(alpha=0.2).mean()
df.dropna(inplace=True)

# #Dropping the outlier rows with standard deviation
factor = 3
upper_lim = df['Global_active_power'].mean () + df['Global_active_power'].std () * factor
lower_lim = df['Global_active_power'].mean () - df['Global_active_power'].std () * factor
df = df[(df['Global_active_power'] < upper_lim) & (df['Global_active_power'] > lower_lim)]

####### fill values summer of 2008........

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



####################################################################################################################################################################
### Create supervised learning


df['future'] = df["Global_active_power"].shift(-FUTURE_PERIOD_PREDICT)
df.dropna(inplace=True)

print(df[['Global_active_power','future']].head(10))

print(f'Yooooooo:{df.shape}')



##############################################################################################################################################################
##### cross validation for time series..... ##################################################################################################################

count = 0

#### Evaluation metrics.....
evaluate_loss = []
evaluate_mse = []
evaluate_mlog = []
evaluate_rmse = []
evaluate_r2 = []
evaluate_mape = []


val_loss = []
loss = []


tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(df):

  #### μόνο τελευταία επανάληψη για το walk forward validation.....
  if (count >= 2 ):

    df2 = df.copy()

    print(f'df:{df2[:len(train_index)].shape}  test_df:{df2[len(train_index):len(train_index)+len(test_index)].shape}')

    test_df = df2[len(train_index):len(train_index)+len(test_index)]
    df2 = df2[:len(train_index)]
    valid_df, df2 = split_data(df2, 0.1)

    print(f'df_shape{df2.shape}')
    print(f'test_shape{test_df.shape}')
    print(f'Valid_shape{valid_df.shape}')

    df2.dropna(inplace=True)
    valid_df.dropna(inplace=True)
    test_df.dropna(inplace=True)

  #### scale our data............ #########################################################################################################

    scaler = MinMaxScaler(feature_range=(0, 1))

    cols = [col for col in df.columns if col not in ['time']]

    df2[cols] = scaler.fit_transform(df2[cols])
    df2.dropna(inplace=True)                                      

    test_df[cols] = scaler.transform(test_df[cols])
    test_df.dropna(inplace=True) 

    valid_df[cols] = scaler.transform(valid_df[cols])
    valid_df.dropna(inplace=True) 


    train_x, train_y = preprocess_df(df2, shuffle=True)
    valid_x, valid_y = preprocess_df(valid_df, shuffle=False)
    test_x, test_y = preprocess_df(test_df, shuffle=False)

    train_len = train_x.shape[0]
    valid_len = valid_x.shape[0]
    test_len = test_x.shape[0]
    print(f'train_len before:{train_len} test_len:{test_len}')

###### For batch size we want modulo na einai mhden(pollaplasia toy batch size) #####################################################################

    train_len = train_length(train_len, BATCH_SIZE)
    valid_len = train_length(valid_len, BATCH_SIZE)
    test_len = train_length(test_len, BATCH_SIZE)

    print(f'train_len after modulo:{train_len} test_len:{test_len}')

    train_x = train_x[:train_len]
    train_y = train_y[:train_len]

    valid_x = valid_x[:valid_len]
    valid_y = valid_y[:valid_len]

    test_x = test_x[:test_len]
    test_y = test_y[:test_len]

    print(f'Train_x shape:{train_x.shape} Train_y shape:{train_y.shape}')
    print(f'valid_x shape:{valid_x.shape} valid_y shape:{valid_y.shape}')
    print(f'Test_x shape:{test_x.shape} Test_y shape:{test_y.shape} \n')


    """.....The model....."""

    model = get_model()
    model.summary()

    """.....train model....."""

    ###Για stateful LSTM ######

    for i in range(EPOCHS):
      print(f'Split: {count} Epochs: {i}/{EPOCHS}')

      history = model.fit(
        train_x, train_y,
        batch_size = BATCH_SIZE,
        epochs = epochs,
        validation_data = (valid_x, valid_y),
      )
      
      
      val_loss.append(history.history['val_loss'])
      loss.append(history.history['loss'])
      model.reset_states()


    """ Evaluate model.......  """ 

    score = model.evaluate(test_x, test_y, verbose=0)
    print('Test loss:', score[0])
    print('Test MSE:', score[1])
    print('Test MAE:', score[2])

    evaluate_loss.append(score[0])
    evaluate_mse.append(score[1])
    
    ######## plot loss & val_loss #######################################################

    plot_history(loss, val_loss, EPOCHS) 

   
    #### make & evaluate our prediction ########################################################################

    yhat = model.predict(test_x)                    # κάνω πρόβλεψη δίνοντας ως input το test data και όχι το train dataset
                                 

    ###### invert scaling for forecast
    predict = np.zeros(shape=(len(yhat), 4) )       # εφτιαξα np.array με τον ιδιο αριθμό στηλων για να μπορει να γίνει το inverse στις τιμες του prediction
    predict[:,0] = yhat[:,0]                        #βάζω τις τιμες που προεβλεψα 

    inv_yhat = scaler.inverse_transform(predict)    #κάνω το inverse_transform
    inv_yhat = inv_yhat[:,0]                        #κρατάω μόνο την στήλη που θέλω για να υπολογίσω το Root mean square error


    ###### invert scaling for actual                παρόμοια βήματα
    actual = np.zeros(shape=(len(test_y), 4) )     
    actual[:,0] = test_y

    inv_y = scaler.inverse_transform(actual)
    inv_y = inv_y[:,0]

    ##### calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print(f'Test RMSE: {rmse}')
    evaluate_rmse.append(rmse)

    ##### calculate R^2 metric
    r2 = r2_score(inv_y, inv_yhat)
    print(f'Test R^2 score: {r2}')
    evaluate_r2.append(r2)

    
  count += 1

#### END OF LOOP #################################

### για να πέρναμε το μέσο όρο των performance metrics για το walk forward validation

avg_loss = np.average(evaluate_loss)
avg_mse = np.average(evaluate_mse)
avg_rmse = np.average(evaluate_rmse)
avg_r2 = np.average(evaluate_r2)
avg_theil = np.average(evaluate_theil)
# avg_mape = np.average(evaluate_mape)

print('\n')
print(f'Avg Loss:{avg_loss}')
print(f'Avg MSE:{avg_mse}')
print(f'Avg RMSE:{avg_rmse}')
print(f'Avg R2:{avg_r2}')


### Save the last model to a HDF5 file #########################################################

model.save('my_model.h5')


######## gia na metatrepsoyme sto keras modelo se tensorflow wste na xrhsimopoihthei gia to tensorflow serving ############################
# Recreate the exact same model purely from the file

new_model = keras.models.load_model('my_model.h5')

#### Export the model to a SavedModel....this model is for the tensorflow serving.......with trained weights and graphs
keras.experimental.export_saved_model(new_model, 'SavedModel')