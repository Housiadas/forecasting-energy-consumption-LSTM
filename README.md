# forecasting-energy-consumption-LSTM
Development of a machine learning application for IoT platform to predict energy consumption in smart building environment in real time.

### Data Acquisition
The dataset, that was used for the development of the machine learning models, was taken from:
https://www.kaggle.com/uciml/electric-power-consumption-data-set

### Data Preprocessing
* Handling missing values.
* Data Smoothing (exponential smoothing).
* Handling outliers (we detected them using standard deviation).
* Data normalization (scaling the values between [0,1]).
* Data resampling ().

### Splitting the Dataset.
* Training set.
* Validation set.
* Test set.


### First Approach (LSTM).
We made use of Long Short-Term Memory (LSTM) cells to create a sequential model using the Keras API.


### Second Approach (Seq2Seq).
We implemented a Sequence-to-Sequence model utilizing the Keras' functional API. 
