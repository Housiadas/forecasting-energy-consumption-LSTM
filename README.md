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

* #### Results of prediction for the next day (1h to 24h).

  1. Case
  ![alt text](https://raw.githubusercontent.com/Housiadas/forecasting-energy-consumption-LSTM/master/results/seq2seq/pred1.png)
  2. Case
  ![alt text](https://raw.githubusercontent.com/Housiadas/forecasting-energy-consumption-LSTM/master/results/seq2seq/pred2.png)
  3. Case
  ![alt text](https://raw.githubusercontent.com/Housiadas/forecasting-energy-consumption-LSTM/master/results/seq2seq/pred3.png)
  4. Case
  ![alt text](https://raw.githubusercontent.com/Housiadas/forecasting-energy-consumption-LSTM/master/results/seq2seq/pred4.png)
  5. Case
  ![alt text](https://raw.githubusercontent.com/Housiadas/forecasting-energy-consumption-LSTM/master/results/seq2seq/pred5.png)

We took random prediction cases from the whole test set to examine the performance of our model visually.

___
### License

Copyright © 2019 Christos Chousiadas

This repository is under the MIT License.
