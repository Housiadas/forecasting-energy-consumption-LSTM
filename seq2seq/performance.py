from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

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


def mean_absolute_percentage_error(y_true, y_pred): 
   
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# test_y_predicted = predict(test_x, encoder_predict_model, decoder_predict_model, 24)


# Select 10 random examples to plot
indices = np.random.choice(range(test_x.shape[0]), replace=False, size=30)

for index in indices:
  
    plot_prediction(test_x[index, :, :], test_y[index, :, :], test_y_predicted[index, :, :])
    
print(test_y_predicted.shape)

yhat = test_y_predicted 
test_y1 = test_y
test_x1 = test_x
    
shape0 = yhat.shape[0]
shape1 = yhat.shape[1]
yhat=yhat.reshape(-1)
print(yhat.shape)  
    
###### invert scaling for forecast
predict = np.zeros(shape=(len(yhat), 1))         # εφτιαξα np.array με τον ιδιο αριθμό στηλων για να μπορει να γίνει το inverse στις τιμες του prediction
predict[:,0] = yhat                              #βάζω τις τιμες που προεβλεψα 

inv_yhat = scaler.inverse_transform(predict)    #κάνω το inverse_transform
                                                #κρατάω μόνο την στήλη που θέλω για να υπολογίσω το Root mean square error
  
inv_yhat = np.reshape(inv_yhat, (shape0, shape1))
print(inv_yhat.shape)


###### invert scaling for actual                παρόμοια βήματα

shape0 = test_y1.shape[0]
shape1 = test_y1.shape[1]
test_y1 = test_y1.reshape(-1)
print(test_y1.shape)  

actual = np.zeros(shape=(len(test_y1), 1) )     
actual[:,0] = test_y1

inv_y = scaler.inverse_transform(actual)    #κάνω το inverse_transform
                                            #κρατάω μόνο την στήλη που θέλω για να υπολογίσω το Root mean square error

inv_y = np.reshape(inv_y, (shape0, shape1))
print(inv_y.shape)
    

####### invert scaling for test_x
shape0 = test_x1.shape[0]
shape1 = test_x1.shape[1]
test_x1 = test_x1.reshape(-1)
print(test_x1.shape)  

actual = np.zeros(shape=(len(test_x1), 1) )     
actual[:,0] = test_x1

inv_x = scaler.inverse_transform(actual)    #κάνω το inverse_transform
                                            #κρατάω μόνο την στήλη που θέλω για να υπολογίσω το Root mean square error

inv_x = np.reshape(inv_x, (shape0, shape1))
print(inv_x.shape)

print("24 hours similarities")
##### calculate RMSE  ##########################################################
rmse = np.sqrt(mean_squared_error(inv_y[:,:24], inv_yhat))
print(f'Test RMSE: {round(rmse,5)}')

##### calculate R^2   ##########################################################
r2 = r2_score(inv_y[:,:24], inv_yhat)
print(f'Test R^2: {round(r2,4)}')

##### calculate MAE   ##########################################################
mae = mean_absolute_error(inv_y[:,:24], inv_yhat)
print(f'Test MAE: {round(mae,4)}')

##### calculate MAPE ###################################################
mape = mean_absolute_percentage_error(inv_y[:,:24], inv_yhat)
print(f'Test MAPE: {round(mape,3)} %\n')


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("6 hours similarities")

##### calculate RMSE  ##########################################################
rmse = np.sqrt(mean_squared_error(inv_y[:,:6], inv_yhat[:,:6]))
print(f'Test RMSE: {round(rmse,5)}')

##### calculate R^2   ##########################################################
r2 = r2_score(inv_y[:,:6], inv_yhat[:,:6])
print(f'Test R^2: {round(r2,4)}')

##### calculate MAE   ##########################################################
mae = mean_absolute_error(inv_y[:,:6], inv_yhat[:,:6])
print(f'Test MAE: {round(mae,4)}')

##### calculate MAPE ###################################################
mape = mean_absolute_percentage_error(inv_y[:,:6], inv_yhat[:,:6])
print(f'Test MAPE: {round(mape,3)} %\n')

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("3 hours similarities")

##### calculate RMSE  ##########################################################
rmse = np.sqrt(mean_squared_error(inv_y[:,:3], inv_yhat[:,:3]))
print(f'Test RMSE: {round(rmse,5)}')

##### calculate R^2   ##########################################################
r2 = r2_score(inv_y[:,:3], inv_yhat[:,:3])
print(f'Test R^2: {round(r2,4)}')

##### calculate MAE   ##########################################################
mae = mean_absolute_error(inv_y[:,:3], inv_yhat[:,:3])
print(f'Test MAE: {round(mae,4)}')

##### calculate MAPE ###################################################
mape = mean_absolute_percentage_error(inv_y[:,:3], inv_yhat[:,:3])
print(f'Test MAPE: {round(mape,3)} %\n')


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("9 hours similarities")

##### calculate RMSE  ##########################################################
rmse = np.sqrt(mean_squared_error(inv_y[:,:9], inv_yhat[:,:9]))
print(f'Test RMSE: {round(rmse,5)}')

##### calculate R^2   ##########################################################
r2 = r2_score(inv_y[:,:9], inv_yhat[:,:9])
print(f'Test R^2: {round(r2,4)}')

##### calculate MAE   ##########################################################
mae = mean_absolute_error(inv_y[:,:9], inv_yhat[:,:9])
print(f'Test MAE: {round(mae,4)}')

##### calculate MAPE ###################################################
mape = mean_absolute_percentage_error(inv_y[:,:9], inv_yhat[:,:9])
print(f'Test MAPE: {round(mape,3)} %\n')