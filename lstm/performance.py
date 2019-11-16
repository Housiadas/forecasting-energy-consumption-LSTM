from sklearn import metrics                               # for the check the error and accuracy of the model
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

def mean_absolute_percentage_error(y_true, y_pred): 
   
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def theil_u(yhat, y):
  
  len_yhat = len(yhat)
  sum1 = 0
  sum2 = 0
  
  for i in range(len_yhat-1):
    
    sum1 += ( (yhat[i+1] - y[i+1]) / y[i] )**2
    sum2 += ( (y[i+1] - y[i]) / y[i] )**2
    
  return sqrt(sum1/sum2)



vals = test_df.values.copy()
cols = [col for col in df.columns if col not in ['time']]
vals = scaler.inverse_transform(vals) 
print(f'vals: {vals.shape}')
print(test_x.shape)
print(test_y.shape)



test_x1 = test_x 
test_y1 = test_y
print(test_x1.shape)
print(test_y1.shape)


yhat1 = model.predict(test_x1)  # κάνω πρόβλεψη δίνοντας ως input το test data και όχι το train dataset..για να μην εχουμε εσφαλμένα συμπεράσματα
                       
  # λόγω overfit (αν συμβαίνει)
  
print("yooooooooooo")

###### invert scaling for forecast
predict1 = np.zeros(shape=(len(yhat1), 4) )       # εφτιαξα np.array με τον ιδιο αριθμό στηλων για να μπορει να γίνει το inverse στις τιμες του prediction
predict1[:,0] = yhat1[:,0]                         #βάζω τις τιμες που προεβλεψα 
inv_yhat1 = scaler.inverse_transform(predict1)    #κάνω το inverse_transform
inv_yhat1 = inv_yhat1[:,0]                        #κρατάω μόνο την στήλη που θέλω για να υπολογίσω το Root mean square error

###### invert scaling for actual                  παρόμοια βήματα
actual1 = np.zeros(shape=(len(test_y1), 4) )     
actual1[:,0] = test_y1
inv_y1 = scaler.inverse_transform(actual1)
inv_y1 = inv_y1[:,0]

##### calculate RMSE
rmse1 = np.sqrt(mean_squared_error(inv_y1, inv_yhat1))
print(f'Test RMSE: {round(rmse1,4)}')

##### calculate RMSE
mae = mean_absolute_error(inv_y1, inv_yhat1)
print(f'Test MAE: {round(mae,4)}')

##### calculate R^2 metric
r21 = r2_score(inv_y1, inv_yhat1)
print(f'Test R^2 score: {round(r21,4)}')


##### calculate MAPE ###################################################
mape = mean_absolute_percentage_error(inv_y, inv_yhat)
print(f'Test MAPE: {round(mape,4)} %\n')


plt.figure(figsize=(20,7))

plt.plot(inv_y1[500:900], color = "blue", label = 'true')
plt.plot(inv_yhat1[500:900], color = "green", label = 'prediction')
plt.xlabel('samples')
plt.ylabel('KWh')

plt.legend()
plt.show()