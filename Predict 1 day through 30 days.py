import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dropout, Dense, LSTM, GRU, SimpleRNN
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier


## ---------- (Optional step) ----------
## Call for GPU resources to grow on demand.
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)


## ---------- (1. Production data reading) ----------
production_row_data = pd.read_csv('all_data.csv')


## ---------- (2. Set up training set and test set) ----------
production_data = production_row_data.iloc[:].values[:, 0:6]
## The production data of the first 6000 days is used as the training set,
## Including Casing_pressure, Tubing_pressure, Gas_production, Water_production, Water_gas_ratio, PI.


## ---------- (3. Manually normalize) ----------
max_test = max(production_data[:, 2])
min_test = min(production_data[:, 2])
hand_sc = max_test - min_test
sc = MinMaxScaler(feature_range=(0, 1)) 
production_data_scaled = sc.fit_transform(production_data)
training_set_scaled = production_data_scaled[0:int(9660)]
test_set_scaled = production_data_scaled[
                  int(9660 * 0.75):9660]  

x_train = []
y_train = []
x_test = []
y_test = []


## ---------- (4. Process the training set) ----------
## Using a for loop, traverse the entire training set and extract the changes in oil pressure,
## casing pressure, production and other variables for 30 consecutive days as input features x_train. 
## The data from the 31st day is used as the label.
for i in range(30, len(training_set_scaled) - 30):
    x_train.append(training_set_scaled[i - 30:i, 0:6])
    y_train.append(training_set_scaled[i, 2])

x_train, y_train = np.array(x_train), np.array(y_train)
## features include casing pressure, oil level, pipeline pressure, gas production, water production and water-gas ratio.
x_train = np.reshape(x_train, (x_train.shape[0], 30, 6))
y_train = np.reshape(y_train, (y_train.shape[0], 1))
print("Shape of training set data")
print(x_train.shape)
print(y_train.shape)


## ---------- (5. Process the test set) ----------
for i in range(30, (len(test_set_scaled) - 30), 1):
    x_test.append(test_set_scaled[i - 30:i, 0:6])
    y_test.append(test_set_scaled[i, 2])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 30, 6))
y_test = np.reshape(y_test, (y_test.shape[0], 1))
print("Shape of test set data")
print(x_test.shape)
print(y_test.shape)


## ---------- (6. Configure neural network parameters) ----------
probability_of_Dropout = input("Please select the probability of Dropout (0, 0.5), input 0 or 0.5: ")
if probability_of_Dropout == "0.5":
    ## LSTM, Dropout: 0.5
    model = tf.keras.Sequential()
    model.add(LSTM(90, input_shape=(30, 6),activation="tanh",return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(90, input_shape=(30, 6),activation="tanh"))
    model.add(Dropout(0.5))
    model.add(Dense(1,activation="tanh"))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.00004),
              loss='mse')
    
if probability_of_Dropout == "0":
    ## LSTM, Dropout: 0
    model=tf.keras.Sequential()
    model.add(LSTM(90, input_shape=(30, 6),activation="tanh",return_sequences=True))
    model.add(LSTM(90, input_shape=(30, 6),activation="tanh"))
    model.add(Dense(1,activation="tanh"))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.00004),
                loss='mse')


## ---------- (7. Start train the model) ----------
history = model.fit(x_train, y_train, batch_size=32, epochs=170, validation_data=(x_test, y_test), verbose=2,
                    shuffle=False)
model.summary()
model.save("./my_model/LSTM_model_30_1.h5")


## ---------- (8. Result Visualization) ----------
## Visualization of loss and accuracy
loss = history.history['loss']  ## Training set loss
val_loss = history.history['val_loss']  ## Test set loss
plt.figure()
plt.subplot(211)
plt.plot(loss)
plt.title('loss')
plt.subplot(212)
plt.plot(val_loss)
plt.title('val_loss')
plt.show()


## ---------- (Optional step) ----------
## Grid search optimization algorithm
# def create_model():
#     ## Create model
#     model=tf.keras.Sequential()
#     model.add(LSTM(80, input_shape=(30,7),activation="tanh"))
#     model.add(Dense(30,activation="tanh"))
#     ## Compile model
#     model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(0.0001) , metrics=['accuracy'])
#     return model
# model = KerasClassifier(build_fn=create_model, verbose=2)
# batch_size = [10, 20, 40, 60, 80, 100]
# epochs = [250]
# param_grid = dict(batch_size=batch_size)
# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
# grid_result =grid.fit(x_train, y_train,epochs=250)
## summarize results
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# for params, mean_score, scores in grid_result.grid_scores_:
#     print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))


## ---------- (9. Comparison between predicted values and actual values) ----------
## Predicting the yield of the test set
predicted_production = model.predict(x_test)
predicted_production = (predicted_production * hand_sc) + min_test
real_production = (y_test * hand_sc) + min_test
predicted_production = np.reshape(predicted_production, (predicted_production.shape[0] * predicted_production.shape[1]))
real_production = np.reshape(real_production, (real_production.shape[0] * real_production.shape[1]))
plt.plot(real_production, color='red', label='Real Production')
plt.plot(predicted_production, color='blue', label='Predicted Production')
plt.title('Production Prediction')
plt.xlabel('Day')
plt.ylabel('Production')
plt.legend()
plt.show()


## ---------- (10. Error calculation) ----------
total_mse=0
total_error=0
total_production=0
for i in range(len(real_production)):
    total_production=total_production+real_production[i]
    total_mse=total_mse+(predicted_production[i]-real_production[i])*(predicted_production[i]-real_production[i])
    total_error=total_mse+(predicted_production[i]-real_production[i])
mse=total_mse/len(real_production)
error=total_error/len(real_production)
day_production=total_production/len(real_production)
print("The average daily output is: ",day_production)
print("The absolute error is: ",mse)
print("The relative error is: ",error)
print("The error percentage is: ",error/day_production)


## ---------- (11. Mean Absolute Error Algorithm) ----------
mse=0
r_2=0
tot=0
for i in range(len(real_production)):
    tot=real_production[i]+tot
average_production=tot/len(real_production)
tot_mse=0
tot_av=0
for i in range(len(real_production)):
    tot_mse=tot_mse+((real_production[i]-predicted_production[i])**2)
    tot_av=tot_av+((average_production-real_production[i])**2)
mse=tot_mse/len(real_production)
print("mse is")
print(mse)
r_2=1-tot_mse/tot_av
print("r2 is")
print(r_2)