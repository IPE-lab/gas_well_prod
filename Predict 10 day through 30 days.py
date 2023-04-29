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
    y_train.append(training_set_scaled[i:i+10, 2])

x_train, y_train = np.array(x_train), np.array(y_train)
## features include casing pressure, oil level, pipeline pressure, gas production, water production and water-gas ratio.
x_train = np.reshape(x_train, (x_train.shape[0], 30, 6))
y_train = np.reshape(y_train, (y_train.shape[0], 10))
print("Shape of training set data")
print(x_train.shape)
print(y_train.shape)


## ---------- (5. Process the test set) ----------
for i in range(30, (len(test_set_scaled) - 30), 10):
    x_test.append(test_set_scaled[i - 30:i, 0:6])
    y_test.append(test_set_scaled[i:i+10, 2])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], 30, 6))
y_test = np.reshape(y_test, (y_test.shape[0], 10))
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
    model.add(Dense(10,activation="tanh"))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.00004),
              loss='mse')
    
if probability_of_Dropout == "0":
    ## LSTM, Dropout: 0
    model=tf.keras.Sequential()
    model.add(LSTM(90, input_shape=(30, 6),activation="tanh",return_sequences=True))
    model.add(LSTM(90, input_shape=(30, 6),activation="tanh"))
    model.add(Dense(10,activation="tanh"))
    model.compile(optimizer=tf.keras.optimizers.Adam(0.00004),
                loss='mse')


## ---------- (7. Start train the model) ----------
history = model.fit(x_train, y_train, batch_size=32, epochs=170, validation_data=(x_test, y_test), verbose=2,
                    shuffle=False)
model.summary()
model.save("./my_model/LSTM_model_30_10.h5")


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
#设计平均绝对误差算法
mse_1=0
mse_2=0
mse_3=0
mse_4=0
mse_5=0
mse_6=0
mse_7=0
mse_8=0
mse_9=0
mse_10=0
r_2_1=0
r_2_2=0
r_2_3=0
r_2_4=0
r_2_5=0
r_2_6=0
r_2_7=0
r_2_8=0
r_2_9=0
r_2_10=0
tot=0
for i in range(len(real_production)):
    tot=real_production[i]+tot
average_production=tot/len(real_production)
tot_av_1=0
tot_av_2=0
tot_av_3=0
tot_av_4=0
tot_av_5=0
tot_av_6=0
tot_av_7=0
tot_av_8=0
tot_av_9=0
tot_av_10=0
for i in range(0,len(real_production),10):
    mse_1=mse_1+((real_production[i]-predicted_production[i])**2)
    tot_av_1=tot_av_1+((average_production-real_production[i])**2)
for i in range(1,len(real_production),10):
    mse_2=mse_2+((real_production[i]-predicted_production[i])**2)
    tot_av_2=tot_av_2+((average_production-real_production[i])**2)
for i in range(2,len(real_production),10):
    mse_3=mse_3+((real_production[i]-predicted_production[i])**2)
    tot_av_3=tot_av_3+((average_production-real_production[i])**2)
for i in range(3,len(real_production),10):
    mse_4=mse_4+((real_production[i]-predicted_production[i])**2)
    tot_av_4=tot_av_4+((average_production-real_production[i])**2)
for i in range(4,len(real_production),10):
    mse_5=mse_5+((real_production[i]-predicted_production[i])**2)
    tot_av_5=tot_av_5+((average_production-real_production[i])**2)
for i in range(5,len(real_production),10):
    mse_6=mse_6+((real_production[i]-predicted_production[i])**2)
    tot_av_6=tot_av_6+((average_production-real_production[i])**2)
for i in range(6,len(real_production),10):
    mse_7=mse_7+((real_production[i]-predicted_production[i])**2)
    tot_av_7=tot_av_7+((average_production-real_production[i])**2)
for i in range(7,len(real_production),10):
    mse_8=mse_8+((real_production[i]-predicted_production[i])**2)
    tot_av_8=tot_av_8+((average_production-real_production[i])**2)
for i in range(8,len(real_production),10):
    mse_9=mse_9+((real_production[i]-predicted_production[i])**2)
    tot_av_9=tot_av_9+((average_production-real_production[i])**2)
for i in range(9,len(real_production),10):
    mse_10=mse_10+((real_production[i]-predicted_production[i])**2)
    tot_av_10=tot_av_10+((average_production-real_production[i])**2)

mse_1_v=mse_1/len(real_production)
mse_2_v=mse_2/len(real_production)
mse_3_v=mse_3/len(real_production)
mse_4_v=mse_4/len(real_production)
mse_5_v=mse_5/len(real_production)
mse_6_v=mse_6/len(real_production)
mse_7_v=mse_7/len(real_production)
mse_8_v=mse_8/len(real_production)
mse_9_v=mse_9/len(real_production)
mse_10_v=mse_10/len(real_production)
print("mse is")
print(mse_1_v,mse_2_v,mse_3_v,mse_4_v,mse_5_v,mse_6_v,mse_7_v,mse_8_v,mse_9_v,mse_10_v)

tot_av=0
for i in range(len(real_production)):
    tot_av=tot_av+((average_production-real_production[i])**2)
tot_av=tot_av/10

r_2_1=1-mse_1/tot_av
r_2_2=1-mse_2/tot_av
r_2_3=1-mse_3/tot_av
r_2_4=1-mse_4/tot_av
r_2_5=1-mse_5/tot_av
r_2_6=1-mse_6/tot_av
r_2_7=1-mse_7/tot_av
r_2_8=1-mse_8/tot_av
r_2_9=1-mse_9/tot_av
r_2_10=1-mse_10/tot_av
print("r2 is")
print(r_2_1,r_2_2,r_2_3,r_2_4,r_2_5,r_2_6,r_2_7,r_2_8,r_2_9,r_2_10)

zong_mse=0
zong_av=0
for i in range(len(real_production)):
    zong_mse=zong_mse+((real_production[i]-predicted_production[i])**2)
    zong_av=zong_av+((average_production-real_production[i])**2)
print("The overall R2 value is")
print(1-(zong_mse/zong_av))