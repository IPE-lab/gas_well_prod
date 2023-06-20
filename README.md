# **Smart gas well**
## Paper Support
- Original information: Intelligent production optimization method for low productivity shale gas well 
- Recruitment Journal: Petroleum Exploration and Development (JCR Q1) 
- Original DOI: https://doi.org/10.11698/PED.20210781
## **1. Introduction to the dataset file :**
- `all_data.csv`, this file contains six columns of data, namely: Casing_pressure, Tubing_pressure, Gas_production, Water_production, Water_gas_ratio, PI.
- `Heatmap of the dataset.ipynb`, this file presents a heatmap visualization of the `all_data.csv` file.



## **2. Code file interpretation :**
- `Predict 1 day through 10 days.py`, to predict the production for the next day based on the known production of 10 days, this file uses multiple neural network methods for yield prediction, with the following parameters: 
  - `GRU, Dropout: 0.2`
  - `LSTM, Dropout: 0.5`
  - `RNN, Dropout: 0.2`
  


- `Predict 1 day through 30 days.py`, to predict the production for the next day based on the known production of 30 days, this file uses multiple neural network methods for yield prediction, with the following parameters: 
  - `LSTM, Dropout: 0.5`
  - `LSTM, Dropout: 0`



- `Predict 5 day through 30 days.py`, to predict the production in the next 5 days based on the known production of 30 days, this file uses multiple neural network methods for yield prediction, with the following parameters: 
  - `LSTM, Dropout: 0.5`
  - `LSTM, Dropout: 0`



- `Predict 10 day through 30 days.py`, to predict the production in the next 10 days based on the known production of 30 days, this file uses multiple neural network methods for yield prediction, with the following parameters: 
  - `LSTM, Dropout: 0.5`
  - `LSTM, Dropout: 0`