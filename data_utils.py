import pandas as pd
import numpy as np

# replace missing values of nominal feature with their modes
def fill_mode(data, col_idx):
    for col in col_idx:
        data.iloc[:,col].fillna(data.iloc[:,col].mode()[0], inplace=True)
    return data

# replace missing values of numerical feature with their mean
def fill_mean(data, col_idx):
    for col in col_idx:
        data.iloc[:,col].fillna((data.iloc[:,col].mean()), inplace=True)
    return data

''' Define a loading scheme for each of the tested datasets '''

def load_zoo(path='./data/zoo.data'):
    data = pd.read_csv(path, header=None)
    data = data.iloc[:,1:]
    data = fill_mode(data, list(range(len(data.columns))))
    x, y = data.iloc[:,:-1], data.iloc[:,-1]
    
    return np.array(x), np.array(y), []

def load_waveform_5000(path='./data/waveform-5000.csv'):
    data = pd.read_csv(path)
    x, y = data.iloc[:,:-1], data.iloc[:,-1]
    x = fill_mean(x, list(range(len(x.columns))))
    
    return np.array(x), np.array(y), list(range(len(x.columns)))

datasets = {
    'zoo': load_zoo,
    'waveform-5000': load_waveform_5000
}