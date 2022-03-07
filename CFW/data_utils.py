import pandas as pd
import numpy as np

# replace missing values of nominal feature with their modes
def fill_mode(data, col_idx):
    if len(col_idx)>0:
        for col in col_idx:
            data.iloc[:,col].fillna(data.iloc[:,col].mode()[0], inplace=True)
    return data

# replace missing values of numerical feature with their mean
def fill_mean(data, col_idx):
    if len(col_idx)>0:
        data.iloc[:,col_idx] = data.iloc[:,col_idx].astype('float32')
        for col in col_idx:
            data.iloc[:,col].fillna((data.iloc[:,col].mean()), inplace=True)
    return data

''' Define a loading scheme for each of the tested datasets '''

def load_data(path, y_col, mode_idxs=[], mean_idxs=[], sep=',', remove_idxs=[]):
    data = pd.read_csv(path, header=None, sep=sep)
    data = data.replace('?', np.nan)
    
    # remove unsued features
    if len(remove_idxs)>0:
        valid_features = [i for i in range(len(data.columns)) if i not in remove_idxs]
        data = data.iloc[:,valid_features]
        
    # split x and y columns
    if y_col==-1: y_col = len(data.columns)-1
    y = data.iloc[:,y_col]
    x = data.iloc[:,[i for i in range(len(data.columns)) if i!=y_col]]
    
    # get numerical and nominal values
        # all options
    if mode_idxs=='all':
        mode_idxs = np.arange(len(x.columns))
        assert len(mean_idxs)==0, 'For all option, the other idexes should have size 0'
    elif mean_idxs=='all':
        mean_idxs = np.arange(len(x.columns))
        assert len(mode_idxs)==0, 'For all option, the other idexes should have size 0'
        # if only one of the idexes is defined, infer the others
    elif len(mode_idxs)>0 and len(mean_idxs)==0:
        mean_idxs = [i for i in range(len(x.columns)) if i not in mode_idxs]
    elif len(mean_idxs)>0 and len(mode_idxs)==0:
        mode_idxs = [i for i in range(len(x.columns)) if i not in mean_idxs]
        
    x = fill_mode(x, mode_idxs)
    x = fill_mean(x, mean_idxs)
    
    return np.array(x), np.array(y), mean_idxs