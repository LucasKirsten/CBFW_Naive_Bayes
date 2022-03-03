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

datasets = {
    'audiology': lambda:load_data('./data/audiology.standardized.data', -1, mode_idxs='all'),
    'balance-scale': lambda:load_data('./data/balance-scale.data', 0, mode_idxs='all'),
    'breast-cancer': lambda:load_data('./data/breast-cancer.data', 0, mode_idxs='all'),
    'breast-cancer-w': lambda:load_data('./data/breast-cancer-wisconsin.data', -1, mode_idxs='all'),
    'colic': lambda:load_data('./data/horse-colic.data', -1, sep=' ', mode_idxs=[0,1,5,6,7,8,9,10,11,12,13,15,16,19,21,22,23,24,25], remove_idxs=[2]),
    'credit-a': lambda:load_data('./data/credit-a.data', -1, mean_idxs=[1,2,7,10,13,14]),
    'credit-g': lambda:load_data('./data/credit-g.data', -1, sep=' ', mean_idxs=[1,4,7,10,12,15,17]),
    'diabetes': lambda:load_data('./data/diabetes.data', -1, mode_idxs='all'),
    'glass': lambda:load_data('./data/glass.data', -1, mean_idxs='all'),
    'heart-c': lambda:load_data('./data/heart-c.data', -1, mean_idxs=[0,3,4,7,9,11]),
    'heart-h': lambda:load_data('./data/heart-h.data', -1, mean_idxs=[0,3,4,7,9,11]),
    'heart-statlog': lambda:load_data('./data/heart-statlog.data', -1, sep=' ', mean_idxs=[0,3,4,7,9,11]),
    'hepatitis': lambda:load_data('./data/hepatitis.data', 0, mean_idxs=[0,13,14,15,16,17]),
    'ionosphere': lambda:load_data('./data/ionosphere.data', -1, mean_idxs='all'),
    'iris': lambda:load_data('./data/iris.data', -1, mean_idxs='all'),
    'kr-vs-kp': lambda:load_data('./data/kr-vs-kp.data', -1, mode_idxs='all'),
    'letter': lambda:load_data('./data/letter.data', 0, mode_idxs='all'),
    'lymph': lambda:load_data('./data/lymphography.data', 0, mean_idxs=[9,10]),
    'mushroom': lambda:load_data('./data/mushroom.data', 0, mode_idxs='all'),
    'primary-tumor': lambda:load_data('./data/primary-tumor.data', 0, mode_idxs='all'),
    'segmentation': lambda:load_data('./data/segmentation.data', 0, mean_idxs='all'),
    'sonar': lambda:load_data('./data/sonar.data', -1, mean_idxs='all'),
    'waveform-5000': lambda:load_data('./data/waveform-5000.data', -1, mean_idxs='all'),
    'zoo': lambda:load_data('./data/zoo.data', -1, mode_idxs='all', remove_idxs=[0]),
}