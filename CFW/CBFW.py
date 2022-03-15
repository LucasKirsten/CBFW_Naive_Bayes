__author__ = 'Lucas N. Kirsten'

import numpy as np
import pickle as pkl
from scipy.special import softmax

from time import time

# load a saved model
def load_model(path_model):
    with open(path_model, 'rb') as handle:
        model = pkl.load(handle)
    return model

# save a model
def save_model(path_save, model):
    with open(path_save, 'wb') as handle:
        pkl.dump(model, handle, protocol=pkl.HIGHEST_PROTOCOL)

# class for the classifier based on SKLEARN standarts
class CBFW(object):
    def __init__(self):
        self.map_features = []
        self.map_labels = {}
        self.priors = []
        self.features_posteriors = []
        self.NIAA = []
        self.NIAC = []
        self.W = []
        
    # fit data to the model
    def fit(self, x, y):
        features = np.array(x)
        labels = np.array(y)
        
        assert features.shape[-1]>1, 'Number of feature for CBFW should be larger than 1!'
            
        # mapping between features values
        map_features = []
        for i in range(features.shape[-1]):
            feat = features[:, i]
            map_features.append({ft:j for j,ft in enumerate(np.unique(feat))})
        Nf = len(map_features)

        # mapping between labels values
        map_labels = {lb:l for l,lb in enumerate(np.unique(labels))}
        Nl = len(map_labels)
        
        # compute priors
        priors = [(len(labels[labels==lb])+1/Nl)/(len(labels)+1) \
                      for lb in map_labels.keys()]
        
        # compute features posteriors and feature-class correlation (relevance)
        features_posteriors = []
        IAC = np.zeros((Nf))
        for i in range(features.shape[-1]):
            feats = features[:, i]
            
            # iterate over feature values and classes
            feat_post = np.zeros((len(map_features[i]), Nl))
            for val,vi in map_features[i].items():
                for lb,li in map_labels.items():
                    # feature posterior
                    p_val_lb = np.count_nonzero(np.logical_and(labels==lb, feats==val))
                    p_lb = np.count_nonzero([labels==lb])
                    feat_post[vi][li] = (p_val_lb+1/len(feats))/(p_lb+1)
                    
                    # feature-class correlation
                    p_ac = np.logical_and(labels==lb, feats==val)
                    p_ac = (np.count_nonzero(p_ac)+1/len(feats))/(len(p_ac)+1)
                    p_a  = (len(feats[feats==val])+1/len(feats))/(len(feats)+1)
                    p_c  = priors[li]
                    
                    IAC[i] += p_ac*np.log(p_ac/(p_a*p_c))

            features_posteriors.append(feat_post)
        
        # normalize relevance
        NIAC = IAC/np.mean(IAC)
        self.NIAC = NIAC

        # compute feature-feature correlation (redundancy)
        aij = [] # all possible feature combinations
        for i in range(Nf):
            for j in range(i+1,Nf):
                aij.append([i,j])
        
        # iterate over feature combinations
        IAA = np.zeros(Nf)
        for i,j in aij:
            # get feature values
            valsi = features[:,i]
            valsj = features[:,j]
            
            # iterate over possible feature values
            for vali in map_features[i]:
                for valj in map_features[j]:
                    # compute probabilities
                    p_aiaj = np.logical_and(valsi==vali, valsj==valj)
                    p_aiaj = np.count_nonzero(p_aiaj)/len(p_aiaj)
                    
                    v_ai = valsi==vali
                    p_ai = np.count_nonzero(v_ai)/len(v_ai)
                    
                    v_aj = valsj==valj
                    p_aj = np.count_nonzero(v_aj)/len(v_aj)
                    
                    iaa = p_aiaj*np.log(p_aiaj/(p_ai*p_aj)+1e-6)
                    IAA[i] += iaa
                    IAA[j] += iaa
        
        # normalize redundancy
        NIAA = IAA/np.mean(IAA)
        self.NIAA = NIAA
        
        # mutual redundancy
        D = np.zeros(Nf)
        for i in range(Nf):
            D[i] = NIAC[i] - np.mean(NIAA[i])

        # logistic sigmoid over mutual redundancy
        W = 1/(1+np.exp(-D))
        
        # store class attributes
        self.map_features = map_features
        self.map_labels = map_labels
        self.labels_map = {v:k for k,v in self.map_labels.items()}
        self.priors = priors
        self.features_posteriors = features_posteriors
        self.W = W
        
    # make predictions on new data
    def predict(self, x, proba=False, naive=False):
        # if proba=True returns the probabilities
        # if naive=True perfoms predicitions using vanilla NB
        
        x = np.array(x)
        assert len(x.shape)==2, 'Invalid input shape!'
        
        # if using standart Naive Bayes, set weights to 1
        if naive:
            W = np.ones_like(self.W)
        else:
            W = self.W

        preds = []
        for xx in x:
            # map the input features
            input_feats = [self.map_features[i][inp] if inp in self.map_features[i] else -1 \
                               for i,inp in enumerate(xx)]

            # compute CBFW
            pred_class = np.copy(self.priors)
            for li in range(len(self.map_labels)):
                for fi,feat in enumerate(input_feats):
                    if feat==-1:
                        pred_class[li] *= (1/len(self.features_posteriors[fi][feat]))**W[fi]
                    else:
                        pred_class[li] *= self.features_posteriors[fi][feat][li]**W[fi]
            
            # if to store probabilities or the predicted class
            if proba:
                preds.append(softmax(pred_class))
            else:
                preds.append(self.labels_map[np.argmax(pred_class)])
            
        return np.array(preds)