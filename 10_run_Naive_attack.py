#import pandas library
import sys
from time import time
from traceback import print_tb

from numpy import argmax
import numpy as np
import random

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, plot_roc_curve, roc_curve, auc

import torch

import data
from utils import fix_randomness
from models import Covid19_MainTaskModel, Adults_MainTaskModel, Fivethirtyeight_MainTaskModel, GSS_MainTaskModel

import yaml

fix_randomness()

# read config file
with open("./config/config.yaml", "r") as ymlfile:
    try:
        cfg = yaml.safe_load(ymlfile)
    except yaml.YAMLError as exc:
        print(exc)

# Step 1: Load data from .csv file
problem = cfg["problem"]
datapath = cfg["dataset"][problem]["path_to_data"]

if problem == 'covid19':
    X, y = data.covid19_load_data(datapath)
elif problem == 'adults':
    X, y, scaler = data.adults_load_data(datapath)
elif problem == 'fivethirtyeight':
    X, y, scaler = data.fivethirtyeight_load_data(datapath)
elif problem == 'gss':
    X, y, scaler = data.gss_load_data(datapath)
else:
    print("The problem was not supported!!!")
    sys.exit()

# Load attack feature configuration file
path_to_cfg_target_feature = "./config/target_attributes_" + problem + ".yaml"
with open(path_to_cfg_target_feature, "r") as ymlfile:
    try:
        cfg_target_attribute = yaml.safe_load(ymlfile)
    except yaml.YAMLError as exc:
        print(exc)

# Step 2: Define target attribute index
target_attribute_index = cfg_target_attribute["target_attribute"]["target_attribute_id"]
possible_attack_features = cfg_target_attribute["target_attribute"]["target_attribute_ids"]

###########################################################
# Step 3: Prepare data for training and test attack model #
###########################################################

# Step 3.1: Shuffle data
from sklearn.utils import shuffle
sklearn_random_state = cfg["random"]["random_state_sklearn"]
X, y = shuffle(X, y, random_state=sklearn_random_state)

# Step 3.2: get remain dataset to use in attack (for both train and test)
propotion_to_train_attack = cfg["attack_model"][problem]["propotion_to_train_attack_model"]
X_attack = X[int(X.shape[0]*(1.0-propotion_to_train_attack)):,:]
if cfg_target_attribute['target_attribute']['isBinary'] == True:
    y_attack = X_attack[:,target_attribute_index]
else:
    X_attack_before_normalized = scaler.inverse_transform(X_attack).astype(np.int64)
    y_attack = X_attack_before_normalized[:,target_attribute_index]

# Dataset to train attack model 
propotion_to_train_attack = cfg["attack_model"]["train_size"]
X_train_attack = X_attack[:int(X_attack.shape[0]*propotion_to_train_attack),:]
y_train_attack = y_attack[:int(X_attack.shape[0]*propotion_to_train_attack)]

# test attack model on remain dataset (e.g: 0.2)
X_test_attack = X_attack[int(X_attack.shape[0]*propotion_to_train_attack):,:]
y_test_attack = y_attack[int(X_attack.shape[0]*propotion_to_train_attack):]

X_train_attack = torch.from_numpy(X_train_attack).type(torch.FloatTensor)
X_test_attack = torch.from_numpy(X_test_attack).type(torch.FloatTensor)

#########################################################
# use X_train_attack... to calculate prior maginal of sensitive attribute
########################################################
prior = [0]*len(possible_attack_features)
count = 0

# NOTE: This attack assumes Adversary knows prior maginal of target attribute
prior = [0]*len(possible_attack_features)
count = 0
if cfg_target_attribute['target_attribute']['isBinary'] == False:
    for x in X_test_attack:
        x_before_nomarlized = scaler.inverse_transform(x.reshape(1, -1)).astype(np.int64)[0]
        for v in possible_attack_features:
            if x_before_nomarlized[target_attribute_index]==v:
                prior[v]=prior[v]+1
        count+=1
    prior = np.array(prior)
    prior = prior/count
else:
    for x in X_attack:
        for v in possible_attack_features:
            if x[target_attribute_index]==v:
                prior[v]=prior[v]+1
        count+=1
    prior = np.array(prior)
    prior = prior/count

major_label = argmax(prior)
print("Prior maginal: ", prior)
print("Major target attribute: ", major_label)

# Naive Attack
x_sensitive_attribute_true=[]
x_sensitive_attribute_pred=[]

# NOTE: This attack assumes Adversary can access to confusion matrix
cms = np.loadtxt('./models/' + problem + '_confusion_matrix.txt', dtype=int)
with torch.no_grad():
    for i, x in enumerate(X_test_attack):
        x_before_nomalized = scaler.inverse_transform(x.reshape(1, -1)).astype(np.int64)[0]
        x_sensitive_attribute_true.append(x_before_nomalized[target_attribute_index])
        x_sensitive_attribute_pred.append(major_label)
        
print("===========Test model (with DP, with noise)===========")

testAcc = accuracy_score(x_sensitive_attribute_true, x_sensitive_attribute_pred)
print("+ Accuracy = ", testAcc)

if cfg_target_attribute['target_attribute']['isBinary'] == True:
    testF1 = f1_score(x_sensitive_attribute_true, x_sensitive_attribute_pred)
    testPrecision = precision_score(x_sensitive_attribute_true, x_sensitive_attribute_pred)
    testRecall = recall_score(x_sensitive_attribute_true, x_sensitive_attribute_pred)
    print("+ F1 score = ", testF1)
    print("+ Precision = ", testPrecision)
    print("+ Recall = ", testRecall)
    
cm = confusion_matrix(x_sensitive_attribute_true, x_sensitive_attribute_pred)
print("+ Confusion matrix:\n", cm)
