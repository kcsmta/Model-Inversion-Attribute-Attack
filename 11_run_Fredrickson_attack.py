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
y_main_task = y[int(X.shape[0]*(1.0-propotion_to_train_attack)):]

# test attack model on remain dataset
propotion_to_train_attack = cfg["attack_model"]["train_size"]
X_test_attack = X_attack[int(X_attack.shape[0]*propotion_to_train_attack):,:]
y_test_main_task = y_main_task[int(X_attack.shape[0]*propotion_to_train_attack):]

X_test_attack = torch.from_numpy(X_test_attack).type(torch.FloatTensor)

#########################################################
# use X_train_attack... to calculate prior maginal of sensitive attribute
########################################################

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

print("Prior maginal: ", prior)
print("Major target attribute: ", argmax(prior))

main_task_model = torch.load(cfg["target_model"][problem]["path_to_target_model"])

# create train dataset for attack model
main_task_model_output = []
main_task_model_output_with_noise = []

# Fredrickson's Attack
x_sensitive_attribute_true=[]
x_sensitive_attribute_pred=[]

# NOTE: This attack assumes Adversary can access to confusion matrix
cms = np.loadtxt('./models/' + problem + '_confusion_matrix.txt', dtype=int)

with torch.no_grad():
    for i, x in enumerate(X_test_attack):
        if cfg_target_attribute['target_attribute']['isBinary'] == False:
            x_before_nomarlized = scaler.inverse_transform(x.reshape(1, -1)).astype(np.int64)[0]
            x_before_nomarlized_copy = x_before_nomarlized # copy to loop all possible values of target attribute
            x_sensitive_attribute_true.append(x_before_nomarlized[target_attribute_index])
        else:
            x_copy = x # copy to loop all possible values of target attribute
            x_sensitive_attribute_true.append(int(x[target_attribute_index].item()))

        maximum = -1

        for v in possible_attack_features:
            # replace sensitive attribute by v
            if cfg_target_attribute['target_attribute']['isBinary'] == False:
                x_before_nomarlized_copy[target_attribute_index] = v
                x_after_normalized = torch.from_numpy(scaler.transform(x_before_nomarlized_copy.reshape(1, -1))[0]).to(torch.float)
                x_v = x_after_normalized
            else:
                x_copy[target_attribute_index] = v
                x_v = x_copy

            x_v = torch.unsqueeze(x_v, 0)
            target_outputs = main_task_model(x_v)

            y_pred = np.argmax(target_outputs, axis=1)
            y_true = int(y_test_main_task[i])
            C_ytrue_ypred = cms[y_true][y_pred]/np.sum(cms, axis=1)[y_true]

            p_v = prior[v] # xác suất possible sensitive attribute = v, gọi là maginal prior
            # print(p_v)

            if C_ytrue_ypred*p_v>maximum:
                maximum = C_ytrue_ypred * p_v
                x_pred_ = v
        
            # print(maximum)
        
        x_sensitive_attribute_pred.append(x_pred_)

        # for debugging    
        # if i==2:
        #     break

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
