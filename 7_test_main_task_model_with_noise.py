from statistics import mode
from numpy import argmax
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, plot_roc_curve, roc_curve, auc
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import SGD

import data
from utils import fix_randomness
from models import Covid19_MainTaskModel, Adults_MainTaskModel, Fivethirtyeight_MainTaskModel, GSS_MainTaskModel

import yaml

import sys

fix_randomness()

# read config file
with open("./config/config.yaml", "r") as ymlfile:
    try:
        cfg = yaml.safe_load(ymlfile)
    except yaml.YAMLError as exc:
        print(exc)

# Step 1: load data from .csv file
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

# Step 2: shuffle data
sklearn_random_state = cfg["random"]["random_state_sklearn"]
X, y = shuffle(X, y, random_state=sklearn_random_state)

# Step 3: Take a part of data to train TARGET model
propotion_to_train_target = cfg["target_model"][problem]["propotion_to_train_target_model"]
X = X[:int(X.shape[0]*propotion_to_train_target),:]
y = y[:int(y.shape[0]*propotion_to_train_target)]

# Step 4: Split data into training set and test set (to train TARGET model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg["target_model"][problem]["test_size"], random_state=sklearn_random_state)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test)
y_test = F.one_hot(y_test).type(torch.FloatTensor)

# NOTE: Set batch size to 1
BATCH_SIZE = 1

# use DataSet and DataLoader class in Pytorch
# NOTE: dont shuffle the data, we already shuffled them before
data_test = data.TabularData(X_test, y_test)
data_test_loader =  DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)


# Step 5: Define pre-trained model to load
if problem == 'covid19':
    model = Covid19_MainTaskModel()
elif problem == 'adults':
    model = Adults_MainTaskModel()
elif problem == 'fivethirtyeight':
    model = Fivethirtyeight_MainTaskModel()
elif problem == 'gss':
    model = GSS_MainTaskModel()
else:
    print("Can not define target model!!!")
    sys.exit()

if cfg['target_model'][problem]['DP']['isDP'] == True:
    print("Test model with DP...")
    model = torch.load(cfg["target_model"][problem]["path_to_target_model_with_DP"])
else:
    print("Test model without DP...")
    model = torch.load(cfg["target_model"][problem]["path_to_target_model"])

BATCH_SIZE = 1
lossFunc = nn.CrossEntropyLoss()

import time
# test trained model
test_preds = []
testLoss = 0
samples = 0

with torch.no_grad():
    start_time = time.time()
    for (batchX, batchY) in data_test_loader: # NOTE: Batch size is set to 1
        output = model(batchX)
        
        output_dimension = output.shape[0]
        
        random_noise = []
        for i_dimension in range(output_dimension):
            random_num = random.uniform(cfg_target_attribute["target_attribute"]["target_attribute_noise_lower_bound"], 
            cfg_target_attribute["target_attribute"]["target_attribute_noise_upper_bound"])
            random_noise.append(random_num)

        random_noise = np.asarray(random_noise)

        output_with_noise = np.add(output, random_noise)

        loss = lossFunc(output, batchY)
        testLoss += loss.item() * batchY.size(0)
        test_preds.append(argmax(output_with_noise))
        samples += batchY.size(0)

    end_time = time.time()
    infer_time = end_time - start_time
    print("+ Inference time = ", infer_time)

    y_true = argmax(y_test, axis=1)
    testAcc = accuracy_score(y_true, test_preds)
    print("+ Accuracy = ", testAcc)

    try:
        testF1 = f1_score(y_true, test_preds)
        testPrecision = precision_score(y_true, test_preds)
        testRecall = recall_score(y_true, test_preds)
        print("+ F1 score = ", testF1)
        print("+ Precision = ", testPrecision)
        print("+ Recall = ", testRecall)
    except:
        print("Multi-classification! Dont compute F1, Precision, and Recall.")
        pass

    cm = confusion_matrix(y_true, test_preds)
    print("+ Confusion matrix:\n", cm)
    np.savetxt('./models/' + problem + '_confusion_matrix.txt', cm, fmt='%d') # save confusion matrix for Fredrickson's attack
    print("Confusion matrix for {} is saved".format(problem))


import matplotlib.pyplot as plt

with torch.no_grad():
    test_preds = model(X_test)
# print(test_preds)

fpr, tpr, thresholds = roc_curve(y_test[:, 0], test_preds[:, 0])
roc_auc = auc(fpr, tpr)

print("+ AUC = ", roc_auc)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.4f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()