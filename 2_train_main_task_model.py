from statistics import mode
from numpy import argmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import SGD

from opacus import PrivacyEngine

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

# Step 2: shuffle data
sklearn_random_state = cfg["random"]["random_state_sklearn"]
X, y = shuffle(X, y, random_state=sklearn_random_state)

# Step 3: Take a part of data to train TARGET model
propotion_to_train_target = cfg["target_model"][problem]["propotion_to_train_target_model"]
X = X[:int(X.shape[0]*propotion_to_train_target),:]
y = y[:int(y.shape[0]*propotion_to_train_target)]

# Step 4: Split data into training set and test set (to train TARGET model)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=cfg["target_model"][problem]["test_size"], random_state=sklearn_random_state)
X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train)
y_train = F.one_hot(y_train).type(torch.FloatTensor)
y_test = torch.from_numpy(y_test)
y_test = F.one_hot(y_test).type(torch.FloatTensor)

# parameters to train target model
BATCH_SIZE = cfg["target_model"][problem]["batch_size"]
EPOCHS = cfg["target_model"][problem]["epochs"]
LR = cfg["target_model"][problem]["learning_rate"]
MOMENTUM = cfg["target_model"][problem]["momentum"]

# use DataSet and DataLoader class in Pytorch
# NOTE: dont shuffle the data, we already shuffled them before
data_train = data.TabularData(X_train, y_train)
data_train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=False)
data_test = data.TabularData(X_test, y_test)
data_test_loader =  DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=False)


# Step 5: Define model to train
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

# Step 6: define optimizer used to train model
# optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
optimizer = SGD(model.parameters(), lr=LR)

# Step 7: define loss function 
# lossFunc = nn.MSELoss()
lossFunc = nn.CrossEntropyLoss()

# Step 8: define training with DP
# Attaching a Differential Privacy Engine to the Optimizer
if cfg['target_model'][problem]['DP']['isDP'] == True:
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_train_loader,
        noise_multiplier=1.1,
        max_grad_norm=1.0,
    )
    delta=cfg['target_model'][problem]['DP']['delta']

for epoch in range(0, EPOCHS):
    print("[INFO] epoch: {}...".format(epoch + 1))

    # Step 1: Train on training set
    trainLoss = 0
    samples = 0
    model.train()

    train_preds = []
    for (batchX, batchY) in data_train_loader:
        outputs = model(batchX)
        loss = lossFunc(outputs, batchY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        trainLoss += loss.item() * batchY.size(0)

        for output in outputs:
            train_preds.append(argmax(output.detach().numpy()))
            # if output[0] > output[1]:
            #     train_preds.append(0)
            # else:
            #     train_preds.append(1)

        samples += batchY.size(0)
    
    y_true  = argmax(y_train, axis=1)
    trainAcc = accuracy_score(y_true, train_preds)
    if cfg['target_model'][problem]['DP']['isDP'] == True:
        print(delta)
        print(type(delta))
        epsilon = privacy_engine.get_epsilon(delta) 
        trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f} (ε = {:.2f}, δ = {})"
        print(trainTemplate.format(epoch + 1, (trainLoss / samples), trainAcc, epsilon, delta))
        # trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
        # print(trainTemplate.format(epoch + 1, (trainLoss / samples), trainAcc))
    else:
        trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
        print(trainTemplate.format(epoch + 1, (trainLoss / samples), trainAcc))

    # Step 2: Evaluate on test set
    testLoss = 0
    samples = 0
    model.eval()

    test_preds = []
    with torch.no_grad():
        for (batchX, batchY) in data_test_loader:
            outputs = model(batchX)
            loss = lossFunc(outputs, batchY)
            testLoss += loss.item() * batchY.size(0)
            for output in outputs:
                test_preds.append(argmax(output))
            samples += batchY.size(0)
        
        y_true  = argmax(y_test, axis=1)
        testAcc = accuracy_score(y_true, test_preds)

        # display model progress on the current training batch
        trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
        print(trainTemplate.format(epoch + 1, (testLoss / samples), testAcc))

# Save trained model
if cfg['target_model'][problem]['DP']['isDP'] == True:
    torch.save(model, cfg["target_model"][problem]["path_to_target_model_with_DP"])
else:
    torch.save(model, cfg["target_model"][problem]["path_to_target_model"])
