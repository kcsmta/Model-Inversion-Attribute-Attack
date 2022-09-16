# imports from captum library
from captum.attr import (
    IntegratedGradients,
    DeepLift,
    DeepLiftShap,
    GradientShap,
    NoiseTunnel,
    FeatureAblation,
    Saliency,
    InputXGradient,
    Deconvolution,
    FeaturePermutation
)

from operator import mod
from time import time
import sys

from numpy import argmax
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F

import data
from utils import fix_randomness
from models import Covid19_MainTaskModel, Adults_MainTaskModel, Fivethirtyeight_MainTaskModel, GSS_MainTaskModel
import pandas as pd
import plotly.express as px

import yaml

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
y_test = torch.from_numpy(y_test)
y_test = F.one_hot(y_test).type(torch.FloatTensor)


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
    model = torch.load(cfg["target_model"][problem]["path_to_target_model"])
else:
    model = torch.load(cfg["target_model"][problem]["path_to_target_model"])

ig = IntegratedGradients(model)

test_input_tensor = torch.from_numpy(X_test).type(torch.FloatTensor)

test_input_tensor.requires_grad_()

print(test_input_tensor.shape)
# print(model)

import time
start_time = time.time()
attr, delta = ig.attribute(test_input_tensor, target=1, return_convergence_delta=True)
# attr, delta = ig.attribute(test_input_tensor[:10,:],target=1, return_convergence_delta=True)
end_time = time.time()
print("Time to calculate importance weights: {:.2f} seconds".format(end_time - start_time))

attr = attr.detach().numpy()

def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True, axis_title="Features"):
    print(title)
    for i in range(len(feature_names)):
        print(feature_names[i], ": ", '%.3f'%(importances[i]))
    x_pos = (np.arange(len(feature_names)))
    if plot:
        plt.figure(figsize=(12,6))
        plt.bar(x_pos, importances, align='center')
        plt.xticks(x_pos, feature_names, wrap=True)
        plt.xlabel(axis_title)
        plt.title(title)
        plt.show()

feature_names = cfg['dataset'][problem]['attributes_names']
print(feature_names)
visualize_importances(feature_names, np.mean(attr, axis=0))

if cfg['target_model'][problem]['DP']['isDP'] == True:
    print("NOTICE: Calculating feature's importance only work for model trained without DP")
    print("This results is calculated on pretrained model withou DP")