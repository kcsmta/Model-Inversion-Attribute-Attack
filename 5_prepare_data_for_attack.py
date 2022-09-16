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


# Step 4: Query to black-box target model
main_task_model = torch.load(cfg["target_model"][problem]["path_to_target_model"])
main_task_model_with_DP = torch.load(cfg["target_model"][problem]["path_to_target_model"])

# create train dataset for attack model
main_task_model_output = []
main_task_model_output_with_DP = []
main_task_model_output_with_noise = []
main_task_model_output_with_DP_with_noise = []
attribute_label = []

with torch.no_grad():
    for i, x in enumerate(X_train_attack):
        # output of main task model
        x = torch.unsqueeze(x, 0)
        target_output = main_task_model(x).numpy()[0]
        output_dimension = target_output.shape[0]
        
        random_noise = []
        for i_dimension in range(output_dimension):
            random_num = random.uniform(cfg_target_attribute["target_attribute"]["target_attribute_noise_lower_bound"], 
            cfg_target_attribute["target_attribute"]["target_attribute_noise_upper_bound"])
            random_noise.append(random_num)

        random_noise = np.asarray(random_noise)

        target_outputs_with_noise = np.add(target_output, random_noise)

        main_task_model_output.append(target_output)
        main_task_model_output_with_noise.append(target_outputs_with_noise)

        # output of main task model with DP
        target_output_with_DP = main_task_model_with_DP(x).numpy()[0]
        
        random_noise = []
        for i_dimension in range(output_dimension):
            random_num = random.uniform(cfg_target_attribute["target_attribute"]["target_attribute_noise_lower_bound"], 
            cfg_target_attribute["target_attribute"]["target_attribute_noise_upper_bound"])
            random_noise.append(random_num)

        random_noise = np.asarray(random_noise)

        target_outputs_with_DP_with_noise = np.add(target_output_with_DP, random_noise)

        main_task_model_output_with_DP.append(target_output_with_DP)
        main_task_model_output_with_DP_with_noise.append(target_outputs_with_DP_with_noise)

        # attribute label
        attribute_label.append(int(y_train_attack[i]))

X_train = np.array(main_task_model_output)
X_train_with_noise = np.array(main_task_model_output_with_noise)
X_train_with_DP = np.array(main_task_model_output_with_DP)
X_train_with_DP_with_noise = np.array(main_task_model_output_with_DP_with_noise)
y_train = np.array(attribute_label)

np.savetxt(cfg["dataset"][problem]["path_to_x_train"], X_train, delimiter=",")
np.savetxt(cfg["dataset"][problem]["path_to_x_train_with_noise"], X_train_with_noise, delimiter=",")
np.savetxt(cfg["dataset"][problem]["path_to_x_train_with_DP"], X_train_with_DP, delimiter=",")
np.savetxt(cfg["dataset"][problem]["path_to_x_train_with_DP_with_noise"], X_train_with_DP_with_noise, delimiter=",")
np.savetxt(cfg["dataset"][problem]["path_to_y_train"], y_train, delimiter=",")

# create test dataset for attack model
main_task_model_output = []
main_task_model_output_with_noise = []
main_task_model_output_with_DP = []
main_task_model_output_with_DP_with_noise = []
attribute_label = []

with torch.no_grad():
    for i, x in enumerate(X_test_attack):
        # output of main task model
        x = torch.unsqueeze(x, 0)
        target_output = main_task_model(x).numpy()[0]

        random_noise = []
        for i_dimension in range(output_dimension):
            random_num = random.uniform(cfg_target_attribute["target_attribute"]["target_attribute_noise_lower_bound"], 
            cfg_target_attribute["target_attribute"]["target_attribute_noise_upper_bound"])
            random_noise.append(random_num)

        random_noise = np.asarray(random_noise)

        target_outputs_with_noise = np.add(target_output, random_noise)

        main_task_model_output.append(target_output)
        main_task_model_output_with_noise.append(target_outputs_with_noise)

        # output of main task model with DP
        target_output_with_DP = main_task_model_with_DP(x).numpy()[0]
        
        random_noise = []
        for i_dimension in range(output_dimension):
            random_num = random.uniform(cfg_target_attribute["target_attribute"]["target_attribute_noise_lower_bound"], 
            cfg_target_attribute["target_attribute"]["target_attribute_noise_upper_bound"])
            random_noise.append(random_num)

        random_noise = np.asarray(random_noise)

        target_outputs_with_DP_with_noise = np.add(target_output_with_DP, random_noise)

        main_task_model_output_with_DP.append(target_output_with_DP)
        main_task_model_output_with_DP_with_noise.append(target_outputs_with_DP_with_noise)

        # attribute label
        attribute_label.append(int(y_test_attack[i]))

X_test = np.array(main_task_model_output)
X_test_with_noise = np.array(main_task_model_output_with_noise)
X_test_with_DP = np.array(main_task_model_output_with_DP)
X_test_with_DP_with_noise = np.array(main_task_model_output_with_DP_with_noise)
y_test = np.array(attribute_label)

# print(X_test)
# print(y_test)
# print(X_test.shape)
# print(y_test.shape)

np.savetxt(cfg["dataset"][problem]["path_to_x_test"], X_test, delimiter=",")
np.savetxt(cfg["dataset"][problem]["path_to_x_test_with_noise"], X_test_with_noise, delimiter=",")
np.savetxt(cfg["dataset"][problem]["path_to_x_test_with_DP"], X_test_with_DP, delimiter=",")
np.savetxt(cfg["dataset"][problem]["path_to_x_test_with_DP_with_noise"], X_test_with_DP_with_noise, delimiter=",")
np.savetxt(cfg["dataset"][problem]["path_to_y_test"], y_test, delimiter=",")