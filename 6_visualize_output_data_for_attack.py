#import pandas library
import sys
from time import time

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from utils import load_data_csv
from utils import fix_randomness

import yaml

fix_randomness()

# read config file
with open("./config/config.yaml", "r") as ymlfile:
    try:
        cfg = yaml.safe_load(ymlfile)
    except yaml.YAMLError as exc:
        print(exc)

problem = cfg["problem"]

# Load attack feature configuration file
path_to_cfg_target_feature = "./config/target_attributes_" + problem + ".yaml"
with open(path_to_cfg_target_feature, "r") as ymlfile:
    try:
        cfg_target_attribute = yaml.safe_load(ymlfile)
    except yaml.YAMLError as exc:
        print(exc)

# load data for attack
X_train = load_data_csv(cfg["dataset"][problem]["path_to_x_train"])
X_train_with_noise = load_data_csv(cfg["dataset"][problem]["path_to_x_train_with_noise"])
X_train_with_DP = load_data_csv(cfg["dataset"][problem]["path_to_x_train_with_DP"])
X_train_with_DP_with_noise = load_data_csv(cfg["dataset"][problem]["path_to_x_train_with_DP_with_noise"])
y_train = load_data_csv(cfg["dataset"][problem]["path_to_y_train"])

# print(X_train)
# print(y_train)
# print(X_train.shape)
# print(y_train.shape)

target_ids = cfg_target_attribute["target_attribute"]["target_attribute_ids"]
target_names = cfg_target_attribute["target_attribute"]["target_attribute_names"]
target_names_with_noise = cfg_target_attribute["target_attribute"]["target_attribute_names_with_noise"]
target_names_with_DP = cfg_target_attribute["target_attribute"]["target_attribute_names_with_DP"]
target_names_with_DP_with_noise = cfg_target_attribute["target_attribute"]["target_attribute_names_with_DP_with_noise"]
colors = cfg_target_attribute["target_attribute"]["target_attribute_represent_colors"]

#############
for i, c, label in zip(target_ids, colors, target_names):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], color=c, label=label, alpha=0.3)

plt.legend()

x = np.linspace(0.15,0.85,100)
y = 0*x + 0.5
plt.plot(x, y, '-b')
plt.title('Main task prediction')
plt.xlabel('negative', color='#1C2833')
plt.ylabel('positive', color='#1C2833')
plt.show()

#############
for i, c, label in zip(target_ids, colors, target_names_with_noise):
    plt.scatter(X_train_with_noise[y_train == i, 0], X_train_with_noise[y_train == i, 1], color=c, label=label, alpha=0.3)

plt.legend()

x = np.linspace(0.15,0.85,100)
y = 0*x + 0.5
plt.plot(x, y, '-b')
plt.title('Main task prediction')
plt.xlabel('negative', color='#1C2833')
plt.ylabel('positive', color='#1C2833')
plt.show()

#############
for i, c, label in zip(target_ids, colors, target_names_with_DP):
    plt.scatter(X_train_with_DP[y_train == i, 0], X_train_with_DP[y_train == i, 1], color=c, label=label, alpha=0.3)

plt.legend()

x = np.linspace(0.15,0.85,100)
y = 0*x + 0.5
plt.plot(x, y, '-b')
plt.title('Main task prediction')
plt.xlabel('negative', color='#1C2833')
plt.ylabel('positive', color='#1C2833')
plt.show()

#############
for i, c, label in zip(target_ids, colors, target_names_with_DP_with_noise):
    plt.scatter(X_train_with_DP_with_noise[y_train == i, 0], X_train_with_DP_with_noise[y_train == i, 1], color=c, label=label, alpha=0.3)

plt.legend()

x = np.linspace(0.15,0.85,100)
y = 0*x + 0.5
plt.plot(x, y, '-b')
plt.title('Main task prediction')
plt.xlabel('negative', color='#1C2833')
plt.ylabel('positive', color='#1C2833')
plt.show()
