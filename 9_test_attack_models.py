from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, plot_roc_curve, roc_curve, auc
import numpy as np
import pickle

from utils import load_data_csv, fix_randomness, visualize_cm

import yaml

fix_randomness()

# read config file
with open("./config/config.yaml", "r") as ymlfile:
    try:
        cfg = yaml.safe_load(ymlfile)
    except yaml.YAMLError as exc:
        print(exc)

problem = cfg['problem']

# Load attack feature configuration file
path_to_cfg_target_feature = "./config/target_attributes_" + problem + ".yaml"
with open(path_to_cfg_target_feature, "r") as ymlfile:
    try:
        cfg_target_attribute = yaml.safe_load(ymlfile)
    except yaml.YAMLError as exc:
        print(exc)


# 
X_test = load_data_csv(cfg["dataset"][problem]["path_to_x_test"])
y_test = load_data_csv(cfg["dataset"][problem]["path_to_y_test"])
y_true = y_test

print("Test {} attack model".format(cfg["attack_model"]["model"]))

# load the attack model from disk
if cfg["attack_model"]["model"] == "KNN":
    attack_model = pickle.load(open(cfg["attack_model"]["KNN"]["model_path"], 'rb'))
elif cfg["attack_model"]["model"] == "NB":
    attack_model = pickle.load(open(cfg["attack_model"]["NB"]["model_path"], 'rb'))
elif cfg["attack_model"]["model"] == "LR":
    attack_model = pickle.load(open(cfg["attack_model"]["LR"]["model_path"], 'rb'))
elif cfg["attack_model"]["model"] == "DT":
    attack_model = pickle.load(open(cfg["attack_model"]["DT"]["model_path"], 'rb'))
elif cfg["attack_model"]["model"] == "MLP":
    attack_model = pickle.load(open(cfg["attack_model"]["MLP"]["model_path"], 'rb'))
elif cfg["attack_model"]["model"] == "XGBoost":
    attack_model = pickle.load(open(cfg["attack_model"]["XGBoost"]["model_path"], 'rb'))

test_preds = attack_model.predict(X_test)

print("===========Test model===========")

testAcc = accuracy_score(y_true, test_preds)
print("+ Accuracy = ", testAcc)

if cfg_target_attribute['target_attribute']['isBinary'] == True:
    testF1 = f1_score(y_true, test_preds)
    testPrecision = precision_score(y_true, test_preds)
    testRecall = recall_score(y_true, test_preds)
    print("+ F1 score = ", testF1)
    print("+ Precision = ", testPrecision)
    print("+ Recall = ", testRecall)

cm = confusion_matrix(y_true, test_preds)
print("+ Confusion matrix:\n", cm)

# 
X_test = load_data_csv(cfg["dataset"][problem]["path_to_x_test_with_noise"])
y_test = load_data_csv(cfg["dataset"][problem]["path_to_y_test"])
y_true = y_test

print("Test {} attack model".format(cfg["attack_model"]["model"]))

# load the attack model from disk
if cfg["attack_model"]["model"] == "KNN":
    attack_model = pickle.load(open(cfg["attack_model"]["KNN"]["model_path_with_noise"], 'rb'))
elif cfg["attack_model"]["model"] == "NB":
    attack_model = pickle.load(open(cfg["attack_model"]["NB"]["model_path_with_noise"], 'rb'))
elif cfg["attack_model"]["model"] == "LR":
    attack_model = pickle.load(open(cfg["attack_model"]["LR"]["model_path_with_noise"], 'rb'))
elif cfg["attack_model"]["model"] == "DT":
    attack_model = pickle.load(open(cfg["attack_model"]["DT"]["model_path_with_noise"], 'rb'))
elif cfg["attack_model"]["model"] == "MLP":
    attack_model = pickle.load(open(cfg["attack_model"]["MLP"]["model_path_with_noise"], 'rb'))
elif cfg["attack_model"]["model"] == "XGBoost":
    attack_model = pickle.load(open(cfg["attack_model"]["XGBoost"]["model_path_with_noise"], 'rb'))

test_preds = attack_model.predict(X_test)

print("===========Test model (with noise)===========")

testAcc = accuracy_score(y_true, test_preds)
print("+ Accuracy = ", testAcc)

if cfg_target_attribute['target_attribute']['isBinary'] == True:
    testF1 = f1_score(y_true, test_preds)
    testPrecision = precision_score(y_true, test_preds)
    testRecall = recall_score(y_true, test_preds)
    print("+ F1 score = ", testF1)
    print("+ Precision = ", testPrecision)
    print("+ Recall = ", testRecall)
    
cm = confusion_matrix(y_true, test_preds)
print("+ Confusion matrix:\n", cm)

# 
X_test = load_data_csv(cfg["dataset"][problem]["path_to_x_test_with_DP"])
y_test = load_data_csv(cfg["dataset"][problem]["path_to_y_test"])
y_true = y_test

print("Test {} attack model".format(cfg["attack_model"]["model"]))

# load the attack model from disk
if cfg["attack_model"]["model"] == "KNN":
    attack_model = pickle.load(open(cfg["attack_model"]["KNN"]["model_path_with_DP"], 'rb'))
elif cfg["attack_model"]["model"] == "NB":
    attack_model = pickle.load(open(cfg["attack_model"]["NB"]["model_path_with_DP"], 'rb'))
elif cfg["attack_model"]["model"] == "LR":
    attack_model = pickle.load(open(cfg["attack_model"]["LR"]["model_path_with_DP"], 'rb'))
elif cfg["attack_model"]["model"] == "DT":
    attack_model = pickle.load(open(cfg["attack_model"]["DT"]["model_path_with_DP"], 'rb'))
elif cfg["attack_model"]["model"] == "MLP":
    attack_model = pickle.load(open(cfg["attack_model"]["MLP"]["model_path_with_DP"], 'rb'))
elif cfg["attack_model"]["model"] == "XGBoost":
    attack_model = pickle.load(open(cfg["attack_model"]["XGBoost"]["model_path_with_DP"], 'rb'))

test_preds = attack_model.predict(X_test)

print("===========Test model (with DP)===========")

testAcc = accuracy_score(y_true, test_preds)
print("+ Accuracy = ", testAcc)

if cfg_target_attribute['target_attribute']['isBinary'] == True:
    testF1 = f1_score(y_true, test_preds)
    testPrecision = precision_score(y_true, test_preds)
    testRecall = recall_score(y_true, test_preds)
    print("+ F1 score = ", testF1)
    print("+ Precision = ", testPrecision)
    print("+ Recall = ", testRecall)
    
cm = confusion_matrix(y_true, test_preds)
print("+ Confusion matrix:\n", cm)

# 
X_test = load_data_csv(cfg["dataset"][problem]["path_to_x_test_with_DP_with_noise"])
y_test = load_data_csv(cfg["dataset"][problem]["path_to_y_test"])
y_true = y_test

print("Test {} attack model".format(cfg["attack_model"]["model"]))

# load the attack model from disk
if cfg["attack_model"]["model"] == "KNN":
    attack_model = pickle.load(open(cfg["attack_model"]["KNN"]["model_path_with_DP_with_noise"], 'rb'))
elif cfg["attack_model"]["model"] == "NB":
    attack_model = pickle.load(open(cfg["attack_model"]["NB"]["model_path_with_DP_with_noise"], 'rb'))
elif cfg["attack_model"]["model"] == "LR":
    attack_model = pickle.load(open(cfg["attack_model"]["LR"]["model_path_with_DP_with_noise"], 'rb'))
elif cfg["attack_model"]["model"] == "DT":
    attack_model = pickle.load(open(cfg["attack_model"]["DT"]["model_path_with_DP_with_noise"], 'rb'))
elif cfg["attack_model"]["model"] == "MLP":
    attack_model = pickle.load(open(cfg["attack_model"]["MLP"]["model_path_with_DP_with_noise"], 'rb'))
elif cfg["attack_model"]["model"] == "XGBoost":
    attack_model = pickle.load(open(cfg["attack_model"]["XGBoost"]["model_path_with_DP_with_noise"], 'rb'))

test_preds = attack_model.predict(X_test)

print("===========Test model (with DP, with noise)===========")

testAcc = accuracy_score(y_true, test_preds)
print("+ Accuracy = ", testAcc)

if cfg_target_attribute['target_attribute']['isBinary'] == True:
    testF1 = f1_score(y_true, test_preds)
    testPrecision = precision_score(y_true, test_preds)
    testRecall = recall_score(y_true, test_preds)
    print("+ F1 score = ", testF1)
    print("+ Precision = ", testPrecision)
    print("+ Recall = ", testRecall)
    
cm = confusion_matrix(y_true, test_preds)
print("+ Confusion matrix:\n", cm)