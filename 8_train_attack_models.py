from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pickle
import sys

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

from utils import load_data_csv, fix_randomness

import yaml

fix_randomness()

# read config file
with open("./config/config.yaml", "r") as ymlfile:
    try:
        cfg = yaml.safe_load(ymlfile)
    except yaml.YAMLError as exc:
        print(exc)

problem = cfg['problem']

# train on X_train data
X = load_data_csv(cfg["dataset"][problem]["path_to_x_train"])
y = load_data_csv(cfg["dataset"][problem]["path_to_y_train"])

# balance the dataset
if cfg['attack_model']['balance_data']['method'] == 'None':
    pass
elif cfg['attack_model']['balance_data']['method'] == 'RandomOverSampler':
    X, y = RandomOverSampler(random_state=0).fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'SMOTE':
    X, y = SMOTE().fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'ADASYN':
    X, y = ADASYN().fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'BorderlineSMOTE':
    X, y = BorderlineSMOTE().fit_resample(X, y)
else:
    print("Balance data method is not supported!!!")
    sys.exit()

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=cfg["attack_model"]["test_size"], random_state=cfg["random"]["random_state_sklearn"])

print("Training {} attack model".format(cfg["attack_model"]["model"]))

if cfg["attack_model"]["model"] == "KNN":
    model = KNeighborsClassifier(n_neighbors=cfg["attack_model"]["KNN"]["n_neighbors"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["KNN"]["model_path"], 'wb'))
elif cfg["attack_model"]["model"] == "NB":
    model = GaussianNB()
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["NB"]["model_path"], 'wb'))
elif cfg["attack_model"]["model"] == "LR":
    model = LogisticRegression(random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["LR"]["model_path"], 'wb'))
elif cfg["attack_model"]["model"] == "DT":
    model = DecisionTreeClassifier(random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["DT"]["model_path"], 'wb'))
elif cfg["attack_model"]["model"] == "MLP":
    model = MLPClassifier(solver=cfg["attack_model"]["MLP"]["solver"], alpha=1e-5, hidden_layer_sizes=(6,2), random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["MLP"]["model_path"], 'wb'))
elif cfg["attack_model"]["model"] == "XGBoost":
    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["XGBoost"]["model_path"], 'wb'))

# train on X_train data (with noise)
X = load_data_csv(cfg["dataset"][problem]["path_to_x_train_with_noise"])
y = load_data_csv(cfg["dataset"][problem]["path_to_y_train"])

# balance the dataset
if cfg['attack_model']['balance_data']['method'] == 'None':
    pass
elif cfg['attack_model']['balance_data']['method'] == 'RandomOverSampler':
    X, y = RandomOverSampler(random_state=0).fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'SMOTE':
    X, y = SMOTE().fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'ADASYN':
    X, y = ADASYN().fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'BorderlineSMOTE':
    X, y = BorderlineSMOTE().fit_resample(X, y)
else:
    print("Balance data method is not supported!!!")
    sys.exit()

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=cfg["attack_model"]["test_size"], random_state=cfg["random"]["random_state_sklearn"])

print("Training {} attack model (with noise data)".format(cfg["attack_model"]["model"]))

if cfg["attack_model"]["model"] == "KNN":
    model = KNeighborsClassifier(n_neighbors=cfg["attack_model"]["KNN"]["n_neighbors"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["KNN"]["model_path_with_noise"], 'wb'))
elif cfg["attack_model"]["model"] == "NB":
    model = GaussianNB()
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["NB"]["model_path_with_noise"], 'wb'))
elif cfg["attack_model"]["model"] == "LR":
    model = LogisticRegression(random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["LR"]["model_path_with_noise"], 'wb'))
elif cfg["attack_model"]["model"] == "DT":
    model = DecisionTreeClassifier(random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["DT"]["model_path_with_noise"], 'wb'))
elif cfg["attack_model"]["model"] == "MLP":
    model = MLPClassifier(solver=cfg["attack_model"]["MLP"]["solver"], alpha=1e-5, hidden_layer_sizes=(6,2), random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["MLP"]["model_path_with_noise"], 'wb'))
elif cfg["attack_model"]["model"] == "XGBoost":
    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["XGBoost"]["model_path_with_noise"], 'wb'))

# train on X_train data (with DP)
X = load_data_csv(cfg["dataset"][problem]["path_to_x_train_with_DP"])
y = load_data_csv(cfg["dataset"][problem]["path_to_y_train"])

# balance the dataset
if cfg['attack_model']['balance_data']['method'] == 'None':
    pass
elif cfg['attack_model']['balance_data']['method'] == 'RandomOverSampler':
    X, y = RandomOverSampler(random_state=0).fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'SMOTE':
    X, y = SMOTE().fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'ADASYN':
    X, y = ADASYN().fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'BorderlineSMOTE':
    X, y = BorderlineSMOTE().fit_resample(X, y)
else:
    print("Balance data method is not supported!!!")
    sys.exit()

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=cfg["attack_model"]["test_size"], random_state=cfg["random"]["random_state_sklearn"])

print("Training {} attack model (trained with DP)".format(cfg["attack_model"]["model"]))

if cfg["attack_model"]["model"] == "KNN":
    model = KNeighborsClassifier(n_neighbors=cfg["attack_model"]["KNN"]["n_neighbors"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["KNN"]["model_path_with_DP"], 'wb'))
elif cfg["attack_model"]["model"] == "NB":
    model = GaussianNB()
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["NB"]["model_path_with_DP"], 'wb'))
elif cfg["attack_model"]["model"] == "LR":
    model = LogisticRegression(random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["LR"]["model_path_with_DP"], 'wb'))
elif cfg["attack_model"]["model"] == "DT":
    model = DecisionTreeClassifier(random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["DT"]["model_path_with_DP"], 'wb'))
elif cfg["attack_model"]["model"] == "MLP":
    model = MLPClassifier(solver=cfg["attack_model"]["MLP"]["solver"], alpha=1e-5, hidden_layer_sizes=(6,2), random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["MLP"]["model_path_with_DP"], 'wb'))
elif cfg["attack_model"]["model"] == "XGBoost":
    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["XGBoost"]["model_path_with_DP"], 'wb'))

# train on X_train data (with DP with noise)
X = load_data_csv(cfg["dataset"][problem]["path_to_x_train_with_DP_with_noise"])
y = load_data_csv(cfg["dataset"][problem]["path_to_y_train"])

# balance the dataset
if cfg['attack_model']['balance_data']['method'] == 'None':
    pass
elif cfg['attack_model']['balance_data']['method'] == 'RandomOverSampler':
    X, y = RandomOverSampler(random_state=0).fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'SMOTE':
    X, y = SMOTE().fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'ADASYN':
    X, y = ADASYN().fit_resample(X, y)
elif cfg['attack_model']['balance_data']['method'] == 'BorderlineSMOTE':
    X, y = BorderlineSMOTE().fit_resample(X, y)
else:
    print("Balance data method is not supported!!!")
    sys.exit()

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=cfg["attack_model"]["test_size"], random_state=cfg["random"]["random_state_sklearn"])

print("Training {} attack model (trained with DP) with noise data".format(cfg["attack_model"]["model"]))

if cfg["attack_model"]["model"] == "KNN":
    model = KNeighborsClassifier(n_neighbors=cfg["attack_model"]["KNN"]["n_neighbors"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["KNN"]["model_path_with_DP_with_noise"], 'wb'))
elif cfg["attack_model"]["model"] == "NB":
    model = GaussianNB()
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["NB"]["model_path_with_DP_with_noise"], 'wb'))
elif cfg["attack_model"]["model"] == "LR":
    model = LogisticRegression(random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["LR"]["model_path_with_DP_with_noise"], 'wb'))
elif cfg["attack_model"]["model"] == "DT":
    model = DecisionTreeClassifier(random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["DT"]["model_path_with_DP_with_noise"], 'wb'))
elif cfg["attack_model"]["model"] == "MLP":
    model = MLPClassifier(solver=cfg["attack_model"]["MLP"]["solver"], alpha=1e-5, hidden_layer_sizes=(6,2), random_state=cfg["random"]["random_state_sklearn"])
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["MLP"]["model_path_with_DP_with_noise"], 'wb'))
elif cfg["attack_model"]["model"] == "XGBoost":
    model = GradientBoostingClassifier()
    model.fit(X_train, Y_train)
    pickle.dump(model, open(cfg["attack_model"]["XGBoost"]["model_path_with_DP_with_noise"], 'wb'))