import torch
import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


def fix_randomness(torch_random_seed=0, random_seed_python=0, random_seed_numpy=0):
	torch.manual_seed(torch_random_seed)
	random.seed(random_seed_python)
	np.random.seed(random_seed_numpy)


# def next_batch(inputs, targets, batchSize):
# 	# loop over the dataset
# 	for i in range(0, inputs.shape[0], batchSize):
# 		# yield a tuple of the current batched data and labels
# 		yield (inputs[i:i + batchSize], targets[i:i + batchSize])

def load_data_csv(file_path):
	from numpy import loadtxt
	# load array
	data = loadtxt(file_path, delimiter=',')
	return data


def calculate_MCC(confusion_matrix):
	return 


def calculate_G_mean(confusion_matrix):
	return


def visualize_cm(confusion_matrix, class_names):
 	# Create pandas dataframe
	dataframe = pd.DataFrame(confusion_matrix)

	plt.figure(figsize=(8, 6))
 
	# Create heatmap
	sns.heatmap(dataframe, annot=True, cbar=None,cmap="YlGnBu",fmt="d")
	
	plt.title("Confusion Matrix"), plt.tight_layout()
	
	plt.ylabel("True Class"), 
	plt.xlabel("Predicted Class")
	plt.show()