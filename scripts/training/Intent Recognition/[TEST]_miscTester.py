# ========================================
# Imports
# ========================================

import os, shutil, warnings, torch, json
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, F1Score
from torch.utils.data import DataLoader, TensorDataset

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from Tools.depFiles.readJson import load_config

# from depFiles.readJson import load_config

# ========================================
# User Def Area
# ========================================

current_dir = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(current_dir,"training_info.json")
config = load_config(config_path)

customName, file_loc, ds_folder, model_name, train_file_custom_name, val_file_custom_name, test_file_custom_name, className, gpuMode, epochs, learningRate, ifPrompt = (
    config["customName"],
    config["file_loc"],
    config["ds_folder"],
    config["model_name"],
    config["train_file_custom_name"],
    config["val_file_custom_name"],
    config["test_file_custom_name"],
    config["className"],
    config["gpuMode"],
    config["epochs"],
    config["learningRate"],
    config["ifPrompt"]
)