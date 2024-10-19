# ========================================
# Imports
# ========================================

import os, shutil, warnings, torch, json
from datetime import datetime
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
from Tools.depFiles.datasetHandler import ds_split_set, eval_img
from Tools.depFiles.modelHandler import model_eval, predict_intent, train_model, train_one_epoch, validate_one_epoch

print("\nLoaded all Imports\n")

# ========================================
# User Def Area
# ========================================

current_dir = os.path.dirname(os.path.abspath(__file__))

config_path = os.path.join(current_dir,"training_info.json")

config = load_config(
    config_path
)

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
    float(config["learningRate"]),
    config["ifPrompt"]
)

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

result_path = os.path.join(current_dir, "Results", f"{current_time}_{customName}_{epochs}_{learningRate}")
os.makedirs(result_path, exist_ok=True)

config_ds = ds_split_set(
    file_loc, ds_folder, className, 
    train_file_custom_name, val_file_custom_name, test_file_custom_name
)

trainfeatures, trainlabels, testfeatures, testlabels, validfeatures, validlabels = (
        config_ds["trainfeatures"],
        config_ds["trainlabels"],
        config_ds["testfeatures"],
        config_ds["testlabels"],
        config_ds["validfeatures"],
        config_ds["validlabels"]
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("\nLoaded JSON\n")

# ========================================
# Feature Def
# ========================================

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

warnings.filterwarnings("ignore")

plt.ioff()
chart = sns.countplot(x=trainlabels, palette=HAPPY_COLORS_PALETTE)
plt.title("Number of texts per intent")
chart.set_xticklabels(chart.get_xticklabels(), rotation=69, horizontalalignment='right', fontsize=8)

x_axis_entries = [tick.get_text() for tick in chart.get_xticklabels()]
len_dims = len(x_axis_entries)

label_encoder = LabelEncoder()
trainlabels = label_encoder.fit_transform(trainlabels.values)
testlabels = label_encoder.transform(testlabels.values)
validlabels = label_encoder.transform(validlabels.values)

print("\nDefined Features\n")

# ========================================
# Model Def
# ========================================

class IntentClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(IntentClassifier, self).__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs.logits

num_labels = len_dims
model = IntentClassifier(model_name, num_labels).to(gpuMode)

def prepare_data(features, labels):
    features = [str(feature) for feature in features]
    features = tokenizer(features, padding=True, truncation=True, return_tensors="pt")
    
    labels = torch.tensor(labels).long().to(gpuMode)  # Change to long tensor
    
    dataset = TensorDataset(features['input_ids'], features['attention_mask'], labels)
    return dataset

# Prepare train, validation, and test datasets
train_dataset = prepare_data(trainfeatures.values, trainlabels)
valid_dataset = prepare_data(validfeatures.values, validlabels)
test_dataset = prepare_data(testfeatures.values, testlabels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# loss_fn = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification
loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=2e-5)
optimizer = optim.Adam(model.parameters(), lr=learningRate)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * epochs)

# accuracy = Accuracy(task='binary').to('cuda') 
# accuracy = Accuracy(task='multiclass', num_classes=num_labels, average='micro').to('cuda') 
accuracy = Accuracy(task='multiclass', num_classes=num_labels).to(gpuMode) 
f1_score = F1Score(task='multiclass', num_classes=num_labels, average='macro').to(gpuMode)

print("\nLoaded and Prepared Datasets via DataLoader\n")

# ========================================
# Training Loop
# ========================================
device = torch.device("cuda" if gpuMode == 'cuda' and torch.cuda.is_available() else "cpu")
model.to(device)

print("\nMoved model to GPU\n")

(
epoch, avg_val_accuracy, train_accuracies, val_accuracies,
train_losses, val_losses, train_f1_scores, val_f1_scores, saved_model_loc
) = train_model(
    result_path, model, learningRate, train_loader, valid_loader, optimizer, 
    scheduler, loss_fn, accuracy, f1_score, epochs, device, customName, 1
).values()

print("\nFinished Training Model\n")

# ========================================
# Plotting Accuracy/F1/Loss
# ========================================
eval_img(
    result_path, epoch, avg_val_accuracy, train_accuracies, val_accuracies, 
    train_losses, val_losses, train_f1_scores, val_f1_scores
)
# ========================================
# Model testing / eval
# ========================================

model.eval() # Putting the model in testing mode
print("\nModel for Evaluation\n")

model_eval(
    model, test_loader, gpuMode, 
    loss_fn, accuracy, f1_score
)

predict_intent(
    ifPrompt, model, tokenizer, label_encoder, 
    gpuMode, IntentClassifier, model_name, num_labels, saved_model_loc
)

print("\nRun Complete\n")