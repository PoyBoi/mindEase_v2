# ==================================================
# Imports
# ==================================================

import os, shutil, warnings, torch
import pandas as pd

import torch.nn as nn
import tensorflow as tf
import torch.optim as optim
import tensorflow_hub as hub
import tensorflow_text as text
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, TensorDataset


import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==================================================
# Basic init Defintions
# ==================================================

tf.get_logger().setLevel('ERROR')

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

warnings.filterwarnings("ignore")

ds_folder = r'C:\Users\parvs\VSC Codes\Python-root\_Projects_Personal\mindEase_v2\datasets\Conversational Training\Intent Based\difDs'

trainfile = os.path.join(ds_folder, 'train.csv')
validfile = os.path.join(ds_folder, 'val.csv')
testfile = os.path.join(ds_folder, 'test.csv')

traindf = pd.read_csv(trainfile)
validdf = pd.read_csv(validfile)
testdf = pd.read_csv(testfile)

# ==================================================
# LLM Download
# ==================================================

model_name = "distilbert/distilbert-base-uncased" 
model = AutoModelForSequenceClassification.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ==================================================
# Setting Features
# ==================================================

trainfeatures = traindf.copy()
trainlabels = trainfeatures.pop("Intent")
trainfeatures = trainfeatures.values

# ==================================================
# Finding dim depth
# ==================================================

chart = sns.countplot(x=trainlabels, palette=HAPPY_COLORS_PALETTE)
plt.title("Number of texts per intent")
chart.set_xticklabels(chart.get_xticklabels(), rotation=69, horizontalalignment='right', fontsize=8)

# Get the x-axis entries
x_axis_entries = [tick.get_text() for tick in chart.get_xticklabels()]
# print(x_axis_entries)
len_dims = len(x_axis_entries)

# ==================================================
# Fitting Features
# ==================================================

binarizer = LabelBinarizer()
trainlabels = binarizer.fit_transform(trainlabels.values)

testfeatures = testdf.copy()
testlabels = testfeatures.pop("Intent")
validfeatures = validdf.copy()
validlabels = validfeatures.pop("Intent")

testfeatures = testfeatures.values
validfeatures = validfeatures.values

testlabels = binarizer.transform(testlabels.values)
validlabels = binarizer.transform(validlabels.values)

# ==================================================
# LLM Class Definition
# ==================================================

class IntentClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(IntentClassifier, self).__init__()
        self.num_labels = num_labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits  # Directly access logits
        return logits

# ==================================================
# LLM Definiton
# ==================================================

# Example usage
num_labels = len_dims
# tokenizer = AutoTokenizer.from_pretrained(model_name)
model = IntentClassifier(model_name, num_labels).to('cuda')

# Ensure trainfeatures and validfeatures are lists of strings
trainfeatures = [str(feature) for feature in trainfeatures]
validfeatures = [str(feature) for feature in validfeatures]

# Convert train and validation data to PyTorch tensors
train_features = tokenizer(trainfeatures, padding=True, truncation=True, return_tensors="pt")
train_labels = torch.tensor(trainlabels)
valid_features = tokenizer(validfeatures, padding=True, truncation=True, return_tensors="pt")
valid_labels = torch.tensor(validlabels)

# Create DataLoader for training and validation
train_dataset = TensorDataset(train_features['input_ids'], train_features['attention_mask'], train_labels)
valid_dataset = TensorDataset(valid_features['input_ids'], valid_features['attention_mask'], valid_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32)

# Define loss function, optimizer, and metrics
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
accuracy = Accuracy(task='multiclass', num_classes=num_labels)

# ==================================================
# Running the training loop
# ==================================================

# Training loop
epochs = 5
model.cuda()

for epoch in range(epochs):
    model.train()
    total_loss = 0
    total_accuracy = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = [x.cuda() for x in batch]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs

        # print(f"Logits: \n\n{logits}\n\nLabels: \n\n{labels.float()}")
        # labels = labels.to("cuda").float()
        labels = labels.cuda().float()
        logits = logits.cuda()
        # labels = labels.cuda()

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy(logits, labels).item()  # Convert labels to float

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Training loss: {avg_loss:.4f}, Training accuracy: {avg_accuracy:.4f}")

    # Validation loop
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0

    with torch.no_grad():
        for batch in valid_loader:
            input_ids, attention_mask, labels = [x.cuda() for x in batch]

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs

            val_loss = loss_fn(logits, labels)
            total_val_loss += val_loss.item()
            total_val_accuracy += accuracy(logits, labels.float()).item()  # Convert labels to float

    avg_val_loss = total_val_loss / len(valid_loader)
    avg_val_accuracy = total_val_accuracy / len(valid_loader)

    print(f"Validation loss: {avg_val_loss:.4f}, Validation accuracy: {avg_val_accuracy:.4f}")