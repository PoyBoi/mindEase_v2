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
from Tools.depFiles.datasetHandler import ds_split, set_files


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

# ========================================
# DS Separation
# ========================================

# Load the dataset
dataset = pd.read_csv(file_loc)

# Split the dataset into training and temporary datasets
train_data, temp_data = train_test_split(dataset, test_size=0.4, random_state=42)

# Split the temporary dataset into validation and test datasets
val_data, test_data = train_test_split(temp_data, test_size=0.4, random_state=42)

# Save the datasets to separate files
save_loc = ds_folder

train_data.to_csv(os.path.join(save_loc, train_file_custom_name), index=False)
val_data.to_csv(os.path.join(save_loc, val_file_custom_name), index=False)
test_data.to_csv(os.path.join(save_loc, test_file_custom_name), index=False)

print("Datasets have been split and saved successfully.")

trainfile = os.path.join(ds_folder, train_file_custom_name)
validfile = os.path.join(ds_folder, val_file_custom_name)
testfile = os.path.join(ds_folder, test_file_custom_name)

traindf = pd.read_csv(trainfile)
validdf = pd.read_csv(validfile)
testdf = pd.read_csv(testfile)

trainfeatures = traindf.copy()
trainlabels = trainfeatures.pop(className)

testfeatures = testdf.copy()
testlabels = testfeatures.pop(className)

validfeatures = validdf.copy()
validlabels = validfeatures.pop(className)

plt.ioff()

# ========================================
# Model DL
# ========================================

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ========================================
# Feature Def
# ========================================

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

warnings.filterwarnings("ignore")

chart = sns.countplot(x=trainlabels, palette=HAPPY_COLORS_PALETTE)
plt.title("Number of texts per intent")
chart.set_xticklabels(chart.get_xticklabels(), rotation=69, horizontalalignment='right', fontsize=8)

# Get the x-axis entries
x_axis_entries = [tick.get_text() for tick in chart.get_xticklabels()]
len_dims = len(x_axis_entries)

label_encoder = LabelEncoder()

trainlabels = label_encoder.fit_transform(trainlabels.values)
testlabels = label_encoder.transform(testlabels.values)
validlabels = label_encoder.transform(validlabels.values)

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

# ========================================
# Training Loop
# ========================================

if gpuMode == 'cuda':
    model.cuda()
elif gpuMode == 'cpu':
    model.cpu()

train_losses, train_accuracies, train_f1_scores, val_losses, val_accuracies, val_f1_scores = [], [], [], [], [], []

for epoch in range(epochs):

    model.train()
    total_loss = 0
    total_accuracy = 0
    total_f1 = 0 

    for batch in train_loader:

        input_ids, attention_mask, labels = [x.to(gpuMode) for x in batch] 

        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        total_accuracy += accuracy(logits, labels)
        total_f1 += f1_score(logits, labels)

    avg_loss = total_loss / len(train_loader)
    avg_accuracy = total_accuracy / len(train_loader)
    avg_f1 = total_f1 / len(train_loader)

    train_losses.append(avg_loss)
    train_accuracies.append(avg_accuracy)
    train_f1_scores.append(avg_f1)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Training loss: {avg_loss:.4f}, Training accuracy: {avg_accuracy:.4f}, Training F1: {avg_f1:.4f}")

    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0
    total_val_f1 = 0

    with torch.no_grad():
        for batch in valid_loader:
            input_ids, attention_mask, labels = [x.to(gpuMode) for x in batch]

            logits = model(input_ids, attention_mask=attention_mask)

            val_loss = loss_fn(logits, labels)
            total_val_loss += val_loss.item()
            total_val_accuracy += accuracy(logits, labels)
            total_val_f1 += f1_score(logits, labels)

    avg_val_loss = total_val_loss / len(valid_loader)
    avg_val_accuracy = total_val_accuracy / len(valid_loader)
    avg_val_f1 = total_val_f1 / len(valid_loader)

    val_losses.append(avg_val_loss)
    val_accuracies.append(avg_val_accuracy)
    val_f1_scores.append(avg_val_f1)

    print(f"Validation loss: {avg_val_loss:.4f}, Validation accuracy: {avg_val_accuracy:.4f}, Validation F1: {avg_val_f1:.4f}")

saved_model_name = f"{customName}_{epochs}_{avg_val_accuracy:.4f}.pth"

torch.save(model.state_dict(), saved_model_name)

print(f"Validation loss: {avg_val_loss:.4f}, Validation accuracy: {avg_val_accuracy:.4f}, Validation F1: {avg_val_f1:.4f}")
print(f"Model saved to {saved_model_name}")

# ========================================
# Plotting Accuracy/F1/Loss
# ========================================
plt.ioff()
# Convert tensors to numpy arrays
train_accuracies_np = [acc.cpu().numpy() for acc in train_accuracies]
val_accuracies_np = [acc.cpu().numpy() for acc in val_accuracies]
train_losses_np = train_losses  # These should already be float values
val_losses_np = val_losses  # These should already be float values
train_f1_scores_np = [f1.cpu().numpy() for f1 in train_f1_scores]
val_f1_scores_np = [f1.cpu().numpy() for f1 in val_f1_scores]

# Plotting Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accuracies_np, label='Training Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies_np, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f'accuracy_plot_{epochs}_{avg_val_accuracy:.4f}.png')
# plt.close()
# plt.show()

# Plotting Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_losses_np, label='Training Loss')
plt.plot(range(1, epochs + 1), val_losses_np, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f'loss_plot_{epochs}_{avg_val_accuracy:.4f}.png')
# plt.close()
# plt.show()

# Plotting F1 Score
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_f1_scores_np, label='Training F1 Score')
plt.plot(range(1, epochs + 1), val_f1_scores_np, label='Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('Training and Validation F1 Score over Epochs')
plt.legend()
plt.grid(True)
plt.savefig(f'f1_score_plot_{epochs}_{avg_val_accuracy:.4f}.png')
# plt.close()
# plt.show()

print(f"Plots have been saved as 'accuracy_plot_{epochs}_{avg_val_accuracy:.4f}.png', 'loss_plot_{epochs}_{avg_val_accuracy:.4f}.png', and 'f1_score_plot_{epochs}_{avg_val_accuracy:.4f}.png'")

# ========================================
# Model testing / eval
# ========================================

# Testing the Model
model.eval()

total_test_loss = 0
total_test_accuracy = 0
total_test_f1 = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(gpuMode) for x in batch]

        logits = model(input_ids, attention_mask=attention_mask)

        test_loss = loss_fn(logits, labels)
        total_test_loss += test_loss.item()
        total_test_accuracy += accuracy(logits, labels)
        total_test_f1 += f1_score(logits, labels)

avg_test_loss = total_test_loss / len(test_loader)
avg_test_accuracy = total_test_accuracy / len(test_loader)
avg_test_f1 = total_test_f1 / len(test_loader)

print(f"Test loss: {avg_test_loss:.4f}, Test accuracy: {avg_test_accuracy:.4f}, Test F1: {avg_test_f1:.4f}")

# ========================================
# Model Inference
# ========================================

def predict_intent(text, model, tokenizer, label_encoder, device):
    # Prepare the input
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Set the model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)

    # Get the predicted class
    predicted_class = torch.argmax(logits, dim=1).item()

    # Convert the predicted class back to the original label
    predicted_intent = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_intent

# Load the saved model
model = IntentClassifier(model_name, num_labels)
model.load_state_dict(torch.load(saved_model_name))
model.to(gpuMode)

# Example usage
text = ifPrompt
predicted_intent = predict_intent(text, model, tokenizer, label_encoder, gpuMode)
print(f"The predicted intent for '{text}' is: {predicted_intent}")