# ==================================================
# Imports
# ==================================================

import os, shutil
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# ==================================================
# Basic init Defintions
# ==================================================

tf.get_logger().setLevel('ERROR')

sns.set_theme(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8
import warnings
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
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ==================================================
# Setting Features
# ==================================================

trainfeatures = traindf.copy()
trainlabels = trainfeatures.pop("Intent")
trainfeatures = trainfeatures.values

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
# LLM Call
# ==================================================

# # Example text
# text = "This will work out"

# # Tokenize the input text
# inputs = tokenizer(text, return_tensors="pt")

# # Perform inference
# with torch.no_grad():
#     outputs = model(**inputs)

# # Get the predicted class
# predictions = torch.argmax(outputs.logits, dim=-1)
# print(f"Predicted class: {predictions.item()}")

# ==================================================
# Testing the LLM
# ==================================================

# Example text from trainfeatures
text_test = trainfeatures[0]

# Tokenize the input text
inputs = tokenizer(text_test, return_tensors="pt")

# Print the keys and shapes of the tokenized inputs
print(f'Keys       : {list(inputs.keys())}')
print(f'Shape      : {inputs["input_ids"].shape}')
print(f'Word Ids   : {inputs["input_ids"][0, :12]}')
print(f'Input Mask : {inputs["attention_mask"][0, :12]}')
# Note: DistilBERT does not use token_type_ids, so this key might not be present
if "token_type_ids" in inputs:
    print(f'Type Ids   : {inputs["token_type_ids"][0, :12]}')
else:
    print('Type Ids   : Not applicable for this model')