from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_name = "distilbert/distilbert-base-uncased"  # Example model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example text
text = "This will work out"

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Perform inference
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted class
predictions = torch.argmax(outputs.logits, dim=-1)
print(f"Predicted class: {predictions.item()}")