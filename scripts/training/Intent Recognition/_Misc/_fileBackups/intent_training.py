import json, os
from rasa.engine import training
from rasa.shared.nlu.training_data.formats.rasa import RasaReader
# from rasa.engine.training import 

# Load the JSON dataset
# try:
with open(r"_Projects_Personal\mindEase_v2\datasets\Conversational Training\Conversation_Intents.json", "r") as f:
    raw_training_data = json.load(f)

print("---> Dataset Loaded")

# 2. Convert JSON to Rasa training data format (optional but recommended) 
training_data = []
for intent in raw_training_data["intents"]:
    for pattern in intent["patterns"]:
        training_data.append({"text": pattern, "intent": intent["tag"]})

print("---> Data Loaded")

# 3. Save the training data to a temporary Rasa training file 
with open(r"_Projects_Personal\mindEase_v2\datasets\temp_training_data.yml", "w") as f:
    for item in training_data:
        f.write(f"- {item['intent']}: {item['text']}\n")

print("---> Temp file written, Starting Training")

# 4. Train the Rasa NLU model
# training.train(
#     training_data="temp_training_data.yml",  # Use the temporary file
#     output="models",  # Directory to save the trained model
#     config="config.yml", # Your Rasa NLU pipeline configuration
#     force_training=True  # Overwrite existing models (optional)
# )
train_nlu(
    domain="domain.yml",  # Path to your domain file (even if empty)
    config="config.yml",  # Path to your NLU config file 
    training_data="temp_training_data.yml", # Path to your training data
    output="models", # Directory to save the trained model 
)

print("---> TRAINING COMPLETE <---")

# except Exception as e:
#     print(f"---> Error Encountered: {e}")

# message = "Hello, I want to order a pizza"
# result = interpreter.parse(message)
# print(result)