import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer

# Set environment variables (optional but recommended)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU (change if needed)
# os.environ["HF_DATASETS_CACHE"] = "/path/to/cache"  # Optional: Set a cache directory

data_path = "...\datasets\Conversational Training\CLEANED_Mental_Health_Conversational.csv"
dataset = load_dataset('csv', data_files=data_path)

# 2. Load your base model and tokenizer
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Define preprocessing function
def preprocess_function(examples):
    # Format your data appropriately for conversational tuning
    # For example, concatenate questions and answers with a separator:
    conversations = []
    for question, answer in zip(examples['question'], examples['answer']):
        conversation = f"Question: {question}\nAnswer: {answer}"
        conversations.append(conversation)
    return tokenizer(conversations, truncation=True, padding="max_length", max_length=512)

# 4. Preprocess the dataset
dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

# 5. Define LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank (dimensionality of the updated matrices)
    lora_alpha=32,  # Scaling factor for the LoRA updates
    lora_dropout=0.05,  # Dropout probability for the LoRA layers
    bias="none",  # Whether to apply LoRA to biases ("none", "all", or "lora_only")
    task_type="CAUSAL_LM",  # Set to "CAUSAL_LM" for conversational/text generation
)

# 6. Load the base model with 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

# 7. Apply LoRA to the model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# 8. Define training arguments
training_args = TrainingArguments(
    output_dir="./lora-bart-large-cnn",
    num_train_epochs=3,  # Adjust as needed
    per_device_train_batch_size=1,  # Adjust based on your GPU memory
    gradient_accumulation_steps=4,  # Adjust based on your GPU memory
    learning_rate=1e-4,  # Adjust as needed
    fp16=True,  # Enable mixed precision training
    save_total_limit=2,  # Only save the last 2 checkpoints
)

# 9. Create the Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)
trainer.train()

# 10. Save the LoRA model
trainer.save_model("./lora-bart-large-cnn")