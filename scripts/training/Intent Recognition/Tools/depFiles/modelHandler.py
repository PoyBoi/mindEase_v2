import os, torch
from tqdm import tqdm
from datetime import datetime

def model_eval(
    model, test_loader, gpuMode, 
    loss_fn, accuracy, f1_score
):
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

def predict_intent(
    text, model, tokenizer, label_encoder, 
    gpuMode, IntentClassifier, model_name, num_labels, saved_model_name
):

    device = torch.device("cuda" if gpuMode=='cuda' else "cpu")

    model = IntentClassifier(model_name, num_labels)
    model.load_state_dict(torch.load(saved_model_name))
    model.to(gpuMode)

    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    model.eval()

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask)

    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_intent = label_encoder.inverse_transform([predicted_class])[0]

    print(f"The predicted intent for '{text}' is: {predicted_intent}")

# ========================================
# Model Training Loop
# ========================================

import torch
from tqdm import tqdm

def train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, accuracy, f1_score, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_f1 = 0

    for batch in train_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

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

    return avg_loss, avg_accuracy, avg_f1

def validate_one_epoch(model, valid_loader, loss_fn, accuracy, f1_score, device):
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0
    total_val_f1 = 0

    with torch.no_grad():
        for batch in valid_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]

            logits = model(input_ids, attention_mask=attention_mask)
            val_loss = loss_fn(logits, labels)

            total_val_loss += val_loss.item()
            total_val_accuracy += accuracy(logits, labels)
            total_val_f1 += f1_score(logits, labels)

    avg_val_loss = total_val_loss / len(valid_loader)
    avg_val_accuracy = total_val_accuracy / len(valid_loader)
    avg_val_f1 = total_val_f1 / len(valid_loader)

    return avg_val_loss, avg_val_accuracy, avg_val_f1

import os
from datetime import datetime
import torch
from tqdm import tqdm

def train_model(
        save_path, model, lr,
        train_loader, valid_loader, optimizer, scheduler, 
        loss_fn, accuracy, f1_score, epochs, device, customName
    ):
    train_losses, train_accuracies, train_f1_scores = [], [], []
    val_losses, val_accuracies, val_f1_scores = [], [], []

    progress_bar = tqdm(range(epochs), desc="Training Progress")
    for epoch in progress_bar:
        train_loss, train_accuracy, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, accuracy, f1_score, device
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)

        val_loss, val_accuracy, val_f1 = validate_one_epoch(
            model, valid_loader, loss_fn, accuracy, f1_score, device
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        progress_bar.set_postfix({
            'Epoch': epoch + 1,
            'Train Loss': f'{train_loss:.4f}',
            'Train Acc': f'{train_accuracy:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val Acc': f'{val_accuracy:.4f}'
        })

    saved_model_name = os.path.join(save_path, f"{customName}_{epochs}_{val_accuracies[-1]:.4f}.pth")
    torch.save(model.state_dict(), saved_model_name)

    print(f"Model saved to {saved_model_name}")

    # Create the table content separately
    table_content = "| Epoch | Train Loss | Train Accuracy | Train F1 | Val Loss | Val Accuracy | Val F1 |\n"
    table_content += "|-------|------------|----------------|----------|----------|--------------|--------|\n"
    for i, (tl, ta, tf, vl, va, vf) in enumerate(zip(train_losses, train_accuracies, train_f1_scores, val_losses, val_accuracies, val_f1_scores)):
        table_content += f"| {i+1} | {tl:.4f} | {ta:.4f} | {tf:.4f} | {vl:.4f} | {va:.4f} | {vf:.4f} |\n"

    # Create and write the Markdown file
    md_content = f"""
# Model Training Information

## Model Name
{customName}

## Training Date
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Architecture
```python
{model.__class__.__name__}
{str(model)}
```

## Training Parameters
- Epochs: {epochs}
- Optimizer: {optimizer.__class__.__name__}
- Learning Rate: {lr}
- Loss Function: {loss_fn.__class__.__name__}
- Device: {device}

## Learning Rate Scheduler
{scheduler.__class__.__name__}

## Performance Metrics
### Training
- Final Loss: {train_losses[-1]:.4f}
- Final Accuracy: {train_accuracies[-1]:.4f}
- Final F1 Score: {train_f1_scores[-1]:.4f}

### Validation
- Final Loss: {val_losses[-1]:.4f}
- Final Accuracy: {val_accuracies[-1]:.4f}
- Final F1 Score: {val_f1_scores[-1]:.4f}

## Training Progress
{table_content}
## Data Loaders
- Training Samples: {len(train_loader.dataset)}
- Validation Samples: {len(valid_loader.dataset)}
- Batch Size: {train_loader.batch_size}

## Additional Notes
- Custom Name: {customName}
- Saved Model Path: {saved_model_name}
"""

    md_file_path = os.path.join(save_path, f"{customName}_training_info.md")
    with open(md_file_path, 'w') as f:
        f.write(md_content)

    print(f"Training information saved to {md_file_path}")

    return {
        "epoch": epochs,
        "avg_val_accuracy": val_accuracies[-1],
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_f1_scores": train_f1_scores,
        "val_f1_scores": val_f1_scores,
        "saved_model_loc": saved_model_name
    }