import os, torch
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

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

def custom_collate(batch):
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    labels = torch.tensor([item[2] for item in batch])
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    
    return input_ids, attention_mask, labels

# ========================================
# Model Training Loop
# ========================================

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

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

def train_fold(save_path, model, lr, optimizer, scheduler, train_loader, valid_loader, loss_fn, accuracy, f1_score, epochs, device, customName):
    train_losses, train_accuracies, train_f1_scores = [], [], []
    val_losses, val_accuracies, val_f1_scores = [], [], []

    best_val_accuracy = 0
    best_model = None
    patience = 10
    no_improvement = 0

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

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        progress_bar.set_postfix({
            'Epoch': epoch + 1,
            'Train Loss': f'{train_loss:.4f}',
            'Train Acc': f'{train_accuracy:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val Acc': f'{val_accuracy:.4f}'
        })

    saved_model_name = os.path.join(save_path, f"{customName}_{epoch+1}_{best_val_accuracy:.4f}.pth")
    torch.save(best_model, saved_model_name)

    print(f"Model saved to {saved_model_name}")

    table_content = "| Epoch | Train Loss | Train Accuracy | Train F1 | Val Loss | Val Accuracy | Val F1 |\n"
    table_content += "|-------|------------|----------------|----------|----------|--------------|--------|\n"
    for i, (tl, ta, tf, vl, va, vf) in enumerate(zip(train_losses, train_accuracies, train_f1_scores, val_losses, val_accuracies, val_f1_scores)):
        table_content += f"| {i+1} | {tl:.4f} | {ta:.4f} | {tf:.4f} | {vl:.4f} | {va:.4f} | {vf:.4f} |\n"

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
        "epoch": epoch + 1,
        "avg_val_accuracy": best_val_accuracy,
        "train_accuracies": train_accuracies,
        "val_accuracies": val_accuracies,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_f1_scores": train_f1_scores,
        "val_f1_scores": val_f1_scores,
        "saved_model_loc": saved_model_name
    }

def train_model(save_path, model, lr, train_loader, valid_loader, optimizer, scheduler, loss_fn, accuracy, f1_score, epochs, device, customName, num_folds=None):
    if num_folds is not None and num_folds > 1:

        # Combine train_loader and valid_loader datasets
        combined_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, valid_loader.dataset])
        
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(combined_dataset))), 1):
            print(f"Fold {fold}")
            
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            fold_train_loader = torch.utils.data.DataLoader(
                combined_dataset, 
                batch_size=train_loader.batch_size, 
                sampler=train_subsampler,
                collate_fn=custom_collate
            )
            fold_valid_loader = torch.utils.data.DataLoader(
                combined_dataset, 
                batch_size=valid_loader.batch_size, 
                sampler=val_subsampler,
                collate_fn=custom_collate
            )
            
            model.apply(lambda m: m.reset_parameters() if hasattr(m, 'reset_parameters') else None)
            fold_optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
            fold_scheduler = ReduceLROnPlateau(fold_optimizer, mode='min', factor=0.1, patience=5, verbose=True)

            fold_result = train_fold(save_path, model, lr, fold_optimizer, fold_scheduler, fold_train_loader, fold_valid_loader, loss_fn, accuracy, f1_score, epochs, device, f"{customName}_fold_{fold}")
            fold_results.append(fold_result)

            (
            epoch, best_val_accuracy, train_accuracies, val_accuracies, train_losses, 
            val_losses, train_f1_scores, val_f1_scores, saved_model_name
            ) = fold_result.values()
        
        # Calculate average performance across folds
        avg_val_accuracy = sum(result['avg_val_accuracy'] for result in fold_results) / num_folds
        avg_val_f1 = sum(result['val_f1_scores'][-1] for result in fold_results) / num_folds
        
        print(f"\nAverage Validation Accuracy across folds: {avg_val_accuracy:.4f}")
        print(f"Average Validation F1 Score across folds: {avg_val_f1:.4f}")

        return {
            "epoch": epoch,
            # "avg_val_accuracy": best_val_accuracy,
            "avg_val_accuracy": avg_val_accuracy,
            "train_accuracies": train_accuracies,
            "val_accuracies": val_accuracies,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_f1_scores": train_f1_scores,
            "val_f1_scores": val_f1_scores,
            "saved_model_loc": saved_model_name
        }
    else:
        # Regular training without cross-validation
        return train_fold(save_path, model, lr, optimizer, scheduler, train_loader, valid_loader, loss_fn, accuracy, f1_score, epochs, device, customName)