import torch
from tqdm import tqdm


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

def train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, accuracy, f1_score, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    total_f1 = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
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
        for batch in tqdm(valid_loader, desc="Validation", leave=False):
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

def train_model(model, train_loader, valid_loader, optimizer, scheduler, loss_fn, accuracy, f1_score, epochs, device, customName):
    train_losses, train_accuracies, train_f1_scores = [], [], []
    val_losses, val_accuracies, val_f1_scores = [], [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_accuracy, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, accuracy, f1_score, device
        )
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)

        print(f"Training loss: {train_loss:.4f}, Training accuracy: {train_accuracy:.4f}, Training F1: {train_f1:.4f}")

        val_loss, val_accuracy, val_f1 = validate_one_epoch(
            model, valid_loader, loss_fn, accuracy, f1_score, device
        )
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        print(f"Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")

    saved_model_name = f"{customName}_{epochs}_{val_accuracies[-1]:.4f}.pth"
    torch.save(model.state_dict(), saved_model_name)

    print(f"Model saved to {saved_model_name}")
