import torch
from nn_model import TorchFashionClassifier
from data_prepare import prepare_data
from config import learning_rate, weight_decay, epoches

train_loader, val_loader, n_features, n_classes = prepare_data()
model = TorchFashionClassifier(n_features, n_classes)

optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = torch.nn.MSELoss()

for epoch in range(epoches):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)

        # Convert labels to one-hot encoding
        targets = torch.nn.functional.one_hot(
            labels, num_classes=outputs.shape[1]).float()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            targets = torch.nn.functional.one_hot(
                labels, num_classes=outputs.shape[1]).float()
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total

    print(
        f"Epoch {epoch+1}/{epoches}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
