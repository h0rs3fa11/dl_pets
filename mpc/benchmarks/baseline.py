import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import numpy as np
from tqdm import tqdm

from mpc_benchmarks import TorchFashionClassifier

parent_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))


def run_plaintext_training(
    X_train_pt,
    labels_one_hot_pt,
    X_val_pt,
    val_labels_one_hot_pt,
    n_features,
    n_classes,
    TorchFashionClassifier_class,
    learning_rate=0.1,
    num_epochs=20,
    batch_size=64,
    opt="ce",
):
    print("--- Starting Plaintext PyTorch Training ---")

    model_plain = TorchFashionClassifier_class(n_features, n_classes)
    # model_plain.to(device)

    if opt == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    num_batches = X_train_pt.size(0) // batch_size

    optimizer = optim.SGD(model_plain.parameters(),
                          lr=learning_rate, momentum=0.99)

    print(f"Plaintext Training Data (Features) Shape: {X_train_pt.size()}")
    print(
        f"Plaintext Training Data (Labels) Shape: {labels_one_hot_pt.size()}")
    print(f"Plaintext Validation Data (Features) Shape: {X_val_pt.size()}")
    print(
        f"Plaintext Validation Data (Labels) Shape: {val_labels_one_hot_pt.size()}")
    print(f"Number of features: {n_features}, Number of classes: {n_classes}")
    print(
        f"Learning rate: {learning_rate}, Epochs: {num_epochs}, Batch size: {batch_size}, Batches per epoch: {num_batches}")

    train_loss = []
    val_losses = []
    val_accuracies = []
    epoch_times = []

    for i in range(num_epochs):
        model_plain.train()
        print(f"Epoch {i+1}/{num_epochs} in progress:")
        epoch_start_time = time.time()
        batch_losses = []

        for batch_idx in tqdm(range(num_batches)):
            start = batch_idx * batch_size
            end = (batch_idx + 1) * batch_size

            # Get mini-batch
            x_batch_pt = X_train_pt[start:end]
            y_batch_pt = labels_one_hot_pt[start:end]
            y_batch_indices = torch.argmax(y_batch_pt, dim=1)

            # x_batch_pt, y_batch_pt = x_batch_pt.to(device), y_batch_pt.to(device)
            x_batch_pt = x_batch_pt.float()
            optimizer.zero_grad()

            outputs_pt = model_plain(x_batch_pt)

            if opt == "ce":
                loss_value_pt = criterion(outputs_pt, y_batch_indices)
            else:
                loss_value_pt = criterion(outputs_pt, y_batch_pt)

            loss_value_pt.backward()

            optimizer.step()

            batch_losses.append(loss_value_pt.item())

        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = sum(batch_losses) / \
            len(batch_losses) if batch_losses else 0

        # --- Validation Phase ---
        # if X_val_pt is not None and val_labels_one_hot_pt is not None:
        val_true_indices_for_loss = torch.argmax(val_labels_one_hot_pt, dim=1)
        model_plain.eval()
        print(f"\nEpoch {i+1}/{num_epochs} (Validation) in progress:")
        X_val_pt = X_val_pt.float()
        with torch.no_grad():
            val_outputs_pt = model_plain(X_val_pt)

            if opt == "mse":
                val_loss_pt = criterion(
                        val_outputs_pt, val_labels_one_hot_pt)
            else:
                val_loss_pt = criterion(
                        val_outputs_pt, val_true_indices_for_loss)

            val_loss_item = val_loss_pt.item()

            # Compute Accuracy
            val_predicted_indices = torch.argmax(val_outputs_pt, dim=1)
            val_true_indices = torch.argmax(val_labels_one_hot_pt, dim=1)

            correct_predictions = (
                    val_predicted_indices == val_true_indices).sum().item()
            total_validation_samples = val_true_indices.numel()

            val_accuracy = 0
            if total_validation_samples > 0:
                    val_accuracy = correct_predictions / total_validation_samples
            else:
                print(
                        "Warning: Total validation samples is 0. Cannot compute accuracy.")

            print(f"Epoch {i+1} Validation Loss: {val_loss_item:.4f}")
            print(f"Epoch {i+1} Validation Accuracy: {val_accuracy:.4f}")

        print(
            f"\tEpoch {i+1}: Avg Training Loss {avg_epoch_loss:.6f}  Time: {epoch_duration:.2f}s")
        train_loss.append(avg_epoch_loss)
        val_losses.append(val_loss_item)
        val_accuracies.append(val_accuracy)
        epoch_times.append(epoch_duration)
        print("--- End of Epoch ---")

    history = {
        "train_loss": train_loss,
        "val_loss": val_losses,
        "accuracy": val_accuracies,
        "epoch_times": epoch_times
    }

    result_dir = os.path.join(parent_dir, 'data', 'models', 'mpc')

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    with open(os.path.join(result_dir, f'plaintext_training_history_{opt}.json'), 'w') as f:
        json.dump(history, f)


train_data = np.load(os.path.join(parent_dir, 'data',
                     'extracted_features', 'train_features.npz'))
X_train, y_train = train_data['X'], train_data['y']
val_data = np.load(os.path.join(parent_dir, 'data',
                   'extracted_features', 'val_features.npz'))
X_val, y_val = val_data['X'], val_data['y']

min_label = np.min(y_train)
max_label = np.max(y_train)
unique_labels = np.unique(y_train)

label_map = {label: i for i, label in enumerate(unique_labels)}
y_train_mapped = np.array([label_map[label] for label in y_train])
y_val_mapped = np.array([label_map[label] for label in y_val])

n_classes = len(unique_labels)
n_features = X_train.shape[1]

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train_mapped).long()

X_val = torch.from_numpy(X_val)
y_val = torch.from_numpy(y_val_mapped).long()

label_eye = torch.eye(n_classes)
labels = y_train.long()
labels_one_hot = label_eye[labels]

# label_eye = torch.eye(n_classes)
val_labels = y_val.long()
val_labels_one_hot = label_eye[val_labels]

trained_plaintext_model = run_plaintext_training(
    X_train,
    labels_one_hot,
    X_val,
    val_labels_one_hot,
    n_features,
    n_classes,
    TorchFashionClassifier,
    learning_rate=0.1,
    num_epochs=20,
    batch_size=64,
    opt="ce"
)
