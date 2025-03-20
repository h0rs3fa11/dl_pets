import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from config import data_dir, batch_size


def load_data():
  print("Loading training data...")
  train_data = np.load(os.path.join(data_dir, 'train_features.npz'))
  X_train, y_train = train_data['X'], train_data['y']

  print("Loading validation data...")
  val_data = np.load(os.path.join(data_dir, 'val_features.npz'))
  X_val, y_val = val_data['X'], val_data['y']

  print(f"Training set shape: {X_train.shape}")
  print(f"Validation set shape: {X_val.shape}")
  print(f"Number of classes: {len(np.unique(y_train))}")

  return X_train, y_train, X_val, y_val


def load_test_data():
  print("Loading test data...")
  test_data = np.load(os.path.join(data_dir, 'test_features.npz'))
  X_test, y_test = test_data['X'], test_data['y']

  unique_labels = np.unique(y_test)
  n_classes = len(unique_labels)
  n_features = X_test.shape[1]

  return X_test, y_test, n_features, n_classes

def prepare_data():
  X_train, y_train, X_val, y_val = load_data()
  min_label = np.min(y_train)
  max_label = np.max(y_train)
  unique_labels = np.unique(y_train)
  print(f"Label range: {min_label} to {max_label}")
  print(f"Number of unique labels: {len(unique_labels)}")

  # If labels don't start at 0 or have gaps, remap them
  label_map = {label: i for i, label in enumerate(unique_labels)}
  y_train_mapped = np.array([label_map[label] for label in y_train])
  y_val_mapped = np.array([label_map[label] for label in y_val])

  # Verify the new label range
  print(
      f"New label range: {np.min(y_train_mapped)} to {np.max(y_train_mapped)}")
  print(f"Number of unique mapped labels: {len(np.unique(y_train_mapped))}")

  # Convert to PyTorch tensors
  X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
  y_train_tensor = torch.tensor(
      y_train_mapped, dtype=torch.long)
  X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
  y_val_tensor = torch.tensor(
      y_val_mapped, dtype=torch.long)

  # Create datasets and dataloaders
  train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
  val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size)

  n_features = X_train.shape[1]
  n_classes = len(unique_labels)

  return train_loader, val_loader, n_features, n_classes
