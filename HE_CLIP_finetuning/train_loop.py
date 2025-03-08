import os
import numpy as np
from he_fashion_classifier import HEFashionClassifier
from he_model_IO import HEModelIO
from helper import preprocess_labels

data_dir = "extracted_features"

# Load the extracted features
print("Loading training data...")
train_data = np.load(os.path.join(data_dir, 'train_features.npz'))
X_train, y_train = train_data['X'], train_data['y']
print("Loading validation data...")
val_data = np.load(os.path.join(data_dir, 'val_features.npz'))
X_val, y_val = val_data['X'], val_data['y']
print("Loading test data...")
test_data = np.load(os.path.join(data_dir, 'test_features.npz'))
X_test, y_test = test_data['X'], test_data['y']

# Map labels if needed (as in your original code)
unique_labels = np.unique(y_train)
label_map = {label: idx for idx, label in enumerate(unique_labels)}
y_train_mapped = np.array([label_map[label] for label in y_train])
y_val_mapped = np.array([label_map[label] for label in y_val])
y_test_mapped = np.array([label_map[label] for label in y_test])

# Get dimensions from the data
n_features = X_train.shape[1]
n_classes = len(unique_labels)
print(f"Features: {n_features}, Classes: {n_classes}")


model = HEFashionClassifier(n_features, n_classes)

y_train_mapped, y_val_mapped, n_classes, label_map = preprocess_labels(y_train, y_val)

epochs = 1

while(epochs < 3):
  if epochs == 1:
    # Create and train HE model
    model = HEFashionClassifier(n_features, n_classes)
  else:
    model = HEModelIO.load_model(os.path.join(data_dir, f'he_fashion_model_{epochs-1}.json'))

  history = model.train_with_reduced_encryption(X_train, y_train_mapped, 1, learning_rate=0.01)
  epochs += 1