import torch
import time
import numpy as np
from abc import ABC, abstractmethod
from config import learning_rate, batch_size, epoches


class TorchFashionClassifier(torch.nn.Module):
    def __init__(self, n_features, n_classes):
        super(TorchFashionClassifier, self).__init__()
        self.classifier = torch.nn.Linear(n_features, n_classes)

    def forward(self, x):
        return self.classifier(x)


class FashionClassifier(ABC):
    def __init__(self, n_features, n_classes):
        std = np.sqrt(2.0 / n_features)
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights = np.random.normal(0, std, (n_features, n_classes))
        self.bias = np.zeros(n_classes)

        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

    def adam_update(self, weight_grads, bias_grads, learning_rate):
        self.t += 1

        # Update weight parameters
        self.m_weights = self.beta1 * self.m_weights + \
            (1 - self.beta1) * weight_grads
        self.v_weights = self.beta2 * self.v_weights + \
            (1 - self.beta2) * (weight_grads ** 2)
        m_weights_hat = self.m_weights / (1 - self.beta1 ** self.t)
        v_weights_hat = self.v_weights / (1 - self.beta2 ** self.t)
        self.weights -= learning_rate * m_weights_hat / \
            (np.sqrt(v_weights_hat) + self.epsilon)

        # Update bias parameters
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * bias_grads
        self.v_bias = self.beta2 * self.v_bias + \
            (1 - self.beta2) * (bias_grads ** 2)
        m_bias_hat = self.m_bias / (1 - self.beta1 ** self.t)
        v_bias_hat = self.v_bias / (1 - self.beta2 ** self.t)
        self.bias -= learning_rate * m_bias_hat / \
            (np.sqrt(v_bias_hat) + self.epsilon)

    @abstractmethod
    def compute_loss(self, outputs, label):
        pass

    @abstractmethod
    def compute_gradients(self, x, outputs, label):
        pass

    def train(self, X_train, y_train, X_val, y_val, epochs=epoches, learning_rate=learning_rate, batch_size=batch_size, verbose=False):
        X_train = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
        y_train = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train

        X_val = X_val.numpy() if isinstance(X_val, torch.Tensor) else X_val
        y_val = y_val.numpy() if isinstance(y_val, torch.Tensor) else y_val

        n_samples = len(X_train)
        history = {'train_loss': [], 'val_loss': [],
                   'val_accuracy': [], 'batch_times': []}

        print(
            f"Training set size: {n_samples} samples with {X_train.shape[1]} features")
        print(f"Validation set size: {len(X_val)} samples")
        print(f"Batch size: {batch_size}")

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_start = time.time()

            # Shuffle indices for this epoch
            indices = np.random.permutation(n_samples)

            # Prepare data in batches
            n_batches = (n_samples + batch_size - 1) // batch_size

            for batch_idx in range(n_batches):
                batch_start = time.time()

                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                if start_idx >= end_idx:
                    continue
                batch_indices = indices[start_idx:end_idx]
                actual_batch_size = len(batch_indices)

                if verbose:
                    print(f"  Processing batch {batch_idx+1}/{n_batches}...")

                # Get batch data
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]

                # Initialize gradient accumulators
                weight_grads_acc = np.zeros_like(self.weights)
                bias_grads_acc = np.zeros_like(self.bias)
                batch_loss = 0.0

                # Process each sample in the batch
                for i in range(actual_batch_size):
                    x = batch_X[i]
                    y = batch_y[i]

                    # Forward pass
                    outputs = self.forward(x)

                    # Compute loss
                    loss_value = self.compute_loss(outputs, y)
                    batch_loss += loss_value

                    # Compute gradients
                    weight_grads, bias_grads = self.compute_gradients(
                        x, outputs, y)

                    # Accumulate gradients
                    weight_grads_acc += weight_grads
                    bias_grads_acc += bias_grads

                # Average gradients
                weight_grads_acc /= actual_batch_size
                bias_grads_acc /= actual_batch_size

                # Update model parameters using Adam
                self.adam_update(weight_grads_acc,
                                 bias_grads_acc, learning_rate)

                # Track metrics
                avg_batch_loss = batch_loss / actual_batch_size
                epoch_loss += batch_loss

                # Record batch timing
                batch_time = time.time() - batch_start
                history['batch_times'].append(batch_time)

                if verbose:
                    print(
                        f"  Batch {batch_idx+1}/{n_batches} completed in {batch_time:.2f}s, loss: {avg_batch_loss:.6f}")

            # Calculate training loss for the epoch
            avg_epoch_loss = epoch_loss / n_samples
            history['train_loss'].append(avg_epoch_loss)

            # Validation phase
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            for i in range(len(X_val)):
                x = X_val[i]
                y = y_val[i]

                # Forward pass
                outputs = self.forward(x)

                # Compute validation loss
                loss_value = self.compute_loss(outputs, y)
                val_loss += loss_value

                # Compute accuracy
                predicted = np.argmax(outputs)
                val_total += 1
                if predicted == y:
                    val_correct += 1

            # Calculate validation metrics
            avg_val_loss = val_loss / len(X_val)
            val_accuracy = val_correct / val_total

            # Store validation metrics
            history['val_loss'].append(avg_val_loss)
            history['val_accuracy'].append(val_accuracy)

            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{epochs}: Train Loss: {avg_epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f}, Val Accuracy: {val_accuracy:.4f}   Time: {epoch_time:.2f}s ({n_samples/epoch_time:.2f} samples/sec)")

        return history

    def evaluate(self, X_test, y_test, verbose=False):
        """Evaluate model on test data."""
        X_test = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_test = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test

        n_samples = len(X_test)
        predictions = np.zeros(n_samples, dtype=int)

        print(f"Starting evaluation on {n_samples} samples...")
        start_time = time.time()

        # Find the actual number of classes in the test data
        unique_labels = np.unique(y_test)
        max_label = int(max(unique_labels))
        actual_n_classes = max(self.n_classes, max_label + 1)

        # Track per-class accuracy with arrays large enough for all classes
        class_correct = np.zeros(actual_n_classes)
        class_total = np.zeros(actual_n_classes)

        # Process batches for faster evaluation
        batch_size = 100
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            print(
                f"Evaluating batch {batch_idx+1}/{n_batches} (samples {start_idx+1}-{end_idx})...", end="\r")

            for idx in range(start_idx, end_idx):
                x = X_test[idx]
                true_label = int(y_test[idx])

                # Skip invalid labels
                if true_label >= actual_n_classes:
                    print(
                        f"Warning: Found label {true_label} which is outside the expected range. Skipping.")
                    continue

                outputs = self.forward(x)

                # Get predicted class
                predicted_class = np.argmax(outputs)
                predictions[idx] = predicted_class

                # Update class-specific accuracy
                class_total[true_label] += 1
                if predicted_class == true_label:
                    class_correct[true_label] += 1

        # Compute overall accuracy
        accuracy = np.sum(class_correct) / np.sum(class_total)

        # Display evaluation results
        eval_time = time.time() - start_time
        print(f"\nEvaluation completed in {eval_time:.2f}s")
        print(f"Overall accuracy: {accuracy:.4f}")

        # Print confusion matrix if scikit-learn is available
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, predictions)
            print("\nConfusion Matrix (first 10x10 section):")
            print(cm[:10, :10])  # Show just a portion for readability
        except:
            pass

        if verbose:
            print("\nPer-class accuracy (for classes with samples):")
            for i in range(actual_n_classes):
                if class_total[i] > 0:
                    class_acc = class_correct[i] / class_total[i]
                    print(
                        f"  Class {i}: {class_acc:.4f} ({int(class_correct[i])}/{int(class_total[i])})")

        return accuracy


class MSEFashionClassifier(FashionClassifier):
    def compute_loss(self, outputs, label):
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1.0

        diff = outputs - one_hot
        return np.sum(diff * diff) / self.n_classes

    def compute_gradients(self, x, outputs, label):
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1.0

        output_grad = 2 * (outputs - one_hot) / self.n_classes

        weight_gradients = np.outer(x, output_grad)

        bias_gradients = output_grad

        return weight_gradients, bias_gradients


class CEFashionClassifier(FashionClassifier):
    def softmax(self, x):
        shifted_x = x - np.max(x)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x)

    def compute_loss(self, outputs, label):
        probabilities = self.softmax(outputs)
        return -np.log(probabilities[label] + self.epsilon)

    def compute_gradients(self, x, outputs, label):
        """Compute gradients for cross entropy loss."""
        probabilities = self.softmax(outputs)

        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1.0

        # Gradient of cross entropy with respect to logits is (p - y)
        output_grad = probabilities - one_hot

        weight_gradients = np.outer(x, output_grad)
        bias_gradients = output_grad

        return weight_gradients, bias_gradients
