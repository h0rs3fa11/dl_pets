import numpy as np
import tenseal as ts
import torch
import time
from sklearn.metrics import accuracy_score

class HEFashionClassifier:
    def __init__(self, n_features, n_classes):
        """
        Initialize a homomorphic encryption compatible fashion classifier.

        Args:
            n_features: Number of input features
            n_classes: Number of output classes
        """
        self.context = self._create_context()

        # Initialize plaintext weights and biases
        # numpy arrays for simplicity
        self.weights = np.random.normal(0, 0.1, (n_features, n_classes))
        self.bias = np.random.normal(0, 0.1, n_classes)

        self.n_features = n_features
        self.n_classes = n_classes

    def _create_context(self):
        """Create a TenSEAL context with appropriate parameters for our task."""
        # # Higher gives more precision but slower
        # poly_mod_degree = 16384
        # # More levels for deeper computation
        # coeff_mod_bit_sizes = [60, 40, 40, 40, 60]
        poly_mod_degree = 8192
        # poly_mod_degree = 4096
        # More levels for deeper computation
        # coeff_mod_bit_sizes = [60, 40, 40, 60]
        coeff_mod_bit_sizes = [60, 40, 40, 60]

        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_mod_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )

        # higher gives more precision
        context.global_scale = 2**30
        context.generate_galois_keys()

        return context

    def encrypt_data(self, data):
        """
        Encrypt input data using TenSEAL.

        Args:
            data: NumPy array to encrypt

        Returns:
            Encrypted TenSEAL vector
        """
        data_list = data.tolist() if isinstance(data, np.ndarray) else data
        encrypted_data = ts.ckks_vector(self.context, data_list)

        return encrypted_data

    def encrypt_one_hot(self, label):
        """
        Encrypt a label as one-hot encoded vector.

        Args:
            label: Class index

        Returns:
            Encrypted one-hot vector
        """
        one_hot = np.zeros(self.n_classes)
        one_hot[label] = 1.0

        return self.encrypt_data(one_hot)

    def forward(self, encrypted_x):
        """
        Forward pass with encrypted input and plaintext weights.

        Args:
            encrypted_x: Encrypted input vector

        Returns:
            List of encrypted logits for each class
        """
        encrypted_outputs = []

        for class_idx in range(self.n_classes):
            class_weights = self.weights[:, class_idx].tolist()
            starttime = time.time()
            result = encrypted_x.dot(class_weights)
            # print(f"dot product {time.time() - starttime}")
            result = result + self.bias[class_idx]

            encrypted_outputs.append(result)

        return encrypted_outputs

    def forward_plaintext(self, x):
        """
        Forward pass using plaintext operations for faster evaluation.

        Args:
            x: Input feature vector (plaintext)

        Returns:
            Array of logits for each class
        """
        # Convert input to numpy array if it's not already
        if isinstance(x, torch.Tensor):
            x = x.numpy()

        # Initialize array to store outputs for each class
        outputs = np.zeros(self.n_classes)

        # Compute logits for each class
        for class_idx in range(self.n_classes):
            # Extract weights for this class
            class_weights = self.weights[:, class_idx]

            # Compute dot product (plaintext)
            result = np.dot(x, class_weights) + self.bias[class_idx]
            outputs[class_idx] = result

        return outputs

    # Approximation functions
    def approximate_exp(self, encrypted_x, degree=3):
        """Polynomial approximation of exp(x) using Taylor series"""
        # Create a constant vector of 1s
        result = ts.ckks_vector(self.context, [1.0])

        # Start with x as the first term
        term = encrypted_x.copy()
        result = result + term

        # Compute higher-order terms
        x_power = encrypted_x.copy()
        factorial = 1.0

        for i in range(2, degree + 1):
            factorial *= i
            x_power = x_power * encrypted_x  # Compute x^i
            term = x_power * (1.0 / factorial)  # Scale by 1/i!
            result = result + term

    def approximate_softmax(self, encrypted_logits, degree=3):
        # Apply exp approximation to each logit
        exp_approx = [self.approximate_exp(self.context, logit, degree)
                      for logit in encrypted_logits]

        # Compute sum (we need to add all vectors)
        exp_sum = exp_approx[0].copy()
        for i in range(1, len(exp_approx)):
            exp_sum = exp_sum + exp_approx[i]

        sum_val = exp_sum.decrypt()[0]
        recip = 1.0 / sum_val

        # Scale each exp_approx by the reciprocal
        softmax_result = [exp_val * recip for exp_val in exp_approx]

        return softmax_result

    def approximate_cross_entropy(self, encrypted_logits, labels, exp_degree=3):
        """Approximate cross-entropy using only additions and multiplications"""
        # Get softmax approximation
        softmax_probs = self.approximate_softmax(
            self.context, encrypted_logits, exp_degree)

        # Extract probability for the true class
        # In a real HE setting, you would use encrypted operations here
        true_probs = []
        for i, label in enumerate(labels):
            if label == 1:
                true_probs.append(softmax_probs[i])

        # Compute negative log approximation
        # Simplified: decrypt, compute log, re-encrypt
        # In a real setting, you would use polynomial approximation
        true_prob_val = true_probs[0].decrypt()[0]
        neg_log_val = -np.log(true_prob_val)

        return neg_log_val

    def compute_encrypted_gradients(self, encrypted_x, encrypted_outputs, plaintext_lable):
        """
        Compute gradients while maintaining encryption of data.

        For cross-entropy loss, gradient simplifies to pred - label,
        which we can compute entirely in the encrypted domain.

        Args:
            encrypted_x: Encrypted input
            encrypted_outputs: Encrypted model outputs
            encrypted_label: Encrypted one-hot label

        Returns:
            Tuple of (weight_gradients, bias_gradients) - these are decrypted
            since weights and biases are plaintext
        """
        # Initialize gradients
        weight_gradients = np.zeros_like(self.weights)
        bias_gradients = np.zeros_like(self.bias)

        # Decrypt label to get one-hot vector
        # decrypted_label = encrypted_label.decrypt()
        one_hot = np.zeros(self.n_classes)
        one_hot[plaintext_lable] = 1.0

        encrypted_features = encrypted_x

        # For each class, compute gradient
        for class_idx in range(self.n_classes):
            # Create encrypted version of this label component
            true_label_val = one_hot[class_idx]
            # true_label = ts.ckks_vector(self.context, [true_label_val])

            # Gradient of output wrt loss: pred - label (approximation of softmax cross-entropy)
            decrypted_pred = encrypted_outputs[class_idx].decrypt()[0]
            output_grad = decrypted_pred - true_label_val

            # Bias gradient is just the output gradient
            # Need to decrypt since bias is plaintext
            # bias_gradients[class_idx] = output_grad.decrypt()[0]
            bias_gradients[class_idx] = output_grad

            # For weight gradients:
            # Multiply each encrypted feature by the output gradient (scalar)
            # This is supported by CKKS homomorphic encryption
            encrypted_grad_vector = encrypted_features * output_grad

            # Now decrypt the gradient vector (only once per class)
            decrypted_grad_vector = encrypted_grad_vector.decrypt()

            # Set the weight gradients for this class
            for feature_idx in range(self.n_features):
                weight_gradients[feature_idx, class_idx] = decrypted_grad_vector[feature_idx]

        return weight_gradients, bias_gradients

    def train_with_reduced_encryption(self, X_train, y_train, epochs=10, learning_rate=0.01, batch_size=16):
        """
        Train the model with reduced encryption/decryption operations.

        Args:
            X_train: Training data features
            y_train: Training data labels
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            batch_size: Size of mini-batches for training

        Returns:
            Training history (losses)
        """
        # Convert training data to numpy if needed
        X_train = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
        y_train = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train

        n_samples = len(X_train)
        history = {'loss': [], 'batch_times': []}

        print(f"Starting optimized training with reduced encryption for {epochs} epochs")
        print(f"Training set size: {n_samples} samples with {X_train.shape[1]} features")
        print(f"Batch size: {batch_size}")
        total_start_time = time.time()

        # Precompute indices for batches
        n_batches = (n_samples + batch_size - 1) // batch_size

        # Pre-encrypt the entire dataset once
        # This is a significant optimization if we have enough memory
        # print("Pre-encrypting labels as one-hot vectors...")
        # encrypted_labels = []
        # for idx in range(n_samples):
        #     y = int(y_train[idx])
        #     encrypted_y = self.encrypt_one_hot(y)
        #     encrypted_labels.append(encrypted_y)
        # print("Labels pre-encrypted.")

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_start = time.time()
            print(f"\nEpoch {epoch+1}/{epochs}")

            # Shuffle indices for this epoch
            indices = np.random.permutation(n_samples)

            for batch_idx in range(n_batches):
                batch_start = time.time()

                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                actual_batch_size = len(batch_indices)

                print(f"  Processing batch {batch_idx+1}/{n_batches} (samples {start_idx+1}-{end_idx})...")

                # Initialize gradient accumulators
                weight_grads_acc = np.zeros_like(self.weights)
                bias_grads_acc = np.zeros_like(self.bias)
                batch_loss = 0.0

                # Process each sample in the batch
                for i, idx in enumerate(batch_indices):
                    x = X_train[idx]

                    # Only encrypt the input once per sample
                    encrypted_x = self.encrypt_data(x)

                    # Get pre-encrypted label
                    # encrypted_y = encrypted_labels[idx]
                    plaintext_y = y_train[idx]

                    # Forward pass
                    encrypted_outputs = self.forward(encrypted_x)

                    # Compute loss (decrypt only for monitoring)
                    # encrypted_loss = self.approximate_cross_entropy(encrypted_outputs, encrypted_y)
                    loss_value = self.approximate_cross_entropy(encrypted_outputs, plaintext_y)
                    # loss_value = encrypted_loss.decrypt()[0]
                    batch_loss += loss_value

                    # Compute gradients (minimize decryption)
                    weight_grads, bias_grads = self.compute_encrypted_gradients(
                        encrypted_x, encrypted_outputs, plaintext_y)

                    # Accumulate gradients
                    weight_grads_acc += weight_grads
                    bias_grads_acc += bias_grads

                    # Print individual sample progress occasionally
                    # if i < 2 or i == actual_batch_size - 1:
                    #     print(f"    Sample {i+1}/{actual_batch_size} processed, loss: {loss_value:.6f}")

                # Average and apply gradients once per batch
                weight_grads_acc /= actual_batch_size
                bias_grads_acc /= actual_batch_size

                # Update model parameters
                self.weights -= learning_rate * weight_grads_acc
                self.bias -= learning_rate * bias_grads_acc

                # Track metrics
                avg_batch_loss = batch_loss / actual_batch_size
                epoch_loss += batch_loss

                # Record batch timing
                batch_time = time.time() - batch_start
                history['batch_times'].append(batch_time)

                # Print batch summary
                print(f"  Batch {batch_idx+1}/{n_batches} completed in {batch_time:.2f}s, loss: {avg_batch_loss:.6f}")

            # Epoch summary
            avg_epoch_loss = epoch_loss / n_samples
            history['loss'].append(avg_epoch_loss)

            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Loss: {avg_epoch_loss:.6f}")
            print(f"  Time: {epoch_time:.2f}s ({n_samples/epoch_time:.2f} samples/sec)")

            # Estimate remaining time
            remaining_epochs = epochs - (epoch + 1)
            if remaining_epochs > 0:
                est_remaining_time = remaining_epochs * epoch_time
                print(f"  Estimated time remaining: {est_remaining_time:.2f}s ({est_remaining_time/60:.2f} minutes)")

        total_time = time.time() - total_start_time
        print(f"\nTraining completed in {total_time:.2f}s ({total_time/60:.2f} minutes)")

        return history

    def predict(self, X_test):
        """
        Make predictions using the trained model.

        Args:
            X_test: Test data features

        Returns:
            Predicted class indices
        """
        # Convert to numpy if needed
        X_test = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test

        n_samples = len(X_test)
        predictions = np.zeros(n_samples, dtype=int)

        for idx in range(n_samples):
            if idx % 100 == 0:
                print(f"Predicting sample {idx}/{n_samples}")

            # Encrypt input
            x = X_test[idx]
            encrypted_x = self.encrypt_data(x)

            # Forward pass
            encrypted_outputs = self.forward(encrypted_x)

            # Decrypt outputs to get logits
            decrypted_outputs = [out.decrypt()[0] for out in encrypted_outputs]

            # Get predicted class
            predictions[idx] = np.argmax(decrypted_outputs)

        return predictions

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.

        Args:
            X_test: Test data features
            y_test: Test data labels

        Returns:
            Accuracy score
        """
        # Convert to numpy if needed
        X_test = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_test = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test

        n_samples = len(X_test)
        predictions = np.zeros(n_samples, dtype=int)

        print(f"Starting evaluation on {n_samples} samples...")
        start_time = time.time()

        # Track per-class accuracy
        class_correct = np.zeros(self.n_classes)
        class_total = np.zeros(self.n_classes)

        for idx in range(n_samples):
            sample_start = time.time()
            print(f"Evaluating sample {idx+1}/{n_samples}...", end="\r")

            # Encrypt input
            x = X_test[idx]
            true_label = int(y_test[idx])
            encrypted_x = self.encrypt_data(x)

            # Forward pass
            encrypted_outputs = self.forward(encrypted_x)

            # Decrypt outputs to get logits
            decrypted_outputs = [out.decrypt()[0] for out in encrypted_outputs]

            # Get predicted class
            predicted_class = np.argmax(decrypted_outputs)
            predictions[idx] = predicted_class

            # Update class-specific accuracy
            class_total[true_label] += 1
            if predicted_class == true_label:
                class_correct[true_label] += 1

            # Print progress for some samples
            if idx < 5 or (idx + 1) % 5 == 0 or idx == n_samples - 1:
                sample_time = time.time() - sample_start
                print(f"Sample {idx+1}/{n_samples}: Predicted {predicted_class}, True {true_label}, " +
                      f"{'✓' if predicted_class == true_label else '✗'}, " +
                      f"Time: {sample_time:.2f}s")

        # Compute accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Display evaluation results
        eval_time = time.time() - start_time
        print(f"\nEvaluation completed in {eval_time:.2f}s")
        print(f"Overall accuracy: {accuracy:.4f}")

        # Print confusion matrix if scikit-learn is available
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, predictions)
            print("\nConfusion Matrix:")
            print(cm)
        except:
            pass

        # Print per-class accuracy
        print("\nPer-class accuracy:")
        for i in range(self.n_classes):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                print(f"  Class {i}: {class_acc:.4f} ({int(class_correct[i])}/{int(class_total[i])})")

        return accuracy

    def evaluate_plaintext(self, X_test, y_test):
        """
        Evaluate model performance using plaintext operations (much faster).

        Args:
            X_test: Test data features
            y_test: Test data labels

        Returns:
            Accuracy score
        """
        # Convert to numpy if needed
        X_test = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
        y_test = y_test.numpy() if isinstance(y_test, torch.Tensor) else y_test

        n_samples = len(X_test)
        predictions = np.zeros(n_samples, dtype=int)

        print(f"Starting plaintext evaluation on {n_samples} samples...")
        start_time = time.time()

        # Track per-class accuracy
        class_correct = np.zeros(self.n_classes)
        class_total = np.zeros(self.n_classes)

        # Process batches for faster evaluation
        batch_size = 100
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_samples)

            print(f"Evaluating batch {batch_idx+1}/{n_batches} (samples {start_idx+1}-{end_idx})...", end="\r")

            # Process each sample in the batch
            for idx in range(start_idx, end_idx):
                # Get sample and true label
                x = X_test[idx]
                true_label = int(y_test[idx])

                # Forward pass (plaintext)
                outputs = self.forward_plaintext(x)

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
        print(f"\nPlaintext evaluation completed in {eval_time:.2f}s")
        print(f"Overall accuracy: {accuracy:.4f}")

        # Print confusion matrix if scikit-learn is available
        try:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, predictions)
            print("\nConfusion Matrix (first 10x10 section):")
            print(cm[:10, :10])  # Show just a portion for readability
        except:
            pass

        # Print per-class accuracy for first 10 classes
        print("\nPer-class accuracy (first 10 classes):")
        for i in range(min(10, self.n_classes)):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                print(f"  Class {i}: {class_acc:.4f} ({int(class_correct[i])}/{int(class_total[i])})")

        return accuracy