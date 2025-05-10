import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def analyze_encrypted_model(folder_path, model_type):
    """
    Analyze encrypted model results (CE or MSE) which are split across multiple epoch files
    """
    # Find all history files for this model
    files = sorted(
        glob(os.path.join(folder_path, "he_training*history*.json")))

    if not files:
        print(f"No files found in {folder_path}")
        return None, None, None

    # Extract epoch numbers from filenames
    epochs = []
    accuracies = []
    all_batch_times = []

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract epoch number from filename
            epoch = int(file_path.split('history_')[-1].split('.json')[0])
            epochs.append(epoch)

            # Extract validation accuracy
            if 'val_accuracy' in data:
                accuracies.append(data['val_accuracy'])

            # Extract batch times
            if 'batch_times' in data:
                all_batch_times.extend(data['batch_times'])

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Sort results by epoch
    sorted_indices = np.argsort(epochs)
    epochs = [epochs[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]

    # Calculate batch time statistics
    avg_batch_time = np.mean(all_batch_times) if all_batch_times else 0
    sum_batch_time = np.sum(all_batch_times) if all_batch_times else 0

    print(f"\n{model_type} Encrypted Model Results:")
    print(f"Average batch time: {avg_batch_time:.4f} seconds")
    print(f"Total batch time: {sum_batch_time:.4f} seconds")

    return epochs, accuracies, {'avg': avg_batch_time, 'sum': sum_batch_time}


def analyze_plaintext_model(file_path, model_type, epoch=20):
    """
    Analyze plaintext model results (CE or MSE) which are in a single file
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract validation accuracies for all epochs
        epochs = list(range(1, len(data.get('val_accuracy', [])) + 1))

        accuracies = data.get('val_accuracy', [])[:epoch]

        # Extract batch times
        all_batch_times = data.get('batch_times', [])

        times_batch = int(len(all_batch_times) / len(epochs) * epoch)

        all_batch_times = all_batch_times[:times_batch]
        epochs = epochs[:epoch]

        # Calculate batch time statistics
        avg_batch_time = np.mean(all_batch_times) if all_batch_times else 0
        sum_batch_time = np.sum(all_batch_times) if all_batch_times else 0

        print(f"\n{model_type} Plaintext Model Results:")
        print(f"Average batch time: {avg_batch_time:.4f} seconds")
        print(f"Total batch time: {sum_batch_time:.4f} seconds")

        return epochs, accuracies, {'avg': avg_batch_time, 'sum': sum_batch_time}

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None, None


def plot_accuracy_curves(results, save_path='accuracy_comparison.png'):
    """
    Plot accuracy curves for all models
    """
    plt.figure(figsize=(12, 8))

    for model_name, (epochs, accuracies) in results.items():
        if epochs and accuracies:
            plt.plot(epochs, accuracies, marker='o',
                     linestyle='-', label=model_name)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Set x-axis to show integer epoch numbers
    plt.xticks(range(1, 21))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    print(f"Plot saved to {save_path}")


if __name__ == "__main__":
    # Define paths to model results

    base_dir = "data/models/"

    ce_enc_dir = os.path.join(base_dir, 'he-ce')
    ce_plaintext_file = os.path.join(
        base_dir, 'plaintext-ce', 'training_history.json')

    mse_enc_dir = os.path.join(base_dir, 'he-mse')
    mse_plaintext_file = os.path.join(
        base_dir, 'plaintext-mse', 'training_history.json')

    # Analyze each model
    ce_enc_epochs, ce_enc_accuracies, ce_enc_batch_stats = analyze_encrypted_model(
        ce_enc_dir, "CE")
    mse_enc_epochs, mse_enc_accuracies, mse_enc_batch_stats = analyze_encrypted_model(
        mse_enc_dir, "MSE")

    ce_plain_epochs, ce_plain_accuracies, ce_plain_batch_stats = analyze_plaintext_model(
        ce_plaintext_file, "CE")
    mse_plain_epochs, mse_plain_accuracies, mse_plain_batch_stats = analyze_plaintext_model(
        mse_plaintext_file, "MSE")

    # Compile results for plotting
    accuracy_results = {
        'CE Encrypted': (ce_enc_epochs, ce_enc_accuracies),
        'MSE Encrypted': (mse_enc_epochs, mse_enc_accuracies),
        'CE Plaintext': (ce_plain_epochs[:20], ce_plain_accuracies[:20]),
        'MSE Plaintext': (mse_plain_epochs[:20], mse_plain_accuracies[:20])
    }

    # Plot accuracy curves
    plot_accuracy_curves(accuracy_results)

    print("\n===== Batch Time Summary =====")
    print("Model               | Average (s) | Total (s)")
    print("-" * 50)
    print(
        f"CE Encrypted        | {ce_enc_batch_stats['avg']:.4f} | {ce_enc_batch_stats['sum']:.4f}")
    print(
        f"MSE Encrypted       | {mse_enc_batch_stats['avg']:.4f} | {mse_enc_batch_stats['sum']:.4f}")
    print(
        f"CE Plaintext        | {ce_plain_batch_stats['avg']:.4f} | {ce_plain_batch_stats['sum']:.4f}")
    print(
        f"MSE Plaintext       | {mse_plain_batch_stats['avg']:.4f} | {mse_plain_batch_stats['sum']:.4f}")
