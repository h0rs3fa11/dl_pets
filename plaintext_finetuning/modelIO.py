import json
import numpy as np


class ModelIO:
    @staticmethod
    def save_model(model, filepath):
        model_state = {
            'weights': model.weights.tolist(),
            'bias': model.bias.tolist(),
            'n_features': model.n_features,
            'n_classes': model.n_classes,
        }

        with open(filepath, 'w') as f:
            json.dump(model_state, f)

        # print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath, model):
        with open(filepath, 'r') as f:
            model_state = json.load(f)

        weights = np.array(model_state['weights'])
        bias = np.array(model_state['bias'])
        n_features = model_state['n_features']
        n_classes = model_state['n_classes']

        model.weights = weights
        model.bias = bias
        if model.n_features != n_features or model.n_classes != n_classes:
            print(
              f"Warning: Model dimensions mismatch. Saved: {n_features}x{n_classes}, Current: {model.n_features}x{model.n_classes}")

        # Keep existing context
        print(f"Updated existing model with weights from {filepath}")
        return model

    @staticmethod
    def save_training_history(history, filepath):
        clean_history = {}
        for key, value in history.items():
            if isinstance(value, np.ndarray):
                clean_history[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                clean_history[key] = [v.tolist() for v in value]
            else:
                clean_history[key] = value

        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(clean_history, f)

        print(f"Training history saved to {filepath}")

    @staticmethod
    def load_training_history(filepath):
        with open(filepath, 'r') as f:
            history = json.load(f)

        print(f"Loaded training history from {filepath}")
        return history
