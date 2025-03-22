import numpy as np
import json
import os
import tenseal as ts


class HEModelIO:
    @staticmethod
    def save_model(model, filepath):
        model_state = {
            # Convert numpy arrays to lists for JSON serialization
            'weights': model.weights.tolist(),
            'bias': model.bias.tolist(),
            'n_features': model.n_features,
            'n_classes': model.n_classes,
        }

        with open(filepath, 'w') as f:
            json.dump(model_state, f)

        print(f"Model saved to {filepath}")

        # Save context parameters separately if needed
        # This is optional but useful if you want to ensure exact same context parameters
        context_filepath = filepath + '.context'
        try:
            # Serialize the context
            serialized_context = model.context.serialize()
            with open(context_filepath, 'wb') as f:
                f.write(serialized_context)
            print(f"Context saved to {context_filepath}")
        except Exception as e:
            print(f"Warning: Could not save context: {e}")

    @staticmethod
    def load_model(filepath, model, load_context=False):
        # Load model state from JSON
        with open(filepath, 'r') as f:
            model_state = json.load(f)

        # Convert lists back to numpy arrays
        weights = np.array(model_state['weights'])
        bias = np.array(model_state['bias'])
        n_features = model_state['n_features']
        n_classes = model_state['n_classes']

        model.weights = weights
        model.bias = bias
        if model.n_features != n_features or model.n_classes != n_classes:
            print(
                f"Warning: Model dimensions mismatch. Saved: {n_features}x{n_classes}, Current: {model.n_features}x{model.n_classes}")
        context = None
        if load_context:
            context_filepath = filepath + '.context'
            if os.path.exists(context_filepath):
                with open(context_filepath, 'rb') as f:
                    serialized_context = f.read()
                    context = ts.context_from(serialized_context)
                    print(f"Loaded context from {context_filepath}")

        if context is not None:
            model.context = context

        print(f"Updated existing model with weights from {filepath}")
        return model

    @staticmethod
    def save_training_history(history, filepath):
        """
        Save training history to a file.

        Args:
            history: Training history dictionary
            filepath: Path to save the history
        """
        # Convert numpy arrays if present
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
