import numpy as np
import json
import os
import tenseal as ts

class HEModelIO:
    """
    Handles saving and loading of homomorphically encrypted models.
    """

    @staticmethod
    def save_model(model, filepath):
        """
        Save the trained model state to a file.

        Args:
            model: The HEFashionClassifier instance
            filepath: Path to save the model state
        """
        # Create model state dictionary
        model_state = {
            'weights': model.weights.tolist(),  # Convert numpy arrays to lists for JSON serialization
            'bias': model.bias.tolist(),
            'n_features': model.n_features,
            'n_classes': model.n_classes,
            # We don't save the TenSEAL context as it will be recreated
        }

        # Save to JSON file
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
    def load_model(filepath, model=None, load_context=False):
        """
        Load a saved model state.

        Args:
            filepath: Path to the saved model state
            model: Optional existing model instance to update
            load_context: Whether to load the saved context or create a new one

        Returns:
            Loaded HEFashionClassifier instance
        """
        # Load model state from JSON
        with open(filepath, 'r') as f:
            model_state = json.load(f)

        # Convert lists back to numpy arrays
        weights = np.array(model_state['weights'])
        bias = np.array(model_state['bias'])
        n_features = model_state['n_features']
        n_classes = model_state['n_classes']

        # Option 1: Update existing model
        if model is not None:
            model.weights = weights
            model.bias = bias
            if model.n_features != n_features or model.n_classes != n_classes:
                print(f"Warning: Model dimensions mismatch. Saved: {n_features}x{n_classes}, Current: {model.n_features}x{model.n_classes}")

            # Keep existing context
            print(f"Updated existing model with weights from {filepath}")
            return model

        # Option 2: Create new model instance
        else:
            # Try to load saved context if requested
            context = None
            if load_context:
                context_filepath = filepath + '.context'
                if os.path.exists(context_filepath):
                    try:
                        with open(context_filepath, 'rb') as f:
                            serialized_context = f.read()
                        context = ts.context_from(serialized_context)
                        print(f"Loaded context from {context_filepath}")
                    except Exception as e:
                        print(f"Warning: Could not load context: {e}")

            # Create new model instance with appropriate dimensions
            new_model = HEFashionClassifier(n_features, n_classes)

            # Update weights and bias
            new_model.weights = weights
            new_model.bias = bias

            # Update context if loaded
            if context is not None:
                new_model.context = context

            print(f"Created new model with weights from {filepath}")
            return new_model

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
        """
        Load training history from a file.

        Args:
            filepath: Path to the saved history

        Returns:
            Loaded history dictionary
        """
        # Load history from JSON
        with open(filepath, 'r') as f:
            history = json.load(f)

        print(f"Loaded training history from {filepath}")
        return history