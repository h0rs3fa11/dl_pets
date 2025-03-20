import os
from config import model_path
from modelIO import ModelIO
from data_prepare import prepare_data, load_test_data, load_data
from helper import preprocess_labels
from nn_model import MSEFashionClassifier, CEFashionClassifier
# load the weights

X_test, y_test, n_features, n_classes = load_test_data()
X_train, y_train, X_val, y_val = load_data()


def evaluate_model(model):
  if model == "mse":
    model_save_path = os.path.join(
        model_path, 'plaintext-mse', 'fashion_model.json')
    model = MSEFashionClassifier(n_features, n_classes)
  elif model == "ce":
    model_save_path = os.path.join(
        model_path, 'plaintext-ce', 'fashion_model.json')
    model = CEFashionClassifier(n_features, n_classes)

  model = ModelIO.load_model(model_save_path, model)
  _, y_test_mapped, _, _ = preprocess_labels(
      y_train, y_test)
  model.evaluate(X_test, y_test_mapped)


print("Evaluate MSE model...")
evaluate_model("mse")
print("="*10)
print("Evaluate Cross Entropy model...")
evaluate_model("ce")
