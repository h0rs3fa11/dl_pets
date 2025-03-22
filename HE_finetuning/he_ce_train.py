import os
from data_prepare import load_data, prepare_data
from config import model_path, epoches, learning_rate, batch_size
from helper import preprocess_labels
from he_fashion_classifier import CE_HEFashionClassifier
from he_model_IO import HEModelIO

X_train, y_train, X_val, y_val = load_data()
_, _, n_features, _ = prepare_data()
model_save_path = os.path.join(model_path, 'he-ce')

y_train_mapped, y_val_mapped, n_classes, label_map = preprocess_labels(
    y_train, y_val)

epochs = 1

while (epochs <= epoches):
  model = CE_HEFashionClassifier(n_features, n_classes)
  if epochs > 1:
    model = HEModelIO.load_model(os.path.join(
        model_save_path, f'he_fashion_model_{epochs-1}.json'), model)

  history = model.train(
      X_train, y_train_mapped, X_val, y_val_mapped, 1, learning_rate=learning_rate, batch_size=batch_size, verbose=True, validation=True)

  HEModelIO.save_model(model, os.path.join(
      model_save_path, f'he_fashion_model_{epochs}.json'))
  HEModelIO.save_training_history(history, os.path.join(
      model_save_path, f'he_training_history_{epochs}.json'))
  epochs += 1
