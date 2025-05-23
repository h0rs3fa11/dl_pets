import os
from data_prepare import load_data, prepare_data
from config import model_path, epoches, learning_rate, batch_size
from helper import preprocess_labels
from nn_model import CEFashionClassifier
from modelIO import ModelIO

X_train, y_train, X_val, y_val = load_data()
_, _, n_features, _ = prepare_data()
model_save_path = os.path.join(model_path, 'plaintext-ce')

y_train_mapped, y_val_mapped, n_classes, label_map = preprocess_labels(
    y_train, y_val)

model = CEFashionClassifier(n_features, n_classes)

history = model.train(X_train, y_train_mapped, X_val, y_val_mapped,
                      epoches, learning_rate=learning_rate, batch_size=batch_size)

ModelIO.save_model(model, os.path.join(
    model_save_path, 'fashion_model.json'))
ModelIO.save_training_history(history, os.path.join(
    model_save_path, 'training_history.json'))
