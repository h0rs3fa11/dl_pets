import numpy as np


def preprocess_labels(y_train, y_val=None):
    unique_labels = np.unique(np.concatenate(
        [y_train, y_val]) if y_val is not None else y_train)

    label_map = {original: idx for idx, original in enumerate(unique_labels)}

    y_train_mapped = np.array([label_map[label] for label in y_train])

    if y_val is not None:
        y_val_mapped = np.array([label_map[label] for label in y_val])
    else:
        y_val_mapped = None

    n_classes = len(unique_labels)

    print(
        f"Mapped {len(unique_labels)} unique labels to consecutive integers 0-{n_classes-1}")
    print(f"Original labels: {unique_labels}")

    return y_train_mapped, y_val_mapped, n_classes, label_map
