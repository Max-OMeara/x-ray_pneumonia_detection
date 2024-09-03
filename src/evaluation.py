import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, test_data, test_labels):
    loss, acc = model.evaluate(test_data, test_labels, batch_size=16)
    print("Loss =", loss, "& Accuracy =", acc)


def plot_confusion_matrix(labels, predictions, classes, normalize=False):
    cm = confusion_matrix(labels, predictions)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def generate_classification_report(test_labels, predictions):
    report = classification_report(
        test_labels, predictions, target_names=["NORMAL", "PNEUMONIA"]
    )
    print(report)
