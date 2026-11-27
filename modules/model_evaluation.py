from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score,
)


def evaluate_regression(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "R-squared": r2_score(y_true, y_pred),
    }


def evaluate_classification(y_true, y_pred, labels=None):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1-score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "Confusion matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def evaluate_clustering(features, labels):
    if len(set(labels)) < 2:
        return {"Silhouette score": None}
    return {"Silhouette score": silhouette_score(features, labels)}
