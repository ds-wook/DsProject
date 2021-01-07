from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, roc_auc_score
import numpy as np


def get_clf_eval(
    y_test: np.ndarray, pred: np.ndarray = None, pred_proba: np.ndarray = None
) -> None:
    """Evaluation fuction
    Args:
        y_test: actaul value
        pred: predict value
        pred_proba: predict values' probablity
    """
    confusion = confusion_matrix(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred_proba)

    print("Confusion Matrix")
    print(confusion)

    print(f"Accuracy: {accuracy: .4f}, Precision: {precision: .4f}, ", end="")
    print(f"Recall: {recall: .4f}, f1: {f1: .4f}, ROC_AUC: {roc_auc: .4f}")
