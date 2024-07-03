from torchmetrics import Accuracy, F1Score

def accuracy_metric():
    return Accuracy(task="multiclass", num_classes=10)

def f1_score_metric():
    return F1Score(task="multiclass", num_classes=10)