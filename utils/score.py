import torch
from sklearn.metrics import f1_score


def evaluate(outputs, labels):
    indices = torch.argmax(outputs, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def f1_score_calc(outputs, labels):
    _, indices = torch.max(outputs, dim=1)
    return f1_score(labels.cpu().numpy(), indices.cpu().numpy(), average="micro")