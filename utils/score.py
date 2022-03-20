import torch
from sklearn.metrics import f1_score


def evaluate(outputs, labels, task_offset=0):
    indices = torch.argmax(outputs, dim=1)
    indices = indices + task_offset
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def f1_score_calc(outputs, labels, task_offset=0):
    indices = torch.argmax(outputs, dim=1)
    indices = indices + task_offset

    return f1_score(labels.cpu().numpy(), indices.cpu().numpy(), average="micro")