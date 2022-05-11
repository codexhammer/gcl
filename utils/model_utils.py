import numpy as np
import torch
from torch.autograd import Variable

class TopAverage(object):
    def __init__(self):
        self.scores = []

    def get_average(self, score):
        if len(self.scores) > 0:
            avg = np.mean(self.scores)
        else:
            avg = 0
        self.scores.append(score)
        return avg

    def get_reward(self, score):
        reward = score - self.get_average(score)
        return np.clip(reward, -0.5, 0.5)

def get_variable(inputs, cuda=False, gpu=0, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(gpu), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out


def to_item(x):
    """Converts x, possibly scalar and possibly tensor, to a Python scalar."""
    if isinstance(x, (float, int)):
        return x

    if float(torch.__version__[0:3]) < 0.4:
        assert (x.dim() == 1) and (len(x) == 1)
        return x[0]

    return x.item()