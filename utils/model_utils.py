import numpy as np

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