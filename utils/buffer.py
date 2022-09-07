import numpy as np


def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


class Buffer:
    def __init__(self, buffer_size=6000):
        self.buffer_size = buffer_size
        self.num_seen_examples = 0
        self.buffer_list = [None]*buffer_size


    def add_data(self, data, logits, task_no):
        index = reservoir(self.num_seen_examples, self.buffer_size)
        self.num_seen_examples += 1
        if index >= 0:
            self.buffer_list[index] = (data,logits, task_no)

    def get_data(self, size: int):
        if size > min(self.num_seen_examples, self.buffer_size):
            size = min(self.num_seen_examples, self.buffer_size)

        choice = np.random.choice(min(self.num_seen_examples, self.buffer_size),
                                  size=size, replace=False)

        return [self.buffer_list[i] for i in choice][0]  # May need change here!

    def is_empty(self) -> bool:
        if self.num_seen_examples == 0:
            return True
        else:
            return False