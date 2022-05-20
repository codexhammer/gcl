from cmath import inf
import torch

import utils.model_utils as utils
from utils.result import result_file 
from copy import deepcopy
from gnn_train import Training
from tqdm import tqdm


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


class Trainer(object):
    """Manage the training process"""

    def __init__(self, args,times,path):
        r"""
        Initialize controller and gnn parameters
        """
        self.args = args
        self.controller_step = 0  # counter for controller
        self.cuda = args.cuda
        self.epoch = 0
        self.start_epoch = 0

        self.n_tasks = args.n_tasks

        self.train_gnn = None
        self.controller = None
        self.build_model()

        controller_optimizer = _get_optimizer(self.args.controller_optim)
        self.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.args.controller_lr)

        self.times = times
        self.path=path
        self.task_param = {}


    def build_model(self):

        from search_space import MacroSearchSpace

        search_space_cls = MacroSearchSpace()
        self.search_space_gnn = search_space_cls.get_search_space()
        self.action_list_gnn = search_space_cls.generate_action_list(len(self.args.channels_gnn))
        # build RNN controller
        from graph_controller_gnn import Controller
        self.controller = Controller(self.args, 
                                    search_space_gnn = self.search_space_gnn,
                                    action_list_gnn = self.action_list_gnn,                                               
                                    cuda=self.args.cuda)


        self.train_gnn = Training(self.args) ### Changed

        if self.cuda:
            self.controller.cuda(self.args.gpu_id)

    def train(self):
        r"""
        In the first task, train the model WITHOUT the controller
        In the next tasks, train the model WITH the controller
        """
        # 2. Training the controller parameters theta
        for self.task_no in tqdm(range(self.n_tasks)):
            
            tqdm.write(f" \n\nTraining Task number {self.task_no} ".center(20, "*"),end="\n\n\n")
            
            self.train_gnn.task_increment()

            if self.args.abl == 0 or self.args.abl == 1:
                self.train_gnn.train()
            else:
                if self.task_no == 0:
                    self.train_gnn.train()
                    self.task_param[self.task_no] = self.train_gnn.model
                else:
                    self.task_param[self.task_no] = deepcopy(self.train_gnn.model)   
                    self.train_controller()

        print(f'\n\n All tasks completed successfully!')

        acc_matrix = self.train_gnn.acc_matrix
        result_file(self.args, acc_matrix, self.times,self.path)


    def train_controller(self):
        r"""
            Train controller for subsequent tasks.
        """
        tqdm.write(f" Training controller ".center(200,'*'))
        self.controller.train()

        baseline = None
        total_loss = 0
        best_reward = -float('inf')

        controller_tqdm = tqdm(range(self.args.controller_max_step), colour='yellow')
        for _ in controller_tqdm:
            controller_tqdm.set_description(f" Controller step ")
            structure_list, log_probs, entropies = self.controller.sample()

            reward = self.get_reward(structure_list, entropies)         # Reward fn.

            torch.cuda.empty_cache()

            if reward is None:  # has reward
                continue

            # moving average baseline
            if baseline is None:
                baseline = reward
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * reward
            
            if reward>best_reward:
                best_reward = reward
                self.task_param[self.task_no] = deepcopy(self.train_gnn.model)           

            adv = reward - baseline

            adv = utils.get_variable(adv, self.cuda, gpu = self.args.gpu_id, requires_grad=False)
            # policy loss
            loss = -log_probs * adv

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(self.controller.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            self.controller_step += 1
            torch.cuda.empty_cache()

        self.train_gnn.model = self.task_param[self.task_no]
        self.train_gnn.train()



    def get_reward(self, gnn_list, entropies):
        """
        Computes the reward of a single sampled model on validation data.
        """
        reward = self.train_gnn.train(gnn_list)

        if reward is None:  # cuda error happened
            return reward

        reward = reward - torch.sum(self.args.entropy_coeff * entropies)

        return reward