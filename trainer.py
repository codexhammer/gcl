import os.path as osp

import numpy as np
import scipy.signal
import torch

import utils.tensor_utils as utils
from utils.result import result_file 

from gnn_train import Training
from tqdm import tqdm

from gnn_test import Testing

logger = utils.get_logger()


def discount(x, amount):
    return scipy.signal.lfilter([1], [1, -amount], x[::-1], axis=0)[::-1]


history = []


def scale(value, last_k=10, scale_value=1):
    '''
    scale value into [-scale_value, scale_value], according last_k history
    '''
    max_reward = np.max(history[-last_k:])
    if max_reward == 0:
        return value
    return scale_value / max_reward * value


def _get_optimizer(name):
    if name.lower() == 'sgd':
        optim = torch.optim.SGD
    elif name.lower() == 'adam':
        optim = torch.optim.Adam

    return optim


class Trainer(object):
    """Manage the training process"""

    def __init__(self, args,times):
        r"""
        Constructor for training algorithm.
        Build sub-model manager and controller.
        Build optimizer and cross entropy loss for controller.

        Args:
            args: From command line, picked up by `argparse`.
        """
        self.args = args
        self.controller_step = 0  # counter for controller
        self.cuda = args.cuda
        self.epoch = 0
        self.start_epoch = 0

        self.max_length = self.args.shared_rnn_max_length
        self.n_tasks = args.n_tasks

        self.train_gnn = None
        self.controller = None
        self.build_model()  # build controller and sub-model

        controller_optimizer = _get_optimizer(self.args.controller_optim)
        self.controller_optim = controller_optimizer(self.controller.parameters(), lr=self.args.controller_lr)

        self.times = times


    def build_model(self):
        

        # from search_space import MacroSearchSpace
        # search_space_cls = MacroSearchSpace()
        # self.search_space_gnn, self.search_space_mlp = search_space_cls.get_search_space()
        # self.action_list_gnn, self.action_list_mlp = search_space_cls.generate_action_list(len(self.args.channels_gnn),len(self.args.channels_mlp))
        # # build RNN controller
        # from graph_controller import Controller
        # self.controller = Controller(self.args, 
        #                                     search_space_gnn = self.search_space_gnn,
        #                                     search_space_mlp = self.search_space_mlp,
        #                                     action_list_gnn = self.action_list_gnn,
        #                                     action_list_mlp = self.action_list_mlp,                                                   
        #                                     cuda=self.args.cuda)




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
            self.controller.cuda()

    def train(self):
        r"""
        Each epoch consists of two phase:
        - In the first phase, shared parameters are trained to exploration.
        - In the second phase, the controller's parameters are trained.
        """
        # 2. Training the controller parameters theta
        for task_no in tqdm(range(self.n_tasks)):
            
            tqdm.write(f" Training Task number {task_no} ".center(200, "*"),end="\n\n\n")
            
            self.train_gnn.task_increment()

            if task_no == 0:
                self.train_init()
            else:                
                self.train_controller()

        print(f'\n\n All tasks completed successfully!')

        self.save_model()
        acc_matrix = self.train_gnn.acc_matrix
        result_file(self.args, acc_matrix, self.times)


    def train_init(self):        
        """
            Train first task withput controller.
        """

        # _, val_score = self.train_gnn.train()
        _, val_score = self.train_gnn.train()
        logger.info(f"Task no. 0: Val_score: {val_score}")


    def train_controller(self):
        r"""
            Train controller for subsequent tasks.
        """
        tqdm.write(f" Training controller ".center(200,'*'))
        model = self.controller
        model.train()

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []

        hidden = self.controller.init_hidden(self.args.batch_size)
        total_loss = 0

        controller_tqdm = tqdm(range(self.args.controller_max_step), colour='yellow')
        for _ in controller_tqdm:
            controller_tqdm.set_description(f" Controller step ")
            structure_list, log_probs, entropies = self.controller.sample(with_details=True)

            # calculate reward
            np_entropies = entropies.data.cpu().numpy()
            results = self.get_reward(structure_list, np_entropies, hidden)
            torch.cuda.empty_cache()

            if results:  # has reward
                rewards, hidden = results
            else:
                continue  # CUDA Error happens, drop structure and step into next iteration

            # discount
            if 1 > self.args.discount > 0:
                rewards = discount(rewards, self.args.discount)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            history.append(adv)
            adv = scale(adv, scale_value=0.5)
            adv_history.extend(adv)

            adv = utils.get_variable(adv, self.cuda, requires_grad=False)
            # policy loss
            loss = -log_probs * adv
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()

            total_loss += utils.to_item(loss.data)

            self.controller_step += 1
            torch.cuda.empty_cache()


    def get_reward(self, gnn_list, entropies, hidden):
        """
        Computes the reward of a single sampled model on validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()
        if isinstance(gnn_list, dict):
            gnn_list = [gnn_list]
        if isinstance(gnn_list[0], list) or isinstance(gnn_list[0], dict):
            pass
        else:
            gnn_list = [gnn_list]  # when structure_list is one structure

        reward_list = []
        for gnn in gnn_list:
            reward = self.train_gnn.train(gnn)

            if reward is None:  # cuda error happened
                reward = 0
            else:
                reward = reward[1]

            reward_list.append(reward)

        if self.args.entropy_mode == 'reward':
            rewards = reward_list + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = reward_list * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards, hidden

    def save_model(self):
        torch.save(self.controller.state_dict(),
                osp.join(f'data/', f'{self.args.dataset}',
                f'{self.args.dataset}_{self.args.mp_nn}_controller.pt'))