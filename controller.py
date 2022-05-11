import torch
import torch.nn.functional as F

import utils.model_utils as utils


class Controller(torch.nn.Module):
    def _construct_action(self, actions):
        structure = []
        for action, action_name in zip(actions, self.action_list):
            predicted_actions = self.search_space[action_name][action]
            structure.append(predicted_actions)
        return structure

    def __init__(self, args, 
                search_space_gnn,
                action_list_gnn,
                controller_hid=100, cuda=True,
                softmax_temperature=5.0, tanh_c=2.5):

        if not self.check_action_list(action_list_gnn, search_space_gnn):
            raise RuntimeError("There are actions not contained in search_space")
        super(Controller, self).__init__()
        
        self.search_space_gnn = search_space_gnn
        self.action_list_gnn = action_list_gnn

        self.controller_hid = controller_hid
        self.is_cuda = cuda

        self.search_space = search_space_gnn
        self.action_list = action_list_gnn 

        # set hyperparameters
        if args and args.softmax_temperature:
            self.softmax_temperature = args.softmax_temperature
        else:
            self.softmax_temperature = softmax_temperature
        if args and args.tanh_c:
            self.tanh_c = args.tanh_c
        else:
            self.tanh_c = tanh_c

        # build encoder
        self.num_tokens = []
        for key in self.search_space:
            self.num_tokens.append(len(self.search_space[key]))

        num_total_tokens = sum(self.num_tokens)  # count action type
        self.encoder = torch.nn.Embedding(num_total_tokens, controller_hid)

        # the core of controller
        self.lstm = torch.nn.LSTMCell(controller_hid, controller_hid)
        # self.lstm = torch.nn.LSTM(controller_hid, controller_hid, num_layers =2)


        # build decoder
        self._decoders = torch.nn.ModuleDict()
        for key in self.search_space:
            size = len(self.search_space[key])
            decoder = torch.nn.Linear(controller_hid, size)
            self._decoders[key] = decoder
        
        self.args = args

        self.reset_parameters()

    # use to scale up the predicted network
    def update_action_list(self, action_list_gnn, search_space_gnn):
        if not self.check_action_list(action_list_gnn, search_space_gnn):
            raise RuntimeError("There are actions not contained in search_space")

        self.action_list_gnn = action_list_gnn

    @staticmethod
    def check_action_list(action_list_gnn, search_space_gnn):
        if isinstance(search_space_gnn, dict):
            keys_gnn = list(search_space_gnn.keys())
        else:
            return False
        for each in action_list_gnn:
            if each in keys_gnn:
                pass
            else:
                return False

        return True

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self._decoders:
            self._decoders[decoder].bias.data.fill_(0)

    def forward(self,
                inputs,
                action_name):

        embed = inputs

        hx, _ = self.lstm(embed)
        logits = self._decoders[action_name](hx)

        logits /= self.softmax_temperature

        logits = (self.tanh_c * torch.tanh(logits))

        return logits

    def action_index(self, action_name):
        key_names = self.search_space.keys()
        for i, key in enumerate(key_names):
            if action_name == key:
                return i

    def sample(self):

        inputs = torch.zeros(1, self.controller_hid) 
        # inputs = torch.zeros([2, 1, self.controller_hid])

        if self.is_cuda:
            inputs = inputs.cuda(self.args.gpu_id)
        entropies = []
        log_probs = []
        actions = []
        for block_idx, action_name in enumerate(self.action_list):
            decoder_index = self.action_index(action_name)

            logits = self.forward(inputs, action_name).flatten()

            probs = F.softmax(logits, dim=0)
            log_prob = F.log_softmax(logits, dim=0)

            entropy = -(log_prob * probs).sum()
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(0,
                 utils.get_variable(action, requires_grad=False, gpu=self.args.gpu_id))

            entropies.append(entropy)
            log_probs.append(selected_log_prob)

            inputs = utils.get_variable(
                action + sum(self.num_tokens[:decoder_index]),
                self.is_cuda,
                requires_grad=False, gpu=self.args.gpu_id)

            inputs = self.encoder(inputs)

            actions.append(action)

        actions = torch.cat(actions)
        dags = self._construct_action(actions)

        return dags, torch.cat(log_probs), torch.stack(entropies)