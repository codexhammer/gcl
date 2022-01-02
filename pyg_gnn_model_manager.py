import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from utils.buffer import Buffer
from search_space import MacroSearchSpace
import copy
from utils.model_utils import EarlyStop, TopAverage, process_action
from pyg_gnn_layer import GraphLayer


def load_data(data_no=0, dataset="Cora", supervised=False, full_data=True, cuda=True):
    '''
    support semi-supervised and supervised
    :param dataset:
    :param supervised:
    :return:
    '''
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["CS", "Physics"]:
        dataset = Coauthor(path, dataset, T.NormalizeFeatures())
    elif dataset in ["Computers", "Photo"]:
        dataset = Amazon(path, dataset, T.NormalizeFeatures())
    elif dataset in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[data_no]
    if supervised:
        if full_data:
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:-1000] = 1
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.test_mask[data.num_nodes - 500:] = 1
        else:
            data.train_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.train_mask[:1000] = 1
            data.val_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.val_mask[data.num_nodes - 1000: data.num_nodes - 500] = 1
            data.test_mask = torch.zeros(data.num_nodes, dtype=torch.uint8)
            data.test_mask[data.num_nodes - 500:] = 1
    if cuda:
        data = data.cuda()
    return data

def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

class GeoCitationManager():
    def __init__(self, args, att_type="gcn",
                 batch_normal=True, residual=False):
        
        self.data_no = 0
        if hasattr(args, "supervised"):
            self.data = load_data(self.data_no, args.dataset, args.supervised, args.cuda)
        else:
            self.data = load_data(self.data_no, args.dataset, args.cuda)
        print("Dataset = ",args.dataset )

        self.num_feat =  self.data.num_features
        self.num_class = self.data.y.max().item() + 1
        self.early_stop_manager = EarlyStop(10)
        self.reward_manager = TopAverage(10)
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.att_type = att_type

        self.loss = nn.CrossEntropyLoss()
        self.buffer = Buffer(args.buffer_size, args.cuda)
        self.alpha = args.alpha
        self.beta = args.beta

        # Copied from pyg_gnn
        self.channels_gnn = copy.deepcopy(args.channels_gnn) # Enter the hidden layer values only
        self.channels_mlp = copy.deepcopy(args.channels_mlp) # Enter the hidden node values only
        self.dropout = args.in_drop
        self.residual = residual
        self.batch_normal = batch_normal
        self.heads = args.heads
        self.max_data = args.max_data

        self.channels_gnn.insert(0,self.num_feat)

        self.model = self.build_gnn(self.channels_gnn, self.channels_mlp, self.num_class, self.heads, self.att_type, self.dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.cuda:
            self.model.cuda()

        self.args = args
    
    def task_increment(self):
        self.data_no = self.data_no + 1

    def evaluate_actions(self, actions_gnn, actions_mlp, state_num_gnn=1, state_num_mlp=2):
        state_length_gnn = len(actions_gnn)
        if state_length_gnn % state_num_gnn!=0:
            raise RuntimeError("Wrong GNN Input: unmatchable input")

        state_length_mlp = len(actions_mlp)
        if state_length_mlp % state_num_mlp != 0:
            raise RuntimeError("Wrong MLP Input: unmatchable input")

    def build_hidden_layers(self, actions):
        if actions:
            search_space_cls = MacroSearchSpace()
            action_list_gnn, _ = search_space_cls.generate_action_list(len(self.args.channels_gnn),len(self.args.channels_mlp))
            actions_gnn = actions[:len(action_list_gnn)]
            actions_mlp = actions[len(action_list_gnn):]
            self.actions_gnn = actions_gnn
            self.actions_mlp = list(map(lambda x,y:x-y,actions_mlp[::2],actions_mlp[1::2]))
            self.evaluate_actions(actions_gnn, actions_mlp, state_num_gnn=1, state_num_mlp=2)
            self.model.weight_update(self.actions_gnn,self.actions_mlp)
            print('**')
        else:
            return

    def evaluate(self, actions=None):
        print("train action:", actions)

        # create model
        try:
            self.model, val_acc, test_acc = self.run_model()
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
                test_acc = 0
            else:
                raise e
        return val_acc, test_acc
    
    def train(self, actions=None):
        
        if self.data_no >= self.max_data:
            raise Exception("Data no. exceeded!")
        if self.data_no>0 and self.data_no<self.max_data:
            load_data(data_no = self.data_no, cuda = self.args.cuda)

        print("*" * 35, " Training Task ",self.data_no, " *" * 35)
        self.build_hidden_layers(actions)
        origin_action = actions
        print("train action:", actions)

        try:
            # use optimizer
            self.model, val_acc = self.run_model()
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
            else:
                raise e
        reward = self.reward_manager.get_reward(val_acc)
        # self.save_param(self.model, update_all=(reward > 0))

        self.record_action_info(origin_action, reward, val_acc)

        return reward, val_acc

    def record_action_info(self, origin_action, reward, val_acc):
        with open(self.args.dataset + "_" + self.args.submanager_log_file, "a") as file:
            # with open(f'{self.args.dataset}_{self.args.search_mode}_{self.args.format}_manager_result.txt', "a") as file:
            file.write(str(origin_action))

            file.write(";")
            file.write(str(reward))

            file.write(";")
            file.write(str(val_acc))
            file.write("\n")
    
    def build_gnn(self, channels_gnn, channels_mlp, num_class, heads, att_type, dropout):
        model = GraphLayer(channels_gnn, channels_mlp, num_class, heads, att_type, dropout)
        return model

    def observe(self):

        self.optimizer.zero_grad()
        # not_aug_inputs = copy.deepcopy(self.data)
        logits = self.model(self.data.x, self.data.edge_index)
        outputs = F.log_softmax(logits, 1)
        loss = self.loss(outputs[self.data.train_mask], self.data.y[self.data.train_mask])

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=None)
            buf_outputs = self.model(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=None)
            buf_outputs = self.model(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward()
        self.optimizer.step()

        self.buffer.add_data(examples=self.data,
                             labels=self.data.y[self.data.train_mask],
                             logits=logits.data)

        return loss.item(), outputs

    
    def run_model(self, return_best=False, show_info=True):
           
        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        model_val_acc = 0
        print("Number of train datas:", self.data.train_mask.sum())
        print("Model = ",self.model)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            t0 = time.time()
            loss, outputs = self.observe()
            # forward
            # logits = self.model(self.data.x, self.data.edge_index)
            # logits = F.log_softmax(logits, 1)
            # loss = self.loss(logits[self.data.train_mask], self.data.y[self.data.train_mask])
            # loss.backward()
            # self.optimizer.step()
            # train_loss = loss.item()

            # evaluate
            if epoch%50==0 and show_info:
                train_acc = evaluate(outputs, self.data.y, self.data.train_mask)                
                print(
                    "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, loss.item(), np.mean(dur), train_acc))
                 # Need change here
        self.model.eval()
        logits = self.model(self.data.x, self.data.edge_index)
        logits = F.log_softmax(logits, 1)
        dur.append(time.time() - t0)

        val_acc = evaluate(logits, self.data.y, self.data.val_mask)
        test_acc = evaluate(logits, self.data.y, self.data.test_mask)
        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, loss.item(), np.mean(dur), val_acc, test_acc))

        loss = self.loss(logits[self.data.val_mask], self.data.y[self.data.val_mask])
        val_loss = loss.item()
        if val_loss < min_val_loss:  # and train_loss < min_train_loss
            min_val_loss = val_loss
            model_val_acc = val_acc
            if test_acc > best_performance:
                best_performance = test_acc


        end_time = time.time()
        print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
        print(f"val_score:{model_val_acc},test_score:{best_performance}")
        if return_best:
            return self.model, model_val_acc, best_performance
        else:
            return self.model, model_val_acc
