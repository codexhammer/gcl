import os.path as osp
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon

from search_space import MacroSearchSpace
import copy
from utils.model_utils import EarlyStop, TopAverage, process_action
from pyg_gnn_layer import GraphLayer


def load_data(data_no=0, dataset="Cora", supervised=False, full_data=True):
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
            self.data = load_data(self.data_no, args.dataset, args.supervised)
        else:
            self.data = load_data(self.data_no, args.dataset)

        self.num_feat =  self.data.num_features
        self.num_class = self.data.y.max().item() + 1
        self.early_stop_manager = EarlyStop(10)
        self.reward_manager = TopAverage(10)
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.att_type = att_type

        self.loss_fn1 = nn.CrossEntropyLoss()

        self.loss_fn2 = 0       #Add here

        device = torch.device('cuda' if args.cuda else 'cpu')
        self.data.to(device)

        # Copied from pyg_gnn
        self.channels_gnn = copy.deepcopy(args.channels_gnn) # Enter the hidden layer values only
        self.channels_mlp = copy.deepcopy(args.channels_mlp) # Enter the hidden node values only
        self.dropout = args.in_drop
        self.residual = residual
        self.batch_normal = batch_normal
        self.heads = args.heads
        self.max_data = args.max_data


        self.channels_gnn.insert(0,self.num_feat)

        self.model = self.build_gnn(self.channels_gnn, self.channels_mlp, self.num_class, self.heads, self.att_type)
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
            self.model, val_acc, test_acc = self.run_model(self.model, self.optimizer, self.loss_fn, self.data, self.epochs,
                                                      cuda=self.args.cuda, return_best=True,
                                                      half_stop_score=max(self.reward_manager.get_top_average() * 0.7,
                                                                          0.4))
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
            load_data(data_no = self.data_no)

        self.build_hidden_layers(actions)
        origin_action = actions
        print("train action:", actions)

        try:
            # use optimizer
            self.model, val_acc = self.run_model(self.model, self.optimizer, self.loss_fn1, self.data, self.epochs, cuda=self.args.cuda,
                                            half_stop_score=max(self.reward_manager.get_top_average() * 0.7, 0.4))
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
    
    def build_gnn(self, channels_gnn, channels_mlp, num_class, heads, att_type):
        model = GraphLayer(channels_gnn, channels_mlp, num_class, heads, att_type)
        return model

    def retrain(self, actions):
        return self.train(actions)

    def test_with_param(self, actions=None):
        return self.train(actions)

    @staticmethod
    def run_model(model, optimizer, loss_fn, data, epochs, early_stop=5, tmp_model_file="geo_citation.pkl",
                  half_stop_score=0, return_best=False, cuda=True, need_early_stop=False, show_info=True):

        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        model_val_acc = 0
        print("Number of train datas:", data.train_mask.sum())
        print("Model = ",model)
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            # forward
            logits = model(data.x, data.edge_index)
            logits = F.log_softmax(logits, 1)
            loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_loss = loss.item()

            # evaluate
            if epoch%50==0: # Need change here
                model.eval()
                logits = model(data.x, data.edge_index)
                logits = F.log_softmax(logits, 1)
                train_acc = evaluate(logits, data.y, data.train_mask)
                dur.append(time.time() - t0)

                val_acc = evaluate(logits, data.y, data.val_mask)
                test_acc = evaluate(logits, data.y, data.test_mask)

                loss = loss_fn(logits[data.val_mask], data.y[data.val_mask])
                val_loss = loss.item()
                if val_loss < min_val_loss:  # and train_loss < min_train_loss
                    min_val_loss = val_loss
                    model_val_acc = val_acc
                    if test_acc > best_performance:
                        best_performance = test_acc
                if show_info:
                    print(
                        "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                            epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

        end_time = time.time()
        print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
        print(f"val_score:{model_val_acc},test_score:{best_performance}")
        if return_best:
            return model, model_val_acc, best_performance
        else:
            return model, model_val_acc
