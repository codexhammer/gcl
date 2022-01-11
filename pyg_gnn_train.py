import os.path as osp
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.loader import NeighborLoader
from utils.buffer import Buffer
from search_space import MacroSearchSpace
import copy
from utils.model_utils import EarlyStop, TopAverage, process_action
from pyg_gnn_layer import GraphLayer
from load_data import DataLoader
from tqdm import tqdm


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

class Training():
    def __init__(self, args, mp_nn="gcn",
                 batch_normal=True, residual=False):
        
        self.task_no = 0
        print("Dataset = ",args.dataset)
        self.data_load = DataLoader(args)
        self.data = self.data_load.load_data()

        self.num_feat =  self.data.num_features
        self.num_class = self.data.y.max().item() + 1
        self.early_stop_manager = EarlyStop(10)
        self.reward_manager = TopAverage(10)
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.mp_nn = mp_nn

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

        self.model = self.build_gnn(self.channels_gnn, self.channels_mlp, self.num_class, self.heads, self.mp_nn, self.dropout)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        if args.cuda:
            self.model.cuda()

        self.args = args
    
    def task_increment(self): # Remove this
        self.task_no = self.task_no + 1

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
        
        # if self.task_no >= self.max_data:
        #     raise Exception("Data no. exceeded!")

        self.build_hidden_layers(actions)
        origin_action = actions
        print("train action:", actions)

        try:
            print("Model = ",self.model)

            task_tqd = tqdm(self.data_load, desc="Task no.", position=0)
            
            for task_i, data_current in enumerate(task_tqd):
                current_task, (train_task_nid, val_task_nid) = data_current

                tqdm.write(f" Training Task number {current_task} ".center(24, "*"))
                task_tqd.set_description(f"Task no.: {task_i}")

                self.model, val_acc = self.run_model(train_nodes = train_task_nid, val_nodes = val_task_nid)

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
    

    def build_gnn(self, channels_gnn, channels_mlp, num_class, heads, mp_nn, dropout):
        model = GraphLayer(channels_gnn, channels_mlp, num_class, heads, mp_nn, dropout)
        return model

    def observe(self, data):

        self.optimizer.zero_grad()
        # not_aug_inputs = copy.deepcopy(data)
        logits = self.model(data.x, data.edge_index)
        outputs = F.log_softmax(logits, 1)
        loss = self.loss(outputs, data.y)

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(
                self.args.minibatch_size, transform=None)
            buf_outputs = self.model(buf_inputs)
            loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

            buf_inputs, buf_labels, _ = self.buffer.get_data(
                self.args.minibatch_size, transform=None)
            buf_outputs = self.model(buf_inputs)
            loss += self.args.beta * self.loss(buf_outputs, buf_labels)

        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.buffer.add_data(examples=torch.tensor([data.x, data.edge_index]),
                             labels=data.y,
                             logits=logits.data)

        return loss.item(), outputs

    
    def run_model(self, train_nodes, val_nodes):
           
        dur = []
        begin_time = time.time()
        best_performance = 0
        min_val_loss = float("inf")
        model_val_acc = 0
        # tqdm.write("Number of train data:", self.data.train_mask.sum())

        epochs = tqdm(range(3), desc = "Epoch", position=1)

        for epoch in epochs:
            epochs.set_description(f"Epoch no.: {epoch}")

            self.model.train()
            t0 = time.time()
            
            train_loader = NeighborLoader(
                    self.data,
                    # Sample 30 neighbors for each node for 2 iterations
                    num_neighbors=[30] * 2,
                    # Use a batch size of 128 for sampling training nodes
                    batch_size=self.args.batch_size,
                    input_nodes = train_nodes,
                    )
            batch = tqdm(train_loader, desc="Batch", position=2, leave=False)

            for batch_i,data_i in enumerate(batch):
                batch.set_description(f"Batch no.: {batch_i}")
                data_i = data_i.to('cuda:0')

                loss, outputs = self.observe(data_i)
            
            if epoch%50==0 :
                train_acc = evaluate(outputs, self.data.y, self.data.train_mask)                
                tqdm.write(
                    "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                        epoch, loss.item(), np.mean(dur), train_acc))
            # forward
            # logits = self.model(self.data.x, self.data.edge_index)
            # logits = F.log_softmax(logits, 1)
            # loss = self.loss(logits[self.data.train_mask], self.data.y[self.data.train_mask])
            # loss.backward()
            # self.optimizer.step()
            # train_loss = loss.item()

            # evaluate
                 # Need change here
        val_loader = NeighborLoader(
        self.data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=self.args.batch_size,
        input_nodes = val_nodes,
        )

        for batch_i,data_i in enumerate(val_loader):
            data_i = data_i.to('cuda:0')

            self.model.eval()
            logits = self.model(data_i.x, data_i.edge_index)
            logits = F.log_softmax(logits, 1)
            dur.append(time.time() - t0)

            val_acc = evaluate(logits, data_i.y, data_i.val_mask)
            test_acc = evaluate(logits, data_i.y, data_i.test_mask)
            print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                            epoch, loss.item(), np.mean(dur), val_acc, test_acc))

            loss = self.loss(logits, data_i.y)
            val_loss = loss.item()
            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                model_val_acc = val_acc
                if test_acc > best_performance:
                    best_performance = test_acc


            end_time = time.time()
            print("Each Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
            print(f"val_score:{model_val_acc},test_score:{best_performance}")
            
        return self.model, model_val_acc