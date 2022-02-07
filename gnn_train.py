import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from utils.buffer import Buffer
from utils.load_data import DataLoader
from utils.model_utils import EarlyStop, TopAverage, process_action
from utils.score import evaluate, f1_score_calc

from search_space import MacroSearchSpace
import copy
from gnn_layer import GraphLayer
from tqdm import tqdm


class Training():
    def __init__(self, args):

        
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        
        print("Dataset = ",args.dataset)
        self.data_load = DataLoader(args)
        self.data = self.data_load.load_data()
        
        self.task_iter = iter(self.data_load)

        num_feat =  self.data.num_features
        num_class = self.data_load.n_class
        self.classes_in_task = self.data_load.classes_in_task
        self.class_per_task = len(self.classes_in_task[0])


        self.early_stop_manager = EarlyStop(10)
        self.reward_manager = TopAverage(10)
        self.epochs = args.epochs

        self.loss = nn.CrossEntropyLoss()

        self.buffer = Buffer(args.buffer_size, self.device)
        self.alpha = args.alpha
        self.beta = args.beta

        # Copied from pyg_gnn
        channels_gnn = copy.deepcopy(args.channels_gnn) # Enter the hidden layer values only
        channels_mlp = copy.deepcopy(args.channels_mlp) # Enter the hidden node values only

        channels_gnn.insert(0,num_feat)
        self.acc_matrix = np.zeros([args.n_tasks, args.n_tasks])

        self.model = self.build_gnn(channels_gnn, channels_mlp, num_class, args.heads, args.mp_nn, args.in_drop).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.mc = copy.deepcopy(self.model)

        self.args = args
    
    def build_gnn(self, channels_gnn, channels_mlp, num_class, heads=1, mp_nn='gcn', dropout=0.5):
        model = GraphLayer(channels_gnn, channels_mlp, num_class, heads, mp_nn, dropout)
        return model
    
    def task_increment(self):
        self.current_task, (self.train_task_nid, self.val_task_nid) = next(self.task_iter)

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
            self.model.to(self.device)
            print('**')
        else:
            return

    def evaluate(self, actions=None):
        print("train action:", actions)

        # create model
        # try:
        self.model, val_acc, test_acc = self.run_model()
        # except RuntimeError as e:
        #     if "cuda" in str(e) or "CUDA" in str(e):
        #         print(e)
        #         val_acc = 0
        #         test_acc = 0
        #     else:
        #         raise e
        return val_acc, test_acc

    
    def train(self, actions=None):

        self.build_hidden_layers(actions)
        origin_action = actions  # Add recorder
        print("train action:", actions)

        # try:
        print("Model = ",self.model,end='\n')
        print(f'Classes in current task: {self.classes_in_task[self.current_task]}\n\n')
        self.model, val_acc = self.run_model()

        tqdm.write(f" Testing Task number {self.current_task} ".center(200, "*"),end="\n")


        for task_i in range(self.current_task+1):
            _, test_mask = self.data_load.test_nodes(task_i)
            self.test_model(copy.deepcopy(self.data), test_mask, task_i)

        # except RuntimeError as e:
        #     if "cuda" in str(e) or "CUDA" in str(e):
        #         print(e)
        #         val_acc = 0
        #     else:
        #         raise e
        
        reward = self.reward_manager.get_reward(val_acc)
        # self.save_param(self.model, update_all=(reward > 0))
        # self.record_action_info(origin_action, reward, val_acc)

        return reward, val_acc

    
    def test_model(self, data, test_mask, task_i):
        self.model.eval()
        data = data.to(self.device)

        outputs = self.model(data.x, data.edge_index)
        outputs = outputs[test_mask][:,self.classes_in_task[task_i]]
        outputs = F.log_softmax(outputs, 1)
        labels = data.y[test_mask]

        acc = f1_score_calc(outputs, labels)
        self.acc_matrix[self.current_task][task_i] = np.round(acc*100,2)
        tqdm.write("Test accuracy {:.4f} ".format(acc))

    def equal_(self):
        for (n,p),(_,pc) in zip(self.model.named_parameters(),self.mc.named_parameters()):
            # print(i,n,p.grad,sep='\t')
            if not torch.all(p.eq(pc)).data:
                print(n,"\n", p.eq(pc),sep='\t')
            else:
                return True

    def observe(self, data, mode='train'):

        conditions = torch.BoolTensor([l in self.classes_in_task[self.current_task] for l in data.y]).to(self.device)
        logits = self.model(data.x, data.edge_index)

        if mode=='train':

            self.optimizer.zero_grad()
            data.train_mask =  data.train_mask * conditions
            logits = logits[data.train_mask][:,self.classes_in_task[self.current_task]]
            outputs = F.log_softmax(logits, 1)
            # loss = self.loss(outputs[data.train_mask], data.y[data.train_mask])
            loss = self.loss(outputs, data.y[data.train_mask]-self.current_task*self.class_per_task) # Scale the labels from [2,3] -> [0,1]

            if not self.buffer.is_empty() and self.current_task:
                
                buf_data, buf_logits, task_no = self.buffer.get_data(
                    self.args.minibatch_size, transform=None)
                buf_outputs = self.model(buf_data.x, buf_data.edge_index)
                buf_outputs = buf_outputs[buf_data.train_mask][:,self.classes_in_task[task_no]]
                loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)

                buf_data, buf_logits, task_no = self.buffer.get_data(
                    self.args.minibatch_size, transform=None)
                buf_outputs = self.model(buf_data.x, buf_data.edge_index)
                buf_outputs = buf_outputs[buf_data.train_mask][:,self.classes_in_task[task_no]]
                buf_outputs = F.log_softmax(buf_outputs, 1)
                loss += self.args.beta * self.loss(buf_outputs, buf_data.y[buf_data.train_mask]-task_no*self.class_per_task)

            self.buffer.add_data(data, logits=logits.data, task_no = self.current_task)
                             
            loss.backward(retain_graph=True) # Class labels larger than n_class issue! https://github.com/pytorch/pytorch/issues/1204#issuecomment-366489299
            self.optimizer.step()
        
        else:
            data.val_mask = data.val_mask * conditions
            logits = logits[data.val_mask][:,self.classes_in_task[self.current_task]]
            outputs = F.log_softmax(logits, 1)
            loss = self.loss(outputs, data.y[data.val_mask]-self.current_task*self.class_per_task)

        return loss.item(), outputs

    
    def run_model(self):

        self.model.train()
           
        dur = []
        min_val_loss = float("inf")
        model_val_acc = 0
        # tqdm.write("Number of train data:", self.data.train_mask.sum())

        t0 = time.time()
        epochs = tqdm(range(self.epochs), desc = "Epoch", position=0, colour='green')

        for epoch in epochs:
            epochs.set_description(f"Epoch no.: {epoch}")

            
            train_loader = NeighborLoader(
                    self.data,
                    # Sample 30 neighbors for each node for 2 iterations
                    num_neighbors=[30] * 2,
                    # Use a batch size of 128 for sampling training nodes
                    batch_size=self.args.batch_size_nei,
                    input_nodes = self.train_task_nid,
                    )
            batch = tqdm(train_loader, desc="Batch", position=1, leave=False, colour='red')

            for batch_i,data_i in enumerate(batch):
                batch.set_description(f"Batch no.: {batch_i}")
                data_i = data_i.to(self.device)

                loss, outputs = self.observe(data_i, mode='train')
            
            if epoch%50==0 :
                train_acc = evaluate(outputs, data_i.y[data_i.train_mask])
                dur.append(time.time() - t0)
                
                tqdm.write(
                    "Epoch {:05d} | Train Loss {:.4f} | Time(s) {:.4f} | Train acc {:.4f}".format(
                        epoch, loss, np.mean(dur), train_acc))
                        
        val_loader = NeighborLoader(
        self.data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=self.args.batch_size,
        input_nodes = val_nodes,
        )

        tqdm.write(f" Validation ".center(200, "*"),end="\n")

        self.model.eval()

        for batch_i,data_i in enumerate(val_loader):
            data_i = data_i.to(self.device)

            val_loss, outputs = self.observe(data_i, mode='eval')

            val_acc = evaluate(outputs, data_i.y[data_i.val_mask])
            
            tqdm.write("Validation Loss: {:.4f}  | Validation accuracy: {:.4f}".format(
                            val_loss, val_acc))

            if val_loss < min_val_loss:  # and train_loss < min_train_loss
                min_val_loss = val_loss
                model_val_acc = val_acc

            
        return self.model, model_val_acc

        
    # def record_action_info(self, origin_action, reward, val_acc):
    #     with open(self.args.dataset + "_" + self.args.logger_file, "a") as file:
    #         # with open(f'{self.args.dataset}_{self.args.search_mode}_{self.args.format}_manager_result.txt', "a") as file:
    #         file.write(str(origin_action))

    #         file.write(";")
    #         file.write(str(reward))

    #         file.write(";")
    #         file.write(str(val_acc))
    #         file.write("\n")
    