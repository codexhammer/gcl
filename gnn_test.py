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

from torch_geometric.data import Data


class Testing():
    def __init__(self,args):

        
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

        self.loss = torch.nn.CrossEntropyLoss()

        self.buffer = Buffer(args.buffer_size, self.device)
        self.alpha = args.alpha
        self.beta = args.beta

        ## Working here
        channels_gnn = copy.deepcopy(args.channels_gnn) # Enter the hidden layer values only
        channels_mlp = copy.deepcopy(args.channels_mlp) # Enter the hidden node values only
        
        channels_gnn.insert(0,num_feat)


        self.model = self.build_gnn(channels_gnn=channels_gnn,channels_mlp=channels_mlp,num_class=num_class).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=0.05)

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

    def train(self, actions=None):

        self.build_hidden_layers(actions)
        origin_action = actions  # Add recorder
        print("train action:", actions)

        # try:
        print("Model = ",self.model,end='\n')
        print(f'Classes in current task: {self.classes_in_task[self.current_task]}\n\n')

        self.run_model()


    def observe(self, data, mode='train'):

        conditions = torch.BoolTensor([l in self.classes_in_task[self.current_task] for l in data.y]).to(self.device)
        logits = self.model(data.x, data.edge_index)
        self.optimizer.zero_grad()
        data.train_mask =  data.train_mask * conditions
        logits = logits[data.train_mask][:,self.classes_in_task[self.current_task]]
        # outputs = F.log_softmax(logits, 1)
        # loss = self.loss(outputs[data.train_mask], data.y[data.train_mask])
        l = data.y[data.train_mask]-self.current_task*self.class_per_task
        loss = self.loss(logits, l)
                        
        loss.backward(retain_graph=True) 
        self.optimizer.step()
        
        self.mc = copy.deepcopy(self.model)

        return loss.item(), logits

    def equal_(self):
        for (n,p),(_,pc) in zip(self.model.named_parameters(),self.mc.named_parameters()):
            # print(i,n,p.grad,sep='\t')
            if not torch.all(p.eq(pc)).data:
                print(n,"\n", p.eq(pc),sep='\t')
            else:
                return True
            
    def run_model(self):

        self.model.train()
            
        model_val_acc = 0
        # tqdm.write("Number of train data:", self.data.train_mask.sum())

        for _ in range(self.epochs):

            
            train_loader = NeighborLoader(
                    self.data,
                    # Sample 30 neighbors for each node for 2 iterations
                    num_neighbors=[30] * 3,
                    # Use a batch size of 128 for sampling training nodes
                    batch_size=self.args.batch_size_nei,
                    input_nodes = self.train_task_nid
                    )

            for _,data_i in enumerate(train_loader):
                data_i = data_i.to(self.device)

                loss, outputs = self.observe(data_i, mode='train')

            
        return self.model, model_val_acc    



# t = Testing()
# t.task_increment()
# t.train()