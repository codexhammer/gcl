import os.path as osp

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from utils.buffer import Buffer
from utils.load_data import DataLoader
from utils.model_utils import TopAverage
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

        self.reward_manager = TopAverage(10)
        self.epochs = args.epochs

        self.loss = nn.CrossEntropyLoss()

        self.buffer = Buffer(args.buffer_size, self.device)
        self.alpha = args.alpha
        self.beta = args.beta

        channels_gnn = copy.deepcopy(args.channels_gnn) # Enter the hidden layer values only
        # channels_mlp = copy.deepcopy(args.channels_mlp) # Enter the hidden node values only

        channels_gnn.insert(0,num_feat)
        self.acc_matrix = np.zeros([args.n_tasks, args.n_tasks])

        self.model = GraphLayer(channels_gnn, channels_mlp=None, num_class=num_class, heads=args.heads, mp_nn=args.mp_nn).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        # self.mc = copy.deepcopy(self.model)

        self.args = args
        
    def task_increment(self):
        self.current_task, (self.train_task_nid, self.val_task_nid) = next(self.task_iter)


    def build_hidden_layers(self, actions):
        if actions:
            search_space_cls = MacroSearchSpace()
            # action_list_gnn, _ = search_space_cls.generate_action_list(len(self.args.channels_gnn),len(self.args.channels_mlp))
            action_list_gnn = search_space_cls.generate_action_list(len(self.args.channels_gnn))
            actions_gnn = actions[:len(action_list_gnn)]
            # actions_mlp = actions[len(action_list_gnn):]

            self.actions_gnn = list(map(lambda x,y:x-y,actions_gnn[::2],actions_gnn[1::2]))
            # self.actions_mlp = list(map(lambda x,y:x-y,actions_mlp[::2],actions_mlp[1::2]))

            # self.model.weight_update(self.actions_gnn,self.actions_mlp)
            self.model.weight_update(self.actions_gnn)

            self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(),lr=self.args.lr)
            # self.mc = copy.deepcopy(self.model)
            print('**')
        else:
            return

    def train(self, actions=None):

        self.build_hidden_layers(actions)
        origin_action = actions  # Add recorder
        print("train action:", actions)

        try:
            print("Model = ",self.model,end='\n')
            print(f'Classes in current task: {self.classes_in_task[self.current_task]}\n\n')
            val_acc = self.run_model()

            tqdm.write(f" Testing Task number {self.current_task} ".center(200, "*"),end="\n")


            for task_i in range(self.current_task+1):
                _, test_mask = self.data_load.test_masking(task_i)
                self.test_model(copy.deepcopy(self.data), test_mask, task_i)

        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
                val_acc = 0
                torch.cuda.empty_cache()
            else:
                raise e
        
        reward = self.reward_manager.get_reward(val_acc)
        self.record_action_info(origin_action, reward, val_acc, update_all=(reward > 0))

        return reward, val_acc
    
    def test_model(self, data, test_mask, task_i):
        with torch.no_grad():
            self.model.eval()
            data = data.to(self.device)

            outputs = self.model(data.x, data.edge_index)

            if self.args.setting == 'task':
                outputs = outputs[test_mask][:,self.classes_in_task[task_i]]
                labels = data.y[test_mask]
                acc = f1_score_calc(outputs, labels, task_i*self.class_per_task)
            else:
                outputs = outputs[test_mask]
                labels = data.y[test_mask]
                acc = f1_score_calc(outputs, labels)

        self.acc_matrix[self.current_task][task_i] = np.round(acc*100,2)
        tqdm.write("Test accuracy {:.4f} ".format(acc))


    def observe(self, data, mode='train'):

        logits = self.model(data.x, data.edge_index)

        if mode=='train':
            self.optimizer.zero_grad()

            if self.args.setting == 'task':
                conditions = torch.BoolTensor([l in self.classes_in_task[self.current_task] for l in data.y]).to(self.device)
                data.train_mask =  data.train_mask * conditions
                logits = logits[data.train_mask][:,self.classes_in_task[self.current_task]]
                loss = self.loss(logits, data.y[data.train_mask]-self.current_task*self.class_per_task)
            else:
                conditions = torch.BoolTensor([l in range(max(self.classes_in_task[self.current_task])+1) for l in data.y]).to(self.device)
                data.train_mask =  data.train_mask * conditions
                logits = logits[data.train_mask]
                loss = self.loss(logits, data.y[data.train_mask])

            if not self.buffer.is_empty() and self.current_task:
                
                buf_data, buf_logits, task_no = self.buffer.get_data(
                    self.args.minibatch_size, transform=None)
                buf_outputs = self.model(buf_data.x, buf_data.edge_index)
                
                if self.args.setting == 'task':
                    buf_outputs = buf_outputs[buf_data.train_mask][:,self.classes_in_task[task_no]]
                else:
                    buf_outputs = buf_outputs[buf_data.train_mask]

                loss += self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
                
                ### Alpha-beta

                buf_data, _, task_no = self.buffer.get_data(
                    self.args.minibatch_size, transform=None)
                buf_outputs = self.model(buf_data.x, buf_data.edge_index)

                if self.args.setting == 'task':
                    buf_outputs = buf_outputs[buf_data.train_mask][:,self.classes_in_task[task_no]]
                    loss += self.args.beta * self.loss(buf_outputs, buf_data.y[buf_data.train_mask]-task_no*self.class_per_task)
                else:
                    buf_outputs = buf_outputs[buf_data.train_mask]
                    loss += self.args.beta * self.loss(buf_outputs, buf_data.y[buf_data.train_mask])

            self.buffer.add_data(data, logits=logits.data, task_no = self.current_task)
                             
            loss.backward(retain_graph=True)
            self.optimizer.step()
            # self.mc = copy.deepcopy(self.model)

        else:
            
            if self.args.setting == 'task':
                conditions = torch.BoolTensor([l in self.classes_in_task[self.current_task] for l in data.y]).to(self.device)
                data.val_mask = data.val_mask * conditions
                logits = logits[data.val_mask][:,self.classes_in_task[self.current_task]]
                loss = self.loss(logits, data.y[data.val_mask]-self.current_task*self.class_per_task)
            else:
                conditions = torch.BoolTensor([l in range(max(self.classes_in_task[self.current_task])+1) for l in data.y]).to(self.device)
                data.val_mask = data.val_mask * conditions
                logits = logits[data.val_mask]
                loss = self.loss(logits, data.y[data.val_mask])

        return loss.item(), logits

    # def equal_(self):
    #     for (n,p),(_,pc) in zip(self.model.named_parameters(),self.mc.named_parameters()):
    #         # print(i,n,p.grad,sep='\t')
    #         if not torch.all(p.eq(pc)).data:
    #             print(n,"\n", p.eq(pc),sep='\t')
    #         else:
    #             return True
    
    def run_model(self):

        self.model.train()
           
        dur = []
        # min_val_loss = float("inf")
        val_acc_list = []
        model_val_acc = 0
        

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
                    input_nodes = self.train_task_nid
                    )
            batch = tqdm(train_loader, desc="Batch", position=1, leave=False, colour='red')

            for batch_i,data_i in enumerate(batch):
                batch.set_description(f"Batch no.: {batch_i}")
                data_i = data_i.to(self.device)

                loss, outputs = self.observe(data_i, mode='train')
            
            if epoch%50==0 :
                train_acc = evaluate(outputs, data_i.y[data_i.train_mask], self.current_task*self.class_per_task)
                dur.append(time.time() - t0)
                
                tqdm.write(
                    "Epoch {:05d} | Train Loss {:.4f} | Time(s) {:.4f} | Train acc {:.4f}".format(
                        epoch, loss, np.mean(dur), train_acc))
                
                torch.cuda.empty_cache()

                        
        val_loader = NeighborLoader(
        self.data,
        # Sample 30 neighbors for each node for 2 iterations
        num_neighbors=[30] * 2,
        # Use a batch size of 128 for sampling training nodes
        batch_size=self.args.batch_size_nei,
        input_nodes = self.val_task_nid,
        )

        tqdm.write(f" Validation ".center(200, "*"),end="\n")

        with torch.no_grad():
            self.model.eval()

            for batch_i,data_i in enumerate(val_loader):
                data_i = data_i.to(self.device)
            # data_eval = copy.deepcopy(self.data).cuda()
                val_loss, outputs = self.observe(data_i, mode='eval')

                val_acc = evaluate(outputs, data_i.y[data_i.val_mask],self.current_task*self.class_per_task)
                val_acc_list.append(val_acc)
            
                tqdm.write("Validation Loss: {:.4f}  | Validation accuracy: {:.4f}".format(
                                val_loss, val_acc))

        # if val_loss < min_val_loss:  # and train_loss < min_train_loss
        #     min_val_loss = val_loss
        #     model_val_acc = val_acc
        model_val_acc = np.mean(val_acc_list)*100

        torch.cuda.empty_cache()

            
        return model_val_acc

        
    def record_action_info(self, origin_action, reward, val_acc, update_all):
        
        if update_all:
            torch.save(self.model.state_dict(),
                    osp.join(f'data/', f'{self.args.dataset}',  f'{self.args.dataset}_{self.args.mp_nn}_{self.args.setting}_gnn.pt') )

        with open(osp.join(f'data/', f'{self.args.dataset}',
                    f'{self.args.dataset}_{self.args.mp_nn}_{self.args.logger_file}'), "a") as file:

            file.write(str(origin_action))

            file.write(";")
            file.write(str(reward))

            file.write(";")
            file.write(str(val_acc))
            file.write("\n")