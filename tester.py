import os.path as osp
import time
from search_space import MacroSearchSpace
from gnn_layer import GraphLayer 
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

def sample(gnn_len=2, mlp_len=2):
    msearch = MacroSearchSpace()
    space_gnn, space_mlp = msearch.get_search_space()
    action_gnn = []
    action_mlp = []

    for _ in range(gnn_len):
        for i in space_gnn.keys():
            action_gnn.append(np.random.choice(space_gnn[i]))
            
    for _ in range(mlp_len):
        action_layer=[]
        for i in space_mlp.keys():
            action_layer.append(np.random.choice(space_mlp[i]))
        action_mlp.append(action_layer[0]-action_layer[1] )
    
    return action_gnn, action_mlp
        

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

    
    def run_model(self, train_nodes, val_nodes):

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
                    input_nodes = train_nodes,
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
