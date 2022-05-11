import torch
from torch_geometric.datasets import Planetoid, Amazon, CoraFull
import torch_geometric.transforms as T
import os.path as osp
import numpy as np
from collections import defaultdict, Counter

class DataLoader():
    def __init__(self, args=None):

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', args.dataset)
        if args.dataset in ["Computers", "Photo"]:
            dataset = Amazon(path, args.dataset, T.NormalizeFeatures()).shuffle()
        elif args.dataset in ["Cora", "Citeseer", "Pubmed"]:
            dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures()).shuffle()
        elif args.dataset=='CoraFull':
            dataset = CoraFull(path, transform=T.NormalizeFeatures()).shuffle()
        # elif args.dataset=='Reddit':
        #     dataset = Reddit(path, transform=T.NormalizeFeatures()).shuffle()
        # elif args.dataset=='ppi':
        #     dataset = PPI(path, transform=T.NormalizeFeatures()).shuffle()

        self.data = dataset[0]
        self.n_class = dataset.num_classes
        self.classes_in_task = {}
        self.n_tasks = args.n_tasks
        self.current_task = 0
        n_class_per_task = self.n_class // self.n_tasks

        if args.dataset=='Computers':
            count = defaultdict(list)

            for i,j in enumerate(self.data.y):
                count[j.item()].append(i)
            
            self.data.train_mask=torch.zeros_like(self.data.y).bool()
            self.data.val_mask=torch.zeros_like(self.data.y).bool()
            self.data.test_mask=torch.zeros_like(self.data.y).bool()

            for v in count.values():
                self.data.train_mask[v[:200]]=True
                self.data.val_mask[v[200:250]]=True
                self.data.test_mask[v[250:]]=True

        elif args.dataset=='CoraFull':
            count, _count  = defaultdict(list), defaultdict(list)
            n_class_per_task = 5
            label = self.data.y.numpy()
            labels= Counter(label)
            labels = {k: v for k, v in labels.items() if v>150}

            for i,j in enumerate(self.data.y):
                if j.item() in labels.keys():
                    count[j.item()].append(i)
            
            for k1,k2 in enumerate(count.keys()):
                _count[k1] = count[k2]
            del count

            self.data.train_mask=torch.zeros_like(self.data.y).bool()
            self.data.val_mask=torch.zeros_like(self.data.y).bool()
            self.data.test_mask=torch.zeros_like(self.data.y).bool()

            for v in _count.values():
                self.data.train_mask[v[:100]]=True
                self.data.val_mask[v[100:140]]=True
                self.data.test_mask[v[140:]]=True

        print(f"Nodes: {self.data.num_nodes}")
        print(f"Edges: {self.data.num_edges}")
        print(f"Features = {dataset.num_node_features}")

        if args.dataset=='CoraFull':
            print(f"Classes = {len(torch.unique(self.data.y[self.data.train_mask]))}")
        else:
            print(f"Classes = {dataset.num_classes}")
        
        assert n_class_per_task >= 1, "Reduce the number of tasks in the args: n_tasks"

        for task_i in range(self.n_tasks):
            self.classes_in_task[task_i] = list(range( task_i * n_class_per_task, (task_i+1) * n_class_per_task ))

        self.test_mask_info = {}

    
    def load_data(self):
        return self.data

    def __iter__(self):
        return self
    
    def next(self):
        return self.__next__()
    
    def __next__(self):
        if self.current_task >= self.n_tasks:
            raise StopIteration
        else:
            _, _, classes, train_mask, val_mask, test_mask = self.data
            current_task = self.current_task
            labels_of_current_task = self.classes_in_task[current_task]
            
            conditions = torch.BoolTensor( [l in labels_of_current_task for l in classes[1]] )
            
            train_mask = (train_mask[0], train_mask[1] * conditions)
            val_mask = (val_mask[0], val_mask[1] * conditions)
            test_mask = (test_mask[0], test_mask[1] * conditions)

            train_task_nid = np.nonzero(train_mask[1])
            train_task_nid = torch.flatten(train_task_nid)

            val_task_nid = np.nonzero(val_mask[1])
            val_task_nid = torch.flatten(val_task_nid)
            
            self.current_task += 1
            self.test_mask_info[current_task] = test_mask

            return current_task, (train_task_nid, val_task_nid)
    
    def test_masking(self, test_task_no):
        return self.test_mask_info[test_task_no]