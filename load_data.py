import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon, Reddit
import torch_geometric.transforms as T
import os.path as osp
import numpy as np


class DataLoader():
    def __init__(self, args):
        dataset = args.dataset

        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
        if dataset in ["CS", "Physics"]:
            dataset = Coauthor(path, dataset, T.NormalizeFeatures())
        elif dataset in ["Computers", "Photo"]:
            dataset = Amazon(path, dataset, T.NormalizeFeatures())
        elif dataset in ["Cora", "Citeseer", "Pubmed"]:
            dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
        elif dataset=='Reddit':
            dataset = Reddit(path, transform=T.NormalizeFeatures())
        self.data = dataset[0]

        print(f"Nodes: {self.data.num_nodes}")
        print(f"Edges: {self.data.num_edges}")
        print(f"Features = {dataset.num_node_features}")
        print(f"Classes = {dataset.num_classes}")

        self.n_class = dataset.num_classes
        self.classes_in_task = {}

        # self.n_tasks = 3
        self.n_tasks = args.n_tasks
        self.args = args
        self.current_task = 0

        n_class_per_task = self.n_class // self.n_tasks

        assert n_class_per_task > 1, "Reduce the number of tasks in the args: n_tasks"

        for task_i in range(self.n_tasks):
            self.classes_in_task[task_i] = list(range( task_i * n_class_per_task, (task_i+1) * n_class_per_task ))

        self.test_mask_info = {}

    
    def load_data(self):
        return self.data

    def get_label_offset(self, task_i):
        return task_i * self.n_labels_per_task, (task_i + 1) * self.n_labels_per_task 

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
            
            test_task_nid = np.nonzero(test_mask[1])
            test_task_nid = torch.flatten(test_task_nid)
            
            
            self.current_task += 1  # Need change here
            # return current_task, (train_mask, val_mask, test_mask, train_task_nid)
            self.test_mask_info[current_task] = test_task_nid

            return current_task, (train_task_nid, val_task_nid)
    
    def test_nodes(self, test_task_no):
        return self.test_mask_info[test_task_no], self.classes_in_task[test_task_no]




# from torch_geometric.loader import NeighborLoader
# from torch_geometric.nn import Sequential, GCNConv



# datas = DataLoader(args=None,dataset='Reddit')
# data = datas.load_data()
# print(data)

# model = Sequential('x, edge_index', [
#     (GCNConv(602, 64), 'x, edge_index -> x'),
#     (GCNConv(64, 1), 'x, edge_index -> x')
# ]).to('cuda:0')

# for i,data_current in enumerate(datas):
#     current_task, (train_mask, val_mask, test_mask, train_task_nid) = data_current
#     for ep in range(2):
#         print(f"Epoch no. = {ep}")
#         loader = NeighborLoader(
#         data,
#         # Sample 30 neighbors for each node for 2 iterations
#         num_neighbors=[30] * 2,
#         # Use a batch size of 128 for sampling training nodes
#         batch_size=200,
#         input_nodes=train_task_nid,
#         directed = False
#         )
#         print(f"Task no. = {i}")
#         for i,dat in enumerate(loader):
#             dat = dat.to('cuda:0')
#             print(i, dat)
#             print(model(dat.x, dat.edge_index))