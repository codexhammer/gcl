import os.path as osp
import time
from search_space import MacroSearchSpace
from pyg_gnn_layer import GraphLayer 
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Coauthor, Amazon

def load_data(dataset="Cora", supervised=False, full_data=True):
    '''
    support semi-supervised and supervised
    :param dataset:
    :param supervised:
    :return:
    '''
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
    data = dataset[0]
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
    return dataset, data

def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

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
        

def run_model():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    channels_gnn = [32, 48]
    channels_mlp = [30, 40]
    action_gnn, action_mlp = sample()
    epochs = 200
    dataset, data = load_data()
    data = data.to(device)
    channels_gnn.insert(0,dataset.num_node_features)

    model = GraphLayer(channels_gnn,channels_mlp, num_class=dataset.num_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.NLLLoss()
    dur = []
    begin_time = time.time()
    best_performance = 0
    min_val_loss = float("inf")
    model_val_acc = 0
    print("Number of train datas:", data.train_mask.sum())
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        t0 = time.time()
        # forward
        logits = model(data.x, data.edge_index)
        logits = F.log_softmax(logits, 1)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        train_loss = loss.item()

        # evaluate
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
        if epoch%10==0:
            print(
                "Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | acc {:.4f} | val_acc {:.4f} | test_acc {:.4f}".format(
                    epoch, loss.item(), np.mean(dur), train_acc, val_acc, test_acc))

            end_time = time.time()
            print("Epoch Cost Time: %f " % ((end_time - begin_time) / epoch))
    print(f"val_score:{model_val_acc},test_score:{best_performance}")
    model.weight_update(action_gnn, action_mlp)

run_model()