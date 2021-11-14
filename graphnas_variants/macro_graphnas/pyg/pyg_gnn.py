import torch
import torch.nn.functional as F

# from gnn import GraphNet as BaseNet
from search_space import act_map
from graphnas_variants.macro_graphnas.pyg.pyg_gnn_layer import GeoLayer
import torch_geometric.nn as nn


class GraphNet(torch.nn.Module):

    def __init__(self, actions_mlp, num_feat, num_label, drop_out=0.6, multi_label=False, batch_normal=True, state_num_gcn=1,state_num_mlp=2,
                 residual=False):
        '''
        :param actions:
        :param multi_label:
        '''
        super(GraphNet, self).__init__()
        # args

        self.multi_label = multi_label
        self.num_feat = num_feat
        self.num_label = num_label
        self.dropout = drop_out
        self.residual = residual
        self.batch_normal = batch_normal
        
        # check structure of GNN
        self.layer_nums_mlp = self.evalate_actions(actions_mlp,state_num_mlp)
        self.head_num = 1

        # layer module
        self.build_model(actions, batch_normal, drop_out, num_feat, num_label, state_num)

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_label, state_num):
        if self.residual:
            self.fcs = torch.nn.ModuleList()
        if self.batch_normal:
            self.bns = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.acts = []
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums_mlp, num_feat, num_label, state_num)

    def evalate_actions(self, actions_mlp,state_num_mlp):
        state_length_mlp = len(actions_mlp)
        if state_length_mlp % state_num_mlp != 0:
            raise RuntimeError("Wrong Input: unmatchable input")
        layer_nums_mlp = state_length_mlp // state_num_mlp
        if self.evaluate_structure(actions_mlp, layer_nums_mlp, state_num=state_num_mlp):
            pass
        else:
            raise RuntimeError("wrong structure")
        return layer_nums_mlp

    def evaluate_structure(self, actions_mlp, layer_nums_mlp, state_num=6):
        # Check for output gnn == input mlp
        pass


        # hidden_units_list = []
        # out_channels_list = []
        # for i in range(layer_nums_mlp):
        #     out_channels = actions[i * state_num + 0]
        #     hidden_units_list.append(self.head_num * out_channels)
        #     out_channels_list.append(out_channels)

        # return out_channels_list[-1] == self.num_label
        
    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums_mlp, num_feat, num_label, state_num=6):

        # build hidden layer
        for i in range(layer_nums_mlp):

            if i == 0:
                in_channels = num_feat
            else:
                in_channels = out_channels * self.head_num

            # extract layer information
            # attention_type = actions[i * state_num + 0]
            # aggregator_type = actions[i * state_num + 1]
            # act = actions[i * state_num + 2]
            # head_num = actions[i * state_num + 3]
            out_channels = actions[i * state_num + 0]
            concat = True
            if i == layer_nums_mlp - 1:
                concat = False
            if self.batch_normal:
                self.bns.append(torch.nn.BatchNorm1d(in_channels, momentum=0.5))
            self.layers.append(
                GeoLayer(in_channels, out_channels, self.head_num, concat, dropout=self.dropout ))
            self.acts.append(act_map())
            if self.residual:
                if concat:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels * self.head_num))
                else:
                    self.fcs.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, x, edge_index_all):
        output = x
        if self.residual:
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)

                output = act(layer(outpu