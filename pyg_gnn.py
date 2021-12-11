import torch
import torch.nn.functional as F
from search_space import act_map
from pyg_gnn_layer import GraphLayer
import torch_geometric.nn as nn


class GraphNet(torch.nn.Module):

    def __init__(self, channels_gnn, channels_mlp, actions_gnn, actions_mlp, num_feat, 
                num_class, drop_out=0.6, multi_label=False, batch_normal=True, 
                state_num_gnn=1,state_num_mlp=2, residual=False):
        '''
        :param actions:
        :param multi_label:
        '''
        super().__init__()
        # args

        self.channels_gnn = channels_gnn # Enter the hidden layer values only
        self.channels_mlp = channels_mlp # Enter the hidden node values only
        self.multi_label = multi_label
        self.num_feat = num_feat
        self.num_class = num_class
        self.dropout = drop_out
        self.residual = residual
        self.batch_normal = batch_normal
        
        # check structure of GNN
        (self.layer_nums_gnn ,self.layer_nums_mlp) = self.evaluate_actions(actions_gnn, actions_mlp, state_num_gnn, state_num_mlp)
        self.head_num = 1

        # layer module
        self.build_model(actions, batch_normal, drop_out, num_feat, num_class, state_num)

    def build_model(self, actions, batch_normal, drop_out, num_feat, num_class, state_num):
        if self.residual:
            self.fcs = torch.nn.ModuleList()
        if self.batch_normal:
            self.bns = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.acts = []
        self.build_hidden_layers(actions, batch_normal, drop_out, self.layer_nums_mlp, num_feat, num_class, state_num)

    def evaluate_actions(self, actions_gnn, actions_mlp, state_num_gnn, state_num_mlp):
        state_length_gnn = len(actions_gnn)
        if state_length_gnn % state_num_gnn!=0:
            raise RuntimeError("Wrong GNN Input: unmatchable input")
        layer_nums_gnn = state_length_gnn // state_num_gnn

        state_length_mlp = len(actions_mlp)
        if state_length_mlp % state_num_mlp != 0:
            raise RuntimeError("Wrong MLP Input: unmatchable input")
        layer_nums_mlp = state_length_mlp // state_num_mlp

        assert args.num_gnn == len(args.channels_gnn), "No. of gnn channels must match channel length"
        assert args.num_mlp == len(args.channels_mlp), "No. of mlp channels must match channel length" # Check here
        return (layer_nums_gnn, layer_nums_mlp)
        
    def build_hidden_layers(self, actions, batch_normal, drop_out, layer_nums_mlp, num_feat, num_class, state_num=6):

        # build hidden layer
        self.channels_gnn.insert(0,self.num_feat)

        # for i in range(layer_nums_mlp):

        #     if i == 0:
        #         in_channels = self.num_feat
        #     else:
        #         in_channels = out_channels * self.head_num

            # extract layer information
            # attention_type = actions[i * state_num + 0]
            # aggregator_type = actions[i * state_num + 1]
            # act = actions[i * state_num + 2]
            # head_num = actions[i * state_num + 3]
            # out_channels = actions[i * state_num + 0]
            # if i == layer_nums_mlp - 1:
            #     concat = False
            # if self.batch_normal:
            #     self.bns.append(torch.nn.BatchNorm1d(in_channels, momentum=0.5))
            # self.layers.append(
            #     GraphLayer(in_channels, out_channels, self.head_num, concat, dropout=self.dropout ))
            # self.acts.append(act_map())
            # if self.residual:
            #     self.fcs.append(torch.nn.Linear(in_channels, out_channels))

    def forward(self, x, edge_index_all):
        output = x
        if self.residual:
            for i, (act, layer, fc) in enumerate(zip(self.acts, self.layers, self.fcs)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)

                output = act(layer(output, edge_index_all) + fc(output))
        else:
            for i, (act, layer) in enumerate(zip(self.acts, self.layers)):
                output = F.dropout(output, p=self.dropout, training=self.training)
                if self.batch_normal:
                    output = self.bns[i](output)
                output = act(layer(output, edge_index_all))
        if not self.multi_label:
            output = F.log_softmax(output, dim=1)
        return output

    def __repr__(self):
        result_lines = ""
        for each in self.layers:
            result_lines += str(each)
        return result_lines

    # @staticmethod
    # def merge_param(old_param, new_param, update_all):
    #     for key in new_param:
    #         if update_all or key not in old_param:
    #             old_param[key] = new_param[key]
    #     return old_param

    # def get_param_dict(self, old_param=None, update_all=True):
    #     if old_param is None:
    #         result = {}
    #     else:
    #         result = old_param
    #     for i in range(self.layer_nums):
    #         key = "layer_%d" % i
    #         new_param = self.layers[i].get_param_dict()
    #         if key in result:
    #             new_param = self.merge_param(result[key], new_param, update_all)
    #             result[key] = new_param
    #         else:
    #             result[key] = new_param
    #     if self.residual:
    #         for i, fc in enumerate(self.fcs):
    #             key = f"layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
    #             result[key] = self.fcs[i]
    #     if self.batch_normal:
    #         for i, bn in enumerate(self.bns):
    #             key = f"layer_{i}_fc_{bn.weight.size(0)}"
    #             result[key] = self.bns[i]
    #     return result

    # def load_param(self, param):
    #     if param is None:
    #         return

    #     for i in range(self.layer_nums):
    #         self.layers[i].load_param(param["layer_%d" % i])

    #     if self.residual:
    #         for i, fc in enumerate(self.fcs):
    #             key = f"layer_{i}_fc_{fc.weight.size(0)}_{fc.weight.size(1)}"
    #             if key in param:
    #                 self.fcs[i] = param[key]
    #     if self.batch_normal:
    #         for i, bn in enumerate(self.bns):
    #             key = f"layer_{i}_fc_{bn.weight.size(0)}"
    #             if key in param:
    #                 self.bns[i] = param[key]