import torch


class MacroSearchSpace():
    def __init__(self):
            self.search_space_gnn = {
                'hidden_units': [8,16,32,64,128]
                          }

                          
            self.search_space_mlp = {
                'add_nodes' : [10,15,20,25,30],
                'del_nodes' : [2, 4, 6, 8, 10]
            }

    def get_search_space(self):
        return self.search_space_gnn, self.search_space_mlp

    # Assign operator category for controller RNN outputs.
    # The controller RNN will select operators from search space according to operator category.
    def generate_action_list(self, num_of_layers_gnn=2, num_of_layers_mlp=2):
        action_name_gnn = list(self.search_space_gnn.keys())
        action_name_mlp = list(self.search_space_mlp.keys())
        action_list_gnn = action_name_gnn * num_of_layers_gnn
        action_list_mlp = action_name_mlp*num_of_layers_mlp
        return action_list_gnn, action_list_mlp


def act_map():
    # if act == "linear":
    #     return lambda x: x
    # elif act == "elu":
    #     return torch.nn.functional.elu
    # elif act == "sigmoid":
    #     return torch.sigmoid
    # elif act == "tanh":
    #     return torch.tanh
    # elif act == "relu":
    #     return torch.nn.functional.relu
    # elif act == "relu6":
    #     return torch.nn.functional.relu6
    # elif act == "softplus":
    #     return torch.nn.functional.softplus
    # elif act == "leaky_relu":
    return torch.nn.functional.leaky_relu
    # else:
    #     raise Exception("wrong activate function")
