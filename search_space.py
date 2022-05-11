
# class MacroSearchSpace():
#     def __init__(self):
#             self.search_space_gnn = {
#                 'gnn_add_nodes': [8,9,10,11,12],
#                 'gnn_del_nodes': [4,5, 6, 7, 8]
#             }

                          
#             self.search_space_mlp = {
#                 'mlp_add_nodes' : [10,11,12,13,14],
#                 'mlp_del_nodes' : [2, 4, 6, 8, 10]
#             }

#     def get_search_space(self):
#         return self.search_space_gnn, self.search_space_mlp

#     # Assign operator category for controller RNN outputs.
#     # The controller RNN will select operators from search space according to operator category.
#     def generate_action_list(self, num_of_layers_gnn=2, num_of_layers_mlp=2):
#         action_name_gnn = list(self.search_space_gnn.keys())
#         action_name_mlp = list(self.search_space_mlp.keys())
#         action_list_gnn = action_name_gnn * num_of_layers_gnn
#         action_list_mlp = action_name_mlp*num_of_layers_mlp
#         return action_list_gnn, action_list_mlp


class MacroSearchSpace():
    def __init__(self):
            self.search_space_gnn = {
                'gnn_add_nodes': [9,11,15,21,29],
                'gnn_del_nodes': [1,2,5,8,9]
            }


    def get_search_space(self):
        return self.search_space_gnn

    # Assign operator category for controller RNN outputs.
    # The controller RNN will select operators from search space according to operator category.
    def generate_action_list(self, num_of_layers_gnn=2):
        action_name_gnn = list(self.search_space_gnn.keys())
        action_list_gnn = action_name_gnn * num_of_layers_gnn
        return action_list_gnn