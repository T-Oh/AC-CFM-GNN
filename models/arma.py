from torch_geometric.nn import ARMAConv, global_mean_pool
from torch.nn import Module, ModuleList, Linear, Sigmoid, ReLU, BatchNorm1d



class ArmaConvModule(Module):
    def __init__(self, num_channels_in, num_channels_out, activation, num_internal_layers, num_internal_stacks, batch_norm=False, shared_weights=False, dropout=0.00):
        super(ArmaConvModule, self).__init__()
        self.activation = activation
        self.batch_norm = batch_norm
        self.conv = ARMAConv(in_channels=num_channels_in, out_channels=num_channels_out,
                             num_stacks=num_internal_stacks, num_layers=num_internal_layers, shared_weights=shared_weights, dropout=dropout)
        #self.endLinear=Linear()
        if batch_norm:
            self.batch_norm_layer = BatchNorm1d(num_channels_out)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        x = self.conv(x, edge_index=edge_index,
                      edge_weight=edge_weight.float())
        if self.batch_norm:
            x = self.batch_norm_layer(x)
        x = apply_activation(x, self.activation)
        return x


class ArmaNet_ray(Module):
    def __init__(self, num_layers, num_channels, activation, num_internal_layers, num_internal_stacks, batch_norm_index, shared_weights, dropout, final_linear_layer):
        super(ArmaNet_ray, self).__init__()
        self.batch_norm_index = convert_binary_array_to_index(batch_norm_index)
        self.final_linear_layer = final_linear_layer

        self.convlist = ModuleList()
        for i in range(0, num_layers):
            num_c_in = num_channels[i]
            num_c_out = num_channels[i+1]
            num_s = num_internal_stacks[i]
            num_l = num_internal_layers[i]
            conv = ArmaConvModule(num_channels_in=num_c_in, num_channels_out=num_c_out, activation=activation[i], num_internal_layers=num_l,
                                  num_internal_stacks=num_s, batch_norm=batch_norm_index[i], shared_weights=shared_weights, dropout=dropout)
            self.convlist.append(conv)
        if final_linear_layer:
            self.endLinear = Linear(1, 1)
        self.pool = global_mean_pool
        self.endSigmoid = Sigmoid()

    def forward(self, data):
        x = data.x
        batch = data.batch
        for i, _ in enumerate(self.convlist):
            x = self.convlist[i](data, x)
        if self.final_linear_layer:
            x = self.endLinear(x)
        x = self.pool(x, batch)
        return x

def convert_binary_array_to_index(binary_array):
    length_input = len(binary_array)
    new_array = []
    for i in range(length_input):
        if binary_array == True:
            new_array.append(i)
    return new_array

def aggregate_list_from_config(config, key_word, index_start, index_end):
    new_list = [config[key_word+str(index_start)]]
    for i in range(index_start+1, index_end+1):
        index_name = key_word + str(i)
        new_list.append(config[index_name])
    return new_list


def make_list_number_of_channels(config):
    key_word = "num_channels"
    index_start = 1
    index_end = config["num_layers"] + 1
    num_channels = aggregate_list_from_config(
        config, key_word, index_start, index_end)
    return num_channels

def make_list_Arma_internal_stacks(config):
    key_word = "ARMA::num_internal_stacks"
    index_start = 1
    index_end = config["num_layers"]
    list_internal_stacks = aggregate_list_from_config(
        config, key_word, index_start, index_end)
    return list_internal_stacks

def make_list_Arma_internal_layers(config):
    key_word = "ARMA::num_internal_layers"
    index_start = 1
    index_end = config["num_layers"]
    list_internal_layers = aggregate_list_from_config(
        config, key_word, index_start, index_end)
    return list_internal_layers


def apply_activation(x, activation):
    if activation == None:
        return x
    if activation == "None":
        return x
    if activation == "relu":
        return ReLU()(x)