import torch
import numpy as np
import networkx as nx
import scipy.io
import os

class state_loss(torch.nn.Module):
    def __init__(self, edge_factor, dataset_path, weight_nodes_by_centrality, weight_nodes_by_voltage_threshold=False, voltage_weight=2.0):
        super(state_loss, self).__init__()
        self.edge_factor = edge_factor
        self.node_loss = torch.nn.MSELoss(reduction='none')
        self.edge_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.weight_nodes_by_centrality = weight_nodes_by_centrality
        self.weight_nodes_by_voltage_threshold = weight_nodes_by_voltage_threshold
        if weight_nodes_by_voltage_threshold:
            #weighting by voltage threshold applies the weight both when imag and/or real part of the voltage is below/above threshold
            #if both are true the weight is applied twice
            data = torch.load(os.path.join(dataset_path, 'processed/data_static.pt'))
            voltage_real = data.x[:,2]  
            voltage_imag = data.x[:,3]  
            weight_mask = voltage_real > 0.7    #thresholds chosen from according scatterplot from check_for_hard_nodes_aggregate.py (Documentation_Time_series plots 33 (c) and (d))
            weight_mask_imag = voltage_imag < 0.31

            self.voltage_node_weights = (torch.ones(2000) + (voltage_weight-1.0) * (weight_mask.float() + weight_mask_imag.float()))
            self.voltage_node_weights = self.voltage_node_weights / self.voltage_node_weights.mean()
            print(self.voltage_node_weights)
            print(self.voltage_node_weights.max())

        if weight_nodes_by_centrality:
            print('Weighting Node Loss by Betweenness Centrality')
            data = torch.load(os.path.join(dataset_path, 'processed/data_static.pt'))
            edge_index = data.edge_index.numpy()
            edge_weight = np.sqrt(data.edge_attr[:, 0].numpy()**2 + data.edge_attr[:,1].numpy()**2) if data.edge_attr is not None else np.ones(edge_index.shape[1])

            G = nx.Graph()
            G.add_nodes_from(range(data.x.size(0)))
            for (u, v), w in zip(edge_index.T, edge_weight):
                G.add_edge(int(u), int(v), weight=float(w))
            betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
            self.centrality_node_weights = torch.tensor(
            [betweenness[i] for i in range(2000)]
            )
            self.centrality_node_weights = self.centrality_node_weights / self.centrality_node_weights.mean()
        print(f'Using State Loss with edge factor {self.edge_factor}')

    def forward(self, node_output, edge_output, node_labels, edge_labels):
        raw_node_loss = self.node_loss(node_output, node_labels)
        if self.weight_nodes_by_centrality:
            #print('Weighting node loss by betweenness centrality')
            weights = torch.tensor(
            [self.centrality_node_weights[i] for i in range(2000)],
            device=node_output.device,
            dtype=node_output.dtype
            )
            weights = weights.repeat(int(len(node_output)/2000))
            # if raw_node_loss has shape [num_nodes, features], average across features first
            if raw_node_loss.dim() > 1:
                """num_blocks = len(node_output) // 2000
                losses = []
                for i in range(num_blocks):
                    block_loss = (raw_node_loss[i*2000:(i+1)*2000] * weights).mean()
                    losses.append(block_loss)
                weighted_node_loss = torch.stack(losses).mean()"""

                """raw_node_loss = raw_node_loss.mean(dim=1)
                for i in range(int(len(node_output)/2000)):
                    weighted_node_loss = (raw_node_loss[i*2000:(i+1)*2000] * weights).mean()"""
                raw_node_loss = raw_node_loss.mean(dim=1)
            weighted_node_loss = (raw_node_loss * weights).mean()
        else:
            weighted_node_loss = raw_node_loss.mean()

        if self.weight_nodes_by_voltage_threshold:
            print('Weighting node loss by voltage thresholding')
            weights = self.voltage_node_weights.to(node_output.device).to(node_output.dtype)
            weights = weights.repeat(int(len(node_output)/2000))
            # if raw_node_loss has shape [num_nodes, features], average across features first
            if raw_node_loss.dim() > 1:
                raw_node_loss = raw_node_loss.mean(dim=1)
            weighted_node_loss = (raw_node_loss * weights).mean()

        edge_loss = self.edge_loss(edge_output, edge_labels.reshape(-1))
        loss = weighted_node_loss + edge_loss*self.edge_factor
        return loss, weighted_node_loss, edge_loss
    
class state_loss_power_injection(torch.nn.Module):
    def __init__(self, cfg, edge_factor, PI_factor, device):
        """
        Initializes the state loss with an edge factor and provides the necessary static information for the power injection loss.
        Parameters
        ----------
        edge_factor : float
            Factor to scale the edge loss.
        b_edge_attr : torch.Tensor, optional
            Edge attributes representing the shunt admittance (jb/2) for each edge.
        Y_raw : torch.Tensor
            Raw admittance matrix of the grid, used to build the Y matrix based on the edge predictions. Used to calculate the power injections.
        basekV : torch.Tensor
            Base voltage of the grid, used to calculate the power injections.
        min_max : dict
            Dictionary containing the min and max values for denormalization of the predictions.
        """

        super(state_loss_power_injection, self).__init__()
        self.edge_factor = edge_factor
        self.PI_factor = PI_factor
        self.node_loss = torch.nn.MSELoss(reduction='mean')
        self.edge_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.PI_loss = torch.nn.MSELoss(reduction='mean')



        self.device=device
        #Things needed for the power injection loss
        static_data = torch.load(os.path.join(cfg['dataset::path'], 'processed/data_static.pt'))    #Used for pytorch branch IDs of fully functioning grid
        pwsdata = scipy.io.loadmat(os.path.join(cfg['dataset::path'], 'raw/pwsdata.mat'))    #  
        self.edge_index = static_data.edge_index
        self.Y_raw = torch.tensor(pwsdata['clusterresult_'][0,0][10] )
        self.Y_raw = torch.complex(torch.tensor(self.Y_raw.real), torch.tensor(self.Y_raw.imag)).type(torch.complex64).to(self.device)

     
        bus_IDs = torch.tensor(pwsdata['clusterresult_'][0,0][2][:,0] )
        branch_data = pwsdata['clusterresult_'][0,0][4]
        b = torch.tensor(branch_data[:,4] )
        self.b_edge_attr = self.create_b_edge_attr(bus_IDs, branch_data, self.edge_index, b).to(self.device)

        print(f'Using State Loss with edge factor {self.edge_factor} and PI factor {self.PI_factor}')

    def forward(self, node_output, edge_output, node_labels, edge_labels):
        #denormalized_output, denormalized_labels = self.denormalize((node_output, edge_output), (node_labels, edge_labels))
        node_loss = self.node_loss(node_output, node_labels)

        edge_loss = self.edge_loss(edge_output, edge_labels.reshape(-1))

        S = self.calculate_S((node_output, edge_output), (node_labels, edge_labels), use_edge_labels=True)
        S_true = self.calculate_S((node_labels, edge_labels), (node_labels, edge_labels), use_edge_labels=True)
        PI_loss = self.PI_loss(torch.view_as_real(S), torch.view_as_real(S_true))
        #PI_R2 = r2_score(torch.view_as_real(S_true).cpu().detach().numpy(), torch.view_as_real(S).cpu().detach().numpy())

        loss = node_loss + self.edge_factor*edge_loss + self.PI_factor*PI_loss
        return loss, node_loss, edge_loss, PI_loss


    def build_Y_matrix_from_predictions(self, edge_predictions):
        Y = self.Y_raw.clone()
        Y = Y.to(torch.complex64)
        inactive_edges = torch.where(edge_predictions.flatten() == 0)[0]

        for idx in inactive_edges:
            #print('inactive_edges: ', len(inactive_edges))
            #print(idx)
            i, j = self.edge_index[:,idx]
            y_ij = Y[i, j]
    
            if i!=j:
                Y[i, i] += y_ij - self.b_edge_attr[idx]
                Y[i, j] = 0


        Y[abs(Y.real)<0.001] = 0j
        return Y #torch.complex(torch.tensor(Y.real), torch.tensor(Y.imag)).type(torch.complex64).to(self.device) 
    
    def calculate_S(self, output, labels, use_edge_labels):


        S_all = []


        for i in range(int(len(output[0])/2000)):
            if use_edge_labels:
                Y_instance = self.build_Y_matrix_from_predictions(labels[1][i])
            else:
                instance_output = (output[0], torch.nn.functional.gumbel_softmax(output[1][i*7064:(i+1)*7064], tau=1.0, hard=True, dim=1))  # shape: [batch_size]
                Y_instance = self.build_Y_matrix_from_predictions(instance_output[1][:,0])

            V = output[0][i*2000:(i+1)*2000,:].float()
            V = torch.complex(V[:, 0], V[:, 1])

            YV= Y_instance.to(dtype=torch.complex64) @ V.to(dtype=torch.complex64)
            S = V * YV.conj()

            S_all.append(S)

        return torch.stack(S_all)
    
    def create_b_edge_attr(self, bus_IDs, branch_data, edge_index, b):
        bus_id_map = {int(bus_id): idx for idx, bus_id in enumerate(bus_IDs)}
        from_buses_raw = branch_data[:, 0].astype(int)
        to_buses_raw   = branch_data[:, 1].astype(int)

        line_to_shunt = {}
        for fb_raw, tb_raw, b in zip(from_buses_raw, to_buses_raw, b):
            fb = bus_id_map[fb_raw]
            tb = bus_id_map[tb_raw]
            i, j = sorted((fb, tb))
            line_to_shunt[(i, j)] = 1j * b / 2  # jb/2

    # Step 4: Assign jb/2 to each edge in edge_index
        shunt_attr = torch.zeros(edge_index.shape[1], dtype=torch.cfloat)
        for k in range(edge_index.shape[1]):
            i = edge_index[0, k].item()
            j = edge_index[1, k].item()
            key = tuple(sorted((i, j)))
            if key in line_to_shunt:
                shunt_attr[k] = torch.tensor(line_to_shunt[key], dtype=torch.cfloat)
        return shunt_attr



class weighted_loss_label(torch.nn.Module):
    """
    weights the loss with a constant factor depending on wether the label is >0 or not
    """
    def __init__(self, factor):
        super(weighted_loss_label, self).__init__()
        self.factor = torch.sqrt(factor)
        self.base_loss= torch.nn.MSELoss(reduction='mean')


    def forward(self, output, labels):
        print('Using weighted loss label')
        output_ = output.clone()
        labels_ = labels.clone()
        output_[labels>0] = output_[labels>0]*self.factor
        labels_[labels>0] = labels_[labels>0].clone()*self.factor
        return self.base_loss(self.factor*output_,self.factor*labels_)


class weighted_loss_var(torch.nn.Module):
    """
    weights the loss depending on the label variance at each node
    """
    def __init__(self, var, device):
        super(weighted_loss_var, self).__init__()
        self.weights = torch.sqrt(var).to(device)
        self.baseloss = torch.nn.MSELoss(reduction='mean').to(device)

    def forward(self, output ,labels):
        output_ = output.reshape(int(len(output)/len(self.weights)),len(self.weights))*self.weights
        labels_ = labels.reshape(int(len(output)/len(self.weights)),len(self.weights))*self.weights
        return self.baseloss(output_.reshape(-1), labels_.reshape(-1))
    


class physics_loss(torch.nn.Module):
    """
    Physics-Informed Loss function from:
    https://ieeexplore.ieee.org/document/9881910
    """

    def __init__(self, w1, w2, w3, device):
        super(physics_loss, self).__init__()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.device = device

    def forward(self, batch, outputs):
        N_BUSES = 2000
        Vpred = torch.view_as_complex(outputs.type(torch.float32)).to(self.device).reshape(-1, N_BUSES, 1).to(
            torch.complex64)
        # print(f'{Vpred=}')
        node_features = batch.x.view(-1, N_BUSES, 6)
        Y_matrix_reshaped = batch.admittance_matrix.reshape(-1, N_BUSES, N_BUSES).to(torch.complex64)
        # print(f'{Y_matrix_reshaped=}')
        S = torch.view_as_complex(node_features[:, :, 0:2].contiguous())  # (2000, 2)
        # S = S/100
        # print(f'{S=}')



        Vm = node_features[:, :, 2]

        bus_type = node_features[:, :, 3:6]

        '''using magnitude-angle representation for complex labels instead of real-imag'''
        # Vreal = outputs[:, 0] * torch.cos(outputs[:, 1])
        # Vimag = outputs[:, 0] * torch.sin(outputs[:, 1])
        # Vpred = torch.view_as_complex(torch.cat((Vreal.unsqueeze(1), Vimag.unsqueeze(1)), dim=1)).reshape(-1, N_BUSES, 1).to(torch.complex64)
        # print(f'{Vpred.dtype=}')
        '''end of alternative magn-angle representation'''
        # print(f'{Y_matrix_reshaped.dtype=}')
        # print(f'{torch.bmm(Y_matrix_reshaped, Vpred).dtype=}')
        # print(f'{torch.bmm(Y_matrix_reshaped, Vpred)=}')

        Spred = torch.mul(Vpred, torch.conj(torch.bmm(Y_matrix_reshaped, Vpred)))
        # print(f'{Spred.size()=}')
        # print(f'{Spred=}')
        # print(f'{torch.mul(S, bus_type[:,:, 0])=}')

        S_mean = S.mean(dim=1, keepdim=True)
        S_inverse = torch.where(S != 0, 1.0 / S, 1.0 / S_mean)
        S_inverse_norm = torch.abs(S_inverse)
        # print(f'{S_inverse_norm=}')
        Spred = torch.squeeze(Spred, 2)

        result1 = torch.mul(torch.abs(Spred - S), bus_type[:, :, 0])
        # print(f'{result1=}')
        result2 = torch.mul(torch.abs(Spred.real - S.real), bus_type[:, :, 1])
        # print(f'{result2=}')
        Vm_mean = Vm.mean(dim=1, keepdim=True)
        Vm_inverse = torch.where(Vm != 0, 1.0 / Vm, 1.0 / Vm_mean)

        D1 = torch.mul(S_inverse_norm, result1 + result2)
        D2 = torch.abs(torch.mul(torch.abs(Vpred.squeeze(2)), bus_type[:, :, 1] + bus_type[:, :, 2]) - Vm) * Vm_inverse
        D3 = torch.abs(Vpred.imag.squeeze(2) * bus_type[:, :, 2]) * Vm_inverse
        print(f'{D1.mean()=}')
        print(f'{D2.mean()=}')
        print(f'{D3.mean()=}')
        physics_loss = self.w1 * D1.mean() + self.w2 * D2.mean() + self.w3 * D3.mean()
        print(f'{physics_loss=}')
        return physics_loss, D1.mean(), D2.mean(), D3.mean()


class MSE_plus_physics_loss(torch.nn.Module):
    """
    MSE plus lmbda*physics_loss
    """

    def __init__(self, w1, w2, w3, lmbda, device):
        super(MSE_plus_physics_loss, self).__init__()
        # self.w1 = torch.nn.Parameter(torch.tensor(w1))
        # self.w2 = torch.nn.Parameter(torch.tensor(w2))
        # self.w3 = torch.nn.Parameter(torch.tensor(w3))
        # self.w1 = torch.cuda.FloatTensor([w1])
        # self.w2 = torch.cuda.FloatTensor([w2])
        # self.w3 = torch.cuda.FloatTensor([w3])
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.lmbda = lmbda
        self.device = device

    def forward(self, batch, outputs, labels):
        '''
        model inputs:
        S = P + iQ given by P, Q
        Vm voltage magnitudes
        one-hot encoding of bus type (ignore fourth column corr. to isolated buses)
                   PQ bus          = 0
                   PV bus          = 1
                   reference bus   = 2
                   isolated bus    = 3
        model outputs:
        Vpred given by Vpred_real and Vpred_imag
        '''
        print('HELLO MSE PLUS PHYSICS LOSS')
        Vpred = torch.view_as_complex(outputs.type(torch.float32))
        Vpred = Vpred.to(torch.complex64)
        Vpred = Vpred.to(self.device)
        Vpred = Vpred.reshape(-1, 2000, 1)

        node_features = batch.x
        edge_features = batch.edge_attr
        edge_index = batch.edge_index
        # print('dirichlet energy', dirichlet_energy(edge_index, node_features))
        print(f'{node_features.size()=}')
        print(f'{edge_features.size()=}')
        print(f'{edge_index.size()=}')
        print(f'{edge_index=}')

        node_features = node_features.view(-1, 2000, 6)
        print(f'{node_features.size()=}')
        batchsize = node_features.size(0)
        print(f'{batchsize=}')

        edge_features = edge_features.view(batchsize, -1, 2)
        print(f'{edge_features.size()=}')
        n_edges = edge_features.shape[1]
        # edge_index_reshaped = torch.tensor([edge_index[:, i:i+n_edges] for i in range(0, edge_index.shape[1], n_edges)])
        edge_index_reshaped = edge_index.view(2, batchsize, n_edges).permute(1, 0, 2)
        edge_index = edge_index_reshaped % 2000
        print(f'{edge_index.size()=}')
        print(f'{edge_index=}')

        S = torch.view_as_complex(node_features[:, :, 0:2].contiguous())  # (2000, 2)
        # S = torch.view_as_complex(node_features[:, 0:2].contiguous()) #(2000, 2)
        print(f'{S.size()=}')
        Vm = node_features[:, :, 2]
        print(f'{Vm.size()=}')

        bus_type = node_features[:, :, 3:6]
        print(f'{bus_type.size()=}')

        admittance = torch.view_as_complex(edge_features).to(torch.complex64)  # (batchsize, n_edges)
        print(f'{admittance.size()=}')
        print('admittance', admittance)
        Y = torch.zeros((batchsize, 2000, 2000), dtype=torch.complex64, device=self.device)

        # Edge indexing (batchsize, 2, n_edges) is split into source and target tensors
        sources = edge_index[:, 0, :]  # (batchsize, n_edges)
        targets = edge_index[:, 1, :]  # (batchsize, n_edges)

        # Use advanced indexing to directly assign the admittance values to the Y matrix
        Y[torch.arange(batchsize).unsqueeze(1), sources, targets] = admittance

        print(f'{Vpred.size()=}')
        print(f'{Vpred=}')
        print(f'{Y.size()=}')
        print(f'{Y=}')
        print(f'{torch.bmm(Y, Vpred)=}')
        print(f'{torch.bmm(Y, Vpred).size()=}')
        print(f'{torch.conj(torch.bmm(Y, Vpred))=}')
        print(f'{torch.conj(torch.bmm(Y, Vpred)).size()=}')

        Spred = torch.mul(Vpred, torch.conj(torch.bmm(Y, Vpred)))  # .to(device=self.device)
        S_mean = torch.mean(S, dim=1)
        print('S_mean', S_mean.size())
        S_mean = torch.unsqueeze(S_mean, 1)
        # print('S_mean', S_mean.size())
        S_inverse = torch.where(S != 0, 1.0 / S, 1.0 / S_mean)
        S_inverse_norm = torch.abs(S_inverse)
        print(f'{S_inverse_norm=}')

        # print(f'{Spred.size()=}')
        # print(f'{S.size()=}')
        Spred = torch.squeeze(Spred, 2)
        # print(f'{S.size()=}')

        # print(f'{torch.abs(Spred - S).size()=}')
        # print(f'{bus_type[:,:,0].size()=}')
        print(f'{Spred=}')
        print(f'{S=}')
        result1 = torch.mul((Spred - S) ** 2, bus_type[:, :, 0])
        result2 = torch.mul((Spred.real - S.real) ** 2, bus_type[:, :, 1])
        print(f'{result1=}')
        print(f'{result2=}')

        # print('Vm size', Vm.size())
        Vm_mean = torch.mean(Vm, dim=1)
        Vm_mean = torch.unsqueeze(Vm_mean, 1)
        Vm_inverse = torch.where(Vm != 0, 1.0 / Vm, 1.0 / Vm_mean)
        D1 = torch.mul(S_inverse_norm, result1 + result2)

        print('torch.abs(Vpred)', torch.abs(Vpred).size())
        print('torch.mul(torch.abs(Vpred), bus_type[:,:,1] + bus_type[:,:,2])',
              torch.mul(torch.squeeze(torch.abs(Vpred), 2), bus_type[:, :, 1] + bus_type[:, :, 2]).size())
        D2 = torch.mul(
            torch.abs(torch.mul(torch.squeeze(torch.abs(Vpred), 2), bus_type[:, :, 1] + bus_type[:, :, 2]) - Vm),
            Vm_inverse)
        print('Vm_inverse', Vm_inverse.size())
        print('torch.squeeze(Vpred.imag, 1),', torch.squeeze(Vpred.imag, 1).size())
        print('bus_type[:,:,2]', bus_type[:, :, 2].size())
        print('torch.mul(torch.squeeze(Vpred.imag, 2), bus_type[:,:,2]))',
              torch.mul(torch.squeeze(Vpred.imag, 2), bus_type[:, :, 2]).size())
        D3 = torch.mul(torch.abs(torch.mul(torch.squeeze(Vpred.imag, 2), bus_type[:, :, 2])), Vm_inverse)
        print('D1 size', D1.size())
        print('D2 size', D2.size())
        print('D3 size', D3.size())

        print('D1', torch.mean(D1))
        print('D2', torch.mean(D2))
        print('D3', torch.mean(D3))

        physics_loss = self.w1 * torch.mean(D1) + self.w2 * torch.mean(D2) + self.w3 * torch.mean(D3)
        print('loss size', physics_loss.size())

        MSE_loss = F.mse_loss(outputs, labels)
        print(f'{MSE_loss.size()=}')
        loss = MSE_loss + self.lmbda * physics_loss

        return loss, physics_loss, torch.mean(D1), torch.mean(D2), torch.mean(D3)
    
