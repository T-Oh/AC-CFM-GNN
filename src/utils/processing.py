from attr import dataclass
import torch
import numpy as np
import scipy
import h5py
import os
from torch_geometric.data import Data

from utils.enums import ZhuIdx, BusTypes
from utils.utils import zhu_perform_bus_check


@dataclass
class ProcessingConfig:
    root: str
    raw_paths: list
    processed_dir: str

    data_type: str
    edge_attr_type: str

    ls_threshold: float
    N_below_threshold: int

    normalize_injection: bool
    multiply_base_voltage: bool
    zhu_check_buses: bool
    check_s_y: bool

def get_initial_damages():
    '''
    returns the sorted initial damages of each scenario in damages [N_scenarios][step,line_id]
    where step is reassigned so that it starts with 0 and increments by 1 except if two lines 
    were destroyed in the same step
    '''
    
    #load scenario file which stores the initial damages
    f = open('raw/Hurricane_Ike_gamma8.3e-5_scenarios.txt','r')
    lines= f.readlines()
    damages = []
    for i in range(len(lines)):
        lines[i] = lines[i].replace("[", '')
        lines[i] = lines[i].replace(']', '')
        lines[i] = lines[i].replace('(', '')
        lines[i] = lines[i].replace(')', '')
        lines[i] = lines[i].replace('"', '')
        lines[i] = lines[i].replace(',', '')
        line = np.array(list(map(int, lines[i].split())))
        scenario_dmgs=np.reshape(line,(-1,2))
        scenario_dmgs=scenario_dmgs[scenario_dmgs[:,0].argsort(axis=0)]
        #rewrite the time steps to count in steps from 0 for easier handling of multiple damages in the same time step
        index = 0
        for j in range(0,len(scenario_dmgs)-1):
            increment = 0
            if scenario_dmgs[j,0] != scenario_dmgs[j+1,0]:
                increment = 1
            scenario_dmgs[j,0] = index
            index += increment
        scenario_dmgs[-1,0] = index

        damages.append(scenario_dmgs)
    return damages

def load_mat_file(file_path):
    try:
    # Attempt to load using scipy (works for MATLAB files below v7.3)
        #data = scipy.io.loadmat(file_path)
        #filetype = 'Zhu'
        return scipy.io.loadmat(file_path), 'Zhu'   #data, filetype

    except Exception as e1:
        try:    #h5py for v7.3 .mat files
            f = h5py.File(file_path, 'r')
            data = f
            return data, 'Zhu_mat73'
        except Exception as e2:
            print(f"Error loading {file_path} with h5py: {e2}")
            return None, 'LOAD_FAILED'
        
def save_static_data(cfg, processed_dir, KEY, init_data, filetype):
    node_data_pre, gen_data_pre, edge_data_pre, _, node_data_post, _ = get_data(cfg, init_data, init_data, KEY, -1, filetype)
    node_feature, node_labels, graph_label = get_node_features(cfg, node_data_pre, node_data_post, gen_data_pre)   #extract node features and labels from data  

    adj, edge_attr = get_edge_attrY(edge_data_pre, [])
    edge_labels = get_edge_labels(adj, adj, edge_attr)

    data = Data(x=node_feature.float(), edge_index=adj, edge_attr=edge_attr.float(), node_labels=node_labels.float(), y=graph_label.float(), 
                y_cummulative=torch.tensor(0).to(torch.float32), edge_labels=edge_labels.to(torch.float32))
    torch.save(data, os.path.join(processed_dir, f'data_static.pt'))
    #returns the adjacency matrix of the static data which is used to determine the edge labels in the LSTM data
    return adj

def get_scenario_of_file(name):
    """
    Input:
    name        name of the processed data file
    
    Returns:
    scenario    index of the scenario of which the datafile stems
    """
    """if name.startswith('./processed'):
        name=name[17:]
    else:
        name=name[26:]
    i=1"""
    i = 1
    while not name[i].isnumeric():
        i+=1
    j = 1
    while name[j+i].isnumeric():
        j+=1
    scenario=int(name[i:i+j])
    return scenario


def get_ls_tot(KEY, filetype, file):
    if filetype == 'Zhu_mat73':
        ls_tot_ref = file[KEY]['ls_total']
        len_scenario = len(ls_tot_ref)              
        ls_tot = []
        for step in range(len_scenario):
            ls_tot_deref = file[ls_tot_ref[step,0]]
            ls_tot.append(ls_tot_deref[()])
    else:
        len_scenario = len(file[KEY][0,:])
        ls_tot = []
        for step in range(len_scenario):
            ls_tot.append(file[KEY][0,step][21])
    return len_scenario,ls_tot


def get_node_features(
        cfg,
        node_data_pre,
        node_data_post,
        gen_data_pre,
        gen_data_post = None,
):
    '''
    extracts the unnormalized node features and labels from the raw data
    
    Input:
        node_data_pre:  node data read from matpower formatted matlab file of the initial state
        node_data_post: node data read from matpower formatted matlab files of the post cascading failure state
    Output:
        node_features:  torch.tensor of node features
        node_labels:    torch.tensor of node labels
    '''
    normalize_injection = cfg.normalize_injection
    multiply_base_voltage = cfg.multiply_base_voltage
    zhu_check_buses = cfg.zhu_check_buses

    #one hot encoded bus types
    N_BUSES = len(node_data_pre[:,2])
    bus_type = torch.zeros([N_BUSES,4], dtype=torch.int32)
    bus_type2 = torch.zeros([N_BUSES,4], dtype=torch.int32)
    for i in range(N_BUSES):
        bus_type[i, int(node_data_pre[i,1]-1)] = 1
        bus_type2[i, int(node_data_post[i,1]-1)] = 1

    P1 = torch.as_tensor(node_data_pre[:,2]) #P of all buses at initial condition - Node feature
    Q1 = torch.as_tensor(node_data_pre[:,3]) #Q of all buses at initial condition - Node feature
    S1 = np.sqrt(P1**2+Q1**2).clone().detach()
    Vm = torch.as_tensor(node_data_pre[:,7]) #Voltage magnitude of all buses at initial condition - Node feature
    Va = torch.as_tensor(node_data_pre[:,8]) #Voltage angle of all buses at initial condition - Node feature
    baseKV = torch.as_tensor(node_data_pre[:,9]) #Base Voltage
    P2 = torch.as_tensor(node_data_post[:,2]) #P of all buses after step - used for calculation of Node labels
    Q2 = torch.as_tensor(node_data_post[:,3]) #Q of all buses after step - used of calculation of Node labels
    S2 = np.sqrt(P2**2+Q2**2).clone().detach()

    if multiply_base_voltage:
        Vm = Vm*baseKV
        Vm2 = Vm2*baseKV



    if cfg.data_type in ['AC', 'n-k']:
        Bs = torch.tensor(node_data_pre[:,5]) #Shunt susceptance
        Bs[bus_type[:,BusTypes.inactive]==1] = 0
    
    #one hot encoded node IDs
    node_ID = torch.eye(N_BUSES)
    
    #adjust features of inactive buses
    P1[bus_type[:,BusTypes.inactive]==1] = 0
    Q1[bus_type[:,BusTypes.inactive]==1] = 0
    S1[bus_type[:,BusTypes.inactive]==1] = 0
    Vm[bus_type[:,BusTypes.inactive]==1] = 0
    Va[bus_type[:,BusTypes.inactive]==1] = 0

    P2[bus_type2[:,BusTypes.inactive]==1] = 0
    Q2[bus_type2[:,BusTypes.inactive]==1] = 0
    S2[bus_type2[:,BusTypes.inactive]==1] = 0
    
    
    gen_features = get_gen_features(cfg, gen_data_pre, node_data_pre)
    gen_features[bus_type[:,BusTypes.inactive]==1,:] = 0

    #node Features for AC (ANGF_CE_Y) and n-k data
    if cfg.data_type in ['AC', 'n-k']:
        node_features = torch.cat([P1.reshape(-1,1), Q1.reshape(-1,1), Vm.reshape(-1,1), Bs.reshape(-1,1), bus_type, gen_features, node_ID], dim=1)
        node_labels = torch.tensor(S1-S2)
    #Node features for Zhu data
    elif cfg.data_type in ['Zhu', 'Zhu_mat73', 'LSTM', 'Zhu_nobustype']:
        P_injection = (gen_features[:,0]-P1) #in p.u.
        Q_injection = (gen_features[:,1]-Q1) #in p.u.
        Vreal = Vm*torch.cos(np.deg2rad(Va))    #in p.u.
        Vimag = Vm*torch.sin(np.deg2rad(Va))    #in p.u.
        #ajust values to bus types according to Zhu paper
        """
        if self.data_type in ['Zhu', 'zhu_mat73']:
            P_injection = P_injection*(bus_type[:,0]+bus_type[:,1])  #P only given for PQ and PV buses
            Q_injection = Q_injection*(bus_type[:,0])  #Q only given for PQ
            Vreal = Vreal*(bus_type[:,1]+bus_type[:,2])  #V only given for PV and slack bus
            Vimag = Vimag*(bus_type[:,1]+bus_type[:,2])
        """

        Vm2 = torch.as_tensor(node_data_post[:,7]) #*node_data_post[:,9]
        Va2 = torch.as_tensor(node_data_post[:,8]) #Q of all buses after step - used of calculation of Node labels

        if normalize_injection:
            P_injection /= 100
            Q_injection /= 100

            if zhu_check_buses:
                gen_features_post = get_gen_features(cfg, gen_data_post, node_data_post)
                P2_injection = (gen_features_post[:, 0] - P2)
                Q2_injection = (gen_features_post[:, 1] - Q2)

                if normalize_injection:
                    P2_injection /= 100
                    Q2_injection /= 100

                zhu_perform_bus_check(
                    bus_type=bus_type,
                    bus_type_post=bus_type2,
                    P1_net=P_injection,
                    Q1_net=Q_injection,
                    Vm=Vm,
                    Va=Va,
                    P2_net=P2_injection,
                    Q2_net=Q2_injection,
                    Vm2=Vm2,
                    Va2=Va2,
                )

            #P_injection = P_injection * (bus_type[:, BusTypes.PQ] + bus_type[:, BusTypes.PV])  # P only given for PQ and PV buses
            #Q_injection = Q_injection * (bus_type[:, BusTypes.PQ])  # Q only given for PQ

            #Vreal = Vreal * (
            #            bus_type[:, BusTypes.PV] + bus_type[:, BusTypes.SL])  # V only given for PV and slack bus
            #Vimag = Vimag * (bus_type[:, BusTypes.PV] + bus_type[:, BusTypes.SL])


        Vm2[bus_type2[:,BusTypes.inactive]==1] = 0
        Va2[bus_type2[:,BusTypes.inactive]==1] = 0
        V2real = Vm2*torch.cos(np.deg2rad(Va2))
        V2imag = Vm2*torch.sin(np.deg2rad(Va2))

        if cfg.data_type == 'MAKSIM TYPE':
            node_features = torch.cat([P_injection.unsqueeze(1), Q_injection.unsqueeze(1), Vreal.unsqueeze(1), Vimag.unsqueeze(1), bus_type], dim=1)
        else:
            node_features = torch.cat([P_injection.unsqueeze(1), Q_injection.unsqueeze(1), Vreal.unsqueeze(1), Vimag.unsqueeze(1)], dim=1)
        node_labels = torch.cat([V2real.unsqueeze(1), V2imag.unsqueeze(1)], dim=1)     #S1-S2 is passed not to be used as node feature but for the graph labels

    #Node features for ANGF_Vcf data
    elif cfg.data_type == 'ANGF_Vcf':
        Vreal = Vm*torch.cos(np.deg2rad(Va))
        Vimag = Vm*torch.sin(np.deg2rad(Va))
        P_injection = (gen_features[:,0]-P1)
        Q_injection = (gen_features[:,1]-Q1)
        node_features = torch.cat([P_injection.reshape(-1,1), Q_injection.reshape(-1,1), Vreal.reshape(-1,1), Vimag.reshape(-1,1), bus_type, gen_features[:,2:]], dim=1)
        node_labels = torch.tensor(S1-S2)

    else:
        node_features = torch.cat([P1.reshape(-1,1), Q1.reshape(-1,1), Vm.reshape(-1,1), gen_features], dim=1)
        node_labels = torch.tensor(S1-S2)
    graph_label = (S1-S2).unsqueeze(1).sum()
        
    return node_features, node_labels, graph_label

def get_gen_features(cfg, gen_data_pre, node_data_pre):
    N_BUSES = node_data_pre.shape[0]

    if cfg.data_type in ['AC', 'n-k', 'ANGF_Vcf']:
        gen_features = torch.zeros(N_BUSES, 9)
    else: gen_features = torch.zeros(N_BUSES,2)
    node_index = 0
    for i in range(len(gen_data_pre)):
        while gen_data_pre[i,0] != node_data_pre[node_index,0]: #get the node belonging to the generator
            node_index += 1
            if node_index >= N_BUSES: node_index = 0
        if gen_data_pre[i,0] == node_data_pre[node_index,0]:

            if gen_data_pre[i,7] >0 and node_data_pre[node_index,1]!=4:    #if generator is active and bus is active
                gen_features[node_index][:2] += torch.as_tensor(gen_data_pre[i,1:3])    #only adds p and q if the generator is active since ac-cfm does not update inactive buses
                if cfg.data_type in ['AC', 'n-k', 'ANGF_Vcf']:  #Features not added for TimeSeries
                    gen_features[node_index][6] = 1
                    if gen_features[node_index][3] == 0:    gen_features[node_index][3]=torch.tensor(gen_data_pre[i,4])
                    else:                                   gen_features[node_index][3]=min([gen_features[node_index][3],torch.tensor(gen_data_pre[i,4])])
                    gen_features[node_index][4] = torch.tensor(gen_data_pre[i,5])
                    if gen_features[node_index][8] == 0:    gen_features[node_index][8]=torch.tensor(gen_data_pre[i,9])
                    else:                                   gen_features[node_index][8]=min([gen_features[node_index][8],torch.tensor(gen_data_pre[i,9])])

            elif node_data_pre[node_index,1] != 4:   #if gen is inactive but bus is active
                if cfg.data_type in ['AC', 'n-k', 'ANGF_Vcf']:  #Features not added for TimeSeries
                    gen_features[node_index][6] = gen_features[node_index][6]   #if bus is active but generator isnt leave state as is since an active gen could be connected
                    #set lower limits and voltage set point only to inactive values if there are no existing values yet
                    if gen_features[node_index][3] == 0: gen_features[node_index][3] = gen_data_pre[i,4]    #Pmin
                    if gen_features[node_index][4] == 0: gen_features[node_index][4] = gen_data_pre[i,5]    #voltage set point
                    if gen_features[node_index][8] == 0: gen_features[node_index][8] = gen_data_pre[i,9]    #Qmin  

            else:   #this case is only entered if bus is inactive then all gens should also be counted as inactive 
                gen_features[node_index][:2] = 0
                if cfg.data_type in ['AC', 'n-k', 'ANGF_Vcf']:  #Features not added for TimeSeries
                    if gen_features[node_index][3] == 0:    gen_features[node_index][3]=torch.tensor(gen_data_pre[i,4])
                    else:                                   gen_features[node_index][3]=min([gen_features[node_index][3],torch.tensor(gen_data_pre[i,4])])
                    gen_features[node_index][4] = torch.tensor(gen_data_pre[i,5])
                    gen_features[node_index][6] = 0     
                    if gen_features[node_index][8] == 0:    gen_features[node_index][8]=torch.tensor(gen_data_pre[i,9])
                    else:                                   gen_features[node_index][8]=min([gen_features[node_index][8],torch.tensor(gen_data_pre[i,9])])
                    
            if cfg.data_type in ['AC', 'n-k', 'ANGF_Vcf']:  #features that are treated equally for active and inactive busses and generatos
                gen_features[node_index][2] += torch.tensor(gen_data_pre[i,3])    
                gen_features[node_index][5] += torch.tensor(gen_data_pre[i,6])
                gen_features[node_index][7] += torch.tensor(gen_data_pre[i,8])
                

            
    if cfg.data_type in ['AC', 'n-k', 'ANGF_Vcf']:
        gen_features = torch.cat([gen_features[:,:6], gen_features[:,7:], gen_features[:,6].reshape(-1,1)], dim=1)

    
    return gen_features


def get_data(cfg, init_data, file, KEY, i, filetype):
    if i <= 0:  #in first iteration load original pwsdata as initial data 
        node_data_pre = init_data[KEY][0,0][2] 
        gen_data_pre = init_data[KEY][0,0][3]
        gen_data_post = gen_data_pre
        if cfg.edge_attr_type == 'Y':                           
            edge_data_pre = init_data[KEY][0,0][10]    #loading the added Admittance matrix instead of the edge data
            edge_IDs = init_data[KEY][0,0][4][:,:2]
        else: 
            edge_data_pre = init_data[KEY][0,0][4]

        if filetype == 'Zhu_mat73':           #Zhu_mat73 is only necessary for Ike and Harvey where the files were too big and need to be saved in the newer mat7.3 format
            node_data_post = []
            bus_data_ref = file[KEY]['bus']
            ref = bus_data_ref[i, 0]  # Get the object reference
            dereferenced_data = file[ref]  # Dereference it
            node_data_post.append(dereferenced_data[()])  # Append the actual data
            node_data_post = torch.tensor(np.array(node_data_post).squeeze()).transpose(0,1)  

            edge_data_post = []
            edge_data_ref = file[KEY]['Ybus_ext']
            ref = edge_data_ref[i,0]
            dereferenced_data = file[ref]
            edge_data_post.append(dereferenced_data[()])          
        else:
            node_data_post = file[KEY][0,i][2]   #node_data after step i for node_label_calculation
            if i == -1: edge_data_post = edge_data_pre
            else:       
                edge_data_post = file[KEY][0,i][ZhuIdx.Y_matrix]   #edge data after step i for edge_label_calculation
            

    else:
        edge_IDs = None
        if filetype == 'Zhu_mat73':
            node_data_pre = []
            bus_data_ref = file[KEY]['bus']
            ref = bus_data_ref[i-1, 0]
            dereferenced_data = file[ref]
            node_data_pre.append(dereferenced_data[()])

            gen_data_pre = []
            gen_data_ref = file[KEY]['gen']
            ref = gen_data_ref[i-1,0]
            dereferenced_data = file[ref]
            gen_data_pre.append(dereferenced_data[()])

            gen_data_post = []
            gen_data_ref_post = file[KEY]['gen']
            ref = gen_data_ref_post[i, 0]
            dereferenced_data = file[ref]
            gen_data_post.append(dereferenced_data[()])

            edge_data_pre = []
            edge_data_ref = file[KEY]['Ybus_ext']
            ref = edge_data_ref[i-1,0]
            dereferenced_data = file[ref]
            edge_data_pre.append(dereferenced_data[()])

            node_data_post = []
            bus_data_ref = file[KEY]['bus']
            ref = bus_data_ref[i, 0]  # Get the object reference
            dereferenced_data = file[ref]  # Dereference it
            node_data_post.append(dereferenced_data[()])  # Append the actual data

            edge_data_post = []
            edge_data_ref = file[KEY]['Ybus_ext']
            ref = edge_data_ref[i,0]
            dereferenced_data = file[ref]
            edge_data_post.append(dereferenced_data[()])

            node_data_pre = torch.tensor(np.array(node_data_pre).squeeze()).transpose(0,1)
            node_data_post = torch.tensor(np.array(node_data_post).squeeze()).transpose(0,1)

            # Convert edge_data to a NumPy array for processing
            edge_data_pre_array = np.array(edge_data_pre)
            edge_data_post_array = np.array(edge_data_post)

            # Check if 'dtype' exists and whether it has named fields
            if hasattr(edge_data_pre_array, 'dtype') and edge_data_pre_array.dtype.names:
                # Extract real and imaginary parts
                real_part_pre = edge_data_pre_array['real'].squeeze()
                imag_part_pre = edge_data_pre_array['imag'].squeeze()
            else:
                # No dtype field, treat the entire array as the real part
                real_part_pre = edge_data_pre_array.squeeze()
                imag_part_pre = np.zeros_like(real_part_pre)

            # Check if 'dtype' exists and whether it has named fields
            if hasattr(edge_data_post_array, 'dtype') and edge_data_post_array.dtype.names:
                # Extract real and imaginary parts
                real_part_post = edge_data_post_array['real'].squeeze()
                imag_part_post = edge_data_post_array['imag'].squeeze()
            else:
                # No dtype field, treat the entire array as the real part
                real_part_post = edge_data_post_array.squeeze()
                imag_part_post = np.zeros_like(real_part_post)

            # Create the complex tensors
            edge_data_pre = torch.complex(torch.tensor(real_part_pre), torch.tensor(imag_part_pre))
            edge_data_post = torch.complex(torch.tensor(real_part_post), torch.tensor(imag_part_post))

            gen_data_pre = torch.tensor(np.array(gen_data_pre).squeeze()).transpose(0,1)
            gen_data_post = torch.tensor(np.array(gen_data_post).squeeze()).transpose(0,1)

        else:
            node_data_pre = []
            gen_data_pre = []
            edge_data_pre = []
            edge_data_post = []

            
            node_data_pre = file[KEY][0,i-1][2]    #node_data of initial condition of step i
            gen_data_pre = file[KEY][0,i-1][3]
            gen_data_post = file[KEY][0,i-1][3]
            if cfg.edge_attr == 'Y':                           
                edge_data_pre = file[KEY][0,i-1][ZhuIdx.Y_matrix]       #loading the added Admittance matrix instead of the edge data
                edge_data_post = file[KEY][0,i][ZhuIdx.Y_matrix]       #loading the added Admittance matrix instead of the edge data
            else:
                edge_data_pre = file[KEY][0,i-1][4]         #edge data of initial condition of step i
            node_data_post = file[KEY][0,i][2]   #node_data after step i for node_label_calculation  
    return node_data_pre, gen_data_pre, edge_data_pre, edge_data_post, node_data_post, edge_IDs


def decode_damage(dmgs, step, node_IDs, edge_IDs):
    """node_IDs must be passed as a list of node IDs
    edge_IDs must be passed as a list of edge IDs
    dmgs must be passed as a list of damages with the format [[step, edge_ID], ...]"""
    decoded_damages = []

    for i in range(len(dmgs)):
        if dmgs[i,0] == step:
            dmg = dmgs[i]
            for busID in range(len(node_IDs)):
                if node_IDs[busID] == edge_IDs[dmg[1]-1, 0]:  busID_a = busID
            for busID in range(len(node_IDs)):
                if node_IDs[busID] == edge_IDs[dmg[1]-1, 1]:  busID_b = busID
            decoded_damages.append([busID_a, busID_b])

    return decoded_damages




def get_edge_attrY(edge_data, decoded_damages):
    "decoded_damages is encoded as [[bus_from, bus_to]], with python indices (0-1999)"
    #Deactivate damaged lines

    if decoded_damages != []:
        for i in range(len(decoded_damages)):
            edge_data[decoded_damages[i][0],decoded_damages[i][1]] = 0
            edge_data[decoded_damages[i][1],decoded_damages[i][0]] = 0
    # Threshold value
    threshold = 1e-8
    # Step 1: Get the indices of entries that satisfy the condition > 1e-8
    if len(edge_data) == 1: 
        if np.all(edge_data == 0):    edge_data = torch.complex(torch.tensor(np.zeros((2000,2000))), torch.tensor(np.zeros((2000,2000))))
        else:                       edge_data = torch.complex(torch.tensor(edge_data[0]['real']), torch.tensor(edge_data[0]['imag']))
        mask = np.abs(edge_data) > threshold
    else:                   
        mask = np.abs(edge_data) > threshold
    edge_index = torch.as_tensor(mask).nonzero().t()

    # Step 2: Extract the corresponding edge attributes (weights)        
    edge_attr = torch.cat([torch.as_tensor(edge_data[edge_index[0], edge_index[1]]).real.unsqueeze(1), torch.as_tensor(edge_data[edge_index[0], edge_index[1]]).imag.unsqueeze(1)], dim=1)

    return edge_index, edge_attr


def get_edge_attrY_Zhumat73(edge_data, decoded_damages):
    #Deactivate damaged lines
    for i in range(len(decoded_damages)):
        edge_data[decoded_damages[i][0],decoded_damages[i][1]] = 0
        edge_data[decoded_damages[i][1],decoded_damages[i][0]] = 0
    # Threshold value
    threshold = 1e-8

    # Step 1: Get the indices of entries that satisfy the condition > 1e-8
    mask = abs(edge_data) > threshold
    edge_index = torch.as_tensor(mask).nonzero().t()

    # Step 2: Extract the corresponding edge attributes (weights)
    edge_attr = torch.cat([torch.as_tensor(edge_data[edge_index[0], edge_index[1]]).real.unsqueeze(1), torch.as_tensor(edge_data[edge_index[0], edge_index[1]]).imag.unsqueeze(1)], dim=1) 
    #edge_attr = torch.cat([torch.complex(torch.as_tensor(edge_data[edge_index[0], edge_index[1]][0]), torch.as_tensor(edge_data[edge_index[0], edge_index[1]][1]))], dim=1)
    return edge_index, edge_attr


def get_edge_features(edge_data, damages, node_data_pre, scenario, i, n_minus_k):
    N_BUSES = node_data_pre.shape[0]

    rating = edge_data[:,5] #long term rating (MVA) - edge feature
    status = edge_data[:,10]  #1 if line is working 0 if line is not - edge feature
    resistance = edge_data[:,2]
    reactance = edge_data[:,3] 
    #power flows
    pf1 = edge_data[:,13]
    qf1 = edge_data[:,14]

    pf2 = edge_data[:,15]
    qf2 = edge_data[:,16]

    Gs = torch.tensor(node_data_pre[:,4]) #Shunt conductance
    Bs = torch.tensor(node_data_pre[:,5]) #Shunt susceptance
            
    #Adjacency Matrix
    bus_id = node_data_pre[:,0] #list of bus ids in order
    bus_from = edge_data[:,0]   
    bus_to = edge_data[:,1] 

    #initial damages
    init_dmg = torch.zeros(len(status)) #edge feature that is 0 except if the line was an initial damage during that step
    #set initially damaged lines to 1
    for step in range(len(damages[scenario])):
        if n_minus_k:
            if damages[scenario][step,0] <= i:
                init_dmg[damages[scenario][step,1]-1] = 1 
        else:       
            if damages[scenario][step,0] == i:
                init_dmg[damages[scenario][step,1]-1] = 1 
                j = 0
                while bus_from[damages[scenario][step,1]-1] == bus_from[damages[scenario][step,1]-1-j] and bus_to[damages[scenario][step,1]-1] == bus_to[damages[scenario][step,1]-1-j]:
                    init_dmg[damages[scenario][step,1]-1-j] = 1 
                    j = j+1
                j = 0
                while bus_from[damages[scenario][step,1]-1] == bus_from[damages[scenario][step,1]-1+j] and bus_to[damages[scenario][step,1]-1] == bus_to[damages[scenario][step,1]-1+j]:
                    init_dmg[damages[scenario][step,1]-1+j] = 1 
                    j = j+1

    
    #Features
    adj_from = []   #adjacency matrix from/to -> no edges appearing twice
    adj_to = []
    rating_feature = [] #new list because orig data contains multiple lines along the same edge which are combined here
    resistance_feature = []
    reactance_feature = []
    init_dmg_feature = [] #always zero except if the line was an initial damage during this step -> then 1
    
    pf_feature = []
    qf_feature = []
    #Add edges and the respective features, edges are always added in both directions, so that the pf can be directional, the other features
    #   are added to both directions
    for j in range(len(bus_from)):
        id_from = int(np.where(bus_id==bus_from[j])[0]) #bus_id where line starts
        id_to = int(np.where(bus_id==bus_to[j])[0])     #bus_id where line ends
        
        #if edge already exists recalculate (add) the features and dont add a new edge
        exists=False
        if (adj_from.count(id_from) > 0):   #check if bus from exists
            for k in range(len(adj_from)):  #check all appeareances of bus from in adj_from
                if adj_from[k] == id_from and adj_to[k] == id_to: #if bus from and bus to are at the same entry update their edge features
                    exists = True                       #mark as edge exists
                    if status[j]==1:
                        rating_feature[k] += rating[j]      #add the capacities (ratings)
                        resistance_feature[k], reactance_feature[k] =calc_total_resistance_reactance(resistance_feature[k], resistance[j], reactance_feature[k], reactance[j])
                        pf_feature[k] += pf1[j]         #add PF
                        qf_feature[k] += qf1[j]         #add PF

                    if init_dmg_feature[k] != 1:
                        init_dmg_feature[k] = init_dmg[j]

        if (adj_to.count(id_from)>0):       #check other way
            for k in range(len(adj_to)):
                if adj_to[k] == id_from and adj_from[k] == id_to:
                    exists = True

                    if status[j] == 1:
                        rating_feature[k] += rating[j]      #add the capacities (ratings)
                        resistance_feature[k], reactance_feature[k] = calc_total_resistance_reactance(resistance_feature[k],resistance[j],reactance_feature[k], reactance[j])
                        pf_feature[k] += qf2[j]
                        qf_feature[k] += qf2[j]

                    if init_dmg_feature[k] != 1:
                        init_dmg_feature[k] = init_dmg[j]
                    
        if exists: 
            continue
        #if edge does not exist yet add it in both directions
        elif status[j]==1 and init_dmg[j] == 0:
            
            #First direction
            
            adj_from.append(id_from)
            adj_to.append(id_to)

            #status_feature.append(status[j])

            init_dmg_feature.append(init_dmg[j])
            if status[j] !=0:
                pf_feature.append(pf1[j])   #pf in first directiong
                qf_feature.append(qf1[j])   #qf in first directiong
                pf_feature.append(pf2[j])   #pf in opposite direction
                qf_feature.append(qf2[j])   #qf in opposite direction
                resistance_feature.append(resistance[j])
                reactance_feature.append(reactance[j])
                rating_feature.append(rating[j])
            else:
                pf_feature.append(0)   #if line inactive set power flows to 0 for both directions
                qf_feature.append(0)   
                pf_feature.append(0)   
                qf_feature.append(0)   
                resistance_feature.append(1)
                reactance_feature.append(1)
                rating_feature.append(0)
            #Opposite direction
            adj_from.append(id_to)
            adj_to.append(id_from)
            rating_feature.append(rating[j])
            #status_feature.append(status[j])
            resistance_feature.append(resistance[j])
            reactance_feature.append(reactance[j])
            init_dmg_feature.append(init_dmg[j])
        


    adj = torch.tensor([adj_from,adj_to])

    if self.edge_attr == 'Y':
        impedance = torch.tensor([resistance_feature, reactance_feature])
        impedance = torch.transpose(impedance, 0, 1).contiguous()   #(5154, 2)
        impedance_complex = torch.view_as_complex(impedance)        #(5154)
        admittance_complex = torch.reciprocal(impedance_complex)    #(5154)
        
        #edge_attr = - torch.view_as_real(admittance_complex)
        edge_attr = -admittance_complex

        Y = torch.zeros((N_BUSES,N_BUSES), dtype=torch.cfloat)
        for idx, edge in enumerate(adj.t().tolist()):                
            source, target = edge
            Y[source, target] = - admittance_complex[idx]
        admittance_sum = torch.sum(Y, dim=0) #(N_BUSES), contains (y12 + y13, y12 + y23, ...)
        
        self_admittance = torch.complex(Gs, Bs) + admittance_sum
        #self_admittance = self_admittance + torch.view_as_real(admittance_sum) #DO POPRAWKI
        
        edge_attr = torch.cat([edge_attr, self_admittance], dim=1)
        #edge_attr = torch.transpose(edge_attr, 0, 1)
        #edge_attr = np.sqrt(edge_attr[0,:]**2+edge_attr[1,:]**2).clone().detach()

        self_connections = torch.stack([torch.arange(N_BUSES), torch.arange(N_BUSES)], dim=0)
        adj = torch.cat([adj, self_connections], dim=1)
    else:
        edge_attr = torch.tensor([rating_feature, pf_feature, qf_feature, resistance_feature, reactance_feature, init_dmg_feature])
        edge_attr = torch.transpose(edge_attr,0,1)

    
    return adj, edge_attr



def get_edge_labels(adj_init, adj_post, edge_attr_post):
    '''
    Calculates the binary edge labels for the LSTM data.
    
    Parameters:
    - adj_init: torch.Tensor (2, E_init) - Initial edge_index
    - adj_post: torch.Tensor (2, E_post) - Updated edge_index
    - edge_attr_post: torch.Tensor (E_post,) - Updated edge attributes (e.g., admittance)
    - threshold: float - Threshold for determining edge labels
    
    Returns:
    - edge_labels: torch.Tensor (E_init,) - Binary labels for edges in adj_init, 1 means edge exists
    '''
    
    threshold = 1e-8
    # Convert edges to set for fast lookup
    adj_post_set = {tuple(edge.tolist()) for edge in adj_post.T}

    # Edge labels
    edge_labels = torch.zeros(adj_init.shape[1], dtype=torch.long)

    # Iterate over edges in the initial adjacency matrix
    for i, edge in enumerate(adj_init.T):
        edge_tuple = tuple(edge.tolist())

        if edge_tuple in adj_post_set:
            # Get index of the edge in adj_post
            idx = (adj_post.T == edge).all(dim=1).nonzero(as_tuple=True)[0]
            if len(idx) > 0:  # Edge exists in updated graph
                edge_labels[i] = 1 if abs(edge_attr_post[idx[0],0]) >= threshold or abs(edge_attr_post[idx[0],1]) >= threshold  else 0

    return edge_labels


def calc_total_resistance_reactance( r1, r2, x1, x2):
    #calculates the total resistance of the existing edge and the line to be added
    a = (r1**2-x1**2)*(r2**2-x2**2)
    G = (r1*r2**2-r1*x2**2+r2*r1**2-r2*x1**2)/a
    B = (x1*x2**2-x1*r2**2-x2*r1**2+x2*x1**2)/a
    r_new = G/(G**2+B**2)
    x_new = B/(G**2 + B**2)
    return r_new, x_new


def damages_from_features(edge_data: np.ndarray):
    inactive_mask = edge_data[:, 10] == 0
    inactive_edges = edge_data[inactive_mask, :2].astype(int)

    return inactive_edges.tolist()           

    

def get_initial_damages_dc(self, scenario_dmgs):
    """
    Extract and process initial damages for a given scenario.
    This function takes damage data for a scenario and reformats it for easier handling.
    It returns a list of damaged edges with time steps reindexed to count sequentially from 0,
    which simplifies processing multiple damages occurring in the same time step.
    Parameters
    ----------
    scenario_dmgs : array-like
        A 2D array where each row contains [edge_ID, time_step] pairs representing damages.
        The input is expected to be convertible to a numpy integer array.
    Returns
    -------
    np.ndarray
        A 2D numpy integer array with shape (n_damages, 2) where each row is [time_step, edge_ID].
        Time steps are reindexed to count sequentially from 0, grouping damages by their
        original time step but renumbering them as [0, 1, 2, ...].
    Notes
    -----
    - Columns are swapped: input [edge_ID, time_step] becomes [time_step, edge_ID]
    - Array is sorted by time step (first column)
    - Time steps are reindexed to increment only when transitioning to a new original time step
    """


    scenario_dmgs = np.array(scenario_dmgs).astype(int)
    scenario_dmgs[:,[0,1]] = scenario_dmgs[:,[1,0]]
    scenario_dmgs = scenario_dmgs[scenario_dmgs[:,0].argsort(axis=0)]        

    #rewrite the time steps to count in steps from 0 for easier handling of multiple damages in the same time step
    index = 0
    for j in range(0,len(scenario_dmgs)-1):
        increment = 0
        if scenario_dmgs[j,0] != scenario_dmgs[j+1,0]:
            increment = 1
        scenario_dmgs[j,0] = index
        index += increment
    scenario_dmgs[-1,0] = index

    return scenario_dmgs

            
