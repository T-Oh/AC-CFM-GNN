def process_dc(self):
        """
        Loads the raw numpy data, converts it to torch
        tensors and normalizes the labels to lie in
        the interval [0,1].
        Then pre-filters and pre-transforms are applied
        and the data is saved in the processed directory.
        """
        #get limits for scaling

        x_min, x_max, x_means, edge_attr_min, edge_attr_max, edge_attr_means, node_labels_min, node_labels_max, node_labels_means, graph_label_min, graph_label_max, graph_label_mean = self.get_min_max_features()
        x_stds, edge_stds = self.get_feature_stds(x_means, edge_attr_means)
        ymax=torch.log(self.get_max_label()+1)
        
        #process data
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            raw_data = np.load(raw_path)
            
            #scale node features
            x = torch.from_numpy(raw_data["x"]).float()
            x[:,0]=(x[:,0]-x_means[0])/x_stds[0]/((x_max[0]-x_means[0])/x_stds[0])
            x[:,1]=(x[:,1]-x_means[1])/x_stds[1]/((x_max[1]-x_means[1])/x_stds[1])
            
            #add supernode
            if self.use_supernode:
                x=torch.cat((x,torch.tensor([[1,1]])),0)            
            
            #get and scale labels according to task (GraphReg or NodeReg)
            y = torch.log(torch.tensor([raw_data["y"].item()]).float()+1)/ymax
            if 'node_labels' in raw_data.keys():
                node_labels=torch.from_numpy(raw_data['node_labels'])
                node_labels= torch.log(node_labels+1)/torch.log(node_labels_max+1)
                node_labels.type(torch.FloatTensor)
            
            #get and scale edges and adjacency matrix
            adj = torch.from_numpy(raw_data["adj"])  
            edge_attr = torch.from_numpy(raw_data["edge_weights"])

            if edge_attr.shape[0]<3:    #multidimensional edge features 
                edge_attr[0]=torch.log(edge_attr[0]+1)/torch.log(edge_attr_max+1)                
                adj, edge_attr = to_undirected(adj,[edge_attr[0].float(),edge_attr[1].float()])
                edge_attr=torch.stack(edge_attr)

                
            else:                
                #transform to undirected graph            
                adj, edge_attr = to_undirected(adj, edge_attr)
            

            #add supernode edges
            if self.use_supernode:
                Nnodes=len(x)
                supernode_edges=torch.zeros(2,Nnodes-1)
                for i in range(len(x)-1):
                    supernode_edges[0,i]=i
                    supernode_edges[1,i]=Nnodes-1
                adj = torch.cat((adj,supernode_edges),1)
                supernode_edge_attr=torch.zeros(supernode_edges.shape[1])+1
                edge_attr=torch.cat((edge_attr,supernode_edge_attr),0)
            

            print('Using Homogenous Data')
            if 'node_labels' in raw_data.keys():
                data = Data(x=x, y=y, edge_index=adj, edge_attr=torch.transpose(edge_attr,0,1),node_labels=node_labels)
            else:
                data = Data(x=x, y=y, edge_index=adj, edge_attr=torch.transpose(edge_attr,0,1))
            

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            print(data)
            #torch.save(data, os.path.join(self.processed_dir, f'data_{idx}.pt'))
            torch.save(data, os.path.join(self.processed_dir, 'data_'+raw_path[15:-4]+'.pt'))        
           
    def get_max_label(self):
        "Computes the max of all labels in the dataset"
        ys = np.zeros(len(self.raw_file_names))
        for i, raw_path in enumerate(self.raw_paths):
            ys[i] = torch.from_numpy(np.load(raw_path)["y"])

        return torch.tensor(ys.max())


    def get_min_max_features(self):
        """
        Returns the min and max of the features
        Probably deprecated as normalization is not part of dataset anymore
        """
        x_max=torch.zeros(2)
        x_min=torch.zeros(2)
        x_means = torch.zeros(2)
        edge_attr_max=0
        edge_attr_min=0
        edge_attr_means = 0
        node_count = 0
        edge_count = 0
        for i in range(5):
            edge_attr_max[i] =  np.NINF
            edge_attr_min[i] = np.Inf
            if i <2:
                x_max[i] = np.NINF
                x_min[i] = np.Inf
        node_labels_max=0
        node_labels_min=1e6
        node_labels_mean = 0
        graph_label_max = np.NINF
        graph_label_min = np.Inf
        graph_label_mean = 0
        graph_count = 0
        
        
        for file in self.raw_paths:
            # Read data from `raw_path`.
            
       
            graph_count += 1
            data = np.load(file)
            #data = torch.load(self.processed_dir +'/' + file)
            x = data['x']
            for i in range(x.shape[0]):
                if x[i,0]>x_max[0]: x_max[0]=x[i,0]
                if x[i,0]<x_min[0]: x_min[0]=x[i,0]
                if x[i,1]>x_max[1]: x_max[1]=x[i,1]
                if x[i,1]<x_min[1]: x_min[1]=x[i,1]
                x_means[0] += x[i,0]
                x_means[1] += x[i,1]
                node_count += 1
                #if x[i,2]>x_max[2]: x_max[2]=x[i,2]    can be used for a third node feature
                #if x[i,2]<x_min[2]: x_min[2]=x[i,2]
            
            edge_attr = data['edge_weights']
            for i in range(len(edge_attr)):
                if self.data_type == 'DC':
                    if edge_attr[i]>edge_attr_max: edge_attr_max=edge_attr[i]
                    if edge_attr[i]<edge_attr_min: edge_attr_min=edge_attr[i]
                    edge_attr_means += edge_attr[i]
                if self.data_type in ['AC', 'n-k']:
                    if edge_attr[i,0]>edge_attr_max[0]: edge_attr_max[0]=edge_attr[i,0]
                    if edge_attr[i,0]<edge_attr_min[0]: edge_attr_min[0]=edge_attr[i,0]
                    edge_attr_means[0] += edge_attr[i,0]
                    if edge_attr[i,1]>edge_attr_max[1]: edge_attr_max[1]=edge_attr[i,1]
                    if edge_attr[i,1]<edge_attr_min[1]: edge_attr_min[1]=edge_attr[i,1]
                    if edge_attr[i,2]>edge_attr_max[2]: edge_attr_max[2]=edge_attr[i,2]
                    if edge_attr[i,2]<edge_attr_min[2]: edge_attr_min[2]=edge_attr[i,2]
                    if edge_attr[i,4]>edge_attr_max[3]: edge_attr_max[3]=edge_attr[i,4]
                    if edge_attr[i,4]<edge_attr_min[3]: edge_attr_min[3]=edge_attr[i,4]
                    if edge_attr[i,5]>edge_attr_max[4]: edge_attr_max[4]=edge_attr[i,5]
                    if edge_attr[i,5]<edge_attr_min[4]: edge_attr_min[4]=edge_attr[i,5]

                    edge_attr_means[1] += edge_attr[i,1]
                    edge_attr_means[2] += edge_attr[i,2]
                    edge_attr_means[3] += edge_attr[i,4]
                    edge_attr_means[4] += edge_attr[i,5]
                edge_count += 1
            graph_label = data['y']
            if graph_label>graph_label_max: graph_label_max=graph_label
            if graph_label<graph_label_min: graph_label_min=graph_label
            graph_label_mean += graph_label
            
            if self.data_type in ['AC', 'n-k']:
                node_labels = data['node_labels']
                for i in range(len(node_labels)):
                    if node_labels[i]>node_labels_max: node_labels_max=node_labels[i]
                    if node_labels[i]<node_labels_min: node_labels_min=node_labels[i]
                    node_labels_mean += node_labels[i]
            else: node_count = 1    #avoid devision by 0
            

        return x_min, x_max, x_means/node_count, edge_attr_min, edge_attr_max, edge_attr_means/edge_count, node_labels_min, node_labels_max, node_labels_mean/node_count, graph_label_min, graph_label_max, graph_label_mean/graph_count
     
                         
    
    def get_feature_stds(self, x_means, edge_means):
        """
        Input 
        x_means     float array
            means of the node features
        edge_means  float array
            means of the edge features
            
        Returns
        np.sqrt(x_stds/node_count)
            the standarad deviations of the node features
        
        np.sqrt(edge_stds/edge_count)
            the standard deviations of the edge features
        """
        
        x_stds = torch.zeros(2)
        edge_stds = torch.zeros(5)
        node_count = 0
        edge_count = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            data = np.load(raw_path)
            #data = torch.load(self.processed_dir +'/' + file)
            x = data['x']
            for i in range(x.shape[0]):
                x_stds[0] += (x[i,0]-x_means[0])**2
                x_stds[1] += (x[i,1]-x_means[1])**2
                node_count += 1
            edge_attr = data['edge_attr']
            for i in range(len(edge_attr)):
                edge_stds[0] += (edge_attr[i,0] - edge_means[0])**2
                if self.data_type in ['AC', 'n-k']:
                    edge_stds[1] += (edge_attr[i,1] - edge_means[1])**2
                    edge_stds[2] += (edge_attr[i,2] - edge_means[2])**2
                    edge_stds[3] += (edge_attr[i,4] - edge_means[3])**2
                    edge_stds[4] += (edge_attr[i,5] - edge_means[4])**2
                edge_count += 1
        return np.sqrt(x_stds/node_count), np.sqrt(edge_stds/edge_count)
                
        