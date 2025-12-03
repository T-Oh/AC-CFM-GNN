
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np

FILE = 'processed/data_static.pt'

# Assuming you have a PyTorch Geometric graph object named `data`
init_data = torch.load('processed/data_static.pt')
data = torch.load(FILE)


# Step 1: Extract node positions
node_positions = np.loadtxt('/home/tohlinger/HUI/Documents/AC-CFM/Data/bus_positions.txt', delimiter='\t', usecols=(14, 15), dtype=float)
node_positions[:,[0,1]] = node_positions[:,[1,0]]

#remove half of the edges (since edges are implemented bidirectional with one edge per direction we remove every 2nd edge)
edge_index = init_data.edge_index[:, ::2]


# Create an array of node types
node_types = (data.x[:,5]+data.x[:,6]*2 + data.x[:,7]*3+data.x[:,8]*4).long()
type_mapping = {1: 'A', 2: 'B', 3: 'C', 4:'D'}
node_colors = {'A': 'green', 'B': 'orange', 'C': 'pink', 'D':'red'}
node_shapes = {'A': 's', 'B': 's', 'C': 's', 'D': 's'}
shapes = [node_shapes[type_mapping[node_type.item()]] for node_type in node_types]
colors = [node_colors[type_mapping[node_type.item()]] for node_type in node_types]

for i in range(len(shapes)):
    if shapes[i] != 's': print(i)
#print(colors)


# Step 2: Create a networkx graph object
graph = nx.Graph()

# Step 3: Add nodes and edges to the graph object
graph.add_nodes_from(range(init_data.num_nodes))
graph.add_edges_from(edge_index.t().tolist())

# Step 4: Set node positions as attributes
for i, pos in enumerate(node_positions):
    graph.nodes[i]['pos'] = pos
    graph.nodes[i]['type'] = node_types[i]
    graph.nodes[i]['node_marker'] = shapes[i]




#Load Background
img = plt.imread('Texas_blank.png')

# Step 5: Plot the graph
#plt.rcParams['font.size'] = 16
#plt.rcParams['figure.dpi'] = 300
fig = plt.figure(figsize=(10,8))
ax = plt.axes()
ax.set_axis_on()
ax.set_xticks([-109, -107, -105, -103, -101, -99, -97, -95, -93, -91])
ax.set_yticks([24,26,28,30,32,34,36, 38])
ax.imshow(img, zorder=0, extent=[-107.824514, -92.367277, 24.4131, 37.330763])
pos = nx.get_node_attributes(graph, 'pos')
nx.draw_networkx_nodes(graph, ax=ax, pos=pos, node_size=5, node_color=colors) #, node_shape=shapes,
nx.draw_networkx_edges(graph, pos=pos, width=1.8, )
plt.show()
fig.savefig('network_plot_test.png', bbox_inches='tight')

