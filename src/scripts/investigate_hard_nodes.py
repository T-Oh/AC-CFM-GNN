import os
import torch
import scipy.io
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed




# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "processed_1_111_PU/"  # contains data_static.pt and scenario folders
HARD_NODES = [1303, 1913, 564, 1396, 1087, 1070, 1259, 1190, 1341, 1115]  # indices of hardest nodes
BEST_NODES = [1168, 1185, 1184, 1183, 1177]   # indices of best nodes
SAVE_DIR = "analysis_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -------------------------
# Helper functions
# -------------------------

def process_scenario(scen, DATA_PATH, Y_raw, b_edge_attr):
    scen_path = os.path.join(DATA_PATH, scen)
    return scen, analyze_scenario(scen_path, use_labels=False, last_n=5, Y_raw=Y_raw, b_edge_attr=b_edge_attr)


def get_graph_metrics(data):
    """Build graph and compute graph-based metrics (degree, weighted degree, betweenness)."""
    
    edge_index = data.edge_index.numpy()
    edge_weight = np.sqrt(data.edge_attr[:, 0].numpy()**2 + data.edge_attr[:,1].numpy()**2) if data.edge_attr is not None else np.ones(edge_index.shape[1])

    G = nx.Graph()
    G.add_nodes_from(range(data.x.size(0)))
    for (u, v), w in zip(edge_index.T, edge_weight):
        G.add_edge(int(u), int(v), weight=float(w))

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    return degree, weighted_degree, betweenness



def collect_metrics_features(data, nodes):
    """Collect metrics from input features x for a list of node indices."""
    degree, w_degree, betweenness = get_graph_metrics(data)
    x = data.x.numpy()

    # average neighbour degree (weighted)
    avg_neigh_degree_dict = nx.average_neighbor_degree(
        nx.Graph(list(zip(data.edge_index[0].numpy(), data.edge_index[1].numpy()))),
        weight="weight" if data.edge_attr is not None else None
    )

    results = {
        "degree": [degree.get(n, 0) for n in nodes],
        "weighted_degree": [w_degree.get(n, 0) for n in nodes],
        "avg_neighbour_degree": [avg_neigh_degree_dict.get(n, 0) for n in nodes],
        "betweenness": [betweenness.get(n, 0.0) for n in nodes],
        "voltage_real": [x[n, 0] for n in nodes],
        "voltage_imag": [x[n, 1] for n in nodes],
        "p_inj_real": [x[n, 2] for n in nodes],
        "p_inj_imag": [x[n, 3] for n in nodes],
    }
    return results



def build_Y_matrix_from_labels(data, Y_raw, b_edge_attr):
    """Re-use your existing build_Y_matrix_from_predictions but with edge_labels."""
    Y = Y_raw.clone()
    inactive_edges = torch.where(data.edge_labels.flatten() == 0)[0]

    for idx in inactive_edges:
        i, j = data.edge_index[:, idx]
        y_ij = Y[i, j]

        if i != j:
            Y[i, i] += y_ij - b_edge_attr[idx]  # adjust diagonal
            Y[i, j] = 0

    Y[abs(Y.real) < 1e-3] = 0j
    return Y


def graph_from_Y(Y):
    """Convert Y-matrix to weighted networkx graph."""
    G = nx.Graph()
    num_nodes = Y.shape[0]
    G.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if Y[i, j] != 0:
                G.add_edge(i, j, weight=abs(Y[i, j].item()))
    return G


def collect_metrics_labels_and_graph(data, nodes, Y_raw, b_edge_attr):
    """
    Collect node_labels (voltages) + graph metrics from Y built with edge_labels.
    """
    labels = data.node_labels.numpy()

    # build Y and graph
    Y = build_Y_matrix_from_labels(data, Y_raw, b_edge_attr)
    G = graph_from_Y(Y)

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)

    results = {
        "voltage_real_label": [labels[n, 0] for n in nodes],
        "voltage_imag_label": [labels[n, 1] for n in nodes],
        "degree_final": [degree.get(n, 0) for n in nodes],
        "weighted_degree_final": [weighted_degree.get(n, 0) for n in nodes],
        "betweenness_final": [betweenness.get(n, 0.0) for n in nodes],
    }
    return results


def aggregate_results(all_results):
    """Stack results across dataset."""
    agg = {k: [] for k in all_results[0].keys()}
    for res in all_results:
        for k, v in res.items():
            agg[k].extend(v)
    return agg


def save_histograms_and_stats(agg, subset_name, node_type):
    """Save histograms + mean/std for each metric."""
    outdir = os.path.join(SAVE_DIR, f"{subset_name}_{node_type}")
    os.makedirs(outdir, exist_ok=True)

    stats = {}
    for metric, values in agg.items():
        arr = np.array(values)
        mean, std = arr.mean(), arr.std()
        stats[metric] = (mean, std)

        # histogram
        plt.figure()
        plt.hist(arr, bins=50, alpha=0.7)
        plt.title(f"{metric} ({subset_name}, {node_type})")
        plt.xlabel(metric)
        plt.ylabel("Count")
        plt.savefig(os.path.join(outdir, f"{metric}_hist.png"))
        plt.close()

    # save stats
    with open(os.path.join(outdir, "stats.txt"), "w") as f:
        for metric, (mean, std) in stats.items():
            f.write(f"{metric}: mean={mean:.4f}, std={std:.4f}\n")


def analyze_subset(subset_name, data_list, use_labels, Y_raw, b_edge_attr):
    """Analyze a subset of data, choose either features or labels + final step graph metrics."""

    for node_type, nodes in [("hard", HARD_NODES), ("best", BEST_NODES)]:
        all_results = []
        for data in tqdm(data_list, desc=f"{subset_name} - {node_type}"):
            if use_labels:
                results = collect_metrics_labels_and_graph(data, nodes, Y_raw, b_edge_attr)
            else:
                results = collect_metrics_features(data, nodes)
            all_results.append(results)

        agg = aggregate_results(all_results)
        save_histograms_and_stats(agg, subset_name, node_type)

def create_b_edge_attr(bus_IDs, branch_data, edge_index, b):
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


def analyze_scenario(scen_path, use_labels, last_n, Y_raw, b_edge_attr):
    files = sorted([f for f in os.listdir(scen_path) if f.endswith(".pt")])
    if len(files) == 0:
        return None  # skip empty scenarios

    results_per_node_type = {"hard": [], "best": []}

    if use_labels:
        data = torch.load(os.path.join(scen_path, files[-1]))
        for node_type, nodes in [("hard", HARD_NODES), ("best", BEST_NODES)]:
            results = collect_metrics_labels_and_graph(data, nodes, Y_raw, b_edge_attr)
            results_per_node_type[node_type].append(results)
    else:
        for f in files[-last_n:]:
            data = torch.load(os.path.join(scen_path, f))
            for node_type, nodes in [("hard", HARD_NODES), ("best", BEST_NODES)]:
                results = collect_metrics_features(data, nodes)
                results_per_node_type[node_type].append(results)

    # aggregate across this scenario
    agg_per_node_type = {nt: aggregate_results(results_per_node_type[nt]) for nt in results_per_node_type}
    return agg_per_node_type


# -------------------------
# Load static data (still only one file, OK)
# -------------------------

print('Number of CPUs:', os.cpu_count())

print("Loading static data...")
data_static = torch.load(os.path.join(DATA_PATH, "data_static.pt"))
pwsdata = scipy.io.loadmat('raw/pwsdata.mat') 
Y_raw = torch.tensor(pwsdata['clusterresult_'][0,0][10])
Y_raw = torch.complex(torch.tensor(Y_raw.real), torch.tensor(Y_raw.imag)).type(torch.complex64)
bus_IDs = torch.tensor(pwsdata['clusterresult_'][0,0][2][:,0])
branch_data = pwsdata['clusterresult_'][0,0][4]
b = torch.tensor(branch_data[:,4])
b_edge_attr = create_b_edge_attr(bus_IDs, branch_data, data_static.edge_index, b)

# Static analysis
analyze_subset("static", [data_static], use_labels=False, Y_raw=Y_raw, b_edge_attr=b_edge_attr)

# -------------------------
# Scenario analysis (on-the-fly)
# -------------------------
scenario_dirs = [d for d in os.listdir(DATA_PATH) if d.startswith("scenario_")]

global_results = {"hard": [], "best": []}

with ProcessPoolExecutor() as executor:
    futures = {executor.submit(process_scenario, scen, DATA_PATH, Y_raw, b_edge_attr): scen for scen in scenario_dirs}
    for fut in tqdm(as_completed(futures), total=len(futures), desc="Aggregating features across all scenarios"):
        scen = futures[fut]
        try:
            scen, agg = fut.result()
            if agg is not None:
                for node_type in ["hard", "best"]:
                    global_results[node_type].append(agg[node_type])
        except Exception as e:
            print(f"Scenario {scen} failed with error: {e}")

# Merge everything into a single aggregated dict per node type
for node_type in ["hard", "best"]:
    global_results[node_type] = aggregate_results(global_results[node_type])

for node_type in ["hard", "best"]:
    save_histograms_and_stats(global_results[node_type], "all_scenarios", node_type)




"""for scen in tqdm(scenario_dirs, desc="Aggregating features across all scenarios"):
    scen_path = os.path.join(DATA_PATH, scen)
    agg = analyze_scenario(scen_path, use_labels=False, last_n=5, Y_raw=Y_raw, b_edge_attr=b_edge_attr)
    if agg is not None:
        for node_type in ["hard", "best"]:
            global_results[node_type].append(agg[node_type])

# Merge everything into a single aggregated dict per node type
for node_type in ["hard", "best"]:
    global_results[node_type] = aggregate_results(global_results[node_type])

for node_type in ["hard", "best"]:
    save_histograms_and_stats(global_results[node_type], "all_scenarios", node_type)"""

