import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import mse_loss
import networkx as nx

# ----------------------------
# Settings
# ----------------------------
MODELS = {
    "TAG": "/home/tohlinger/HUI/Documents/hi-accf-ml/test/",
    "GAT": "/home/tohlinger/HUI/Documents/hi-accf-ml/test/"
}
SUBSETS = ["sl_5", "sl_25", "sl_50"]  # expected subfolders
TOP_K = 30

# ----------------------------
# Helper: process one run
# ----------------------------
def process_run(results_dir, top_k=TOP_K):
    epoch_files = glob.glob(os.path.join(results_dir, "test_output*.pt"))
    if len(epoch_files) == 0:
        return None, None, None, None, None, None, None

    all_epochs = []
    epoch_map = {}

    for f in epoch_files:
        suffix = f.split("test_output")[-1].split(".pt")[0]
        if suffix == "_final":
            epoch = 100
        elif suffix == "":
            continue
        else:
            epoch = int(suffix)
        all_epochs.append(epoch)
        epoch_map[epoch] = suffix

    all_epochs = sorted(all_epochs)

    worst_masks_real = []
    worst_masks_imag = []
    worst_masks_combined = []
    best_masks_real = []
    best_masks_imag = []
    best_masks_combined = []

    for epoch in all_epochs:
        suffix = epoch_map[epoch]
        out_path = os.path.join(results_dir, f"test_output{suffix}.pt")
        lab_path = os.path.join(results_dir, f"test_labels{suffix}.pt")

        if not (os.path.exists(out_path) and os.path.exists(lab_path)):
            continue

        out = torch.load(out_path)[0]
        lab = torch.load(lab_path)[0]

        N_INSTANCES = int(out.shape[0] / 2000)
        out_nodes = out.reshape(N_INSTANCES, 2000, 2)
        lab_nodes = lab.reshape(N_INSTANCES, 2000, 2)

        node_MSEs_real = []
        node_MSEs_imag = []
        node_MSEs_combined = []

        for n in range(2000):
            y_pred_real = out_nodes[:, n, 0].detach().cpu().numpy()
            y_true_real = lab_nodes[:, n, 0].detach().cpu().numpy()
            y_pred_imag = out_nodes[:, n, 1].detach().cpu().numpy()
            y_true_imag = lab_nodes[:, n, 1].detach().cpu().numpy()

            node_MSEs_real.append(np.mean((y_pred_real - y_true_real) ** 2))
            node_MSEs_imag.append(np.mean((y_pred_imag - y_true_imag) ** 2))
            node_MSEs_combined.append(mse_loss(out_nodes[:, n, :], lab_nodes[:, n, :], reduction='mean'))

        node_MSEs_real = np.array(node_MSEs_real)
        node_MSEs_imag = np.array(node_MSEs_imag)
        node_MSEs_combined = np.array(node_MSEs_combined)

        worst_real_idx = np.argsort(-node_MSEs_real)[:top_k]
        worst_imag_idx = np.argsort(-node_MSEs_imag)[:top_k]
        worst_combined_idx= np.argsort(-node_MSEs_combined)[:top_k]

        best_real_idx = np.argsort(node_MSEs_real)[:top_k]
        best_imag_idx = np.argsort(node_MSEs_imag)[:top_k]
        best_combined_idx = np.argsort(node_MSEs_combined)[:top_k]

        mask_real = np.zeros(2000, dtype=int)
        mask_imag = np.zeros(2000, dtype=int)
        mask_combined = np.zeros(2000, dtype=int)
        mask_real[worst_real_idx] = 1
        mask_imag[worst_imag_idx] = 1
        mask_combined[worst_combined_idx] = 1

        worst_masks_real.append(mask_real)
        worst_masks_imag.append(mask_imag)
        worst_masks_combined.append(mask_combined)

        mask_real_best = np.zeros(2000, dtype=int)
        mask_imag_best = np.zeros(2000, dtype=int)
        mask_combined_best = np.zeros(2000, dtype=int)
        mask_real_best[best_real_idx] = 1
        mask_imag_best[best_imag_idx] = 1
        mask_combined_best[best_combined_idx] =1

        best_masks_real.append(mask_real_best)
        best_masks_imag.append(mask_imag_best)
        best_masks_combined.append(mask_combined_best)

    if len(worst_masks_real) == 0:
        return None, None, None, None, None, None, None

    print(np.stack(worst_masks_combined).shape)
    print(f'Node mses shape: {np.stack(node_MSEs_combined).shape}')

    return np.stack(worst_masks_real), np.stack(worst_masks_imag), np.stack(best_masks_real), np.stack(best_masks_imag), np.stack(worst_masks_combined), np.stack(best_masks_combined), np.stack(node_MSEs_combined)

# ----------------------------
# Aggregation helper
# ----------------------------
def aggregate(runs_real, runs_imag, runs_combined, runs_combined_best, title, save_prefix):
    agg_real = np.stack(runs_real) if len(runs_real) > 0 else None
    agg_imag = np.stack(runs_imag) if len(runs_imag) > 0 else None
    agg_combined = np.stack(runs_combined) if len(runs_combined) > 0 else None
    agg_combined_best = np.stack(runs_combined_best) if len(runs_combined_best) > 0 else None

    if agg_real is not None:
        mean_real = agg_real.mean(axis=0)
        top_real = np.argsort(-mean_real)[:10]
        print(f"\n===== Top Persistent Nodes (Real) {title} =====")
        for i in top_real:
            print(f"Node {i} -> {mean_real[i]*100:.1f}% of epochs across runs")

        plt.figure(figsize=(15,6))
        sns.heatmap(agg_real, cmap="Reds", cbar=True)
        plt.xlabel("Node index")
        plt.ylabel("Run index")
        plt.title(f"{title}: fraction of epochs in top {TOP_K} worst (Real)")
        plt.savefig(f"{save_prefix}_real.png", dpi=150)

    if agg_imag is not None:
        mean_imag = agg_imag.mean(axis=0)
        top_imag = np.argsort(-mean_imag)[:10]
        print(f"\n===== Top Persistent Nodes (Imag) {title} =====")
        for i in top_imag:
            print(f"Node {i} -> {mean_imag[i]*100:.1f}% of epochs across runs")

        plt.figure(figsize=(15,6))
        sns.heatmap(agg_imag, cmap="Blues", cbar=True)
        plt.xlabel("Node index")
        plt.ylabel("Run index")
        plt.title(f"{title}: fraction of epochs in top {TOP_K} worst (Imag)")
        plt.savefig(f"{save_prefix}_imag.png", dpi=150)

    if agg_combined is not None:
        mean_combined = agg_combined.mean(axis=0)
        top_combined = np.argsort(-mean_combined)[:10]
        print(f"\n===== Top Persistent Nodes (combined) {title} =====")
        for i in top_combined:
            print(f"Node {i} -> {mean_combined[i]*100:.1f}% of epochs across runs")

        plt.figure(figsize=(15,6))
        sns.heatmap(agg_combined, cmap="Greens", cbar=True)
        plt.xlabel("Node index")
        plt.ylabel("Run index")
        plt.title(f"{title}: fraction of epochs in top {TOP_K} worst (combined)")
        plt.savefig(f"{save_prefix}_combined.png", dpi=150)

    if agg_combined_best is not None:
        mean_combined_best = agg_combined_best.mean(axis=0)
        top_combined_best = np.argsort(mean_combined_best)[:10]
        print(f"\n===== Top Easy Nodes (combined) {title} =====")
        for i in top_combined_best:
            print(f"Node {i} -> {mean_combined_best[i]*100:.1f}% of epochs across runs")

        plt.figure(figsize=(15,6))
        sns.heatmap(agg_combined_best, cmap="Greens", cbar=True)
        plt.xlabel("Node index")
        plt.ylabel("Run index")
        plt.title(f"{title}: fraction of epochs in top {TOP_K} best (combined)")
        plt.savefig(f"{save_prefix}_combined_best.png", dpi=150)

# ----------------------------
# Compute graph metrics
# ----------------------------
def get_graph_metrics(data):
    """Build graph and compute degree, weighted degree, betweenness centrality."""
    edge_index = data.edge_index.numpy()
    edge_weight = np.sqrt(data.edge_attr[:,0].numpy()**2 + data.edge_attr[:,1].numpy()**2) \
                  if data.edge_attr is not None else np.ones(edge_index.shape[1])

    G = nx.Graph()
    G.add_nodes_from(range(data.x.size(0)))
    for (u, v), w in zip(edge_index.T, edge_weight):
        G.add_edge(int(u), int(v), weight=float(w))

    degree = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))
    betweenness = nx.betweenness_centrality(G, weight="weight", normalized=True)
    return degree, weighted_degree, betweenness


# ----------------------------
# Plot graph metrics and features vs average node MSE
# ----------------------------
def plot_metrics_vs_mse(data, agg_combined, subset_name, save_prefix):
    """
    data: single data object used to build the graph
    agg_combined: list of arrays [runs x nodes] containing fraction of epochs in top worst nodes
    """
    # Average nodal MSE across runs
    print(agg_combined)
    avg_mse = np.mean(np.stack(agg_combined), axis=0)

    # ----------------------------
    # Compute graph metrics
    # ----------------------------
    degree_dict, weighted_degree_dict, _ = get_graph_metrics(data)
    degree = np.array([degree_dict[i] for i in range(len(degree_dict))])
    weighted_degree = np.array([weighted_degree_dict[i] for i in range(len(weighted_degree_dict))])
    
    # Average neighbor degree
    G = nx.Graph()
    edge_index = data.edge_index.numpy()
    edge_weight = np.sqrt(data.edge_attr[:,0].numpy()**2 + data.edge_attr[:,1].numpy()**2) \
                  if data.edge_attr is not None else np.ones(edge_index.shape[1])
    G.add_nodes_from(range(data.x.size(0)))
    for (u, v), w in zip(edge_index.T, edge_weight):
        G.add_edge(int(u), int(v), weight=float(w))
    avg_neigh_degree = np.array(list(nx.average_neighbor_degree(G, weight='weight').values()))

    # ----------------------------
    # Node features
    # ----------------------------
    node_features = data.x.numpy()  # shape [num_nodes, num_features]

    metrics = {
        "Degree": degree,
        "Weighted degree": weighted_degree,
        "Average neighbor degree": avg_neigh_degree
    }

    # Add node features to metrics
    for i in range(node_features.shape[1]):
        metrics[f"Feature {i}"] = node_features[:, i]

    # ----------------------------
    # Plot each metric vs avg MSE
    # ----------------------------
    for metric_name, metric_vals in metrics.items():
        print('YELLO')
        print(metric_vals.shape)
        print(avg_mse.shape)
        plt.figure(figsize=(10,6))
        plt.scatter(metric_vals, avg_mse, alpha=0.6)
        plt.xlabel(metric_name)
        plt.ylabel("Average nodal MSE")
        plt.title(f"{subset_name}: {metric_name} vs Average Nodal MSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{subset_name}_{metric_name.replace(' ','_')}_vs_mse.png", dpi=150)
        plt.show()



# ----------------------------
# Main processing
# ----------------------------
all_runs_real, all_runs_imag, all_runs_combined, all_runs_combined_best, all_runs_node_mses_combined = [], [], [], [], []
sl5_runs_real, sl5_runs_imag, sl5_runs_combined, sl_5_runs_combined_best, sl_5_runs_node_mses_combined= [], [], [], [], []
per_model_sl5 = {m: ([], [], [], [], []) for m in MODELS}


for model, base_path in MODELS.items():
    for subset in SUBSETS:
        subset_path = os.path.join(base_path, subset)
        if not os.path.exists(subset_path):
            continue

        run_folders = [os.path.join(subset_path, d, "results") for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
        print(f"\nModel {model}, Subset {subset}: found {len(run_folders)} runs")

        for run_dir in run_folders:
            worst_real, worst_imag, _, _, worst_combined, best_combined, node_mses_combined = process_run(run_dir)
            if worst_real is None:
                continue

            frac_real = worst_real.mean(axis=0)
            frac_imag = worst_imag.mean(axis=0)
            frac_combined = worst_combined.mean(axis=0)
            frac_combined_best = best_combined.mean(axis=0)

            all_runs_real.append(frac_real)
            all_runs_imag.append(frac_imag)
            all_runs_combined.append(frac_combined)
            all_runs_combined_best.append(frac_combined_best)
            print(node_mses_combined)
            all_runs_node_mses_combined.append(node_mses_combined)

            if subset == "sl_5":
                sl5_runs_real.append(frac_real)
                sl5_runs_imag.append(frac_imag)
                sl5_runs_combined.append(frac_combined)
                sl_5_runs_combined_best.append(frac_combined_best)
                sl_5_runs_node_mses_combined.append(node_mses_combined)
                per_model_sl5[model][0].append(frac_real)
                per_model_sl5[model][1].append(frac_imag)
                per_model_sl5[model][2].append(frac_combined)
                per_model_sl5[model][3].append(frac_combined_best)
                per_model_sl5[model][4].append(node_mses_combined)

# ----------------------------
# Analyses
# ----------------------------
aggregate(all_runs_real, all_runs_imag, all_runs_combined, all_runs_combined_best, "All models, all subsets", "all_models_all_subsets")
aggregate(sl5_runs_real, sl5_runs_imag, sl5_runs_combined, sl_5_runs_combined_best, "All models, sl_5 only", "all_models_sl5")

for model, (r_real, r_imag, r_combined, r_combined_best, _) in per_model_sl5.items():
    aggregate(r_real, r_imag, r_combined, r_combined_best, f"{model}, sl_5 only", f"{model}_sl5")

# ----------------------------
# Load graph data and plot centrality vs MSE
# ----------------------------

data_path = "processed/data_static.pt"
data = torch.load(data_path)

# For all models, all subsets
plot_metrics_vs_mse(data, all_runs_node_mses_combined, "All models, all subsets", "all_models_all_subsets")

# For sl_5 subset
plot_metrics_vs_mse(data, sl_5_runs_node_mses_combined, "All models, sl_5", "all_models_sl5")

# Per model for sl_5
for model, (_, _, _, _, node_mses_combined) in per_model_sl5.items():
    plot_metrics_vs_mse(data, node_mses_combined, f"{model}, sl_5", f"{model}_sl5")


