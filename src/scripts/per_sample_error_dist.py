import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, f1_score
import numpy as np


def plot_metric_comparison(good_vals, bad_vals, title, xlabel, savepath=None):
    plt.figure(figsize=(12, 5))

    def get_outlier_indices(data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        # return indices instead of values
        outliers = {i for i, x in enumerate(data) if x < lower_bound or x > upper_bound}
        return outliers, len(outliers), len(outliers) / len(data) if len(data) > 0 else 0

    # Histogram comparison
    plt.subplot(1, 2, 1)
    plt.hist(good_vals, bins=30, alpha=0.7, label="Good epoch")
    plt.hist(bad_vals, bins=30, alpha=0.7, label="Bad epoch")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(f"{title} - Histogram")
    plt.legend()

    # Boxplot comparison
    plt.subplot(1, 2, 2)
    plt.boxplot([good_vals, bad_vals], labels=["Good", "Bad"])
    plt.ylabel(xlabel)
    plt.title(f"{title} - Boxplot")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150)
    plt.close()

    # Outlier indices
    good_outliers, good_abs, good_rel = get_outlier_indices(good_vals)
    bad_outliers, bad_abs, bad_rel = get_outlier_indices(bad_vals)

    # Intersection of instance indices
    overlap = good_outliers.intersection(bad_outliers)
    overlap_abs = len(overlap)
    overlap_rel_good = overlap_abs / good_abs if good_abs > 0 else 0
    overlap_rel_bad = overlap_abs / bad_abs if bad_abs > 0 else 0

    # Print stats
    print(f"Outliers in Good epoch: {good_abs} ({good_rel:.2%})")
    print(f"Outliers in Bad epoch: {bad_abs} ({bad_rel:.2%})")
    print(f"Overlap in outlier instances: {overlap_abs} "
          f"(= {overlap_rel_good:.2%} of Good, {overlap_rel_bad:.2%} of Bad)")

    # --------------------------
    # 1) Venn diagram
    # --------------------------
    """
    plt.figure(figsize=(5, 5))
    venn2([good_outliers, bad_outliers], set_labels=("Good outliers", "Bad outliers"))
    plt.title(f"Outlier Overlap - {title}")
    plt.savefig(savepath.replace(".png", "_venn.png"), dpi=150)
    plt.close()"""

    # --------------------------
    # 2) Scatter plot of metrics
    # --------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(good_vals, bad_vals, alpha=0.5, label="Instances", s=10)
    # Highlight outliers
    plt.scatter(
        [good_vals[i] for i in good_outliers],
        [bad_vals[i] for i in good_outliers],
        color="red", label="Good outliers", s=20
    )
    plt.scatter(
        [good_vals[i] for i in bad_outliers],
        [bad_vals[i] for i in bad_outliers],
        color="blue", label="Bad outliers", s=20
    )
    plt.scatter(
        [good_vals[i] for i in overlap],
        [bad_vals[i] for i in overlap],
        color="purple", label="Overlap", s=30
    )
    plt.xlabel("Good epoch " + xlabel)
    plt.ylabel("Bad epoch " + xlabel)
    plt.legend()
    plt.title(f"Good vs Bad scatter - {title}")
    plt.savefig(savepath.replace(".png", "_scatter.png"), dpi=150)
    plt.close()

    # --------------------------
    # 3) Bar chart (UpSet-like)
    # --------------------------
    plt.figure(figsize=(5, 5))
    bars = [
        ("Good only", len(good_outliers - bad_outliers)),
        ("Bad only", len(bad_outliers - good_outliers)),
        ("Both", len(overlap))
    ]
    labels, values = zip(*bars)
    plt.bar(labels, values, color=["red", "blue", "purple"])
    plt.ylabel("Number of outliers")
    plt.title(f"Outlier distribution - {title}")
    plt.savefig(savepath.replace(".png", "_bars.png"), dpi=150)
    plt.close()

GOOD_EPOCH = 11
BAD_EPOCH = 12

good_MSEs_real = []
good_MSEs_imag = []
bad_MSEs_real = []
bad_MSEs_imag = []
good_R2s_real = []
good_R2s_imag = []
bad_R2s_real = []
bad_R2s_imag = []
good_F1s = []
bad_F1s = []


good_output = torch.load(f'results/test_output{GOOD_EPOCH}.pt')
good_labels = torch.load(f'results/test_labels{GOOD_EPOCH}.pt')
bad_output = torch.load(f'results/test_output{BAD_EPOCH}.pt')
bad_labels = torch.load(f'results/test_labels{BAD_EPOCH}.pt')
print(good_output[0].shape, good_labels[0].shape, good_output[1].shape, good_labels[1].shape)

N_INSTANCES = int(good_output[0].shape[0]/2000)

for i in range(N_INSTANCES):
    good_MSEs_real.append(torch.nn.functional.mse_loss(good_output[0][i*2000:(i+1)*2000,0], good_labels[0][i*2000:(i+1)*2000,0]).item())
    good_MSEs_imag.append(torch.nn.functional.mse_loss(good_output[0][i*2000:(i+1)*2000,1], good_labels[0][i*2000:(i+1)*2000,1]).item())
    bad_MSEs_real.append(torch.nn.functional.mse_loss(bad_output[0][i*2000:(i+1)*2000,0], bad_labels[0][i*2000:(i+1)*2000,0]).item())
    bad_MSEs_imag.append(torch.nn.functional.mse_loss(bad_output[0][i*2000:(i+1)*2000,1], bad_labels[0][i*2000:(i+1)*2000,1]).item())
    good_R2s_real.append(r2_score( good_labels[0][i*2000:(i+1)*2000,0], good_output[0][i*2000:(i+1)*2000,0]).item())
    good_R2s_imag.append(r2_score( good_labels[0][i*2000:(i+1)*2000,1], good_output[0][i*2000:(i+1)*2000,1]).item())
    bad_R2s_real.append(r2_score(bad_labels[0][i*2000:(i+1)*2000,0], bad_output[0][i*2000:(i+1)*2000,0]).item())
    bad_R2s_imag.append(r2_score(bad_labels[0][i*2000:(i+1)*2000,1], bad_output[0][i*2000:(i+1)*2000,1]).item())
    good_F1s.append(f1_score( good_labels[1][i], good_output[1][i*7064:(i+1)*7064].argmax(dim=1), average='macro').item())
    bad_F1s.append(f1_score( bad_labels[1][i], bad_output[1][i*7064:(i+1)*7064].argmax(dim=1), average='macro').item())


torch.save(good_MSEs_real, f'results/per_sample_good_MSEs_real_epoch{GOOD_EPOCH}.pt')
torch.save(good_MSEs_imag, f'results/per_sample_good_MSEs_imag_epoch{GOOD_EPOCH}.pt')
torch.save(bad_MSEs_real, f'results/per_sample_bad_MSEs_real_epoch{BAD_EPOCH}.pt')
torch.save(bad_MSEs_imag, f'results/per_sample_bad_MSEs_imag_epoch{BAD_EPOCH}.pt')
torch.save(good_R2s_real, f'results/per_sample_good_R2s_real_epoch{GOOD_EPOCH}.pt')
torch.save(good_R2s_imag, f'results/per_sample_good_R2s_imag_epoch{GOOD_EPOCH}.pt')
torch.save(bad_R2s_real, f'results/per_sample_bad_R2s_real_epoch{BAD_EPOCH}.pt')
torch.save(bad_R2s_imag, f'results/per_sample_bad_R2s_imag_epoch{BAD_EPOCH}.pt')
torch.save(good_F1s, f'results/per_sample_good_F1s_real_epoch{GOOD_EPOCH}.pt')

torch.save(bad_F1s, f'results/per_sample_bad_F1s_real_epoch{BAD_EPOCH}.pt')


# ----------------------------
# Plot comparisons
# ----------------------------
plot_metric_comparison(good_MSEs_real, bad_MSEs_real, "Per-sample MSE (Real)", "MSE", "results/MSE_real_comparison.png")
plot_metric_comparison(good_MSEs_imag, bad_MSEs_imag, "Per-sample MSE (Imag)", "MSE", "results/MSE_imag_comparison.png")

plot_metric_comparison(good_R2s_real, bad_R2s_real, "Per-sample R² (Real)", "R²", "results/R2_real_comparison.png")
plot_metric_comparison(good_R2s_imag, bad_R2s_imag, "Per-sample R² (Imag)", "R²", "results/R2_imag_comparison.png")

plot_metric_comparison(good_F1s, bad_F1s, "Per-sample F1 (Real)", "F1", "results/F1_real_comparison.png")




# ----------------------------
# Get node average metrics
# ----------------------------
good_node_MSEs_real = []
good_node_MSEs_imag = []
bad_node_MSEs_real = []
bad_node_MSEs_imag = []
good_node_MSEs_combined = []
bad_node_MSEs_combined = []
good_node_R2s_real = []
good_node_R2s_imag = []
bad_node_R2s_real = []
bad_node_R2s_imag = []
# reshape into [N_INSTANCES, 2000, 2]
good_nodes_out = good_output[0].reshape(N_INSTANCES, 2000, 2)
good_nodes_lab = good_labels[0].reshape(N_INSTANCES, 2000, 2)
bad_nodes_out = bad_output[0].reshape(N_INSTANCES, 2000, 2)
bad_nodes_lab = bad_labels[0].reshape(N_INSTANCES, 2000, 2)

for n in range(2000):
    # collect all instance predictions/labels for this node
    y_pred_good = good_nodes_out[:, n, 0].detach().cpu().numpy()
    y_true_good = good_nodes_lab[:, n, 0].detach().cpu().numpy()
    y_pred_bad  = bad_nodes_out[:, n, 0].detach().cpu().numpy()
    y_true_bad  = bad_nodes_lab[:, n, 0].detach().cpu().numpy()

    good_node_MSEs_real.append(np.mean((y_pred_good - y_true_good) ** 2))
    bad_node_MSEs_real.append(np.mean((y_pred_bad - y_true_bad) ** 2))
    good_node_R2s_real.append(r2_score(y_true_good, y_pred_good))
    bad_node_R2s_real.append(r2_score(y_true_bad, y_pred_bad))

    # repeat for imag channel
    y_pred_good = good_nodes_out[:, n, 1].detach().cpu().numpy()
    y_true_good = good_nodes_lab[:, n, 1].detach().cpu().numpy()
    y_pred_bad  = bad_nodes_out[:, n, 1].detach().cpu().numpy()
    y_true_bad  = bad_nodes_lab[:, n, 1].detach().cpu().numpy()

    good_node_MSEs_imag.append(np.mean((y_pred_good - y_true_good) ** 2))
    bad_node_MSEs_imag.append(np.mean((y_pred_bad - y_true_bad) ** 2))
    good_node_R2s_imag.append(r2_score(y_true_good, y_pred_good))
    bad_node_R2s_imag.append(r2_score(y_true_bad, y_pred_bad))

    y_pred_good = good_nodes_out[:, n, :].detach().cpu()#.numpy()
    y_true_good = good_nodes_lab[:, n, :].detach().cpu()#.numpy()
    y_pred_bad  = bad_nodes_out[:, n, :].detach().cpu()#.numpy()
    y_true_bad  = bad_nodes_lab[:, n, :].detach().cpu()#.numpy

    good_node_MSEs_combined.append(torch.nn.functional.mse_loss(y_pred_good, y_true_good))
    bad_node_MSEs_combined.append(torch.nn.functional.mse_loss(y_pred_bad, y_true_bad))

print(f'Total MSE difference between Epochs: {np.array(bad_node_MSEs_combined).sum()-np.array(good_node_MSEs_combined).sum()}')
# convert to numpy arrays
good_combined = np.array([m.item() if torch.is_tensor(m) else m for m in good_node_MSEs_combined])
bad_combined  = np.array([m.item() if torch.is_tensor(m) else m for m in bad_node_MSEs_combined])

# per-node differences
mse_diffs = bad_combined - good_combined

# evaluate multiple k values
for k in [5, 10, 25, 50]:
    topk_idx = np.argsort(mse_diffs)[-k:]
    topk_diff_sum = mse_diffs[topk_idx].sum()
    print(f"Top-{k} nodes contribute {topk_diff_sum:.6f} to MSE difference (indices: {topk_idx})")
# ----------------------------


# bar plots
plt.figure(figsize=(15,5))
plt.bar(range(2000), good_node_MSEs_real, alpha=0.6, label="Good epoch")
plt.bar(range(2000), bad_node_MSEs_real, alpha=0.6, label="Bad epoch")
plt.xlabel("Node index")
plt.ylabel("MSE (Real)")
plt.legend()
plt.title("Per-node average MSE (Real)")
plt.savefig("results/node_MSE_real.png", dpi=150)

plt.figure(figsize=(15,5))
plt.bar(range(2000), np.array(bad_node_MSEs_real) - np.array(good_node_MSEs_real), alpha=0.6)
#plt.bar(range(2000), bad_node_MSEs_real, alpha=0.6, label="Bad epoch")
plt.xlabel("Node index")
plt.ylabel("MSE Diff (Real)")
plt.legend()
plt.title("Per-node average MSE (Real)")
plt.savefig("results/node_MSE_real_diff.png", dpi=150)


plt.figure(figsize=(15,5))
plt.bar(range(2000), good_node_MSEs_imag, alpha=0.6, label="Good epoch")
plt.bar(range(2000), bad_node_MSEs_imag, alpha=0.6, label="Bad epoch")
plt.xlabel("Node index")
plt.ylabel("MSE (Imag)")
plt.legend()
plt.title("Per-node average MSE (Imag)")
plt.savefig("results/node_MSE_imag.png", dpi=150)

plt.figure(figsize=(15,5))
plt.bar(range(2000), np.array(bad_node_MSEs_imag) - np.array(good_node_MSEs_imag), alpha=0.6)
#plt.bar(range(2000), bad_node_MSEs_imag, alpha=0.6, label="Bad epoch")
plt.xlabel("Node index")
plt.ylabel("MSE Diff (Imag)")
plt.legend()
plt.title("Per-node average MSE (Imag) Diff")
plt.savefig("results/node_MSE_imag_diff.png", dpi=150)


plt.figure(figsize=(15,5))
plt.bar(range(2000), np.array(bad_node_MSEs_combined) - np.array(good_node_MSEs_combined), alpha=0.6)
#plt.bar(range(2000), bad_node_MSEs_imag, alpha=0.6, label="Bad epoch")
plt.xlabel("Node index")
plt.ylabel("MSE Diff (Combined)")
plt.legend()
plt.title("Per-node average MSE (Combined) Diff")
plt.savefig("results/node_MSE_combined_diff.png", dpi=150)


plt.figure(figsize=(15,5))
plt.bar(range(2000), good_node_R2s_real, alpha=0.6, label="Good epoch")
plt.bar(range(2000), bad_node_R2s_real, alpha=0.6, label="Bad epoch")
plt.xlabel("Node index")
plt.ylabel("R² (Real)")
plt.legend()
plt.title("Per-node average R² (Real)")
plt.savefig("results/node_R2_real.png", dpi=150)

plt.figure(figsize=(15,5))
plt.bar(range(2000), good_node_R2s_imag, alpha=0.6, label="Good epoch")
plt.bar(range(2000), bad_node_R2s_imag, alpha=0.6, label="Bad epoch")
plt.xlabel("Node index")
plt.ylabel("R² (Imag)")
plt.legend()
plt.title("Per-node average R² (Imag)")
plt.savefig("results/node_R2_imag.png", dpi=150)


# ----------------------------
# Per-edge F1 scores (7064 edges)
# ----------------------------
good_edges_out = good_output[1].reshape(N_INSTANCES, 7064, 2)
good_edges_lab = good_labels[1].reshape(N_INSTANCES, 7064)
bad_edges_out  = bad_output[1].reshape(N_INSTANCES, 7064, 2)
bad_edges_lab  = bad_labels[1].reshape(N_INSTANCES, 7064)

good_edge_F1s = []
bad_edge_F1s = []

for e in range(7064):
    y_true_good = good_edges_lab[:, e].cpu().numpy()
    y_pred_good = good_edges_out[:, e, :].argmax(axis=1).cpu().numpy()
    y_true_bad  = bad_edges_lab[:, e].cpu().numpy()
    y_pred_bad  = bad_edges_out[:, e, :].argmax(axis=1).cpu().numpy()

    good_edge_F1s.append(f1_score(y_true_good, y_pred_good, average="macro"))
    bad_edge_F1s.append(f1_score(y_true_bad, y_pred_bad, average="macro"))

# bar plot for edges
plt.figure(figsize=(15,5))
plt.bar(range(7064), good_edge_F1s, alpha=0.6, label="Good epoch")
plt.bar(range(7064), bad_edge_F1s, alpha=0.6, label="Bad epoch")
plt.xlabel("Edge index")
plt.ylabel("F1")
plt.legend()
plt.title("Per-edge average F1")
plt.savefig("results/edge_F1.png", dpi=150)


# ----------------------------
# Threshold analysis
# ----------------------------
MSE_THRESHOLD_REAL = 0.025      # example: lower is better
MSE_THRESHOLD_IMAG = 0.01      # example: lower is better
R2_THRESHOLD = -0.01e12        # example: higher is better


# --- Nodes ---
hard_nodes_good = {
    "MSE_real": [i for i, v in enumerate(good_node_MSEs_real) if v > MSE_THRESHOLD_REAL],
    "MSE_imag": [i for i, v in enumerate(good_node_MSEs_imag) if v > MSE_THRESHOLD_IMAG],
    "R2_real":  [i for i, v in enumerate(good_node_R2s_real) if v < R2_THRESHOLD],
    "R2_imag":  [i for i, v in enumerate(good_node_R2s_imag) if v < R2_THRESHOLD],
}

hard_nodes_bad = {
    "MSE_real": [i for i, v in enumerate(bad_node_MSEs_real) if v > MSE_THRESHOLD_REAL],
    "MSE_imag": [i for i, v in enumerate(bad_node_MSEs_imag) if v > MSE_THRESHOLD_IMAG],
    "R2_real":  [i for i, v in enumerate(bad_node_R2s_real) if v < R2_THRESHOLD],
    "R2_imag":  [i for i, v in enumerate(bad_node_R2s_imag) if v < R2_THRESHOLD],
}



# ----------------------------
# Print results
# ----------------------------
print("\n===== Nodes (Good epoch) =====")
for k, v in hard_nodes_good.items():
    print(f"{k} : {len(v)} nodes -> {v[:]}")

print("\n===== Nodes (Bad epoch) =====")
for k, v in hard_nodes_bad.items():
    print(f"{k} : {len(v)} nodes -> {v[:]}")





