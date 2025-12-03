import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# ----------------------------
# Load all epochs (handles "final" -> 100)
# ----------------------------
epoch_files = glob.glob("results/test_output*.pt")
all_epochs = []
epoch_map = {}   # maps epoch index -> filename

for f in epoch_files:
    suffix = f.split("test_output")[-1].split(".pt")[0]
    if suffix == "_final":
        epoch = 100   # map final -> 100
    elif suffix == "":
        continue
    else:
        epoch = int(suffix)
    all_epochs.append(epoch)
    epoch_map[epoch] = suffix

all_epochs = sorted(all_epochs)
print(f"Found {len(all_epochs)} epochs: {all_epochs[:10]}...")

# Store per-epoch metrics
all_node_MSEs_real = []
all_node_MSEs_imag = []

# Worst nodes tracking
worst_masks_real = []  # binary masks [E, 2000]
worst_masks_imag = []

TOP_K = 30  # number of worst nodes per epoch

for epoch in all_epochs:
    print(f"Processing epoch {epoch}...")
    suffix = epoch_map[epoch]
    out = torch.load(f"results/test_output{suffix}.pt")[0]
    lab = torch.load(f"results/test_labels{suffix}.pt")[0]

    N_INSTANCES = int(out.shape[0] / 2000)
    out_nodes = out.reshape(N_INSTANCES, 2000, 2)
    lab_nodes = lab.reshape(N_INSTANCES, 2000, 2)

    node_MSEs_real = []
    node_MSEs_imag = []

    for n in range(2000):
        y_pred_real = out_nodes[:, n, 0].detach().cpu().numpy()
        y_true_real = lab_nodes[:, n, 0].detach().cpu().numpy()
        y_pred_imag = out_nodes[:, n, 1].detach().cpu().numpy()
        y_true_imag = lab_nodes[:, n, 1].detach().cpu().numpy()

        node_MSEs_real.append(np.mean((y_pred_real - y_true_real) ** 2))
        node_MSEs_imag.append(np.mean((y_pred_imag - y_true_imag) ** 2))

    node_MSEs_real = np.array(node_MSEs_real)
    node_MSEs_imag = np.array(node_MSEs_imag)

    all_node_MSEs_real.append(node_MSEs_real)
    all_node_MSEs_imag.append(node_MSEs_imag)

    # Select worst K nodes per epoch
    worst_real_idx = np.argsort(-node_MSEs_real)[:TOP_K]  # highest MSE
    worst_imag_idx = np.argsort(-node_MSEs_imag)[:TOP_K]

    mask_real = np.zeros(2000, dtype=int)
    mask_imag = np.zeros(2000, dtype=int)
    mask_real[worst_real_idx] = 1
    mask_imag[worst_imag_idx] = 1

    worst_masks_real.append(mask_real)
    worst_masks_imag.append(mask_imag)

# ----------------------------
# Stack into arrays [epochs, nodes]
# ----------------------------
all_node_MSEs_real = np.stack(all_node_MSEs_real)   # [E, 2000]
all_node_MSEs_imag = np.stack(all_node_MSEs_imag)
worst_masks_real = np.stack(worst_masks_real)       # [E, 2000]
worst_masks_imag = np.stack(worst_masks_imag)

# Fraction of epochs each node is among the worst
worst_fraction_real = worst_masks_real.mean(axis=0)
worst_fraction_imag = worst_masks_imag.mean(axis=0)

# ----------------------------
# Combined heatmap (real + imag stacked)
# ----------------------------
fig, axes = plt.subplots(2, 1, figsize=(15,10), sharex=True)

sns.heatmap(worst_masks_real, cmap="Reds", cbar=True, ax=axes[0])
axes[0].set_ylabel("Epoch index")
axes[0].set_title(f"Top {TOP_K} worst nodes per epoch (Real channel)")

sns.heatmap(worst_masks_imag, cmap="Blues", cbar=True, ax=axes[1])
axes[1].set_xlabel("Node index")
axes[1].set_ylabel("Epoch index")
axes[1].set_title(f"Top {TOP_K} worst nodes per epoch (Imag channel)")

plt.tight_layout()
plt.savefig("results/worst_nodes_combined.png", dpi=150)

# ----------------------------
# Top persistent worst nodes
# ----------------------------
top_real = np.argsort(-worst_fraction_real)[:10]
top_imag = np.argsort(-worst_fraction_imag)[:10]

# Average MSE across all epochs (per node, then mean)
avg_node_MSE_real = all_node_MSEs_real.mean(axis=0)
avg_node_MSE_imag = all_node_MSEs_imag.mean(axis=0)
overall_avg_real = avg_node_MSE_real.mean()
overall_avg_imag = avg_node_MSE_imag.mean()

print("\n===== Real channel =====")
print(f"Overall average MSE (all nodes): {overall_avg_real:.6f}")
for i in top_real:
    print(f"Node {i} -> avg MSE: {avg_node_MSE_real[i]:.6f}, "
          f"in worst {TOP_K} for {worst_fraction_real[i]*100:.1f}% of epochs")

print("\n===== Imag channel =====")
print(f"Overall average MSE (all nodes): {overall_avg_imag:.6f}")
for i in top_imag:
    print(f"Node {i} -> avg MSE: {avg_node_MSE_imag[i]:.6f}, "
          f"in worst {TOP_K} for {worst_fraction_imag[i]*100:.1f}% of epochs")
