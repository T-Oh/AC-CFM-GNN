import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle as pkl

def aggregate_files(result_folder):
    all_files = os.listdir(result_folder)
    grad_files = [f for f in all_files if f.startswith('gradients_') and f.endswith('.pt')]
    gradients = None
    
    with open(os.path.join(result_folder, 'results_.pkl'), 'rb') as f:
        metrics = pkl.load(f)
    for file in grad_files:
        grads = torch.load(os.path.join(result_folder, file))
        if gradients == None:
            gradients = {}

            for key in grads.keys():
                gradients[key] = []

        for key in grads.keys():

            gradients[key].append(grads[key])
    return gradients, metrics

def plot_gradients_vs_loss(gradients, metrics, result_folder, edge_weight = 1, PI_weight = 1):


    losses = np.array(metrics['test_loss'])
    epochs = range(len(losses))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for key in gradients.keys():
        grads = torch.stack(gradients[key]).cpu().numpy()
        if key == 'g_edge_loss':
            grads = grads * edge_weight
        if key == 'g_pi':
            grads = grads * PI_weight
        ax1.plot(range(len(losses)), grads, label=f'Gradient Norm ({key})')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('Gradient Norm vs Test Loss')

    ax2.plot(epochs, losses, color='blue', label='Loss')
    ax2.set_ylabel('Test Loss', color='blue')
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Gradient Norm vs Test Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, 'gradients_vs_testloss.png'))
    plt.close(fig)


    losses = np.array(metrics['train_loss'])
    epochs = range(len(losses))

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    for key in gradients.keys():
        grads = torch.stack(gradients[key]).cpu().numpy()
        ax1.plot(range(len(losses)), grads, label=f'Gradient Norm ({key})')

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Gradient Norm')
    ax1.set_title('Gradient Norm vs Train Loss')

    ax2.plot(epochs, losses, color='blue', label='Loss')
    ax2.set_ylabel('Train Loss', color='blue')

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Gradient Norm vs Train Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, 'gradients_vs_trainloss.png'))
    plt.close(fig)

if __name__ == "__main__":
    result_folder = 'results'  # Change this to your results folder
    edge_weight = 0.2
    PI_weight = 0.05
    gradients, metrics = aggregate_files(result_folder)
    plot_gradients_vs_loss(gradients, metrics, result_folder, edge_weight, PI_weight)