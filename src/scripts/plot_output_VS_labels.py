import torch
import matplotlib.pyplot as plt
import numpy as np

FOLDER = '/home/tohlinger/LEO/Running/TAG_Zhu_NR/results/'
NODE_REGRESSION = True  # Set to False for graph regression

output = np.array(torch.load(FOLDER + 'output.pt'))
labels = np.array(torch.load(FOLDER + 'labels.pt'))
XVALUES = np.arange(2000 if NODE_REGRESSION else len(output))

if NODE_REGRESSION:
    for i in range(int(len(output)/2000)):
        # Plot Vreal
        fig, ax = plt.subplots()
        ax.bar(XVALUES, output[2000*i:2000*(i+1), 0], label='Output')
        ax.bar(XVALUES, labels[2000*i:2000*(i+1), 0], label='Labels')
        ax.set_title(f'Re(V) - Chunk {i}')
        ax.legend()
        fig.savefig(FOLDER + f'outputVSlabel{i}Vreal.png', bbox_inches='tight')
        plt.close()

        # Scatter plot Vreal
        fig, ax = plt.subplots()
        ax.scatter(labels[2000*i:2000*(i+1), 0], output[2000*i:2000*(i+1), 0], alpha=0.5)
        ax.set_xlabel('Labels')
        ax.set_ylabel('Output')
        ax.set_title(f'Scatter Re(V) - Chunk {i}')
        fig.savefig(FOLDER + f'scatter_outputVSlabel{i}Vreal.png', bbox_inches='tight')
        plt.close()

        # Plot Vimag
        fig, ax = plt.subplots()
        ax.bar(XVALUES, output[2000*i:2000*(i+1), 1], label='Output')
        ax.bar(XVALUES, labels[2000*i:2000*(i+1), 1], label='Labels')
        ax.set_title(f'Imag(V) - Chunk {i}')
        ax.legend()
        fig.savefig(FOLDER + f'outputVSlabel{i}Vimag.png', bbox_inches='tight')
        plt.close()

        # Scatter plot Vimag
        fig, ax = plt.subplots()
        ax.scatter(labels[2000*i:2000*(i+1), 1], output[2000*i:2000*(i+1), 1], alpha=0.5)
        ax.set_xlabel('Labels')
        ax.set_ylabel('Output')
        ax.set_title(f'Scatter Imag(V) - Chunk {i}')
        fig.savefig(FOLDER + f'scatter_outputVSlabel{i}Vimag.png', bbox_inches='tight')
        plt.close()

else:
    # Graph regression: Single plot for entire dataset
    fig, ax = plt.subplots()
    ax.bar(XVALUES, output[:, 0], label='Output')
    ax.bar(XVALUES, labels[:, 0], label='Labels')
    ax.set_title('Re(V) - Graph Level')
    ax.legend()
    fig.savefig(FOLDER + 'outputVSlabel_graph_Vreal.png', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    ax.scatter(labels[:, 0], output[:, 0], alpha=0.5)
    ax.set_xlabel('Labels')
    ax.set_ylabel('Output')
    ax.set_title('Scatter Re(V) - Graph Level')
    fig.savefig(FOLDER + 'scatter_outputVSlabel_graph_Vreal.png', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    ax.bar(XVALUES, output[:, 1], label='Output')
    ax.bar(XVALUES, labels[:, 1], label='Labels')
    ax.set_title('Imag(V) - Graph Level')
    ax.legend()
    fig.savefig(FOLDER + 'outputVSlabel_graph_Vimag.png', bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    ax.scatter(labels[:, 1], output[:, 1], alpha=0.5)
    ax.set_xlabel('Labels')
    ax.set_ylabel('Output')
    ax.set_title('Scatter Imag(V) - Graph Level')
    fig.savefig(FOLDER + 'scatter_outputVSlabel_graph_Vimag.png', bbox_inches='tight')
    plt.close()
