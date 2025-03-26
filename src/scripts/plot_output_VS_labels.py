import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

FOLDER = 'results/'
NODE_REGRESSION = False  # Set to False for graph regression
STATE_REGRESSION = True
assert not (NODE_REGRESSION and STATE_REGRESSION), 'Cannot have both node and state regression'

output = torch.load(FOLDER + 'output.pt')
labels = torch.load(FOLDER + 'labels.pt')
XVALUES = np.arange(2000 if NODE_REGRESSION or STATE_REGRESSION else len(output))

if NODE_REGRESSION:
    output = np.array(output)
    labels = np.array(labels)
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
elif STATE_REGRESSION:
    for i in range(int(len(output[0])/2000)):
        print(FOLDER)
        # Scatter plot Vreal
        fig, ax = plt.subplots()
        ax.scatter(labels[0][2000*i:2000*(i+1), 0], output[0][2000*i:2000*(i+1), 0], alpha=0.5)
        ax.set_xlabel('Labels')
        ax.set_ylabel('Output')
        ax.set_title(f'Scatter Re(V) - Chunk {i}')
        fig.savefig(FOLDER + f'scatter_outputVSlabel{i}Vreal.png', bbox_inches='tight')
        plt.close()

        # Scatter plot Vimag
        fig, ax = plt.subplots()
        ax.scatter(labels[0][2000*i:2000*(i+1), 1], output[0][2000*i:2000*(i+1), 1], alpha=0.5)
        ax.set_xlabel('Labels')
        ax.set_ylabel('Output')
        ax.set_title(f'Scatter Re(V) - Chunk {i}')
        fig.savefig(FOLDER + f'scatter_outputVSlabel{i}Vimag.png', bbox_inches='tight')
        plt.close()
        

    # Plot confusion matrix for edge status
    # Convert logits to predicted class
    predictions = np.argmax(output[1], axis=1)
    true_labels = labels[1].reshape(-1)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    classes = np.unique(true_labels)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Labeling
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # Annotate cells with values
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

    # Save the figure
    plt.savefig(FOLDER+'confusion_matrix_edge_status.png', bbox_inches='tight', dpi=300)
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
