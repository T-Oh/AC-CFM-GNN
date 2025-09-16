import torch
import os
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

# Define path to results
PATH = '/home/tohlinger/PIK/Results/GTSF_StateReg/GAT/test/'
RESULT_FOLDER = os.path.join(PATH, 'results/')

# Load outputs and labels
output_path = os.path.join(RESULT_FOLDER, 'output_final.pt')
label_path = os.path.join(RESULT_FOLDER, 'labels_final.pt')

outputs = torch.load(output_path)[1]  # Shape: [num_samples, num_classes]
labels = torch.load(label_path)[1]    # Shape: [num_samples, num_classes] or [num_samples]
# Flatten labels and convert to binary class labels
labels = labels.reshape(-1)  # Shape: [num_edges_total]
print(f'Labels: {labels.shape}')
print('Output')
print(outputs.shape)

# Compute predictions and labels for 'StateReg' task
preds = torch.argmax(outputs, dim=1).cpu()
true_labels = labels.reshape(-1).cpu()

# Compute metrics
f1 = f1_score(true_labels, preds, average=None)
precision = precision_score(true_labels, preds, average=None)
recall = recall_score(true_labels, preds, average=None)
accuracy = accuracy_score(true_labels, preds)

f1, precision, recall, accuracy
print(f'F1 Score: {f1}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'Accuracy: {accuracy}')
