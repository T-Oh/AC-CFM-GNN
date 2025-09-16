# %% Init
import sys
from pathlib import Path
import torch
import json5

# Add parent directory (project root) to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
PATH = "/home/tohlinger/HUI/Documents/hi-accf-ml/"


from utils.utils import setup_datasets_and_loaders

# Loading training configuration
configfile = PATH + "configurations/configuration.json"
with open(configfile, "r") as io:
    cfg = json5.load(io)



# %% Load through dataloader logic
max_seq_len_LDTSF, trainset, trainloader, _ = setup_datasets_and_loaders(cfg, 0, False)
batch = next(iter(trainloader))
print(batch)
x0 = batch[0][0][0].x
x1 = batch[0][0][1].x
x2 = batch[0][0][2].x
x3 = batch[0][0][3].x
x4 = batch[0][0][4].x
#batch = compile_batch(batch)


# %%
# Load directly
static = torch.load(PATH + 'processed/data_static.pt')
y1 = torch.load(PATH + 'processed/scenario_111/data_111_0.pt')
#y2 = torch.load(PATH + 'processed/scenario_111/data_111_1.pt').edge_index
#y3 = torch.load(PATH + 'processed/scenario_111/data_111_2.pt').edge_index
#y4 = torch.load(PATH + 'processed/scenario_111/data_111_3.pt').edge_index
#y5 = torch.load(PATH + 'processed/scenario_111/data_111_4.pt').edge_index






# %%
#Check create_data_from_prediction
from utils.utils import create_data_from_prediction
test_edge_pred = torch.zeros(7064)
test_edge_pred[0] = 1
new_data = create_data_from_prediction(y1.x[:,:2], test_edge_pred, static, y1.node_labels, y1.edge_labels)
print(f'new_data X: {new_data.x}')
print(f'original X: {y1.x}')
print(f'new_data edge_index: {new_data.edge_index}')
print(f'original edge_index: {y1.edge_index}')
print(f'new_data edge_attr: {new_data.edge_attr}')
print(f'original edge_attr: {y1.edge_attr}')
print(f'new_data y: {new_data.node_labels}')
print(f'original y: {y1.node_labels}')
print(f'new_data y: {new_data.edge_labels}')
print(f'original y: {y1.edge_labels}')
# %%
