import logging
import torch
import json
from torch_geometric.transforms import ToUndirected, Compose, RemoveIsolatedNodes, NormalizeScale
from numpy.random import seed as numpy_seed

from training.engine import Engine
from training.training import run_training, run_tuning
from datasets.dataset import create_datasets, create_loaders
from models.get_models import get_model
from utils.get_optimizers import get_optimizer
from utils.utils import  plot_loss, plot_R2, ImbalancedSampler

#TO
#import hiddenlayer as hl #for GNN visualization
import shutil


#save config in results
shutil.copyfile("configurations/configuration.json","results/configuration.json")



logging.basicConfig(filename="results/regression.log", filemode="w", level=logging.INFO)

#Loading training configuration
with open("configurations/configuration.json", "r") as io:
    cfg = json.load(io)


#Loading and pre-transforming data
#trainset, testset = create_datasets(cfg["dataset::path"], pre_transform=ToUndirected()) 
trainset, testset = create_datasets(cfg["dataset::path"],cfg=cfg, pre_transform=None)
trainloader, testloader = create_loaders(cfg, trainset, testset)                        #TO the loaders contain the data and get batchsize and shuffle from cfg


#getting feature and target sizes
num_features = trainset.__getitem__(0).x.shape[1]
num_targets = 1


#Setting model parameters (these are tunable in hyperoptimization)
params = {
    "num_layers" : cfg["num_layers"],
    "hidden_size" : cfg["hidden_size"],
    "dropout" : cfg["dropout"]
}


#choosing device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #TO device represents the 'device' on which a torch.tensor is placed (cpu or cuda) -> cuda uses gpus

#setting seeds
torch.manual_seed(cfg["manual_seed"])
torch.cuda.manual_seed(cfg["manual_seed"])
numpy_seed(cfg["manual_seed"])
if device == "cuda":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#Loading GNN model
model = get_model(cfg, params, num_features, num_targets)   #TO get_model does not load an old model but create a new one 
model.to(device)


#TO for network visualization
#data=next(iter(trainloader))
#graph=hl.build_graph(model,(data.x,data.edge_index,data.edge_attr))
#graph.save("gnn_hiddenlayer",format="png")
#TO end
            
            
#choosing criterion
criterion = torch.nn.MSELoss()  #TO defines the loss
criterion.to(device)

#Choosing optimizer
optimizer = get_optimizer(cfg, model)

#Initializing engine
engine = Engine(model, optimizer, device, criterion, tol=cfg["accuracy_tolerance"])

#Runs study if set in configuration file
if cfg["study::run"]:
    objective = Objective(trainloader, testloader, engine, cfg["epochs"], cfg["study::lr::lower"], cfg["study::lr::upper"])
    optimal_params = run_tuning(cfg, objective)

    with open(f"optimal_params.json", "w+") as out:
        out.write(json.dumps(optimal_params))

    engine.optimizer.lr = optimal_params["lr"]


losses, evaluations, final_eval = run_training(trainloader, testloader, engine, epochs=cfg["epochs"])

logging.info("Final results:")
logging.info(f"Accuracy: {final_eval[2]}")
logging.info(f"R2: {final_eval[1]}")


save_model = True
plot = False

if save_model:
    torch.save(model.state_dict(), "results/" + cfg["model"] + ".pt")
    #torch.onnx.export(model,data,"supernode.onnx")
elif plot:
    train_loss = losses
    test_lost = evaluations[0]
    plot_loss(train_loss, test_loss, save=False)

torch.save(list(losses), "results/" + cfg["dataset::path"] + "losses.pt") #saving train losses
torch.save(list(evaluations), "results/" + cfg["dataset::path"] + "evaluations.pt") #saving test evaluations
