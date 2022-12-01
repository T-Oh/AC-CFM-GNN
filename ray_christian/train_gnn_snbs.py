import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch_geometric.data import DataLoader

from ray import tune
from ray.tune.suggest.optuna import OptunaSearch
import time

from gnn_models import initialize_model, gnn_snbs


print("import finished")


cfg = {}
cfg["manual_seed"] = 1
cfg["dataset::path"] = "/home/chris/work/pik/projects/dataset_gnn/N020/4Pytorch"
cfg["train_set::start_index"] = 0
cfg["train_set::end_index"] = 799
cfg["test_set::start_index"] = 800
cfg["test_set::end_index"] = 999
cfg["train_set::batchsize"] = 100
cfg["test_set::batchsize"] = cfg["test_set::end_index"] - \
    cfg["test_set::start_index"] + 1
cfg["train_set::shuffle"] = True
cfg["test_set::shuffle"] = False
cfg["epochs"] = 50
cfg["model"] = "ArmaNet_optuna"
cfg["optim::optimizer"] = "SGD"
cfg["optim::LR"] = .3
cfg["optim::momentum"] = .9
cfg["optim::weight_decay"] = 1e-9
cfg["cfg_path"] = "./"
cfg["criterion"] = "MSELoss"
cfg["eval::threshold"] = .1


json_config = json.dumps(cfg)
f = open(cfg["cfg_path"] + "training_cfg.json", "w")
f.write(json_config)
f.close()


def train_epoch(model, data_loader, device, optimizer, criterion):
    loss = 0.0
    for iter, (batch) in enumerate(data_loader):
        batch.to(device)
        model.train()
        optimizer.zero_grad()
        outputs = model.forward(batch)
        labels = batch.y
        temp_loss = criterion(outputs, labels)
        temp_loss.backward()
        optimizer.step()
        loss += temp_loss.item()
        # if (iter*cfg["train_set::batchsize"]) % (.1*len(train_set)) == 0:
        #     print('[' + '{:5}'.format(iter * cfg["train_set::batchsize"]) + '/' + '{:5}'.format(len(train_set)) +
        #           ' (' + '{:3.0f}'.format(100 * iter / len(train_loader)) + '%)] Train Loss: ' +
        #           '{:6.4f}'.format(temp_loss.item()))
    return loss


def eval(model, data_loader, tol, device, optimizer, criterion):
    model.eval()
    torch.no_grad()
    N = data_loader.dataset[0].x.shape[0]
    loss = 0.
    correct = 0
    for batch in data_loader:
        batch.to(device)
        labels = batch.y
        output = model(batch)
        temp_loss = criterion(output, labels)
        loss += temp_loss.item()
        correct += get_prediction(output, labels, tol)
    accuracy = 100 * correct / (N*batch.num_graphs*len(data_loader))
    print(f"Test loss: {loss/len(data_loader):.3f}.. "
          f"Test accuracy: {accuracy:.3f} %"
          )
    return loss, accuracy


def get_prediction(output, label, tol):
    count = 0
    output = output.view(-1, 1).view(-1)
    label = label.view(-1, 1).view(-1)
    batchSize = output.size(-1)
    for i in range(batchSize):
        if ((abs(output[i] - label[i]) < tol).item() == True):
            count += 1
    return count


def compute_R2(model, data_loader, device):
    # R**2 = 1 - mse(y,t) / mse(t_mean,t)
    model.eval()
    torch.no_grad()
    mse_trained = 0
    all_labels = []
    for batch in data_loader:
        batch.to(device)
        labels = batch.y
        output = model(batch)
        mse_trained += torch.sum((output - labels) ** 2)
        all_labels.append(labels)
    mean_labels = torch.mean(all_labels[0])
    array_ones = torch.ones(all_labels[0].shape[0], 1)
    if torch.cuda.is_available():
        array_ones = array_ones.cuda()
    output_mean = mean_labels * array_ones
    mse_mean = torch.sum((output_mean-all_labels[0])**2)
    return (1 - mse_trained/mse_mean).item()


def copy_weights(old_model, new_model):
    old_model_dict = old_model.state_dict()
    new_model_dict = new_model.state_dict()
    new_dict = {k: v for k, v in old_model_dict.items() if
                (k in new_model_dict) and (new_model_dict[k].shape == old_model_dict[k].shape)}
    new_model_dict.update(new_model_dict)
    return new_model


def gnn_model_summary(model):

    model_params_list = list(model.named_parameters())
    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new)
    print("----------------------------------------------------------------")
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>20}  {:>25} {:>15}".format(
            p_name, str(p_shape), str(p_count))
        print(line_new)
    print("----------------------------------------------------------------")
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params)
    num_trainable_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params)
    print("Non-trainable params:", total_params - num_trainable_params)

# from torch_geometric.nn import ARMAConv, GCNConv
# from torch_geometric.nn import Sequential
# conv1 = ARMAConv(in_channels=1, out_channels=16, num_stacks=3, num_layers=4, shared_weights=True, dropout=0)
# conv2 = ARMAConv(in_channels=16, out_channels=1, num_stacks=3, num_layers=4, shared_weights=True, dropout=0)
# x0=train_set[0].x
# edge_index = train_set[0].edge_index
# edge_weight = train_set[0].edge_attr

# x_out = conv1(x=x0.float(), edge_index=edge_index)
# x_out = conv2(x=x_out.float(), edge_index=edge_index)

# conv1 = ARMAConv(in_channels=1, out_channels=16, num_stacks=3, num_layers=4, shared_weights=True, dropout=0)
# conv2 = ARMAConv(in_channels=16, out_channels=1, num_stacks=3, num_layers=4, shared_weights=True, dropout=0)
# x_out2 = conv1(x=x0.float(), edge_index=edge_index)
# x_out2 = conv2(x=x_out2.float(), edge_index=edge_index)


# model1 = initialize_model("ArmaNet_optuna",16)
# model2 = initialize_model("ArmaNet_optuna2",16)
# model3 = initialize_model("ArmaNet_optuna2",16)

# new_model1=copy_weights(model2,model1)
# new_model2=copy_weights(model1,model2)


# model1.to(device)
# model2.to(device)
# model3.to(device)

# model1.double()
# model2.double()
# model3.double()

# model1.eval()
# model2.eval()
# model3.eval()

# new_model1.to(device)
# new_model2.to(device)
# new_model1.double()
# new_model2.double()
# new_model1.eval()
# new_model2.eval()


def train_multiple_epochs(config, checkpoint_dir=None):
    cfg = {}
    cfg["manual_seed"] = 1
    cfg["dataset::path"] = "/home/chris/work/pik/projects/dataset_gnn/N020/4Pytorch"
    cfg["train_set::start_index"] = 0
    cfg["train_set::end_index"] = 799
    cfg["test_set::start_index"] = 800
    cfg["test_set::end_index"] = 999
    cfg["train_set::batchsize"] = 100
    cfg["test_set::batchsize"] = cfg["test_set::end_index"] - \
        cfg["test_set::start_index"] + 1
    cfg["train_set::shuffle"] = True
    cfg["test_set::shuffle"] = False
    cfg["epochs"] = 10
    cfg["model"] = "ArmaNet_optuna"
    cfg["optim::optimizer"] = "SGD"
    cfg["optim::LR"] = .3
    cfg["optim::momentum"] = .9
    cfg["optim::weight_decay"] = 1e-9
    cfg["cfg_path"] = "./"
    cfg["criterion"] = "MSELoss"
    cfg["eval::threshold"] = .1

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting the seeds
    torch.manual_seed(cfg["manual_seed"])
    torch.cuda.manual_seed(cfg["manual_seed"])
    np.random.seed(cfg["manual_seed"])
    # if device == "cuda":
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False

    train_set = gnn_snbs(cfg["dataset::path"], slice_index=slice(
        cfg["train_set::start_index"], cfg["train_set::end_index"]))
    test_set = gnn_snbs(cfg["dataset::path"], slice_index=slice(
        cfg["test_set::start_index"], cfg["test_set::end_index"]))

    model = initialize_model(cfg["model"], num_layers=2, num_channels=[
        1, 30, 1], num_internal_layers=[4, 4], num_internal_stacks=[config["stack1"], config["stack2"]])

    model.to(device)
    model.double()

    if cfg["criterion"] == "MSELoss":
        criterion = nn.MSELoss()
    if cfg["optim::optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=cfg["optim::LR"], momentum=cfg["optim::momentum"])
    if cfg["optim::optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(
        ), lr=cfg["optim::LR"], weight_decay=cfg["optim::weight_decay"])

    train_loader = DataLoader(
        train_set, batch_size=cfg["train_set::batchsize"], shuffle=cfg["train_set::shuffle"])
    test_loader = DataLoader(
        test_set, batch_size=cfg["test_set::batchsize"], shuffle=cfg["test_set::shuffle"])

    train_loss, test_loss = [], []
    test_accuracy = []
    R2_score = []
    epochs = cfg["epochs"]
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}.. ")
        temp_loss = train_epoch(model, train_loader,
                                device, optimizer, criterion)
        train_loss.append(temp_loss)
        temp_test_loss, test_accu = eval(
            model, test_loader, cfg["eval::threshold"], device, optimizer, criterion)
        test_loss.append(temp_test_loss)
        test_accuracy.append(test_accu)
        R2 = compute_R2(model, test_loader, device)
        print('R2: ''{:3.2f}'.format(100 * R2) + '%')
        R2_score.append(R2)
        tune.report(mean_accuracy=R2)


def train_multiple_epochs_old(config, checkpoint_dir="./ray"):
    cfg = {}
    cfg["manual_seed"] = 1
    cfg["dataset::path"] = "../../4Pytorch"
    cfg["train_set::start_index"] = 0
    cfg["train_set::end_index"] = 799
    cfg["test_set::start_index"] = 800
    cfg["test_set::end_index"] = 999
    cfg["train_set::batchsize"] = 100
    cfg["test_set::batchsize"] = cfg["test_set::end_index"] - \
        cfg["test_set::start_index"] + 1
    cfg["train_set::shuffle"] = True
    cfg["test_set::shuffle"] = False
    cfg["epochs"] = 10
    cfg["model"] = "ArmaNet_optuna"
    cfg["optim::optimizer"] = "SGD"
    cfg["optim::LR"] = .3
    cfg["optim::momentum"] = .9
    cfg["optim::weight_decay"] = 1e-9
    cfg["cfg_path"] = "./"
    cfg["criterion"] = "MSELoss"
    cfg["eval::threshold"] = .1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setting the seeds
    torch.manual_seed(cfg["manual_seed"])
    torch.cuda.manual_seed(cfg["manual_seed"])
    np.random.seed(cfg["manual_seed"])
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_set = gnn_snbs(cfg["dataset::path"], slice_index=slice(
        cfg["train_set::start_index"], cfg["train_set::end_index"]))
    test_set = gnn_snbs(cfg["dataset::path"], slice_index=slice(
        cfg["test_set::start_index"], cfg["test_set::end_index"]))

    model = initialize_model(cfg["model"], num_layers=2, num_channels=[
        #  1, 30, 1], num_internal_layers=[4, 4], num_internal_stacks=[config["stack1"], config["stack2"]])
        1, 30, 1], num_internal_layers=[4, 4], num_internal_stacks=[6, 6])
    # model = initialize_model(cfg["model"], num_layers=2, num_channels=[
    #                          1, 30, 1], num_internal_layers=[4, 4], num_internal_stacks=[6, 6])

    model.to(device)
    model.double()

    if cfg["criterion"] == "MSELoss":
        criterion = nn.MSELoss()
    if cfg["optim::optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              #   lr=cfg["optim::LR"], momentum=cfg["optim::momentum"])
                              lr=config["lr"], momentum=config["momentum"])
    if cfg["optim::optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(
        ), lr=cfg["optim::LR"], weight_decay=cfg["optim::weight_decay"])

    train_loader = DataLoader(
        train_set, batch_size=cfg["train_set::batchsize"], shuffle=cfg["train_set::shuffle"])
    test_loader = DataLoader(
        test_set, batch_size=cfg["test_set::batchsize"], shuffle=cfg["test_set::shuffle"])

    train_loss, test_loss = [], []
    test_accuracy = []
    R2_score = []
    epochs = cfg["epochs"]
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}.. ")
        temp_loss = train_epoch(model, train_loader,
                                device, optimizer, criterion)
        train_loss.append(temp_loss)
        temp_test_loss, test_accu = eval(
            model, test_loader, cfg["eval::threshold"], device, optimizer, criterion)
        test_loss.append(temp_test_loss)
        test_accuracy.append(test_accu)
        R2 = compute_R2(model, test_loader, device)
        # if len(R2_score) > 1:
        #     if R2 > max(R2_score):
        #         torch.save(model.state_dict(), cfg["cfg_path"] + "best_model.pt")
        print('R2: ''{:3.2f}'.format(100 * R2) + '%')
        R2_score.append(R2)
        tune.report(mean_accuracy=R2)

    # best_accuracy_index = test_accuracy.index(max(test_accuracy))
    # best_R2_index = R2_score.index(max(R2_score))
    # print("Epoch of best test_accuracy: ", best_accuracy_index+1,
    #     "  Accuracy: ", test_accuracy[best_accuracy_index], '%')
    # print("Epoch of best R2_score: ", best_R2_index+1,
    #     '   R2: ''{:3.2f}'.format(100 * R2_score[best_R2_index]) + '%')

    # training_results = {}
    # training_results["train_loss"] = train_loss
    # training_results["test_loss"] = test_loss
    # training_results["test_accuracy"] = test_accuracy
    # training_results["R2_score"] = R2_score

    # json_results = json.dumps(training_results)
    # f = open(cfg["cfg_path"] + "training_results.json", "w")
    # f.write(json_results)
    # f.close()
    # print("results of training are stored in " + cfg["cfg_path"])


start = time.time()
# train_multiple_epochs(cfg,config)
# import os
# os.environ["RAY_PICKLE_VERBOSE_DEBUG"] = "1"

# analysis = tune.run(train_multiple_epochs)


# search_space = {
#     "lr": tune.loguniform(1e-4, 1e-2),
#     "momentum": tune.uniform(0.1, 0.9)
#     # "stack1": tune.choice([1, 2, 3]),
#     # "stack2": tune.choice([6, 7, 8]),
# }

# RAY_PICKLE_VERBOSE_DEBUG=1
# analysis = tune.run(
#     train_multiple_epochs,
#     config={
#         "lr": tune.loguniform(1e-4, 1e-2),
#         "momentum": tune.uniform(0.1, 0.9),
#     },
#     metric="mean_accuracy",
#     mode="max",
#     search_alg=OptunaSearch(),
#     num_samples=2)


analysis = tune.run(
    train_multiple_epochs,
    config={
        "stack1": tune.choice([1, 2, 3]),
        "stack2": tune.choice([6, 7, 8]),
    },
    metric="mean_accuracy",
    mode="max",
    search_alg=OptunaSearch(),
    num_samples=6)
taken = time.time() - start

# taken = time.time() - start
# analysis = tune.run(
#     train_mnist,
#     config={
#         "lr": tune.loguniform(1e-4, 1e-2),
#         "momentum": tune.uniform(0.1, 0.9),
#     },
#     metric="mean_accuracy",
#     mode="max",
#     search_alg=OptunaSearch(),
#     num_samples=10)


print("finished")
