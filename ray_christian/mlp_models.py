
from torchmetrics import F1Score, FBetaScore, Recall, Precision
from torch.utils.data import Dataset, DataLoader


import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os as os
import pandas as pd
import h5py


class NetworkMeasures_MLP_Dataset(Dataset):
    def __init__(self, path_scikit, path_dataset, task, task_type, selected_measures, slice_index=slice(0, 0), normalize_targets=False):
        self.normalize_targets = normalize_targets
        if slice_index.stop == 0:
            self.data_len, self.start_index, digits = self.get_length_of_dataset(
                path_dataset)
        else:
            _, _, digits = self.get_length_of_dataset(path_dataset)
            self.start_index = slice_index.start + 1
            self.data_len = slice_index.stop - slice_index.start + 1

        self.path_dataset = path_dataset
        self.path_scikit = path_scikit
        self.num_digits = '0' + str(digits.__str__().__len__())
        self.task = task
        self.task_type = task_type
        self.selected_measures = selected_measures
        self.labels = self.read_in_labels()
        self.input_data = self.prepare_input_data()
        self.positive_weight = self.compute_pos_weight()

    def read_in_network_measures(self):
        df_all_data = {}
        for i in range(self.start_index, self.start_index+self.data_len):
            id = format(i, self.num_digits)
            file_to_read = self.path_scikit + \
                'network_measures_'+str(id) + '.csv'
            df_one_grid = pd.read_csv(file_to_read, index_col="node")
            df_all_data[i] = df_one_grid
        return df_all_data

    def __get_label__(self, index):
        id_index = format(index, self.num_digits)
        if self.task == "snbs":
            file_to_read = str(self.path_dataset) + \
                '/snbs_'+str(id_index) + '.h5'
        elif self.task == "surv":
            file_to_read = str(self.path_dataset) + \
                '/surv_'+str(id_index) + '.h5'
        elif "troublemaker" in self.task:
            task_number = str(self.task[14:])
            file_to_read = str(self.path_dataset)+'/trouble_maker_init_threshold_2-5percentile_name_' + \
                str(task_number) + '_id_' + str(id_index) + '.h5'
        elif "surv" in self.task:
            task_number = str(self.task)[5:]
            file_to_read = str(self.path_dataset) + '/surv_threshold_' + \
                str(task_number) + '_id_' + str(id_index) + '.h5'
        hf = h5py.File(file_to_read, 'r')
        if self.task == "snbs":
            dataset_target = hf.get('snbs')
            targets = np.array(dataset_target)
        elif "surv" in self.task:
            dataset_target = hf.get('surv')
            targets = np.array(dataset_target)
            if self.task_type == "classification":
                targets = np.where(targets == 1.0, 0., 1.)
        elif "troublemaker" in self.task:
            dataset_target = hf.get("nodal_max_freq_dev")
            targets = np.array(dataset_target)
            targets = targets/self.normalize_targets
            if self.task_type == "classification":
                targets = np.where(targets < 15.0, 0., 1.)
        hf.close()
        return torch.tensor(targets).unsqueeze(1)

    def read_in_labels(self):
        return_labels = torch.empty(0)
        for i in range(self.start_index, self.start_index+self.data_len):
            return_labels = torch.cat(
                [return_labels, self.__get_label__(i).unsqueeze(0)])
        return return_labels

    def select_measures(self, dataset_input):
        dataset_input_selected = {}
        for i, data_df in dataset_input.items():
            if self.selected_measures != "all":
                data_df_selected = data_df.loc[:, self.selected_measures]
            else:
                data_df_selected = data_df.measures
            if "node_cat" in data_df_selected.columns:
                data_df_selected = data_df_selected.drop(columns=["node_cat"])
            dataset_input_selected[i] = data_df_selected
        return dataset_input_selected

    def prepare_input_data(self):
        input_data = torch.empty(0)
        network_measures = self.read_in_network_measures()
        selected_measures = self.select_measures(network_measures)
        for key, item in selected_measures.items():
            data_one_grid = torch.tensor(item.to_numpy()).unsqueeze(0)
            input_data = torch.cat([input_data, data_one_grid])
        return input_data

    def get_length_of_dataset(self, grid_path):
        count = 0
        for file in sorted(os.listdir(grid_path)):
            if file.startswith('grid_data_'):
                if count == 0:
                    startIndex = int(os.path.splitext(
                        file)[0].split('grid_data_')[1])
                    digits = (os.path.splitext(
                        file)[0].split('grid_data_')[1])
                count += 1
        return count, startIndex, digits

    def compute_pos_weight(self):
        count_positive = 0
        num_samples = 0
        for y in self.labels:
            count_positive += (y == 1).sum()
            num_samples += len(y)
        return (num_samples-count_positive)/count_positive

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        return self.input_data[idx], self.labels[idx]


class HiddenLayerModule(torch.nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super(HiddenLayerModule, self).__init__()
        self.activation = activation
        self.layer = nn.Linear(dim_in, dim_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = F.relu(x)
        return x


class MLP01(torch.nn.Module):
    def __init__(self, num_classes=1, num_input_features=6, num_hidden_layers=1, num_hidden_unit_per_layer=30, final_sigmoid_layer=False):
        super(MLP01, self).__init__()
        self.input_layer = nn.Linear(
            num_input_features, num_hidden_unit_per_layer)
        self.internal_layers = nn.ModuleList()

        for i in range(0, num_hidden_layers):
            hidden_layer = HiddenLayerModule(
                num_hidden_unit_per_layer, num_hidden_unit_per_layer, activation=True)
            self.internal_layers.append(hidden_layer)

        self.output_layer = nn.Linear(num_hidden_unit_per_layer, num_classes)

        if final_sigmoid_layer == True:
            self.endSigmoid = nn.Sigmoid()
        self.final_sigmoid_layer = final_sigmoid_layer

    def forward(self, data):
        x = self.input_layer(data)
        x = F.relu(x)

        for i, _ in enumerate(self.internal_layers):
            x = self.internal_layers[i](x)
        x = self.output_layer(x)
        if self.final_sigmoid_layer == True:
            x = self.endSigmoid(x)
        return x


class MLPModule(nn.Module):
    def __init__(self, config, criterion_positive_weight=False):
        super(MLPModule, self).__init__()
        cuda = config["cuda"]
        if "Fbeta::beta" in config:
            self.beta = config["Fbeta::beta"]
        if cuda and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.cuda = True
            print("cuda availabe:: send model to GPU")
        else:
            self.cuda = False
            self.device = torch.device("cpu")
            print("cuda unavailable:: train model on cpu")

        # seeds
        torch.manual_seed(config["manual_seed"])
        torch.cuda.manual_seed(config["manual_seed"])
        np.random.seed(config["manual_seed"])
        if self.cuda:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # set model
        num_classes = config["MLP::num_classes"]
        num_input_features = config["MLP::num_input_features"]
        num_hidden_layers = config["MLP::num_hidden_layers"]
        num_hidden_unit_per_layer = config["MLP::num_hidden_unit_per_layer"]
        final_sigmoid_layer = config["final_sigmoid_layer"]

        model = MLP01(num_classes, num_input_features, num_hidden_layers,
                      num_hidden_unit_per_layer, final_sigmoid_layer)
        model.double()
        model.to(self.device)

        self.model = model

        # criterion
        if config["criterion"] == "MSELoss":
            self.criterion = nn.MSELoss()
        if config["criterion"] == "BCEWithLogitsLoss":
            if criterion_positive_weight == False:
                self.criterion = nn.BCEWithLogitsLoss()
            else:
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(criterion_positive_weight))
                print("positive_weigt used for criterion: ",
                      criterion_positive_weight)
        if config["criterion"] == "BCELoss":
            self.criterion = nn.BCELoss()
        self.criterion.to(self.device)

        # set opimizer
        if config["optim::optimizer"] == "SGD":
            optimizer = optim.SGD(model.parameters(),
                                  lr=config["optim::LR"], momentum=config["optim::momentum"], weight_decay=config["optim::weight_decay"])
        if config["optim::optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(
            ), lr=config["optim::LR"], weight_decay=config["optim::weight_decay"])
        self.optimizer = optimizer

        # scheduler
        scheduler_name = config["optim::scheduler"]
        self.scheduler_name = scheduler_name
        if scheduler_name == "ReduceLROnPlateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=config["optim::ReducePlat_patience"], factor=config["optim::LR_reduce_factor"])
        elif scheduler_name == "stepLR":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=config["optim::stepLR_step_size"], gamma=config["optim::LR_reduce_factor"])
        elif scheduler_name == "ExponentialLR":
            scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=.1, last_epoch=-1)
        elif scheduler_name == "None":
            scheduler = None
        elif scheduler_name == None:
            scheduler = None
        self.scheduler = scheduler

    def forward(self, x):
        y = self.model(x)
        return y

    def save_model(self, epoch, perf_dict, path=None):
        if path is not None:
            fname = path.joinpath(f"model_epoch_{epoch}.ptf")
            perf_dict["state_dict"] = self.model.state_dict()
            torch.save(perf_dict, fname)
        return None

    def scheduler_step(self, criterion):
        scheduler_name = self.scheduler_name
        if scheduler_name == "ReduceLROnPlateau":
            self.scheduler.step(criterion)
        if scheduler_name == "stepLR":
            self.scheduler.step()
        if scheduler_name == "ExponentialLR":
            self.scheduler.step()

    def train_epoch_regression(self, data_loader, threshold):
        self.model.train()
        # scheduler = self.scheduler
        loss = 0.
        correct = 0
        mse_trained = 0.
        all_labels = torch.Tensor(0).to(self.device)
        for iter, (input_data, labels) in enumerate(data_loader):
            input_data = input_data.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(input_data)
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.optimizer.step()
            correct += torch.sum((torch.abs(output - labels) < threshold))
            loss += temp_loss.item()
            # R2
            mse_trained += torch.sum((output - labels) ** 2)
            all_labels = torch.cat([all_labels, labels])
        # self.epoch += 1
        # R2
        mean_labels = torch.mean(all_labels)
        array_ones = torch.ones(all_labels.shape)
        array_ones = array_ones.to(self.device)
        output_mean = mean_labels * array_ones
        mse_mean = torch.sum((output_mean-all_labels)**2)
        R2 = (1 - mse_trained/mse_mean).item()
        # R2 = 1 - mse_trained/mse_mean
        # accuracy
        accuracy = 100 * correct / all_labels.flatten().shape[0]
        self.scheduler_step(loss)
        return loss, accuracy.item(), R2

    def train_epoch_regressionThresholding(self, data_loader, threshold):
        self.model.train()
        # scheduler = self.scheduler
        loss = 0.
        mse_trained = 0.
        all_labels = torch.Tensor(0).to(self.device)
        all_outputs = torch.Tensor(0).to(self.device)
        for iter, (input_data, labels) in enumerate(data_loader):
            input_data = input_data.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(input_data)
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.optimizer.step()
            loss += temp_loss.item()
            # R2
            mse_trained += torch.sum((output - labels) ** 2)
            all_labels = torch.cat([all_labels, labels])
            all_outputs = torch.cat([all_outputs, output])
        # self.epoch += 1
        # R2 = 1 - mse_trained/mse_mean
        mean_labels = torch.mean(all_labels)
        array_ones = torch.ones(all_labels.shape)
        array_ones = array_ones.to(self.device)
        output_mean = mean_labels * array_ones
        mse_mean = torch.sum((output_mean-all_labels)**2)
        R2 = (1 - mse_trained/mse_mean).item()
        
        labels_classification = torch.where(all_labels < threshold, 0., 1.)
        outputs_classification = torch.where(all_outputs < threshold, 0., 1.)
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(outputs_classification, labels_classification.int())
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(outputs_classification, labels_classification.int())
        self.scheduler_step(loss)
        return loss, R2, fbeta.item(), recall.item()

    def train_epoch_classification(self, data_loader):
        self.model.train()
        loss = 0.
        correct = 0
        all_labels = torch.Tensor(0).to(self.device)
        all_outputs = torch.Tensor(0).to(self.device)
        for _, (input_data, labels) in enumerate(data_loader):
            input_data = input_data.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(input_data)
            temp_loss = self.criterion(output, labels)
            temp_loss.backward()
            self.optimizer.step()
            loss += temp_loss.item()
            all_labels = torch.cat([all_labels, labels])
            all_outputs = torch.cat([all_outputs, output])
        sigmoid_layer = torch.nn.Sigmoid().to(self.device)
        all_outputs_bin = sigmoid_layer(all_outputs) > .5
        correct += (all_outputs_bin == all_labels).sum()
        f1 = F1Score(multiclass=False).to(self.device)
        f1 = f1(all_outputs_bin.squeeze(1), all_labels.squeeze(1).int())
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(all_outputs_bin.squeeze(1), all_labels.squeeze(1).int())
        accuracy = 100 * correct / all_labels.flatten().shape[0]
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(all_outputs_bin, all_labels.int())
        precision = Precision(multiclass=False)
        precision = precision.to(self.device)
        precision = precision(all_outputs_bin, all_labels.int())
        self.scheduler_step(loss)
        return loss, accuracy, f1.item(), fbeta.item(), recall.item(), precision.item()

    def eval_model_regression(self, data_loader, threshold):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            correct = 0
            mse_trained = 0.
            all_labels = torch.Tensor(0).to(self.device)
            for _, (input_data, labels) in enumerate(data_loader):
                input_data = input_data.to(self.device)
                labels = labels.to(self.device)
                output = self.model(input_data)
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                correct += torch.sum((torch.abs(output - labels) < threshold))
                mse_trained += torch.sum((output - labels) ** 2)
                all_labels = torch.cat([all_labels, labels])
            accuracy = 100 * correct / all_labels.flatten().shape[0]
        mean_labels = torch.mean(all_labels)
        array_ones = torch.ones(all_labels.shape)
        array_ones = array_ones.to(self.device)
        output_mean = mean_labels * array_ones
        mse_mean = torch.sum((output_mean-all_labels)**2)
        R2 = (1 - mse_trained/mse_mean).item()
        return loss, accuracy.item(), R2
    
    def eval_model_regressionThresholding(self, data_loader, threshold):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            mse_trained = 0.
            all_labels = torch.Tensor(0).to(self.device)
            all_outputs = torch.Tensor(0).to(self.device)
            for iter, (input_data, labels) in enumerate(data_loader):
                input_data = input_data.to(self.device)
                labels = labels.to(self.device)
                output = self.model(input_data)
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                mse_trained += torch.sum((output - labels) ** 2)
                all_labels = torch.cat([all_labels, labels])
                all_outputs = torch.cat([all_outputs, output])
        mean_labels = torch.mean(all_labels)
        array_ones = torch.ones(all_labels.shape)
        array_ones = array_ones.to(self.device)
        output_mean = mean_labels * array_ones
        mse_mean = torch.sum((output_mean-all_labels)**2)
        R2 = (1 - mse_trained/mse_mean).item()

        labels_classification = torch.where(all_labels < threshold, 0., 1.)
        outputs_classification = torch.where(all_outputs < threshold, 0., 1.)
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(outputs_classification, labels_classification.int())
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(outputs_classification, labels_classification.int())
        self.scheduler_step(loss)

        return loss, R2, fbeta.item(), recall.item()

    def eval_model_classification(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            loss = 0.
            correct = 0
            mse_trained = 0.
            all_labels = torch.Tensor(0).to(self.device)
            all_outputs = torch.Tensor(0).to(self.device)
            for iter, (input_data, labels) in enumerate(data_loader):
                input_data = input_data.to(self.device)
                labels = labels.to(self.device)
                output = self.model(input_data)
                temp_loss = self.criterion(output, labels)
                loss += temp_loss.item()
                all_labels = torch.cat([all_labels, labels])
                all_outputs = torch.cat([all_outputs, output])
        sigmoid_layer = torch.nn.Sigmoid().to(self.device)
        all_outputs_bin = sigmoid_layer(all_outputs) > .5
        correct += (all_outputs_bin == all_labels).sum()
        f1 = F1Score(multiclass=False).to(self.device)
        f1 = f1(all_outputs_bin.squeeze(1), all_labels.squeeze(1).int())
        fbeta = FBetaScore(multiclass=False, beta=self.beta).to(self.device)
        fbeta = fbeta(all_outputs_bin.squeeze(1), all_labels.squeeze(1).int())
        accuracy = 100 * correct / all_labels.flatten().shape[0]
        recall = Recall(multiclass=False)
        recall = recall.to(self.device)
        recall = recall(all_outputs_bin, all_labels.int())
        precision = Precision(multiclass=False)
        precision = precision.to(self.device)
        precision = precision(all_outputs_bin, all_labels.int())
        return loss, accuracy, f1.item(), fbeta.item(), recall.item(), precision.item()
