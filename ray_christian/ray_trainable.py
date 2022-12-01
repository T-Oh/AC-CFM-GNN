from ray.tune import Trainable

from torch_geometric.data import DataLoader

from gnn_models import GNNmodule, gnn_snbs_surv
from mlp_models import MLPModule, NetworkMeasures_MLP_Dataset

import torch
import sys
import json
from pathlib import Path


class NN_tune_trainable(Trainable):
    def setup(self, config):
        print("task_type: ", config["task_type"])
        self.config = config
        self.seed = config["manual_seed"]
        self.cuda = config["cuda"]
        self.task_type = config["task_type"]
        task_type = self.task_type
        task = config["task"]
        if "MLP::use_MLP" in config.keys():
            self.use_MLP = config["MLP::use_MLP"]
        else:
            self.use_MLP = False
        if "troublemakers::normalize_targets" in config:
            self.normalize_targets= config["troublemakers::normalize_targets"]
        else:
            self.normalize_targets = False

        # torch.set_num_threads(config["num_threads"])
        # torch.set_num_interop_threads(config["num_threads"])

        if self.cuda and torch.cuda.is_available():
            pin_memory = True
        else:
            pin_memory = False

        if "num_workers" in config.keys():
            num_workers = config["num_workers"]
        else:
            num_workers = 1

        # data set
        if "train_set::start_index" in config.keys() and "train_set::end_index" in config.keys():
            train_slice = slice(config["train_set::start_index"], config["train_set::end_index"])
            print("train_slice: ", train_slice)
        else:
            train_slice = slice(0,0)
        
        if "valid_set::start_index" in config.keys() and "valid_set::end_index" in config.keys():
            valid_slice = slice(config["valid_set::start_index"], config["valid_set::end_index"])
            print("valid_slice: ", valid_slice)
        else:
            valid_slice = slice(0,0)
        if self.use_MLP:
            self.train_set = NetworkMeasures_MLP_Dataset(config["MLP::scikit_dataset_path"], config["dataset::path"] / 'train', task, task_type, config["MLP::selected_measures"], slice_index=train_slice, normalize_targets = self.normalize_targets)
            self.valid_set = NetworkMeasures_MLP_Dataset(
                config["MLP::scikit_dataset_path"], config["dataset::path"] / 'valid', task, task_type, config["MLP::selected_measures"], slice_index=valid_slice, normalize_targets = self.normalize_targets)
            self.test_set = NetworkMeasures_MLP_Dataset(
            config["MLP::scikit_dataset_path"], config["dataset::path"] / 'test', task, task_type, config["MLP::selected_measures"], normalize_targets = self.normalize_targets)
        else:
            self.train_set = gnn_snbs_surv(config["dataset::path"] / 'train', task, task_type, slice_index=train_slice, normalize_targets = self.normalize_targets)
            self.valid_set = gnn_snbs_surv(
                config["dataset::path"] / 'valid', task, task_type, slice_index=valid_slice, normalize_targets = self.normalize_targets)
            self.test_set = gnn_snbs_surv(
                config["dataset::path"] / 'test', task, task_type, normalize_targets = self.normalize_targets)
        
        # print("#################",config["train_set::batchsize"] )


        self.train_loader = DataLoader(
            self.train_set, batch_size=config["train_set::batchsize"], shuffle=config["train_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
        self.valid_loader = DataLoader(
            self.valid_set, batch_size=config["valid_set::batchsize"], shuffle=config["valid_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)
        self.test_loader = DataLoader(
            self.test_set, batch_size=config["test_set::batchsize"], shuffle=config["test_set::shuffle"], num_workers=num_workers, pin_memory=pin_memory)

        if self.use_MLP == False:
            if "criterion::positive_weight" in config.keys():
                if config["criterion::positive_weight"] == True:
                    train_set_positive_weight = self.train_set.positive_weight.clone().detach()
                    self.NN = GNNmodule(config, train_set_positive_weight.numpy())
                else:
                    self.NN = GNNmodule(config)
            else:
                self.NN = GNNmodule(config)
        else:
            if "criterion::positive_weight" in config.keys():
                if config["criterion::positive_weight"] == True:
                    train_set_positive_weight = self.train_set.positive_weight.clone().detach()
                    self.NN = MLPModule(config, train_set_positive_weight)
                else:
                    self.NN = MLPModule(config)
            else:
                self.NN = MLPModule(config)
    def step(self):
        if self.task_type == "regression":
            return self.step_regression()
        elif self.task_type == "classification":
            return self.step_classification()
        elif self.task_type == "regressionThresholding":
            return self.step_regressionThresholding()

    def step_regression(self):
        threshold = self.config["eval::threshold"]
        # train
        loss_train, acc_train, R2_train = self.NN.train_epoch_regression(
            self.train_loader, threshold)
        # valid
        loss_valid, acc_valid, R2_valid = self.NN.eval_model_regression(
            self.valid_loader, threshold)
        # test
        loss_test, acc_test, R2_test = self.NN.eval_model_regression(
            self.test_loader, threshold)

        result_dict = {
            "train_loss": loss_train,
            "train_acc": acc_train,
            "train_R2": R2_train,
            "test_loss": loss_test,
            "test_acc": acc_test,
            "test_R2": R2_test,
            "valid_loss": loss_valid,
            "valid_acc": acc_valid,
            "valid_R2": R2_valid,
        }

        return result_dict

    def step_regressionThresholding(self):
        threshold = self.config["eval::threshold"]
        # train
        loss_train, R2_train, fbeta_train, recall_train = self.NN.train_epoch_regressionThresholding(
            self.train_loader, threshold)
        # valid
        loss_valid, R2_valid, fbeta_valid, recall_valid = self.NN.eval_model_regressionThresholding(
            self.valid_loader, threshold)
        # test
        loss_test, R2_test, fbeta_test, recall_test = self.NN.eval_model_regressionThresholding(
            self.test_loader, threshold)

        result_dict = {
            "train_loss": loss_train,
            "train_R2": R2_train,
            "train_fbeta": fbeta_train,
            "train_recall": recall_train,
            "valid_loss": loss_valid,
            "valid_R2": R2_valid,
            "valid_fbeta": fbeta_valid,
            "valid_recall": recall_valid,
            "test_loss": loss_test,
            "test_R2": R2_test,
            "test_fbeta": fbeta_test,
            "test_recall": recall_test,
        }

        return result_dict

    def step_classification(self):
        threshold = self.config["eval::threshold"]
        # train
        loss_train, acc_train, f1_train, fbeta_train, recall_train, precision_train = self.NN.train_epoch_classification(
            self.train_loader)
        # valid
        loss_valid, acc_valid, f1_valid, fbeta_valid, recall_valid, precision_valid = self.NN.eval_model_classification(
            self.valid_loader)
        # test
        loss_test, acc_test, f1_test, fbeta_test, recall_test, precision_test = self.NN.eval_model_classification(
            self.test_loader)

        result_dict = {
            "train_loss": loss_train,
            "train_acc": acc_train,
            "train_f1": f1_train,
            "train_fbeta": fbeta_train,
            "train_recall": recall_train,
            "train_precision": precision_train,
            "valid_loss": loss_valid,
            "valid_acc": acc_valid,
            "valid_f1": f1_valid,
            "valid_fbeta": fbeta_valid,
            "valid_recall": recall_valid,
            "valid_precision": precision_valid,
            "test_loss": loss_test,
            "test_acc": acc_test,
            "test_f1": f1_test,
            "test_fbeta": fbeta_test,
            "test_recall": recall_test,
            "test_precision": precision_test
        }

        return result_dict

    def save_checkpoint(self, experiment_dir):
        # save model state dict
        path = Path(experiment_dir).joinpath("model_state_dict")
        torch.save(self.NN.model.state_dict(), path)
        # save optimizer state dict
        path = Path(experiment_dir).joinpath("opt_state_dict")
        torch.save(self.NN.optimizer.state_dict(), path)
        # save scheduler state dict
        if self.NN.scheduler != None:
            path = Path(experiment_dir).joinpath("scheduler_state_dict")
            torch.save(self.NN.scheduler.state_dict(), path)

        return experiment_dir

    def load_checkpoint(self, experiment_dir):
        # load model state dict
        path = Path(experiment_dir).joinpath("model_state_dict")
        checkpoint = torch.load(path)
        self.NN.model.load_state_dict(checkpoint)
        # load optimizer state dict
        path = Path(experiment_dir).joinpath("opt_state_dict")
        checkpoint = torch.load(path)
        self.NN.optimizer.load_state_dict(checkpoint)
        # load scheduler state dict
        if self.NN.scheduler != None:
            path = Path(experiment_dir).joinpath("scheduler_state_dict")
            checkpoint = torch.load(path)
            self.NN.scheduler.load_state_dict(checkpoint)
