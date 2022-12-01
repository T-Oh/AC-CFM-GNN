from torch import no_grad
from torch import sum as torch_sum
from torch import cat,save, mean
import logging
#TO
from torchmetrics import R2Score
import torch
#from torchviz import make_dot


class Engine(object):
    """
    Contains the necessary code for training and evaluating
    a single epoch
    """

    def __init__(self, model, optimizer, device, criterion, tol=0.1):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.tol = tol

    def train_epoch(self, dataloader):

        loss = 0.0
        self.model.train()                                  #TO does not train the model but sets the mode to training (layers can behave differently in training than testing)
        #TO
        first=True;
        #End TO
        for (i, batch) in enumerate(dataloader):
            batch.to(self.device)
            self.optimizer.zero_grad()
            output = self.model.forward(batch).reshape(-1)
            #make_dot(output,params=dict(self.model.named_parameters())).render("gnn_torchviz", format="png")
            labels = batch.y
            #TO            
            if first:
                total_output=output
                total_labels=labels
                first=False
            else:
                total_output=cat((total_output,output),0)
                total_labels=cat((total_labels,labels),0)
            #End TO
            temp_loss = self.criterion(output, labels)      
            temp_loss.backward()
            self.optimizer.step()
            loss += temp_loss.item()
        #TO
        save(total_output, 'output.pt')
        save(total_labels, 'labels.pt')
        #End TO
        return loss

    def eval(self, dataloader):
        "Evaluates model"
        self.model.eval()
        with no_grad():
            loss = 0.
            correct = 0
            mse_trained = 0.
            total_error = 0.
            total_output=[]
            total_labels=[]
            for batch in dataloader:
                batch.to(self.device)
                labels = batch.y
                output = self.model(batch).reshape(-1)
                total_output.append(output)
                total_labels.append(labels)
                correct += ((labels - output).abs() < self.tol).sum().item()
                loss += ((labels.to(float) -  output) ** 2).mean().item()
                mse_trained += torch_sum((output -labels.reshape(-1)) **2).item()
                #total_error += torch_sum((output - dataloader.mean_labels) ** 2).item()
                total_error += torch_sum((labels -dataloader.mean_labels) ** 2).item()  #correct version

            accuracy = correct / len(dataloader.dataset)
            if total_error == 0.:
                R2 = 1
            else:
                R2 = (1 - mse_trained/total_error)
            R2torch=R2Score()
            #TO
            #R2=R2Score(output,labels)
            #TO end
        logging.debug(f"Accuracy: {accuracy}")
        return loss, R2, accuracy