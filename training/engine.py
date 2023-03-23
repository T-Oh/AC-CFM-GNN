from torch import no_grad
from torch import sum as torch_sum
from torch import cat, save, mean, stack
import numpy as np
import logging
#TO
from torchmetrics import R2Score
import torch
from utils.utils import discrete_loss
#from torchviz import make_dot
import math


class Engine(object):
    """
    Contains the necessary code for training and evaluating
    a single epoch
    """

    def __init__(self, model, optimizer, device, criterion, tol=0.1,task="NodeReg"):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion
        self.tol = tol
        self.task = task

    def train_epoch(self, dataloader, gradclip, epoch=0):
        
        #TO print weight matrices for debugging
        """print('\nBefore TRAINING\n')
        for param in self.model.parameters():
            print(param)"""
            
        loss = 0.0
        self.model.train()  #sets the mode to training (layers can behave differently in training than testing)
        R2score=R2Score().to(self.device)
        first=True;

        count = 0
        for (i, batch) in enumerate(dataloader):
            self.optimizer.zero_grad()
            count +=1
            batch.to(self.device)


            output = self.model.forward(batch, epoch).reshape(-1)  #reshape used to make sure that output is 1 dimensional
            output.to(self.device)
            
            if self.task == "GraphReg": #set labels according to task (GraphReg or NodeReg)
                labels = batch.y
            elif self.task == "NodeReg":
                labels = batch.node_labels.type(torch.FloatTensor).to(self.device)

            
            #compile outputs and labels for saving
                        
            if first:
                if self.task == "NodeReg":
                    total_output=output[None,:] 
                    total_labels=labels[None,:]
                else:
                    total_output=output
                    total_labels=labels
                first=False
            else:
                if self.task == "NodeReg":
                    total_output=cat((total_output,output[None,:]),0)   
                    total_labels=cat((total_labels,labels[None,:]),0)
                else:
                    total_output=cat((total_output,output),0)  
                    total_labels=cat((total_labels,labels),0)
            
            #total_output = output    #REMOVE if total output of every epoch should be saved
            #total_labels = labels
            #calc and backpropagate loss


            temp_loss = self.criterion(output, labels)#.float()
            print(f'Temp Loss:{temp_loss}') 


            
            temp_loss.backward()
            #print(temp_loss.grad)
            if gradclip != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradclip)
            
            self.optimizer.step()
            loss += temp_loss.item()
        R2 = R2score(total_output.reshape(-1), total_labels.reshape(-1))
            
        #TO print weight matrices for debugging
        """print('\nAFTER TRAINING\n')
        for param in self.model.parameters():
            print(param)"""

        #End TO
        return loss/count, R2, total_output, total_labels

    def eval(self, dataloader):
        "Evaluates model"
        self.model.eval()
        with no_grad():
            loss = 0.
            discrete_measure = 0.
            correct = 0
            
            first = True
            #second = True
            count = 0
            for batch in dataloader:
                count += 1
                batch.to(self.device)
                if self.task == 'GraphReg':
                    temp_labels=batch.y
                else:
                    temp_labels = batch.node_labels.type(torch.FloatTensor)
                    temp_output = self.model.forward(batch,100).reshape(-1).to(self.device)
                if first:
                    labels=temp_labels.clone()
                    output= temp_output.clone()
                    first = False
                    """elif second:
                    print(labels.shape)
                    print(temp_labels.shape)
                    labels = torch.stack([labels,temp_labels])
                    output = torch.stack([output, temp_output])
                    second = False"""
                else:
                    #labels = torch.cat([labels, temp_labels])     #.unsqueeze(0)
                    #output = torch.cat([output, temp_output])     #.unsqueeze(0)
                    output=cat((output,temp_output),0)  
                    labels=cat((labels,temp_labels),0)

            #TO
            R2torch=R2Score().to(self.device)
            labels = labels.to(self.device)
            output = output.to(self.device)
            loss = self.criterion(output, labels)
            discrete_measure = discrete_loss(output.clone(), labels.clone())
            

            R2=R2torch(output.reshape(-1), labels.reshape(-1))
            correct = ((labels-output).abs() < self.tol).sum().item()
            accuracy = correct/len(dataloader.dataset)
            #TO end
        evaluation = [loss, R2, accuracy, discrete_measure/count]
        return evaluation, np.array(output[0:16000].cpu()), np.array(labels[0:16000].cpu())
