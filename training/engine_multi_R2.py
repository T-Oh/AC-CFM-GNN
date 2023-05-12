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
from datasets.dataset import mask_probs_add_bias


class Engine(object):
    """
    Contains the necessary code for training and evaluating
    a single epoch
    """

    def __init__(self, model, optimizer, device, criterion, tol=0.1, task="NodeReg", var=None, masking=False, mask_bias=0.0, zero_buses = np.zeros(2000)):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.tol = tol
        self.task = task
        self.vars = var.clone()
        self.mask_probs = mask_probs_add_bias(var, mask_bias)
        self.masks = torch.bernoulli(self.mask_probs)
        self.masking = masking
        self.criterion = criterion
        self.zero_buses = zero_buses
        
    def calc_R2(self, output, labels, zero_buses, R2score):
        """
        R2 = R2score(output, labels)
        R2adjusted=0
        N = 2000
        for i in range(2000):
            if zero_buses[i] == 0:
                N -= 1
            else:
                R2adjusted += R2[i]"""
        
        labels[0,zero_buses==0] += 0.001
        R2adjusted = R2score(output, labels)
        return R2adjusted.sum()


    def train_epoch(self, dataloader, gradclip):
        
        #TO print weight matrices for debugging
        """print('\nBefore TRAINING\n')
        for param in self.model.parameters():
            print(param)"""
        loss = 0.0
        self.model.train()  #sets the mode to training (layers can behave differently in training than testing)
        R2score=R2Score(num_outputs=2000, multioutput='raw_values').to(self.device)
        first=True;

        count = 0
        
        for (i, batch) in enumerate(dataloader):
            self.optimizer.zero_grad()
            count +=1
            batch.to(self.device)
            N_instances = int(len(batch.x)/2000)


            output = self.model.forward(batch)  #reshape used to make sure that output is 1 dimensional
            output.to(self.device)
            
            if self.task == "GraphReg": #set labels according to task (GraphReg or NodeReg)
                labels = batch.y
            elif self.task == "NodeReg":
                labels = batch.node_labels.type(torch.FloatTensor).to(self.device)

            
            #compile outputs and labels for saving
                        
            if first:
                if self.task == "NodeReg":
                    total_output=output.reshape(N_instances,2000)
                    total_labels=labels.reshape(N_instances,2000)
                else:
                    total_output=output
                    total_labels=labels
                first=False
            else:
                if self.task == "NodeReg":
                    total_output=cat((total_output,output.reshape(N_instances,2000)),0)   
                    total_labels=cat((total_labels,labels.reshape(N_instances,2000)),0)
                else:
                    total_output=cat((total_output,output),0)  
                    total_labels=cat((total_labels,labels),0)
            
            #total_output = output    #REMOVE if total output of every epoch should be saved
            #total_labels = labels
            #calc and backpropagate loss
            if self.masking:
                for j in range(int(len(output)/2000)):
                   if j==0:
                       self.masks=torch.bernoulli(self.mask_probs)
                       print(torch.bincount(self.masks.to(int))[1]/2000)
                   else:
                       self.masks=torch.cat((self.masks,torch.bernoulli(self.mask_probs)))
                   #self.masks= self.masks.to('cuda:0')
                output = output*self.masks
                labels = labels*self.masks
            

            temp_loss = self.criterion(output.to(self.device), labels.to(self.device))#.float()

            temp_loss.backward()
            #print(temp_loss.grad)
            if gradclip != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradclip)
            
            self.optimizer.step()
            loss += temp_loss.item()
        #R2 = R2score(total_output, total_labels)
        R2 = self.calc_R2(total_output, total_labels, self.zero_buses, R2score)
        example_output = total_output[0:16000]
        example_labels = total_labels[0:16000]
        del total_output
        del total_labels
            
        #TO print weight matrices for debugging
        """print('\nAFTER TRAINING\n')
        for param in self.model.parameters():
            print(param)"""

        #End TO
        return loss/count, R2, example_output, example_labels

    def eval(self, dataloader, full_output=False):
        "Evaluates model"
        self.model.eval()
        first = True
        with no_grad():
            loss = 0.
            discrete_measure = 0.
            correct = 0

            count = 0
            index = 0
            for batch in dataloader:
                N_instances = int(len(batch.x)/2000)
                print(N_instances)
                count += 1
                batch.to(self.device)
                if self.task == 'GraphReg':
                    temp_labels=batch.y
                else:
                    if first:
                        output = self.model.forward(batch).reshape(N_instances, 2000)
                        labels = batch.node_labels.reshape(N_instances, 2000)
                        first = False
                    else:
                        output = torch.cat([output, self.model.forward(batch).reshape(N_instances,2000)], dim=0)
                        labels = torch.cat([labels, batch.node_labels.reshape(N_instances, 2000)], dim=0)
                            
                    print(labels.shape)
                    print(batch.node_labels.reshape(N_instances, 2000).shape)
                    #labels[index: index+N_instances] = batch.node_labels.reshape(N_instances, 2000).type(torch.FloatTensor)
                    #output[index: index+N_instances] = self.model.forward(batch).reshape(N_instances, 2000)#.to(self.device)
                    index += N_instances
                    print('Labels and Output')
                    print(labels)
                    print(output)


            #TO
            R2torch = R2Score(num_outputs=2000,multioutput='raw_values').to(self.device)
            labels = labels.to(self.device)
            output = output.to(self.device)

            #R2=R2torch(output, labels)
            R2 = self.calc_R2(output, labels, self.zero_buses, R2torch)
            loss = self.criterion(output, labels)

            discrete_measure = discrete_loss(output.clone(), labels.clone())



            correct = ((labels-output).abs() < self.tol).sum().item()
            accuracy = correct/len(dataloader.dataset)
            #TO end
        evaluation = [loss, R2, accuracy, discrete_measure/count]
        if not full_output:
            example_output = np.array(output[0:16000].cpu())
            example_labels = np.array(labels[0:16000].cpu())
            return evaluation, example_output, example_labels
        else:
            return evaluation, output, labels
