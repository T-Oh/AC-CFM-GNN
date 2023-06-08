import torch
import numpy as np

from torch import no_grad, cat
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import R2Score


from utils.utils import discrete_loss
from datasets.dataset import mask_probs_add_bias


class Engine(object):
    """
    Contains the necessary code for training and evaluating
    a single epoch
    """

    def __init__(self, model, optimizer, device, criterion, tol=0.1, task="NodeReg", var=None, masking=False, mask_bias=0.0, return_full_output = False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.tol = tol
        self.task = task
        self.vars = var.clone()
        print(f'Mask Probs Before adding Bias:\n{var[0:50]}')
        self.mask_probs = mask_probs_add_bias(var, mask_bias)
        print(f'Mask Probs after adding Bias:\n{self.mask_probs[0:50]}')
        self.masks = torch.bernoulli(self.mask_probs)
        self.masking = bool(int(masking))
        self.criterion = criterion
        self.return_full_output = return_full_output
        
        self.scaler = GradScaler()  #necessary for mixed precision learning

        


    def train_epoch(self, dataloader, gradclip):
        
        print('\nENGINE OUTPUT')
            
        loss = 0.0
        self.model.train()  #sets the mode to training (layers can behave differently in training than testing)
        R2score=R2Score()
        first=True;

        count = 0
        self.optimizer.zero_grad()
        for (i, batch) in enumerate(dataloader):
            self.optimizer.zero_grad()
            count +=1
            batch.to(self.device)

            #with autocast(dtype=torch.float16):
            output = self.model.forward(batch).reshape(-1)  #reshape used to make sure that output is 1 dimensional
            output.to(self.device)
        
            if self.task == "GraphReg": #set labels according to task (GraphReg or NodeReg)
                labels = batch.y
            elif self.task == "NodeReg":
                labels = batch.node_labels.type(torch.FloatTensor).to(self.device)

 
            #calc and backpropagate loss
            if self.masking:
                print('Applying Masking')
                for j in range(int(len(output)/2000)):
                   if j==0:
                       self.masks=torch.bernoulli(self.mask_probs)
                       print(torch.bincount(self.masks.to(int))[1]/2000)
                   else:
                       self.masks=torch.cat((self.masks,torch.bernoulli(self.mask_probs).to('cuda:0')))
                   self.masks= self.masks.to('cuda:0')
                   #self.masks.to(self.device)
                print('Masks:')
                print(self.masks)
                print(f'Output before masking:\n{output}')
                output = output*self.masks
                labels = labels*self.masks
                print(f'Output after masking:\n{output}')
            

            temp_loss = self.criterion(output.to(self.device), labels.to(self.device))#.float()
            """self.scaler.scale(temp_loss).backward() 
            self.scaler.step(self.optimizer)
            self.scaler.update()"""
 
            #compile outputs and labels for saving                        
            if first:
                if self.task == "NodeReg":
                    total_output=output.detach().cpu()
                    total_labels=labels.detach().cpu()
                else:
                    total_output=output.detach().cpu()
                    total_labels=labels.detach().cpu()
                first=False
            else:
                if self.task == "NodeReg":
                    total_output=cat((total_output,output.detach().cpu()),0)   
                    total_labels=cat((total_labels,labels.detach().cpu()),0)
                else:
                    total_output=cat((total_output,output.detach().cpu()),0)  
                    total_labels=cat((total_labels,labels.detach().cpu()),0)

                      
            temp_loss.backward()
            #print(temp_loss.grad)
            if gradclip >= 0.02:
                print(f'Using Gradclip with treshold: {gradclip}')
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradclip)
            self.optimizer.step()
            #Gradient accumulation
            """
            if (i+1)%8 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            """
            loss += temp_loss.item()
        R2 = R2score(total_output.reshape(-1), total_labels.reshape(-1))
        
        if not self.return_full_output:
            example_output = total_output[0:16000]
            example_labels = total_labels[0:16000]
            del total_output
            del total_labels
            return loss/count, R2, example_output, example_labels
        


        #End TO
        return loss/count, R2, total_output, total_labels

    def eval(self, dataloader, full_output=False):
        """
        

        Parameters
        ----------
        dataloader : Dataloader
        full_output : boolean, optional
            if True saves output of all instances. The default is False.

        Returns
        -------
        evaluation :[loss, R2, accuracy, discrete_measure/count]
        output : list of outputs
        labels : list of labels

        """
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
                    temp_output = self.model.forward(batch).reshape(-1)#.to(self.device)
                if first:
                    labels=temp_labels.detach().cpu()
                    output= temp_output.detach().cpu()
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
                    output=cat((output,temp_output.detach().cpu()),0)  
                    labels=cat((labels,temp_labels.detach().cpu()),0)

            #TO
            R2torch=R2Score()


            R2=R2torch(output.reshape(-1), labels.reshape(-1))
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
