import torch
import numpy as np
import time

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
        """
        Initializes the Engine

        Parameters
        ----------
        model : torch.nn.module
            the configured model
        optimizer : torch.optim.Optimizer
            the configured optimizer
        device : torch.device
            the device on which computation takes place (on cluster usually gpu)
        criterion : torch.criterion
            the criterion used for training and evaluation
        tol : float, optional
            tolerance for accuracy calculation. The default is 0.1.
        task : string, optional
            the task ('NodeReg' or 'GraphReg') in this version only 'NodeReg' is implemented. The default is "NodeReg".
        var : float array (1D), optional
            the variances of the nodelabels. Used for loss masking or weighting. The default is None.
        masking : bool, optional
            wether masking should be applied or not. The default is False.
        mask_bias : float, optional
            float from the interval [0,1]. The bias to be added to the masking probabilities. The default is 0.0.
        return_full_output : bool, optional
            Wether the full output or only an example output of a few instances should be saved. The default is False.

        Returns
        -------
        None.

        """
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
        """
        Executes the training of a single epoch

        Parameters
        ----------
        dataloader : torch.dataloader
            the loader of the dataset to be sued (testloader or trainloader)
        gradclip : float
            value for gradient clipping. If gradclip<0.02 no gradient clipping is applied.

        Returns
        -------
        loss/count : float
            the mean squared error
        R2 : float
            R2 score
        total_output : torch.tensor (1D)
            the complete output of the model
        total_labelstorch.tensor (1D)
            the complete output of the model

        """
        

        
        loss = 0.0
        self.model.train()  #sets the mode to training (layers can behave differently in training than testing)
        R2score=R2Score()
        first=True;

        count = 0
        for (i, batch) in enumerate(dataloader):
            """t1 = time.time()
            print('Training Batch')
            print('STart Ttime Batch:', t1, flush=True)"""
            #self.optimizer.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()
            count +=1
            batch.to(self.device)

            with autocast():
                output = self.model.forward(batch).reshape(-1)  #reshape used to make sure that output is 1 dimensional
                output.to(self.device)
            
                if self.task == "GraphReg": #set labels according to task (GraphReg or NodeReg)
                    labels = batch.y.type(torch.FloatTensor).to(self.device)
                elif self.task == "NodeReg":
                    labels = batch.node_labels.type(torch.FloatTensor).to(self.device)

 
                #calc and backpropagate loss
                if self.masking:

                    for j in range(int(len(output)/2000)):
                        if j==0:
                            self.masks=torch.bernoulli(self.mask_probs)

                        else:
                            self.masks=torch.cat((self.masks,torch.bernoulli(self.mask_probs).to('cuda:0')))
                        self.masks= self.masks.to('cuda:0')
                        #self.masks.to(self.device)



                    output = output*self.masks
                    labels = labels*self.masks

                

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

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradclip)
            self.optimizer.step()
            #Gradient accumulation
            """
            if (i+1)%8 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            """
            loss += temp_loss.item()
            #t2=time.time()
            #print(f'Training Batch took {(t1-t2)/60} mins', flush=True)
        R2 = R2score(total_output.reshape(-1), total_labels.reshape(-1))
        
        if not self.return_full_output:
            example_output = total_output[0:16000]
            example_labels = total_labels[0:16000]
            del total_output
            del total_labels
            return loss/count, R2, example_output, example_labels
        
        t2 = time.time()

        


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
            with autocast():
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

                #discrete_measure = discrete_loss(output.clone(), labels.clone())



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
