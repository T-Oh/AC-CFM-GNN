import torch
import numpy as np
import time

from torch import no_grad, cat
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import R2Score
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy

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
        self.model = model.to(device).to(torch.float32)
        self.optimizer = optimizer
        self.device = device
        self.tol = tol
        self.task = task
        self.vars = var.clone()
        self.mask_probs = mask_probs_add_bias(var, mask_bias)
        self.masks = torch.bernoulli(self.mask_probs)
        self.masking = bool(int(masking))
        self.criterion = criterion
        self.return_full_output = return_full_output
        self.scaler = GradScaler()  #necessary for mixed precision learning
        self.R2Score = R2Score()
        if 'Class' in self.task:
            self.f1_metric = MulticlassF1Score(num_classes=4, average=None).to(device)
            self.precision_metric = MulticlassPrecision(num_classes=4, average=None).to(device)
            self.recall_metric = MulticlassRecall(num_classes=4, average=None).to(device)
            self.accuracy_metric = MulticlassAccuracy(num_classes=4, average='micro').to(device)

        


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
        

        first=True
        for param in self.model.parameters():
            assert param.requires_grad, "Parameter does not require grad"

        count = 0
        for (i, batch) in enumerate(dataloader):
            """t1 = time.time()
            print('Training Batch')
            print('STart Ttime Batch:', t1, flush=True)"""

            self.optimizer.zero_grad(set_to_none=True)
            count +=1
            
            #compile batch depending on task - ifinstance implies that the data is LDTSF
            if isinstance(batch, tuple) and not self.model.__class__.__name__ == 'GAT_LSTM':
                if self.task == 'GraphReg':     batch = (batch[0].to(self.device), batch[1].to(self.device), batch[2])  #batch[1] contains the regression label
                elif self.task == 'GraphClass': batch = (batch[0].to(self.device), batch[3].to(self.device), batch[2])  #batch[3] contains the classification label                   
            elif not self.model.__class__.__name__ == 'GAT_LSTM':   
                batch.to(self.device)

            with autocast():
                output = self.model.forward(batch)#.reshape(-1)  #reshape used to make sure that output is 1 dimensional
                output.to(self.device)

                output, labels = shape_and_cast_labels_and_output(output, batch, self.task, self.device)
 
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

                temp_loss = self.criterion(output.reshape(-1).to(self.device), labels.reshape(-1).to(self.device)).float()
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

            if 'Class' in self.task:

                preds = torch.argmax(output, dim=1)
                self.f1_metric.update(preds.reshape(-1), labels)
                self.precision_metric.update(preds.reshape(-1), labels.reshape(-1))
                self.recall_metric.update(preds.reshape(-1), labels.reshape(-1)) 
                self.accuracy_metric.update(preds, labels)


        
        metrics = self.compute_metrics(total_output, total_labels, loss, count)

        del batch

        if not self.return_full_output:
            example_output = total_output[0:16000]
            example_labels = total_labels[0:16000]
            del total_output
            del total_labels
            return metrics, example_output, example_labels
        
        t2 = time.time()



        return metrics, total_output, total_labels


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
                
                first = True

                count = 0
                for batch in dataloader:
                    count += 1
                    #compile batch depending on task - ifinstance implies that the data is LDTSF
                    if isinstance(batch, tuple) and not self.model.__class__.__name__ == 'GAT_LSTM':
                        if self.task == 'GraphReg':     batch = (batch[0].to(self.device), batch[1].to(self.device), batch[2])  #batch[1] contains the regression label
                        elif self.task == 'GraphClass': batch = (batch[0].to(self.device), batch[3].to(self.device), batch[2])  #batch[3] contains the classification label                   
                    elif not self.model.__class__.__name__ == 'GAT_LSTM':   
                        batch.to(self.device)
                    temp_output = self.model.forward(batch).to(self.device)#.reshape(-1)#


                    temp_output, temp_labels = shape_and_cast_labels_and_output(temp_output, batch, self.task, self.device)                    

                    if first:
                        labels = temp_labels.detach().cpu()
                        output = temp_output.detach().cpu()
                        first = False
                    else:
                        output=cat((output,temp_output.detach().cpu()),0)  
                        labels=cat((labels,temp_labels.detach().cpu()),0)
                        
                    if 'Class' in self.task:
                        preds = torch.argmax(temp_output, dim=1)
                        self.f1_metric.update(preds, temp_labels.reshape(-1))
                        self.precision_metric.update(preds, temp_labels.reshape(-1))
                        self.recall_metric.update(preds, temp_labels.reshape(-1)) 
                        self.accuracy_metric.update(preds, temp_labels.reshape(-1))
                    temp_loss = self.criterion(temp_output.reshape(-1), temp_labels.reshape(-1)).tolist()
                loss += temp_loss

            metrics = self.compute_metrics(output, labels, loss, count)

            if not full_output:
                example_output = np.array(output[0:16000].cpu())
                example_labels = np.array(labels[0:16000].cpu())
                return metrics, example_output, example_labels
            else:
                return metrics, output, labels



    def compute_metrics(self, total_output, total_labels, loss, count):
        if not 'Class' in self.task:  
            if total_output.dim()==2:
                R2_1 = self.R2Score(total_output[:,0], total_labels[:,0])
                R2_2 = self.R2Score(total_output[:,1], total_labels[:,1])  
                metrics = {
                    'loss'  : float(loss/count),
                    'R2'    : float(R2_1),
                    'R2_2'  : float(R2_2)
                }
            else:
                R2 = self.R2Score(total_output.reshape(-1), total_labels.reshape(-1))
                metrics = {
                    'loss'  : float(loss/count),
                    'R2'    : float(R2)
                }
        else:
            F1 = self.f1_metric.compute()
            precision = self.precision_metric.compute()
            recall = self.recall_metric.compute()
            accuracy = self.accuracy_metric.compute()
            metrics = {
                'loss'  : float(loss/count),
                'F1'    : F1.tolist(),
                'precision' : precision.tolist(),
                'recall'    : recall.tolist(),
                'accuracy'  : accuracy.tolist()
            }
        return metrics
            


def shape_and_cast_labels_and_output(output, batch, task, device):
    if task == "GraphReg": #set labels according to task (GraphReg or NodeReg)
        if isinstance(batch, tuple):    #tuple for TSF  
            if isinstance(batch[1][0], int):    #for GATLSTM the tuple contains the sequence lenghts at batch[1] which is int
                labels = [batch[0][i].y.sum() for i in range(len(batch[0]))]
                labels = torch.stack(labels).to(torch.double)
            else:                               #otherwise its LDTSF
                labels = batch[1].to(torch.double)#.reshape(-1)
            output = output.to(torch.double).reshape(-1)
        else:                                                  
            labels = batch.y.type(torch.FloatTensor).to(device)
            output = output.reshape(-1)
    elif task == 'GraphClass':
            labels = batch[1]
    elif task == "NodeReg":

        labels = batch.node_labels.type(torch.FloatTensor).to(device)
        #if output.dim() > 1:   output = output.reshape(-1)
    print('Output:', output)
    print('Labels:', labels)

    return output.to(device), labels.to(device)


