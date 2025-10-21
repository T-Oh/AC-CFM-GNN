import torch
import numpy as np
import time

from torch import no_grad, cat
from torch.cuda.amp import GradScaler, autocast
from torchmetrics import R2Score
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall, MulticlassAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryAccuracy


from datasets.dataset import mask_probs_add_bias

from utils.utils import grad_norm, state_loss, physics_loss

def safe_grad_norm(loss, model, retain_graph=False):
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        retain_graph=retain_graph,
        allow_unused=True
    )
    grads = [g for g in grads if g is not None]  # filter out unused params
    if len(grads) == 0:
        return torch.tensor(0.0, device=loss.device)
    return torch.norm(torch.stack([torch.norm(g) for g in grads]))




class Engine(object):
    """
    Contains the necessary code for training and evaluating
    a single epoch
    """

    def __init__(self, model, optimizer, device, criterion, tol=0.1, task="NodeReg", var=None, masking=False, mask_bias=0.0, return_full_output = False, 
                 physics_loss_func=None, track_gradients=False, track_test_gradients=False):
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
        self.VERBOSE = False
        self.model = model.to(device).to(torch.float32)
        self.optimizer = optimizer
        self.device = device
        self.tol = tol
        self.task = task
        self.vars = var.clone()
        self.mask_probs = mask_probs_add_bias(var, mask_bias)
        self.masks = torch.bernoulli(self.mask_probs)
        self.masking = bool(int(masking))
        self.criterion = criterion.to(self.device)
        self.return_full_output = return_full_output
        self.scaler = GradScaler()  #necessary for mixed precision learning
        self.R2Score = R2Score()
        #self.track_loss = state_loss(0.2).to(self.device)  #TO BE REMOVED. ONLY USED FOR TRACKING PI LOSS WHILE TRAINING WITH REGULAR STATE LOSS
        self.physics_loss_func = physics_loss_func  # if passed will try to compute pl and add it to metrics
        self.track_gradients = track_gradients
        self.track_test_gradients = track_test_gradients


        if 'Class' in self.task or self.task == 'typeII':
            self.f1_metric = MulticlassF1Score(num_classes=4, average=None)
            self.precision_metric = MulticlassPrecision(num_classes=4, average=None)
            self.recall_metric = MulticlassRecall(num_classes=4, average=None)
            self.accuracy_metric = MulticlassAccuracy(num_classes=4, average='micro')
        elif self.task in ['StateReg', 'StateRegPI']:
            self.f1_metric = BinaryF1Score()
            self.precision_metric = BinaryPrecision()
            self.recall_metric = BinaryRecall()
            self.accuracy_metric = BinaryAccuracy()

        


    def train_epoch(self, dataloader, gradclip, full_output):
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
        node_loss = 0.
        edge_loss = 0.
        PI_loss = 0.
        self.R2Score.reset()
        self.f1_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.accuracy_metric.reset()
        temp_node_loss = 0.
        temp_edge_loss = 0.
        temp_PI_loss = 0.
        first=True
        count = 0
        total_output = None #shape: tuple ([2,2000*N_instances], [7064*N_instances, 2]) where the first element is the node output and the second element is the edge output
        total_labels = None #shape: tuple ([2,2000*N_instances], [1, 7064*N_instances]) where the first element is the node labels and the second element is the edge labels

        data_loading_time = 0.
        gpu_processing_time = 0.
        g_node_list, g_edge_list, g_pi_list, g_total_list = [], [], [], [] #for gradient tracking

        self.model.train()  #sets the mode to training (layers can behave differently in training than testing)
        torch.set_grad_enabled(True)
        if self.VERBOSE:
            print('Pre empty cache')
            self.log_gpu_usage()
            torch.cuda.empty_cache()
            print('Post empty cache')
            self.log_gpu_usage()
        for param in self.model.parameters():
            assert param.requires_grad, "Parameter does not require grad"

        for (i, batch) in enumerate(dataloader):
            data_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            count +=1           
            #compile batch depending on task - ifinstance implies that the data is LDTSF
            batch = self.compile_batch(batch)


            data_loading_time += data_start-time.time()

            if self.VERBOSE:
                print(f"Total Loading Time so far: {data_loading_time:.2f} s", flush=True)
                gpu_start = time.time()
                print('Pre forward pass: ')
                self.log_gpu_usage()
                
            #with autocast():
            output = self.model.forward(batch)
            output, labels = shape_and_cast_labels_and_output(output, batch, self.task, self.device)    #, edge_labels
            if self.VERBOSE: 
                if any(torch.isnan(output[0].reshape(-1))) or any(torch.isinf(output[0].reshape(-1))):
                    print("Node Output contains NaN or Inf values. Skipping this batch.")
                if any(torch.isnan(output[1].reshape(-1))) or any(torch.isinf(output[1].reshape(-1))):
                    print("Edge Output contains NaN or Inf values. Skipping this batch.")
                if any(torch.isnan(labels[0].reshape(-1))) or any(torch.isinf(labels[0].reshape(-1))):
                    print("Node Labels contain NaN or Inf values. Skipping this batch.")
                if any(torch.isnan(labels[1].reshape(-1))) or any(torch.isinf(labels[1].reshape(-1))):
                        print("Edge Labels contain NaN or Inf values. Skipping this batch.")
            #calc and backpropagate loss
            loss_start = time.time()
            if self.masking:    output, labels = self.apply_masking(output, labels)
            if self.task == 'typeIIClass':  temp_loss = self.criterion(output.to(self.device), labels.to(self.device)).float()
            elif self.task in ['StateReg']:   temp_loss, temp_node_loss, temp_edge_loss = self.criterion(output[0], output[1], labels[0], labels[1])
            elif self.task == 'StateRegPI': temp_loss, temp_node_loss, temp_edge_loss, temp_PI_loss = self.criterion(output[0], output[1], labels[0], labels[1])
            elif isinstance(self.criterion, physics_loss): temp_loss, temp_D1, temp_D2, temp_D3 = self.criterion(batch, output)
            else:                           temp_loss = self.criterion(output.reshape(-1).to(self.device), labels.reshape(-1).to(self.device)).float()
            print('Time to calculate loss of one batch: ', time.time()-loss_start)
            #compile outputs and labels for saving                        
            total_output, total_labels = self.compile_labels_output_for_saving(first, output, labels, total_output, total_labels)  
            if self.VERBOSE:
                print('Pre backward pass: ')
                self.log_gpu_usage()
            
            backward_pass_start = time.time()
            #print('HARDCODED USING STATE LOSS (NOT STATE LOSS PI) TO TRACK PI LOSS WHILE TRAINING WITH REGULAR STATE LOSS')
            #temp_loss, temp_node_loss, temp_edge_loss = self.track_loss(output[0], output[1], labels[0], labels[1])


            # Track gradients for inspection, not training
            if self.track_gradients:
                g_node_loss = safe_grad_norm(temp_node_loss, self.model, retain_graph=True)
                g_edge_loss = safe_grad_norm(temp_edge_loss, self.model, retain_graph=True)
                #g_pi_loss   = safe_grad_norm(temp_PI_loss, self.model, retain_graph=True)
                print(f"grad norms: mse={g_node_loss:.3e}, ce={g_edge_loss:.3e}")#, pi={g_pi_loss:.3e}")

            #self.scaler.scale(loss).backward()
            #self.scaler.unscale_(self.optimizer)

                g_total = grad_norm(self.model)

                g_node_list.append(g_node_loss.detach().cpu())
                g_edge_list.append(g_edge_loss.detach().cpu())
                #g_pi_list.append(g_pi_loss.detach().cpu())
                g_total_list.append(torch.tensor(g_total)) 

            
            temp_loss.backward()

            print('Time for backward pass of one batch: ', time.time()-backward_pass_start)
            if gradclip >= 0.02:    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradclip)   #Gradient Clipping
            self.optimizer.step()
            #self.scaler.step(self.optimizer)   #used for mixed precision training
            #self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

           
            if self.VERBOSE: 
                 gpu_processing_time += time.time()-gpu_start
                 print(f'Total GPU processing time so far {gpu_processing_time:.2f}s', flush=True)
            #Gradient accumulation
            """
            if (i+1)%8 == 0:
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            """
            loss += temp_loss.item()
            if self.task in ['StateReg', 'StateRegPI']:
                node_loss += temp_node_loss.item()
                edge_loss += temp_edge_loss.item()
                if self.task == 'StateRegPI':
                    PI_loss += temp_PI_loss.item()

            if 'Class' in self.task or self.task in ['StateReg', 'StateRegPI']:
                if self.task == 'typeIIClass':  
                    labels = labels.reshape(-1).detach().cpu()
                    preds = torch.argmax(output, dim=1)
                elif self.task in ['StateReg','StateRegPI']:
                    preds = 1-torch.argmax(output[1], dim=1).detach().cpu()
                    labels = 1-labels[1].reshape(-1).detach().cpu()
                else:
                    preds = torch.argmax(output, dim=0)
                self.f1_metric.update(preds.reshape(-1), labels)
                self.precision_metric.update(preds.reshape(-1), labels.reshape(-1))
                self.recall_metric.update(preds.reshape(-1), labels.reshape(-1)) 
                self.accuracy_metric.update(preds, labels)
            first = False

        #save gradients
        if self.track_gradients:
            gradients = {
                    'g_node_loss': torch.stack(g_node_list).mean(),
                    'g_edge_loss': torch.stack(g_edge_list).mean(),
                    #'g_pi': torch.stack(g_pi_list).mean(),
                    'g_total': torch.stack(g_total_list).mean()
                    }
        else: gradients = None
        

        metrics = self.compute_metrics(total_output, total_labels, loss, node_loss, edge_loss, PI_loss, count)
        if self.VERBOSE: 
            print(f'Total Loading time during epoch: {data_loading_time:.2f} s', flush=True)
            print(f'Total processing time during epoch: {gpu_processing_time:.2f} s', flush=True)
        del batch, output, labels  
        t2 = time.time()
        if full_output:
            return metrics, total_output, total_labels, gradients
        else:
            return metrics, (total_output[0][:20000], total_output[1][:7064*10]), (total_labels[0][:2000], total_labels[1][0,7064*10]), gradients  #return only the first 2000 instances for saving memory



    def eval(self, dataloader, full_output=True):
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
        if self.VERBOSE:
            print('Eval: Pre empty cache: ')
            self.log_gpu_usage()
            torch.cuda.empty_cache()
            print('Eval: Post empty cache: ')
            self.log_gpu_usage()
        start = time.time()
        if self.track_test_gradients:
            self.model.train()
            grad_context = torch.enable_grad()
            torch.set_grad_enabled(True)
        else:
            self.model.eval()
            grad_context = torch.no_grad()
            torch.set_grad_enabled(False)
        g_node_list, g_edge_list, g_pi_list, g_total_list = [], [], [], []

        
        with grad_context:
            #with autocast():
            loss, node_loss, edge_loss, PI_loss = 0., 0., 0., 0.
            temp_node_loss, temp_edge_loss, temp_PI_loss = 0., 0., 0.
            self.R2Score.reset()
            self.f1_metric.reset()
            self.precision_metric.reset()
            self.recall_metric.reset()
            self.accuracy_metric.reset()

            first = True
            count = 0
            data_loading_time = 0.
            gpu_processing_time = 0.
            total_output = None
            total_labels = None

            for batch in dataloader:
                count += 1
                #compile batch depending on task - ifinstance implies that the data is LDTSF
                data_start = time.time()
                batch = self.compile_batch(batch)
                data_loading_time += data_start-time.time()
                if self.VERBOSE: 
                    print(f'Total Loading Time so far (EVAL): {data_loading_time:.2f} s', flush=True)
                    print('Eval: Pre forward pass: ')
                    self.log_gpu_usage()
                gpu_start = time.time()
                
                output = self.model.forward(batch)
                gpu_processing_time += time.time()-gpu_start
                if self.VERBOSE: print(f'Total GPU processing time so far (EVAL) {gpu_processing_time:.2f}s', flush=True)
                output, labels = shape_and_cast_labels_and_output(output, batch, self.task, self.device) #, edge_labels
                print(f'Output: {output}')
                print(f'Labels: {labels}')
                
                #compile labels and output fo saving
                total_output, total_labels = self.compile_labels_output_for_saving(first, output, labels, total_output, total_labels)         
                first = False
                if 'Class' in self.task or self.task in ['StateReg', 'StateRegPI']:
                    if self.task == 'typeIIClass': 
                        temp_loss = self.criterion(output, labels).tolist() 
                        labels = labels.reshape(-1).detach().cpu()
                        preds = torch.argmax(output, dim=1).detach().cpu()
                    elif self.task in ['StateReg', 'StateRegPI']:
                        if self.task == 'StateReg':
                            temp_loss, temp_node_loss, temp_edge_loss = self.criterion(output[0], output[1], labels[0], labels[1])
                        else:
                            temp_loss, temp_node_loss, temp_edge_loss, temp_PI_loss = self.criterion(output[0], output[1], labels[0], labels[1])
                        if self.VERBOSE:
                            if torch.isnan(temp_loss) or torch.isinf(temp_loss):
                                print("Loss contains NaN or Inf values. Skipping this batch.")
                            if torch.isnan(temp_node_loss) or torch.isinf(temp_node_loss):
                                print("Node Loss contains NaN or Inf values. Skipping this batch.")
                            if torch.isnan(temp_edge_loss) or torch.isinf(temp_edge_loss):
                                print("Edge Loss contains NaN or Inf values. Skipping this batch.")
                        if self.task == 'StateRegPI':
                            if torch.isnan(temp_PI_loss) or torch.isinf(temp_PI_loss):
                                print("Power Injection Loss contains NaN or Inf values. Skipping this batch.")
                        preds = 1-torch.argmax(output[1], dim=1).detach().cpu()
                        labels = 1-labels[1].reshape(-1).detach().cpu()
                    else:
                        temp_loss = self.criterion(output.reshape(-1), labels.reshape(-1)).tolist()
                        preds = torch.argmax(output, dim=1).detach().cpu()

                    self.f1_metric.update(preds, labels.reshape(-1))
                    self.precision_metric.update(preds, labels.reshape(-1))
                    self.recall_metric.update(preds, labels.reshape(-1)) 
                    self.accuracy_metric.update(preds, labels.reshape(-1))


                elif isinstance(self.criterion, physics_loss):
                    temp_loss, temp_D1, temp_D2, temp_D3 = self.criterion(batch, output)
                else:
                    temp_loss = self.criterion(output.reshape(-1).to(self.device), labels.reshape(-1).to(self.device)).tolist()

                #print('HARDCODED USING STATE LOSS (NOT STATE LOSS PI) TO TRACK PI LOSS WHILE TRAINING WITH REGULAR STATE LOSS')
                #temp_loss, temp_node_loss, temp_edge_loss = self.track_loss(output[0], output[1], labels[0], labels[1])
                if self.track_test_gradients:
                    g_node_loss = safe_grad_norm(temp_node_loss, self.model, retain_graph=True)
                    g_edge_loss = safe_grad_norm(temp_edge_loss, self.model, retain_graph=True)
                    #g_pi_loss   = safe_grad_norm(temp_PI_loss, self.model, retain_graph=True)

                    print(f"grad norms: mse={g_node_loss:.3e}, ce={g_edge_loss:.3e}")#, pi={g_pi_loss:.3e}")
                    g_total = grad_norm(self.model)


                    g_node_list.append(g_node_loss.detach().cpu())
                    g_edge_list.append(g_edge_loss.detach().cpu())
                    #g_pi_list.append(g_pi_loss.detach().cpu())
                    g_total_list.append(torch.tensor(g_total)) 
                    

                loss += float(temp_loss)
                node_loss += temp_node_loss
                edge_loss += temp_edge_loss
                PI_loss += temp_PI_loss if self.task == 'StateRegPI' else 0

            metrics = self.compute_metrics(total_output, total_labels, loss, node_loss, edge_loss, PI_loss, count)
            if self.track_test_gradients:
                gradients = {
                    'g_node_loss': torch.stack(g_node_list).mean(),
                    'g_edge_loss': torch.stack(g_edge_list).mean(),
                    #'g_pi': torch.stack(g_pi_list).mean(),
                    'g_total': torch.stack(g_total_list).mean()
                    }
            else: gradients = None
            if self.VERBOSE:
                print(f"Total Loading time during EVAL: {data_loading_time:.2f} s", flush=True)
                print(f"Total processing time during EVAL: {gpu_processing_time:.2f} s", flush=True)
                print(f"Total time for EVAL: {time.time()-start:.2f} s", flush=True)
            if full_output:
                return metrics, total_output, total_labels, gradients
            else:
                return metrics, (total_output[0][:20000], total_output[1][:7064*10]), (total_labels[0][:2000], total_labels[1][0,7064*10]), gradients  #return only the first 2000 instances for saving memory
                


    def log_gpu_usage(self):
        allocated = torch.cuda.memory_allocated() / 1024**2  # Convert to MB
        reserved = torch.cuda.memory_reserved() / 1024**2  # Convert to MB
        print(f"GPU Memory Allocated: {allocated:.2f} MB")
        print(f"GPU Memory Reserved: {reserved:.2f} MB")



    def compile_batch(self, batch):
        if isinstance(batch, tuple) and not self.model.__class__.__name__ == 'GAT_LSTM':
            if self.task == 'GraphReg':     batch = (batch[0].to(self.device), batch[1].to(self.device), batch[2])  #batch[1] contains the regression label
            elif self.task == 'GraphClass': batch = (batch[0].to(self.device), batch[3].to(self.device), batch[2])  #batch[3] contains the classification label                   
            elif self.task == 'typeIIClass':     batch = (batch[0].to(self.device), batch[3].to(self.device), batch[2])  #batch[1] contains the regression label
        elif not self.model.__class__.__name__ == 'GAT_LSTM':   
            batch = batch.to(self.device)
        return batch


    def compile_labels_output_for_saving(self, first, output, labels, total_output, total_labels):
        #first is flagged for the first batch the 2nd batch and onwards will be concatenated to the first batch
        if first:                   
            if self.task in ['StateReg', 'StateRegPI']:
                total_output=(output[0].detach().cpu(), output[1].detach().cpu())
                total_labels=(labels[0].detach().cpu(), labels[1].detach().cpu())
            else:
                total_output=output.detach().cpu()
                total_labels=labels.detach().cpu()
        else:
            if self.task in ['StateReg', 'StateRegPI']:
                total_output = (cat((total_output[0],output[0].detach().cpu()),0), cat((total_output[1],output[1].detach().cpu()),0))
                total_labels = (cat((total_labels[0],labels[0].detach().cpu()),0), cat((total_labels[1],labels[1].detach().cpu()),0))
            else:
                total_output=cat((total_output,output.detach().cpu()),0)  
                total_labels=cat((total_labels,labels.detach().cpu()),0)
        return total_output, total_labels

    def apply_masking(self, output, labels):
        for j in range(int(len(output)/2000)):
            if j==0:    self.masks=torch.bernoulli(self.mask_probs)
            else:       self.masks=torch.cat((self.masks,torch.bernoulli(self.mask_probs).to('cuda:0')))
            self.masks= self.masks.to('cuda:0')
        output = output*self.masks
        labels = labels*self.masks
        return output,labels



    def compute_metrics(self, total_output, total_labels, loss, node_loss, edge_loss, PI_loss, count):
        if not 'Class' in self.task:  
            #GTSF with edge_labels
            if isinstance(total_output, tuple): 
                R2_1 = self.R2Score(total_output[0][:,0], total_labels[0][:,0])
                R2_2 = self.R2Score(total_output[0][:,1], total_labels[0][:,1])
                F1 = self.f1_metric.compute()
                precision = self.precision_metric.compute()
                recall = self.recall_metric.compute()
                accuracy = self.accuracy_metric.compute()
                metrics = {
                    'loss'  : float(loss/count),
                    'R2'    : float(R2_1),
                    'R2_2'  : float(R2_2),
                    'F1'    : F1.tolist(),
                    'precision' : precision.tolist(),
                    'recall'    : recall.tolist(),
                    'accuracy'  : accuracy.tolist(),
                    'node_loss' : float(node_loss/count),
                    'edge_loss' : float(edge_loss/count)
                }
                if self.task == 'StateRegPI':
                    metrics['PI_loss'] = float(PI_loss/count)
            elif total_output.dim() == 3:
                total_output = total_output.reshape(-1, 2)
                total_labels = total_labels.reshape(-1, 2)
                R2_1 = self.R2Score(total_output[:,0], total_labels[:,0])
                R2_2 = self.R2Score(total_output[:,1], total_labels[:,1])
                metrics = {
                    'loss'  : float(loss/count),
                    'R2'    : float(R2_1),
                    'R2_2'  : float(R2_2)
                }
            elif total_output.dim()==2:
                if self.model.__class__.__name__ =='LSTM_LDTSF':
                    R2 = self.R2Score(total_output.reshape(-1), total_labels.reshape(-1))
                    metrics = {
                        'loss'  : float(loss/count),
                        'R2'    : float(R2)
                    }
                else:
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

            if self.physics_loss_func is not None:
                temp_D_loss, temp_D1, temp_D2, temp_D3 = self.criterion(batch, output.to(
                        self.device))  # , labels.to(self.device))#.float()

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
    edge_labels = None  #edge_labels only used in GTSF StateReg
    #Graph Regressions
    if task == "GraphReg": #set labels according to task (GraphReg or NodeReg)
        #Time Series Forecasting
        if isinstance(batch, tuple):    #tuple for TSF  
            #Graph Time Series Forecasting (GTSF)
            if isinstance(batch[1][0], int):    #for GATLSTM the tuple contains the sequence lenghts at batch[1] which is int
                labels = [batch[0][i].y_cummulative[-1] for i in range(len(batch[0]))]
                labels = torch.stack(labels).to(torch.double)
                output = output.to(torch.double).reshape(-1)
            #Line Damge Time Series Forecasting (LDTSF)
            else:                            
                labels = batch[1].to(torch.double)#.reshape(-1)    
        else:   #Regular GraphReg (no TSF)                                                  
            labels = batch.y.type(torch.FloatTensor).to(device)
            output = output.reshape(-1)

    elif task in ['GraphClass', 'typeIIClass']:
            labels = batch[1].to(torch.long).reshape(-1) 
            output = output.transpose(0,1)

    elif task in ["NodeReg", "StateReg", "StateRegPI"]:
        #Time Series Forecasting
        if isinstance(batch, tuple):    #tuple for TSF  
            #Graph Time Series Forecasting (GTSF)
            if isinstance(batch[1][0], int):    #for GATLSTM the tuple contains the sequence lenghts at batch[1] which is int
                labels = [batch[0][i].node_labels[-2000:,:] for i in range(len(batch[0]))]
                labels = torch.stack(labels)
                if task in ['StateReg', 'StateRegPI']:
                    edge_labels = [batch[0][i].edge_labels[-7064:] for i in range(len(batch[0]))]
                    edge_labels = torch.stack(edge_labels).to(torch.long)
                    labels = (labels.reshape(-1,2).to(device), edge_labels.to(device))
                    output = (output[0].reshape(-1,2).to(device), output[1].permute(0, 2, 1).reshape(-1, 2).to(device))  
            else:   assert False, 'Using wrong data with GTSF'
                
        else:
            labels = batch.node_labels.type(torch.FloatTensor).to(device)
    return output, labels 


