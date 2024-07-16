import torch

import logging
import time
from training.engine import Engine
from models.get_models import get_model
from utils.get_optimizers import get_optimizer

#from ray import tune
from ray.air import session

from utils.utils import weighted_loss_label, setup_params, setup_params_from_search_space
from models.run_node2vec import run_node2vec
from datasets.dataset import create_datasets, create_loaders, get_attribute_sizes
from torch.optim.lr_scheduler import ReduceLROnPlateau

def run_training(trainloader, testloader, engine, cfg, LRScheduler):
    """
    

    Parameters
    ----------
    trainloader : Dataloader
    testloader : Dataloader
    engine : Engine
    cfg : search_space file

    Returns
    -------
    metrics : dict {train_loss, train_R2, test_loss, test_R2} each metric is a list of the metric at every epoch
    final_eval : [testloss, testR2, testacc, testDM]
    output : list of outputs through time (entries saved with output freq)
    labels : list of labels through time

    """

    train_loss = []
    train_R2 = []
    test_loss = []
    test_R2 = []
    output=[]
    labels=[]


    for i in range(1, cfg['epochs'] + 1):

        print(f'Epoch: {i}', flush=True)
        t1 = time.time()

        temp_train_loss, R2, temp_output, temp_labels = engine.train_epoch(trainloader, cfg['gradclip'])

        if cfg['train_size'] == 1:
            temp_eval, _, _ = engine.eval(trainloader)    #TO change back to testloader if train_size <1
        else:
            temp_eval, _, _ = engine.eval(testloader)
        LRScheduler.step(temp_eval[1])

        #Logging
        print(f'TrainLoss: {temp_train_loss}')
        print(f'Train R2: {R2}')
        train_loss.append(temp_train_loss)
        train_R2.append(R2)
        test_loss.append(temp_eval[0])
        test_R2.append(temp_eval[1])
        if i % cfg['output_freq'] == 0:
            logging.info(f"Epoch {i}: training loss {temp_train_loss} / test_loss {temp_eval[0]} / test accuracy {temp_eval[2]} / train R2 {R2}/ test R2 {temp_eval[1]}")
            print(f'TestLoss: {temp_eval[0]}')
            print(f'Test R2: {temp_eval[1]}')
            #output.append(temp_output)  #can be added to return to save full output instead of last outpu
            #labels.append(temp_labels)
        t2 = time.time()
        print(f'Training Epoch took {(t1-t2)/60} mins', flush=True)
            
    if cfg['train_size'] == 1 and cfg['stormsplit'] == 0 and cfg['crossvalidation'] == 0:
        final_eval, _, _ =  engine.eval(trainloader)
    else:
        final_eval, _, _ =  engine.eval(testloader)

    metrics = {
        'train_loss' : train_loss,
        'train_R2' : train_R2,
        'test_loss' : test_loss,
        'test_R2' : test_R2
        }
    output = temp_output    #REMOVE TO SAVE FULL OUTPUT
    labels = temp_labels
    return metrics, final_eval, output, labels



def objective(search_space, cfg, device,
              mask_probs, pin_memory, N_CPUS):

    #Create Datasets and Loaders
    trainset, testset, data_list = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'])
    trainloader, testloader = create_loaders(cfg, trainset, testset, pin_memory=pin_memory, data_type=cfg['data'])  #, num_workers=int(N_CPUS)

    num_features, num_edge_features, num_targets = get_attribute_sizes(cfg, trainset)

    params = setup_params(cfg, mask_probs, num_features, num_edge_features, num_targets)
    params = setup_params_from_search_space(search_space, params)

    print(params)  
    print('\nSEARCH_SPACE:\n')
    print(search_space)
    print('PARAMS')
    print(params, flush=True)
    #if device=='cuda':
        #print('CUDA')
        #tune.utils.wait_for_gpu(target_util=0.66)

    model = get_model(cfg, params)
    #model = torch.compile(model_)
    model.to(device)

    #Choose Criterion
    if cfg['task'] == 'GraphClass':
        criterion = torch.nn.CrossEntropyLoss()
    elif cfg['weighted_loss_label']:
        criterion = weighted_loss_label(factor=torch.tensor(cfg['weighted_loss_factor']))
    else:
        criterion = torch.nn.MSELoss(reduction='mean') 

    #criterion.to(device)
    optimizer = get_optimizer(cfg, model, params)
    #Init LR Scheduler
    LRScheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001, verbose=True)
    
    engine = Engine(model,optimizer, device, criterion, tol=cfg["accuracy_tolerance"], task = cfg['task'], var=mask_probs,
                    masking = params['use_masking'], mask_bias=params['mask_bias'])
    
    used_lr=engine.optimizer.param_groups[0]['lr']
    used_weight_decay = engine.optimizer.param_groups[0]['weight_decay']
    print(f'LR: {used_lr}')
    print(f'Weight Decay: {used_weight_decay}',flush=True)
    logging.info('New parameters suggested:')
    for key in search_space.keys():
        logging.info(f"{key}: {search_space[key]}")
    
    test_losses = []
    train_losses = []
    test_R2 = []
    train_R2 = []
    start = time.time()
    savename = '.pt'

    for i in range(1, cfg['epochs'] + 1):
        train_loss, temp_train_R2, output, labels = engine.train_epoch(trainloader, params['gradclip'])
        train_losses.append(train_loss)
        #report
        if i % cfg['output_freq'] == 0:
            eval_score, _, _ =  engine.eval(testloader)  
            test_losses.append(eval_score[0].cpu())
            #discrete_measure.append(eval_score[3].cpu())
            test_R2.append(eval_score[1].cpu())
            train_R2.append(temp_train_R2.cpu())
            logging.info(f"Epoch {i}: Train Loss: {train_loss} // Test Loss: {eval_score[0]} // Train R2: {temp_train_R2} // Test R2: {eval_score[1]} // accuracy {eval_score[2]}")
            result = {
                'train_loss' : train_loss,
                'test_loss' : eval_score[0].detach(),
                'train_R2' : temp_train_R2.detach(),
                'test_R2' : eval_score[1].detach()
                }
            session.report(result)
            torch.save(list(test_losses), cfg['dataset::path'] + "results/" + 'test_losses' + savename) #saving train losses
            torch.save(list(train_losses), cfg['dataset::path'] + "results/" + 'train_losses' + savename) #saving train losses
            torch.save(list(output), cfg['dataset::path'] + "results/" + 'output' + savename) #saving train losses
            torch.save(list(labels), cfg['dataset::path'] + "results/" + 'labels' + savename) #saving train losses
            torch.save(list(test_R2), cfg['dataset::path'] + "results/" + 'test_R2' + savename) #saving test R2
            torch.save(list(train_R2), cfg['dataset::path'] + "results/" + 'train_R2' + savename) #saving trai R2
            

        #LR Scheduler
        LRScheduler.step(eval_score[1].cpu())
    
    final_eval, output, labels = engine.eval(testloader) 
    

    for key in search_space.keys():
        if key in ['num_layers', 'hidden_size', 'embedding_dim', 'walk_length', 'reghead_size', 'reghead_layers', 'K', 'num_heads',
                   'loss_type', 'use_batchnorm', 'use_masking', 'use_skipcon']:
            savename = '_' + key + '_' + str(int(params[key])) + savename
        else:
            savename = '_' + key + '_' + f'{params[key]:.3}'  + savename


    #tune.report(np.array(discrete_measure).min())  #only necessary for intermediate results


    end = time.time()
    logging.info(f'Runtime: {(end-start)/60} min')
    #discrete_measure = torch.tensor(discrete_measure).detach()
    train_losses = torch.tensor(train_losses).detach()
    test_losses = torch.tensor(test_losses).detach()
    train_R2 = torch.tensor(train_R2).cpu().detach()
    test_R2 = torch.tensor(test_R2).cpu().detach()
    result = {
        #'discrete_measure' : discrete_measure.min(),
        'train_loss' : train_losses.min(),
        'test_loss' : test_losses.min(),
        'train_R2' : train_R2.cpu().min(),
        'test_R2' : test_R2.max(),
        'runtime' : (end-start)/60,
        }

    return result

