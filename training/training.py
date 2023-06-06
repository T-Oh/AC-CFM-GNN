import torch

import logging
import time
from training.engine import Engine
from models.get_models import get_model
from utils.get_optimizers import get_optimizer

from ray import tune
from ray.air import session
from os.path import isfile

from utils.utils import weighted_loss_label, weighted_loss_var, setup_params, setup_params_from_search_space
from models.run_node2vec import run_node2vec
from datasets.dataset import create_datasets, create_loaders, calc_mask_probs

def run_training(trainloader, testloader, engine, cfg):
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
        print(f'Epoch: {i}')
        temp_train_loss, R2, temp_output, temp_labels = engine.train_epoch(trainloader, cfg['gradclip'])
        if cfg['train_size'] == 1:
            temp_eval, eval_output, eval_labels = engine.eval(trainloader)    #TO change back to testloader if train_size <1
        else:
            temp_eval, eval_output, eval_labels = engine.eval(testloader)

        print(f'Train R2: {R2}')
        train_loss.append(temp_train_loss)
        train_R2.append(R2)
        test_loss.append(temp_eval[0])
        test_R2.append(temp_eval[2])
        if i % cfg['output_freq'] == 0:
            logging.info(f"Epoch {i}: training loss {temp_train_loss} / test_loss {temp_eval[0]} / test accuracy {temp_eval[2]} / train R2 {R2}/ test R2 {temp_eval[1]} / test discrete measure {temp_eval[3]}")
            print(f'TrainLoss: {temp_train_loss}')
            print(f'Train R2: {R2}')
            #output.append(temp_output)  #can be added to return to save best output instead of last outpu
            #labels.append(temp_labels)
    
    if cfg['train_size'] == 1 and cfg['stormsplit'] == 0 and cfg['crossvalidation'] == 0:
        final_eval, final_output, final_labels =  engine.eval(trainloader)
    else:
        final_eval, final_output, final_labels =  engine.eval(testloader)
    metrics = {
        'train_loss' : train_loss,
        'train_R2' : train_R2,
        'test_loss' : test_loss,
        'test_R2' : test_R2
        }
    output = temp_output    #REMOVE TO SAVE FULL OUTPUT
    labels = temp_labels
    return metrics, final_eval, output, labels



def objective(search_space, trainloader, testloader, cfg, num_features, num_edge_features, num_targets, device, mask_probs):
    params = setup_params(cfg, mask_probs, num_features, num_edge_features)
    params = setup_params_from_search_space(search_space, params)
    print(params)
    
    
    
    if cfg['model'] == 'Node2Vec':
        print('Creating Node2Vec Embedding')
        
        embedding = run_node2vec(cfg, trainloader, device, params, 0)
        normalized_embedding = embedding.data
        #Normalize the Embedding
        print(embedding.shape)
        for i in range(embedding.shape[1]):
            normalized_embedding[:,i] = embedding[:,i].data/embedding[:,i].data.max()
            
        # Create Datasets and Dataloaders
        trainset, testset, data_list = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], embedding=normalized_embedding.to(device))
        trainloader, testloader = create_loaders(cfg, trainset, testset)


        # Calculate probabilities for masking of nodes if necessary
        if cfg['use_masking'] or cfg['weighted_loss_var'] or (cfg['study::run'] and (cfg['study::masking'] or cfg['study::loss_type'])):
            if isfile('node_label_vars.pt'):
                print('Using existing Node Label Variances for masking')
                mask_probs = torch.load('node_label_vars.pt')
            else:
                print('No node label variance file found\nCalculating Node Variances for Masking')
                mask_probs = calc_mask_probs(trainloader)
                torch.save(mask_probs, 'node_label_vars.pt')
        else:
            #Masks are set to one in case it is wrongly used somewhere (when set to 1 masking results in multiplication with 1)
            mask_probs = torch.zeros(2000)+1
    
    
        # getting feature and target sizes
        num_features = trainset.__getitem__(0).x.shape[1]
        print(f'New number of features: {num_features}')
        num_edge_features = trainset.__getitem__(0).edge_attr.shape[1]
        
        #Setup params for following task (MLP)
        params['num_features'] = num_features
        
    
    
    
    print('\nSEARCH_SPACE:\n')
    print(search_space)
    print('PARAMS')
    print(params)
    if device=='cuda':
        print('CUDA')
        tune.utils.wait_for_gpu(target_util=0.66)
    model = get_model(cfg, params)
    model.to(device)
    #Choose Criterion
    if cfg['weighted_loss_label']:
        criterion = weighted_loss_label(factor = torch.tensor(params['loss_weight']))
    elif cfg['weighted_loss_var']:
        criterion = weighted_loss_var(mask_probs, device)    
    else:    
        criterion = torch.nn.MSELoss(reduction = 'mean')  #TO defines the loss
    criterion.to(device)
    optimizer = get_optimizer(cfg, model)
    engine = Engine(model,optimizer, device, criterion, tol=cfg["accuracy_tolerance"], task = cfg['task'], var=mask_probs, masking = params['use_masking'], mask_bias=cfg['mask_bias'])
    engine.optimizer.lr = params['LR']
    
    logging.info('New parameters suggested:')
    for key in search_space.keys():
        logging.info(f"{key}: {search_space[key]}")
    
    test_losses = []
    train_losses = []
    discrete_measure = []
    test_R2 = []
    train_R2 = []
    start = time.time()
    for i in range(1, cfg['epochs'] + 1):
        train_loss, temp_train_R2, _,_ = engine.train_epoch(trainloader, params['gradclip'])
        train_losses.append(train_loss)
        #report
        if i % cfg['output_freq'] == 0:
            eval_score, _, _ =  engine.eval(testloader)  
            test_losses.append(eval_score[0].cpu())
            discrete_measure.append(eval_score[3].cpu())
            test_R2.append(eval_score[1].cpu())
            train_R2.append(temp_train_R2.cpu())
            logging.info(f"Epoch {i}: Train Loss: {train_loss} // Test Loss: {eval_score[0]} // Train R2: {temp_train_R2} // Test R2: {eval_score[1]} // accuracy {eval_score[2]} // discrete measure {eval_score[3]}")
            result = {
                'discrete_measure' : eval_score[3].detach(),
                'train_loss' : train_loss,
                'test_loss' : eval_score[0].detach(),
                'train_R2' : temp_train_R2.detach(),
                'test_R2' : eval_score[1].detach()
                }
            session.report(result)

        """if trial.should_prune():
            logging.warning("Trial pruned")
            raise optuna.exceptions.TrialPruned()"""
    
    final_eval, output, labels = engine.eval(testloader) 
    torch.save(list(test_losses), cfg['dataset::path'] + "results/" + f"test_losses_{params['num_layers']}L_{params['hidden_size']}HF_{params['LR']:.{3}f}lr_{params['gradclip']}GC_{params['use_skipcon']}SC_{params['reghead_size']}RHS_{params['reghead_layers']}RHL.pt") #saving train losses
    torch.save(list(train_losses), cfg['dataset::path'] + "results/" + f"train_losses_{params['num_layers']}L_{params['hidden_size']}HF_{params['LR']:.{3}f}lr_{params['gradclip']}GC_{params['use_skipcon']}SC_{params['reghead_size']}RHS_{params['reghead_layers']}RHL.pt") #saving train losses
    torch.save(list(output), cfg['dataset::path'] + "results/" + f"output_{params['num_layers']}L_{params['hidden_size']}HF_{params['LR']:.{3}f}lr_{params['gradclip']}GC_{params['use_skipcon']}SC_{params['reghead_size']}RHS_{params['reghead_layers']}RHL.pt") #saving train losses
    torch.save(list(labels), cfg['dataset::path'] + "results/" + f"labels_{params['num_layers']}L_{params['hidden_size']}HF_{params['LR']:.{3}f}lr_{params['gradclip']}GC_{params['use_skipcon']}SC_{params['reghead_size']}RHS_{params['reghead_layers']}RHL.pt") #saving train losses
    #tune.report(np.array(discrete_measure).min())  #only necessary for intermediate results


    end = time.time()
    logging.info(f'Runtime: {(end-start)/60} min')
    discrete_measure = torch.tensor(discrete_measure).detach()
    train_losses = torch.tensor(train_losses).detach()
    test_losses = torch.tensor(test_losses).detach()
    train_R2 = torch.tensor(train_R2).cpu().detach()
    test_R2 = torch.tensor(test_R2).cpu().detach()
    result = {
        'discrete_measure' : discrete_measure.min(),
        'train_loss' : train_losses.min(),
        'test_loss' : test_losses.min(),
        'train_R2' : train_R2.cpu().min(),
        'test_R2' : test_R2.max(),
        'runtime' : (end-start)/60,
        }

    return result

