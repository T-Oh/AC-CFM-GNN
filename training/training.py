import torch
import pickle
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


def run_training(trainloader, testloader, engine, cfg, LRScheduler, fold = -1):
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
    TASK = cfg['task']
    if fold == -1:  SAVENAME = ''   #name to add to the saved metrics file -> used during crossval to save the results of the different folds
    else:           SAVENAME = str(fold)

    #Setup dictionary for performance metrics, metrics holds the training results and eval the evaluation results
    metrics, eval = init_metrics_vars(TASK)



    for i in range(1, cfg['epochs'] + 1):
        #print('EVAL: ', eval)

        print(f'Epoch: {i}', flush=True)
        #torch.cuda.synchronize()
        t1 = time.time()

        temp_metrics, temp_output, temp_labels = engine.train_epoch(trainloader, cfg['gradclip'])

        if cfg['train_size'] == 1:
            temp_eval, _, _ = engine.eval(trainloader)    #TO change back to testloader if train_size <1
        else:
            temp_eval, _, _ = engine.eval(testloader)
        LRScheduler.step(temp_eval['loss'])

        #Logging
        temp_train_loss = metrics['loss']
        #print(f'TrainLoss: {temp_train_loss}')

        result, metrics, eval = log_metrics(temp_metrics, temp_eval, metrics, eval, i, TASK, cfg['dataset::path'], SAVENAME)
        t2 = time.time()
        print(f'Training Epoch took {(t2-t1)/60} mins', flush=True)


    output = temp_output   
    labels = temp_labels

    
    return metrics, eval, output, labels



def objective(search_space, cfg, device,
              mask_probs, pin_memory, N_CPUS):

    TASK = cfg['task']

    #Create Datasets and Loaders
    trainset, testset = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'])
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
    if TASK == 'GraphClass':
        criterion = torch.nn.CrossEntropyLoss().to(device)
    elif cfg['weighted_loss_label']:
        criterion = weighted_loss_label(factor=torch.tensor(cfg['weighted_loss_factor']))
    else:
        criterion = torch.nn.MSELoss(reduction='mean') 

    #criterion.to(device)
    optimizer = get_optimizer(cfg, model, params)
    #Init LR Scheduler
    LRScheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001)
    #used_lr = LRScheduler.get_last_lr()
    
    engine = Engine(model,optimizer, device, criterion, tol=cfg["accuracy_tolerance"], task = TASK, var=mask_probs,
                    masking = params['use_masking'], mask_bias=params['mask_bias'])
    
    used_lr=engine.optimizer.param_groups[0]['lr']
    used_weight_decay = engine.optimizer.param_groups[0]['weight_decay']
    print(f'LR: {used_lr}')
    print(f'Weight Decay: {used_weight_decay}',flush=True)
    logging.info('New parameters suggested:')
    for key in search_space.keys():
        logging.info(f"{key}: {search_space[key]}")
    
    metrics, evaluation = init_metrics_vars(TASK)
    start = time.time()

    for i in range(1, cfg['epochs'] + 1):
        temp_metrics, output, labels = engine.train_epoch(trainloader, params['gradclip'])

        #report
        if i % cfg['output_freq'] == 0:
            savename = ''
            temp_eval, _, _ =  engine.eval(testloader)
            for key in search_space.keys():
                if key in ['num_layers', 'hidden_size', 'embedding_dim', 'walk_length', 'reghead_size', 'reghead_layers', 'K', 'num_heads',
                        'loss_type', 'use_batchnorm', 'use_masking', 'use_skipcon']:
                    savename = '_' + key + '_' + str(int(params[key])) + savename
                else:
                    savename = '_' + key + '_' + f'{params[key]:.3}'  + savename
            result, metrics, evaluation = log_metrics(temp_metrics, temp_eval, metrics, evaluation, i, TASK, cfg['dataset::path'], savename)              
            session.report(temp_eval)

            torch.save(list(output), cfg['dataset::path'] + "results/" + 'output.pt') #saving train losses
            torch.save(list(labels), cfg['dataset::path'] + "results/" + 'labels.pt') #saving train losses
            

        #LR Scheduler
        LRScheduler.step(temp_eval['loss']) #.cpu()
    
    #final_eval, output, labels = engine.eval(testloader) 
    




    #tune.report(np.array(discrete_measure).min())  #only necessary for intermediate results


    end = time.time()
    logging.info(f'Runtime: {(end-start)/60} min')
    #discrete_measure = torch.tensor(discrete_measure).detach()


    return result


def init_metrics_vars(TASK):
    if 'Class' in TASK:
        metrics = { 
            'loss'  : [],
            'accuracy'  : [],
            'precision' : [],
            'recall'    : [],
            'F1'        : []
        }
        eval = {
            'loss'  : [],
            'accuracy'  : [],
            'precision' : [],
            'recall'    : [],
            'F1'        : []
        }
    else:
        metrics = {
            'loss'  : [],
            'R2'    : [],
            'R2_2'  : []
        }
        eval = {
            'loss'  : [],
            'R2'    : [],
            'R2_2'  : []
        }
    return metrics, eval


def log_metrics(temp_metrics, temp_eval, metrics, eval, epoch, TASK, path, savename):
    temp_train_loss = temp_metrics['loss']
    temp_test_loss = temp_eval['loss']

    for key in temp_metrics.keys(): 
        metrics[key].append(temp_metrics[key])
        eval[key].append(temp_eval[key])



    if 'Class' not in TASK:
        temp_train_R2 = temp_metrics['R2']
        temp_test_R2 = temp_eval['R2']
        if 'R2_2' in temp_eval.keys():
            temp_train_R2_2 = temp_metrics['R2_2']
            temp_test_R2_2 = temp_eval['R2_2']

        print(f'TrainLoss: {temp_train_loss}')
        print(f'Train R2: {temp_train_R2}')
        print(f'TestLoss: {temp_test_loss}')
        print(f'Test R2: {temp_test_R2}')

        logging.info(f"Epoch {epoch}: training loss {temp_train_loss} / test_loss {temp_test_loss} / train R2 {temp_train_R2}/ test R2 {temp_eval['R2']}")
        result = {
            'train_loss' : metrics['loss'],
            'test_loss' : eval['loss'],
            'train_R2' : metrics['R2'],
            'test_R2' : eval['R2']
            }
        
        if 'R2_2' in temp_eval.keys():
            print(f'Train R2_2: {temp_train_R2_2}')
            print(f'Test R2_2 {temp_test_R2_2}')
            result['train_R2_2'] = metrics['R2_2']
            result['test_R2_2'] = eval['R2_2']


    else:
        temp_train_accuracy = temp_metrics['accuracy']
        temp_train_precision = temp_metrics['precision']
        temp_train_recall = temp_metrics['recall']
        temp_train_f1 = temp_metrics['F1']
        temp_test_accuracy = temp_eval['accuracy']
        temp_test_precision = temp_eval['precision']
        temp_test_recall = temp_eval['recall']
        temp_test_f1 = temp_eval['F1']

        print(f'Train Loss: {temp_train_loss}')
        print(f'Train Accuracy: {temp_train_accuracy}')
        print('Train Precision: ', temp_train_precision)
        print(f'Train Recall: {temp_train_recall}')
        print(f'Train F1: {temp_train_f1}')
        print(f'Test Loss: {temp_test_loss}')
        print(f'Test Accuracy: {temp_test_accuracy}')
        print(f'Test Precision: {temp_test_precision}')
        print(f'Test Recall: {temp_test_recall}')
        print(f'Test F1: {temp_test_f1}')
        logging.info(f"Epoch {epoch}: training loss {temp_train_loss} / test_loss {temp_test_loss} / Train Accuracy {temp_train_accuracy} / Test Accuracy {temp_test_accuracy} / Train Precision {temp_train_precision} / Test Precision {temp_test_precision} / Train Recall {temp_train_recall} / Test Recall {temp_test_recall} / Train F1 {temp_train_f1} / Test F1 {temp_test_f1}")

        result = {
            'train_loss' : metrics['loss'],
            'test_loss' : eval['loss'],
            'train_acc' : metrics['accuracy'],
            'test_acc'  : eval['accuracy'],
            'train_prec'    : metrics['precision'],
            'test_prec' : eval['precision'],
            'train_recall'  : metrics['recall'],
            'test_recall'   : eval['recall'],
            'train_F1'  : metrics['F1'],
            'test_F1'   : eval['F1']
        }
    
    with open(path + 'results/results_' + savename + '.pkl', 'wb') as f:
        pickle.dump(result, f)
        
    #torch.save(list(metrics['loss']), cfg['dataset::path'] + "results/" + 'train_losses' + savename) #saving train losses
               

    return result, metrics, eval
        

