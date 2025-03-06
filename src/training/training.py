import torch
import pickle
import logging
import time

from ray.air import session
from ray import train
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Matplotlib
import matplotlib.pyplot as plt


from training.engine import Engine
from models.get_models import get_model
from utils.get_optimizers import get_optimizer
from utils.utils import weighted_loss_label, setup_params, setup_params_from_search_space
from datasets.dataset import create_datasets, create_loaders, get_attribute_sizes
from datasets.dataset_graphlstm import create_lstm_datasets, create_lstm_dataloader




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

        _, metrics_, eval = log_metrics(temp_metrics, temp_eval, metrics, eval, i, TASK, cfg['cfg_path'], SAVENAME)
        t2 = time.time()
        print(f'Training Epoch took {(t2-t1)/60} mins', flush=True)


    output = temp_output  
    labels = temp_labels
    #Plotting
    plotting(metrics_, eval, output, labels, 'results/plots/', '', TASK)

    
    return metrics, eval, output, labels



def objective(search_space, cfg, device,
              mask_probs, pin_memory, N_CPUS):

    TASK = cfg['task']
    max_seq_length = -1
    NAME = session.get_trial_id() if session else "unknown_trial"

    #Create Datasets and Loaders
    if cfg['model'] == 'Node2Vec':
            trainset, testset = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'], edge_attr=cfg['edge_attr'])
            trainloader, testloader, max_seq_length = create_loaders(cfg, trainset, testset, Node2Vec=True)    #If Node2Vec is applied the embeddings must be calculated first which needs a trainloader with batchsize 1
    elif cfg['model'] == 'GATLSTM':
        # Split dataset into train and test indices
        trainset, testset = create_lstm_datasets(cfg["dataset::path"], cfg['train_size'], cfg['manual_seed'])
        # Create DataLoaders for train and test sets
        trainloader = create_lstm_dataloader(trainset, batch_size=cfg['train_set::batchsize'], shuffle=True)
        testloader = create_lstm_dataloader(testset, batch_size=cfg['test_set::batchsize'], shuffle=False)
    else:
        trainset, testset = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'], edge_attr=cfg['edge_attr'])
        trainloader, testloader, max_seq_length = create_loaders(cfg, trainset, testset, num_workers=N_CPUS, pin_memory=pin_memory, data_type=cfg['data'], task=cfg['task'])

    num_features, num_edge_features, num_targets = get_attribute_sizes(cfg, trainset)

    params = setup_params(cfg, mask_probs, num_features, num_edge_features, num_targets, max_seq_length)
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
        print(f'Epoch: {i}', flush=True)
        temp_metrics, output, labels = engine.train_epoch(trainloader, params['gradclip'])

        #report
        if i % cfg['output_freq'] == 0:
            savename = ''
            temp_eval, _, _ =  engine.eval(testloader)
            """for key in search_space.keys():
                if key in ['num_layers', 'hidden_size', 'embedding_dim', 'walk_length', 'reghead_size', 'reghead_layers', 'K', 'num_lstm_layers',
                        'lstm_hidden_size', 'num_conv_targets', 'loss_type', 'use_batchnorm', 'use_masking', 'use_skipcon', 'heads', ]:
                    savename = '_' + key + '_' + str(int(params[key])) + savename
                else:
                    savename = '_' + key + '_' + f'{params[key]:.2}'  + savename"""
            result, metrics, evaluation = log_metrics(temp_metrics, temp_eval, metrics, evaluation, i, TASK, cfg['cfg_path'], NAME)              
            temp_report = { 'train_loss' : temp_metrics['loss'],
                            'test_loss' : temp_eval['loss'],
                            'train_R2' : temp_metrics['R2'],
                            'test_R2' : temp_eval['R2'],
                            'train_R2_2' : temp_metrics['R2_2'],
                            'test_R2_2' : temp_eval['R2_2']}
            session.report(temp_report)   

            torch.save(list(output), cfg['cfg_path'] + "results/" + 'output.pt') #saving train losses
            torch.save(list(labels), cfg['cfg_path'] + "results/" + 'labels.pt') #saving train losses
            

        #LR Scheduler
        LRScheduler.step(temp_eval['loss']) #.cpu()
    
        if i == cfg['epochs']-1:
            end = time.time()
            logging.info(f'Runtime: {(end-start)/60} min')
            #Plotting
            #session = train.get_session()
            plotting(metrics,evaluation, output, labels, cfg['cfg_path']+ 'results/plots/', NAME, TASK)



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
        
def plotting(metrics_, eval, output, labels, folder, NAME, task):
    print('Plotting...')
    for key in metrics_:
        print('metrics')
        fig1, ax1 = plt.subplots()
        ax1.plot(metrics_[key], label='Train' + key)
        ax1.plot(eval[key], label='Test' + key)
        ax1.legend()

        fig1.savefig(folder+key+'_'+NAME+'.png', bbox_inches='tight')

        plt.close()

    if task == 'GraphReg':
        print('Scatter Plot, graph reg')
        fig, ax = plt.subplots()
        ax.scatter(labels, output, alpha=0.5)
        ax.plot([0,1],[0,1])
        ax.set_xlabel('Labels')
        ax.set_ylabel('Output')
        ax.set_title('Scatter - Graph Level')

        fig.savefig( folder + 'scatter_outputVSlabel_graph_Vreal_' + NAME + '.png', bbox_inches='tight')

        plt.close()

    else:
        print('Scatter Plot, node reg')
        for i in range(int(len(output)/2000)):
            # Scatter plot Vreal
            fig, ax = plt.subplots()
            ax.scatter(labels[2000*i:2000*(i+1), 0], output[2000*i:2000*(i+1), 0], alpha=0.5)
            ax.plot([-1,1],[-1,1])
            ax.set_xlabel('Labels')
            ax.set_ylabel('Output')
            ax.set_title(f'Scatter Re(V) - Chunk {i}')
            fig.savefig(folder + f'scatter_outputVSlabel_{i}_Vreal_' + NAME+ '.png', bbox_inches='tight')
            plt.close()

            # Scatter plot Vimag
            fig, ax = plt.subplots()
            ax.scatter(labels[2000*i:2000*(i+1), 1], output[2000*i:2000*(i+1), 1], alpha=0.5)
            ax.plot([-1,1],[-1,1])
            ax.set_xlabel('Labels')
            ax.set_ylabel('Output')
            ax.set_title(f'Scatter Imag(V) - Chunk {i}')
            fig.savefig(folder + f'scatter_outputVSlabel_{i}_Vimag_' + NAME + '.png', bbox_inches='tight')
            plt.close()




