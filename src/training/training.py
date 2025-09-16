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
import numpy as np


from training.engine import Engine
from models.get_models import get_model
from utils.get_optimizers import get_optimizer
from utils.utils import weighted_loss_label, setup_params, setup_params_from_search_space, state_loss, save_params, save_output
from datasets.dataset import create_datasets, create_loaders, get_attribute_sizes
from datasets.dataset_graphlstm import create_lstm_datasets, create_lstm_dataloader
from sklearn.metrics import confusion_matrix




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
    
    output=[]
    labels=[]
    TASK = cfg['task']
    overfit = False
    if fold == -1:  SAVENAME = ''   #name to add to the saved metrics file -> used during crossval to save the results of the different folds
    else:           SAVENAME = str(fold)

    #Setup dictionary for performance metrics, metrics holds the training results and eval the evaluation results
    metrics, eval = init_metrics_vars(TASK)



    for i in range(1, cfg['epochs'] + 1):

        print(f'Epoch: {i}', flush=True)
        #torch.cuda.synchronize()
        t1 = time.time()
        temp_metrics, train_output, train_labels = engine.train_epoch(trainloader, cfg['gradclip'], cfg['full_output'])

        if cfg['train_size'] == 1:
            temp_eval, test_output, test_labels = engine.eval(trainloader, cfg['full_output'])    #TO change back to testloader if train_size <1
        else:
            temp_eval, test_output, test_labels = engine.eval(testloader, cfg['full_output'])
        LRScheduler.step(temp_eval['loss'])

        _, metrics_, eval = log_metrics(temp_metrics, temp_eval, metrics, eval, i, TASK, cfg['cfg_path'], SAVENAME)
        t2 = time.time()
        print(f'Training Epoch took {(t2-t1)/60} mins', flush=True)

        #Check for overfit and save the outputs of relevatn epochs in that case
        if abs(max(eval['R2'])-temp_eval['R2']) > 0.15 or overfit or (TASK != 'GraphReg' and abs(max(eval['R2_2'])-temp_eval['R2_2']) > 0.15):
            #the logic used here is to ensure that outputs from non-overfitting epochs are saved as well
            if overfit and not (max(eval['R2'])-temp_eval['R2'] > 0.15 or (TASK!='GraphReg' and max(eval['R2_2'])-temp_eval['R2_2'] > 0.15)):  overfit = False
            else:   overfit = True
            save_output(train_output, train_labels, test_output, test_labels, str(i))

    save_output(train_output, train_labels, test_output, test_labels, '_final')

    #Plotting
    plotting(metrics_, eval, train_output, train_labels, 'results/plots/', '', TASK)
  
    return metrics, eval, train_output, train_labels, test_output, test_labels



def objective(search_space, cfg, device,
              mask_probs, pin_memory, N_CPUS):

    TASK = cfg['task']
    max_seq_length = -1
    NAME = session.get_trial_id() if session else "unknown_trial"

    #Create Datasets and Loaders
    if cfg['model'] == 'Node2Vec':
            trainset, testset = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'], edge_attr=cfg['edge_attr'])
            trainloader, testloader, max_seq_length = create_loaders(cfg, trainset, testset, Node2Vec=True)    #If Node2Vec is applied the embeddings must be calculated first which needs a trainloader with batchsize 1
    elif cfg['model'] in ['GATLSTM', 'TAGLSTM', 'MLPLSTM']:
        # Split dataset into train and test indices
        trainset, testset = create_lstm_datasets(cfg["dataset::path"], cfg['train_size'], cfg['manual_seed'], cfg['stormsplit'], cfg['max_seq_length'])
        # Create DataLoaders for train and test sets
        trainloader = create_lstm_dataloader(trainset, batch_size=cfg['train_set::batchsize'], shuffle=True, num_workers=N_CPUS, pin_memory=pin_memory)
        testloader = create_lstm_dataloader(testset, batch_size=cfg['test_set::batchsize'], shuffle=False, num_workers=N_CPUS, pin_memory=pin_memory)
    else:
        trainset, testset = create_datasets(cfg["dataset::path"], cfg=cfg, pre_transform=None, stormsplit=cfg['stormsplit'], data_type=cfg['data'], edge_attr=cfg['edge_attr'])
        trainloader, testloader, max_seq_length = create_loaders(cfg, trainset, testset, num_workers=N_CPUS, pin_memory=pin_memory, data_type=cfg['data'], task=cfg['task'])

    num_features, num_edge_features, num_targets = get_attribute_sizes(cfg, trainset)

    params = setup_params(cfg, mask_probs, num_features, num_edge_features, num_targets, max_seq_length)
    params = setup_params_from_search_space(search_space, params, save=True, path=cfg['cfg_path'], ID=NAME)
    save_params(cfg['cfg_path'], params, ID=NAME)

 
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
    elif TASK == 'StateReg':
        criterion = state_loss(params['loss_weight'])
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
        temp_metrics, output, labels = engine.train_epoch(trainloader, params['gradclip'], cfg['full_output'])

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
            if TASK == 'StateReg':        
                temp_report = { 'train_loss' : temp_metrics['loss'],
                                'test_loss' : temp_eval['loss'],
                                'train_R2' : temp_metrics['R2'],
                                'test_R2' : temp_eval['R2'],
                                'train_R2_2' : temp_metrics['R2_2'],
                                'test_R2_2' : temp_eval['R2_2'],
                                'train_accuracy' : temp_metrics['accuracy'],
                                'test_accuracy' : temp_eval['accuracy'],
                                'train_precision' : temp_metrics['precision'],
                                'test_precision' : temp_eval['precision'],
                                'train_recall' : temp_metrics['recall'],
                                'test_recall' : temp_eval['recall'],
                                'train_F1' : temp_metrics['F1'],
                                'test_F1' : temp_eval['F1'],
                                'train_node_loss' : temp_metrics['node_loss'],
                                'test_node_loss' : temp_eval['node_loss'],
                                'train_edge_loss' : temp_metrics['edge_loss'],
                                'test_edge_loss' : temp_eval['edge_loss']
                                }
            elif TASK == 'NodeReg':
                temp_report = { 'train_loss' : temp_metrics['loss'],
                                'test_loss' : temp_eval['loss'],
                                'train_R2' : temp_metrics['R2'],
                                'test_R2' : temp_eval['R2'],
                                'train_R2_2' : temp_metrics['R2_2'],
                                'test_R2_2' : temp_eval['R2_2']}
            else:
                temp_report = { 'train_loss' : temp_metrics['loss'],
                                'test_loss' : temp_eval['loss'],
                                'train_R2' : temp_metrics['R2'],
                                'test_R2' : temp_eval['R2']
                }
                               
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
    elif TASK == 'StateReg':
        metrics = { 
            'loss'  : [],
            'R2'    : [],
            'R2_2'  : [],
            'accuracy'  : [],
            'precision' : [],
            'recall'    : [],
            'F1'        : [],
            'node_loss' : [],
            'edge_loss' : []
        }
        eval = {
            'loss'  : [],
            'R2'    : [],
            'R2_2'  : [],
            'accuracy'  : [],
            'precision' : [],
            'recall'    : [],
            'F1'        : [],
            'node_loss' : [],
            'edge_loss' : []
        }
    elif TASK == 'StateRegPI':
        metrics = { 
            'loss'  : [],
            'R2'    : [],
            'R2_2'  : [],
            'accuracy'  : [],
            'precision' : [],
            'recall'    : [],
            'F1'        : [],
            'node_loss' : [],
            'edge_loss' : [],
            'PI_loss' : []
        }
        eval = {
            'loss'  : [],
            'R2'    : [],
            'R2_2'  : [],
            'accuracy'  : [],
            'precision' : [],
            'recall'    : [],
            'F1'        : [],
            'node_loss' : [],
            'edge_loss' : [],
            'PI_loss' : []
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

        if 'StateReg' in TASK :
            temp_train_accuracy = temp_metrics['accuracy']
            temp_train_precision = temp_metrics['precision']
            temp_train_recall = temp_metrics['recall']
            temp_train_f1 = temp_metrics['F1']
            temp_test_accuracy = temp_eval['accuracy']
            temp_test_precision = temp_eval['precision']
            temp_test_recall = temp_eval['recall']
            temp_test_f1 = temp_eval['F1']

            print(f'Train Node Loss: {temp_metrics["node_loss"]}')
            print(f'Test Node Loss: {temp_eval["node_loss"]}')
            print(f'Train Edge Loss: {temp_metrics["edge_loss"]}')
            print(f'Test Edge Loss: {temp_eval["edge_loss"]}')
            print(f'Train Accuracy: {temp_train_accuracy}')
            print('Train Precision: ', temp_train_precision)
            print(f'Train Recall: {temp_train_recall}')
            print(f'Train F1: {temp_train_f1}')
            print(f'Test Accuracy: {temp_test_accuracy}')
            print(f'Test Precision: {temp_test_precision}')
            print(f'Test Recall: {temp_test_recall}')
            print(f'Test F1: {temp_test_f1}')
            logging.info(f"Epoch {epoch}: training loss {temp_train_loss} / test_loss {temp_test_loss} / Train Accuracy {temp_train_accuracy} / Test Accuracy {temp_test_accuracy} / Train Precision {temp_train_precision} / Test Precision {temp_test_precision} / Train Recall {temp_train_recall} / Test Recall {temp_test_recall} / Train F1 {temp_train_f1} / Test F1 {temp_test_f1}")

            result = {
                'train_loss' : metrics['loss'],
                'test_loss' : eval['loss'],
                'train_R2' : metrics['R2'],
                'test_R2' : eval['R2'],
                'train_R2_2' : metrics['R2_2'],
                'test_R2_2' : eval['R2_2'],
                'train_acc' : metrics['accuracy'],
                'test_acc'  : eval['accuracy'],
                'train_prec'    : metrics['precision'],
                'test_prec' : eval['precision'],
                'train_recall'  : metrics['recall'],
                'test_recall'   : eval['recall'],
                'train_F1'  : metrics['F1'],
                'test_F1'   : eval['F1'],
                'train_node_loss' : metrics['node_loss'],
                'test_node_loss' : eval['node_loss'],
                'train_edge_loss' : metrics['edge_loss'],
                'test_edge_loss' : eval['edge_loss']

            }
            if 'PI_loss' in temp_eval.keys():
                print(f'Train PI Loss: {temp_metrics["PI_loss"]}')
                print(f'Test PI Loss: {temp_eval["PI_loss"]}')
                result['train_PI_loss'] = metrics['PI_loss']
                result['test_PI_loss'] = eval['PI_loss']
                logging.info(f"Epoch {epoch}: training loss {temp_train_loss} / test_loss {temp_test_loss} / Train Accuracy {temp_train_accuracy} / Test Accuracy {temp_test_accuracy} / Train Precision {temp_train_precision} / Test Precision {temp_test_precision} / Train Recall {temp_train_recall} / Test Recall {temp_test_recall} / Train F1 {temp_train_f1} / Test F1 {temp_test_f1} / Train PI Loss {temp_metrics['PI_loss']} / Test PI Loss {temp_eval['PI_loss']}")



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
        print(key)
        fig1, ax1 = plt.subplots()
        ax1.plot(metrics_[key], label='Train' + key)
        ax1.plot(eval[key], label='Test' + key)
        ax1.legend()
        fig1.savefig(folder+key+'_'+NAME+'.png', bbox_inches='tight')
        plt.close()

        if 'R2' in key:
            fig1, ax1 = plt.subplots()
            ax1.plot(metrics_[key], label='Train' + key)
            ax1.plot(eval[key], label='Test' + key)
            ax1.set_ylim(0,1)
            ax1.legend()
            fig1.savefig(folder+key+'_zoom_'+NAME+'.png', bbox_inches='tight')
            plt.close()
    
    print('Scatter Plot')
    if task == 'GraphReg':
        print('GraphReg')
        fig, ax = plt.subplots()
        ax.scatter(labels, output, alpha=0.5)
        ax.plot([0,1],[0,1])
        ax.set_xlabel('Labels')
        ax.set_ylabel('Output')
        ax.set_title('Scatter - Graph Level')
        fig.savefig( folder + 'scatter_outputVSlabel_graph_Vreal_' + NAME + '.png', bbox_inches='tight')
        plt.close()
    elif task == 'StateReg':
        print('StateReg')
        node_labels = labels[0]
        edge_labels = labels[1]
        node_output = output[0]
        edge_output = output[1]
        for i in range(int(min(20,len(node_labels)/2000))):
            fig, ax = plt.subplots()
            ax.scatter(node_labels[2000*i:2000*(i+1), 0], node_output[2000*i:2000*(i+1), 0], alpha=0.5)
            ax.plot([-1,1],[-1,1])
            ax.set_xlabel('Node Labels')
            ax.set_ylabel('Node Output')
            ax.set_title(f'Scatter Re(V) - Chunk {i}')
            fig.savefig(folder + f'scatter_outputVSlabel_{i}_Vreal_' + NAME+ '.png', bbox_inches='tight')
            plt.close()

            # Scatter plot Vimag
            fig, ax = plt.subplots()
            ax.scatter(node_labels[2000*i:2000*(i+1), 1], node_output[2000*i:2000*(i+1), 1], alpha=0.5)
            ax.plot([-1,1],[-1,1])
            ax.set_xlabel('Node Labels')
            ax.set_ylabel('Node Output')
            ax.set_title(f'Scatter Imag(V) - Chunk {i}')
            fig.savefig(folder + f'scatter_outputVSlabel_{i}_Vimag_' + NAME + '.png', bbox_inches='tight')
            plt.close()

        # Plot confusion matrix for edge status
        # Convert logits to predicted class
        predictions = 1-np.argmax(output[1], axis=1)
        true_labels = 1-labels[1].reshape(-1)
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        classes = np.unique(true_labels)

        # Plot confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

        # Add colorbar
        plt.colorbar(im, ax=ax)

        # Labeling
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')

        # Annotate cells with values
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

        # Save the figure
        plt.savefig(folder+'confusion_matrix_edge_status.png', bbox_inches='tight', dpi=300)
        plt.close()

    else:
        for i in range(min(int(len(output)/2000),20)):
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




