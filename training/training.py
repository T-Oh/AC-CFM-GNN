import optuna
import torch
import numpy as np
import logging
import time
from training.engine import Engine
from models.get_models import get_model
from utils.get_optimizers import get_optimizer
from ray import tune
from ray.air import session

def run_training(trainloader, testloader, engine, cfg):
    train_loss = []
    test_loss = []
    eval_scores = []
    output=[]
    labels=[]

    for i in range(1, cfg['epochs'] + 1):
        print(f'Epoch: {i}')
        temp_train_loss,temp_output, temp_labels = engine.train_epoch(trainloader)
        temp_eval, _, _ = engine.eval(testloader)    #TO change back to testloader if train_size <1

        train_loss.append(temp_train_loss)
        test_loss.append(temp_eval[0])
        
        if i % cfg['output_freq'] == 0:
            logging.info(f"Epoch {i}: training loss {temp_train_loss} / test_loss {temp_eval[0]} / test accuracy {temp_eval[2]} / test R2 {temp_eval[1]} / test discrete measure {temp_eval[3]}")
            output.append(temp_output)  #can be added to return to save best output instead of last outpu
            labels.append(temp_labels)

    final_eval, final_output, final_labels =  engine.eval(testloader)  #TO change back to testloader if trainsiz<1
    #print('USING TRAINLOADER FOR EVALUATION!')

    #logging.info("Final R2: ", final_eval[1])
    #logging.info("Final accuracy: ", final_eval[2])
    eval_scores=0

    return train_loss, final_eval, output, labels



def objective(config, trainloader, testloader, cfg, num_features, num_edge_features, num_targets, device, criterion):
    params = {
        "num_layers" : config['layers'],
        "hidden_size" : config['HF'],
        "dropout" : config["dropout"],
        "heads" : config['heads'],
        "num_features" : num_features,
        "num_edge_features" : num_edge_features,
        "num_targets" : num_targets
    }
    model = get_model(cfg, params)
    model.to(device)
    optimizer = get_optimizer(cfg, model)
    engine = Engine(model,optimizer, device, criterion, tol=cfg["accuracy_tolerance"], task = cfg['task'])
    engine.optimizer.lr = config['LR']
    
    logging.info(f"\n\nNew Parameters suggested:\n LR : {config['LR']} \n Layers : {config['layers']} \n HF : {config['HF']} \n Heads : {config['heads']}\n")
    
    test_losses = []
    train_losses = []
    discrete_measure = []
    r2 = []
    start = time.time()
    for i in range(1, cfg['epochs'] + 1):
        train_loss,_,_ = engine.train_epoch(trainloader)
        train_losses.append(train_loss)
        #report
        if i % cfg['output_freq'] == 0:
            eval_score, _, _ =  engine.eval(trainloader)    #change back to TESTLOADER
            test_losses.append(eval_score[0].cpu())
            discrete_measure.append(eval_score[3].cpu())
            r2.append(eval_score[1].cpu())
            logging.info(f"Epoch {i}: loss {eval_score[0]} // accuracy {eval_score[2]} // discrete measure {eval_score[3]}")
            result = {
                'discrete_measure' : eval_score[3],
                'loss' : eval_score[0],
                'r2' : eval_score[1]
                }
            session.report(result)

        """if trial.should_prune():
            logging.warning("Trial pruned")
            raise optuna.exceptions.TrialPruned()"""
    
    final_eval, output, labels = engine.eval(trainloader) #change back to TESTLOADER
    torch.save(list(test_losses), cfg['dataset::path'] + "results/" + f"test_losses_{params['num_layers']}L_{params['hidden_size']}HF_{params['heads']}heads_{config['LR']:.{3}f}lr_{config['dropout']}dropout.pt") #saving train losses
    torch.save(list(train_losses), cfg['dataset::path'] + "results/" + f"train_losses_{params['num_layers']}L_{params['hidden_size']}HF_{params['heads']}heads_{config['LR']:.{3}f}lr_{config['dropout']}dropout.pt") #saving train losses
    torch.save(list(output), cfg['dataset::path'] + "results/" + f"output_{params['num_layers']}L_{params['hidden_size']}HF_{params['heads']}heads_{config['LR']:.{3}f}lr_{config['dropout']}dropout.pt") #saving train losses
    torch.save(list(labels), cfg['dataset::path'] + "results/" + f"labels_{params['num_layers']}L_{params['hidden_size']}HF_{params['heads']}heads_{config['LR']:.{3}f}lr_{config['dropout']}dropout.pt") #saving train losses
    #tune.report(np.array(discrete_measure).min())  #only necessary for intermediate results


    end = time.time()
    logging.info(f'Runtime: {(end-start)/60} min')
    result = {
        'discrete_measure' : np.array(discrete_measure).min(),
        'test_loss' : np.array(test_losses).min(),
        'r2' : np.array(r2).min(),
        'runtime' : (end-start)/60,
        'train_loss' : np.array(train_losses).min()
        }

    return result



"""class Objective(object):
    def __init__(self, trainloader, testloader, cfg, params, device, criterion, ray_config):
        self.trainloader = trainloader
        self.testloader = testloader
        self.cfg = cfg
        self.params = params
        self.epochs = cfg['epochs']
        self.lower, self.upper = cfg['study::lr::lower'], cfg['study::lr::upper']
        self.device = device
        self.criterion = criterion

    def __call__(self, trial):
        
        model = get_model(self.params)
        model.to(self.device)
        optimizer = get_optimizer(self.cfg, model)

        lr = trial.suggest_loguniform("lr", self.lower, self.upper)
        
        engine = Engine(model,optimizer, self.device, self.criterion, tol=self.cfg["accuracy_tolerance"], task = self.cfg['task'])
        engine.optimizer.lr = lr
        

        logging.info(f"\n\nNew learning rate suggested: {lr}")
        losses=[]
        discrete_measure = []
        start = time.time()
        for i in range(1, self.epochs + 1):
            _,output,labels = engine.train_epoch(self.trainloader)
            eval_score, output, labels =  engine.eval(self.trainloader)    #change back to TESTLOADER
            losses.append(eval_score[0])
            discrete_measure.append(eval_score[3])
            if i % 1 == 0:
                logging.info(f"Epoch {i}: loss {eval_score[0]} // accuracy {eval_score[2]} // discrete measure {eval_score[3]}")

            trial.report(eval_score[3], i)

            if trial.should_prune():
                logging.warning("Trial pruned")
                raise optuna.exceptions.TrialPruned()

        final_eval, output, labels = engine.eval(self.trainloader) #change back to TESTLOADER


        logging.info(f"Final accuracy: {final_eval[2]}")
        logging.info(f"Final R2: {final_eval[1]}")
        logging.info(f"Final loss: {final_eval[0]}")
        logging.info(f"Final discrete measure: {final_eval[3]}")
        end = time.time()
        logging.info(f'Runtime: {(end-start)/60} min')
        

        return np.array(discrete_measure).min()

def run_tuning(cfg, objective):

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=cfg["study::pruner:n_warmup_trials"],
        n_warmup_steps=cfg["study::pruner:n_warmup_steps"],
        n_min_trials=cfg["study::pruner:n_min_trials"]
    )
    study = optuna.create_study(direction="minimize", study_name="parallel_test", pruner=pruner,load_if_exists=True)
    study.optimize(objective, n_trials=cfg["study::n_trials"])

    #Best trial
    logging.info("Best trial:")

    logging.info(f"  Value:  {study.best_value}")

    for key, value in study.best_params:
        logging.info(f"{key} : {value}")
    best_params=study.best_params

    return best_params"""
