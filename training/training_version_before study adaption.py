import optuna
import torch
import numpy as np
import logging
import time

def run_training(trainloader, testloader, engine, epochs=1):
    loss = []
    eval_scores = []
    #output=[]
    #labels=[]

    for i in range(1, epochs + 1):
        print(f'Epoch: {i}')
        temp_loss, temp_output, temp_labels = engine.train_epoch(trainloader)
        #temp_eval = engine.eval(trainloader)    #TO change back to testloader if train_size <1

        loss.append(temp_loss)
        #output.append(temp_output)  #can be added to return to save best output instead of last outpu
        #labels.append(temp_labels)
        #eval_scores.append(temp_eval)


        #if i % 10 == 0:
            #logging.info(f"Epoch {i}: training loss {temp_loss}, accuracy {temp_eval[2]} / R2 {temp_eval[1]}")

    final_eval, output, labels =  engine.eval(trainloader)  #TO change back to testloader if trainsiz<1
    print('USING TRAINLOADER FOR EVALUATION!')

    #logging.info("Final R2: ", final_eval[1])
    #logging.info("Final accuracy: ", final_eval[2])
    eval_scores=0

    return loss, final_eval, output, labels


class Objective(object):
    def __init__(self, trainloader, testloader, engine, epochs, lower, upper):
        self.trainloader = trainloader
        self.testloader = testloader
        self.engine = engine
        self.epochs = epochs
        self.lower, self.upper = lower, upper

    def __call__(self, trial):
        lr = trial.suggest_uniform("lr", self.lower, self.upper)
        self.engine.optimizer.lr = lr
        

        logging.info(f"\n\nNew learning rate suggested: {lr}")
        losses=[]
        discrete_measure = []
        start = time.time()
        for i in range(1, self.epochs + 1):
            _,output,labels = self.engine.train_epoch(self.trainloader)
            eval_score, output, labels =  self.engine.eval(self.trainloader)    #change back to TESTLOADER
            losses.append(eval_score[0])
            discrete_measure.append(eval_score[3])
            if i % 1000 == 0:
                logging.info(f"Epoch {i}: loss {eval_score[0]} // accuracy {eval_score[2]} // discrete measure {eval_score[3]}")

            trial.report(eval_score[3], i)

            if trial.should_prune():
                logging.warning("Trial pruned")
                raise optuna.exceptions.TrialPruned()

        final_eval, output, labels = self.engine.eval(self.trainloader) #change back to TESTLOADER


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
    study = optuna.create_study(direction="minimize", study_name="line_priority_regression", pruner=pruner)
    study.optimize(objective, n_trials=cfg["study::n_trials"])

    #Best trial
    logging.info("Best trial:")

    logging.info(f"  Value:  {study.best_value}")

    for key, value in study.best_params:
        logging.info(f"{key} : {value}")
    best_params=study.best_params

    return best_params
