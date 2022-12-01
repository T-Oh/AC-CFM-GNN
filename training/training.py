import optuna
import logging

def run_training(trainloader, testloader, engine, epochs=1):
    loss = []
    eval_scores = []

    for i in range(1, epochs + 1):
        print(i)
        temp_loss = engine.train_epoch(trainloader)
        temp_eval = engine.eval(testloader)

        loss.append(temp_loss)
        eval_scores.append(temp_eval)


        if i % 10 == 0:
            logging.info(f"Epoch {i}: training loss {temp_loss}, accuracy {temp_eval[2]} / R2 {temp_eval[1]}")

    final_eval =  engine.eval(testloader)

    #logging.info("Final R2: ", final_eval[1])
    #logging.info("Final accuracy: ", final_eval[2])

    return loss, eval_scores, final_eval


class Objective(object):
    def __init__(self, trainloader, testloader, engine, epochs, lower, upper):
        self.trainloader = trainloader
        self.testloader = testloader
        self.engine = engine
        self.epochs = epochs
        self.lower, self.upper = lower, upper

    def __call__(self, trial):
        lr = trial.suggest_loguniform("lr", self.lower, self.upper)
        self.engine.optimizer.lr = lr

        logging.info(f"New learning rate suggested: {lr}")

        for i in range(1, self.epochs + 1):
            _ = self.engine.train_epoch(self.trainloader)
            eval_score =  self.engine.eval(self.testloader)

            if i % 10 == 0:
                logging.info(f"Epoch {i}: loss {eval_score[0]} // accuracy {eval_score[2]}")

            trial.report(eval_score[0], i)

            if trial.should_prune():
                logging.warning("Trial pruned")
                raise optuna.exceptions.TrialPruned()

        final_eval = self.engine.eval(self.testloader)

        logging.info(f"Final accuracy: {final_eval[2]}")

        return final_eval[0]

def run_tuning(cfg, objective):

    pruner = optuna.pruners.MedianPruner(
        n_warmup_trials=cfg["study::pruner:n_warmup_trials"],
        n_warmup_steps=cfg["study::pruner::n_warmup_steps"],
        n_min_trials=cfg["study::pruner::n_min_trials"]
    )
    study = optuna.create_study(direction="minimize", study_name="line_priority_regression", pruner=pruner)
    study.optimize(objective, n_trials=cfg["study::n_trials"])

    #Best trial
    logging.info("Best trial:")

    logging.info(f"  Value:  {study.best_value}")

    for key, value in study.best_params:
        logging.info(f"{key} : {value}")

    return study.best_params
