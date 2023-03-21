from torch.optim import SGD, Adam

def get_optimizer(cfg, model):
    """
    Helper function which gets the desired optimizer
    and initializes it
    """
    if cfg["optim::optimizer"] == "SGD":
        optimizer = SGD(
            model.parameters(),
            lr = cfg["optim::LR"],
            momentum=cfg["optim::momentum"],
            nesterov=cfg["optim::nesterov"],
            weight_decay=cfg["optim::weight_decay"]
        )
        print(cfg["optim::LR"])
    elif cfg["optim::optimizer"] == "Adam":
        optimizer = Adam(
            model.parameters(),
            lr = cfg["optim::LR"],
            betas=cfg["optim::betas"],
            weight_decay=cfg["optim::weight_decay"],
            eps = cfg['optim::epsilon']
        )
    else:
        raise ValueError("Optimizer {} does not exist".format(cfg["optim::optimizer"]))

    return optimizer