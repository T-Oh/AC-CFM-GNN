from pathlib import Path
import ray as ray
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import CombinedStopper, MaximumIterationStopper, TrialPlateauStopper
from ray import tune
import ray as ray
from ray_trainable import NN_tune_trainable
from sys import argv

dataset_path = Path("/home/chris/work/pik/projects/dataset_gnn/N020/4Pytorch")
result_path = Path(
    "/home/chris/work/pik/projects/dataset_gnn/N020/ray_arma/run003")

N_cpus = int(argv[1])
port_dashboard = int(argv[2])
ray.init(num_cpus=N_cpus, num_gpus = 1, include_dashboard=True,dashboard_port=port_dashboard)

cfg = {}

# dataset
cfg["dataset::path"] = dataset_path
cfg["task"] = "snbs"
cfg["task_type"] = "regression"
cfg["Fbeta::beta"] = 2.0
# cfg["train_set::start_index"] = 0
# cfg["train_set::end_index"] = 799

# dataset batch sizes
cfg["train_set::batchsize"] = 150
# cfg["train_set::batchsize"] = tune.choice([10, 20])
# cfg["train_set::batchsize"] = tune.randint(10,800)
cfg["test_set::batchsize"] = 150
cfg["valid_set::batchsize"] = 150
cfg["train_set::shuffle"] = True
cfg["test_set::shuffle"] = False
cfg["valid_set::shuffle"] = False


# ray settings
cfg["save_after_epochs"] = 10
cfg["checkpoint_freq"] = 10
cfg["num_samples"] = 6
cfg["ray_name"] = "ArmaNet_temp_bs"

# model settings
cfg["model_name"] = "ArmaNet_ray"
cfg["final_linear_layer"] = False
cfg["final_sigmoid_layer"] = False

cfg["num_layers"] = 2

cfg["max_num_channels"] = 200
cfg["num_channels1"] = 1
cfg["num_channels2"] = tune.randint(1, cfg["max_num_channels"])
cfg["num_channels3"] = 1
# cfg["num_channels4"]  = tune.randint(1,cfg["max_num_channels"])
# cfg["num_channels5"]  = tune.randint(1,cfg["max_num_channels"])
# cfg["num_channels6"]  = tune.randint(1,cfg["max_num_channels"])
# cfg["num_channels7"]  = tune.randint(1,cfg["max_num_channels"])
# cfg["num_channels8"]  = tune.randint(1,cfg["max_num_channels"])
# cfg["num_channels9"]  = tune.randint(1,cfg["max_num_channels"])
# cfg["num_channels10"] = tune.randint(1,cfg["max_num_channels"])

cfg["batch_norm_index"] = [True, True]
cfg["activation"] = ["relu", "None"]
# ARMA
cfg["ARMA::max_num_internal_layers"] = 4
cfg["ARMA::num_internal_layers1"] = tune.randint(1,cfg["ARMA::max_num_internal_layers"])
cfg["ARMA::num_internal_layers2"] = tune.randint(1,cfg["ARMA::max_num_internal_layers"])
# cfg["ARMA::num_internal_layers3"] = 4
# cfg["ARMA::num_internal_layers4"] = 4
# cfg["ARMA::num_internal_layers5"] = 4
# cfg["ARMA::num_internal_layers6"] = 4


cfg["ARMA::max_num_internal_stacks"] = 3
cfg["ARMA::num_internal_stacks1"] = tune.randint(1,cfg["ARMA::max_num_internal_stacks"])
cfg["ARMA::num_internal_stacks2"] = tune.randint(1,cfg["ARMA::max_num_internal_stacks"])


cfg["ARMA::dropout"] = .25
cfg["ARMA::shared_weights"] = True
# GCN
cfg["GCN::improved"] = True

# TAG
cfg["TAG::max_K_hops"] = 12
cfg["TAG::K_hops1"] = 3
cfg["TAG::K_hops2"] = 3
# cfg["TAG::K_hops3"] = tune.randint(1,cfg["TAG::max_K_hops"])
# cfg["TAG::K_hops4"] = 3
# cfg["TAG::K_hops5"] = 3
# cfg["TAG::K_hops6"] = 3
# cfg["TAG::K_hops7"] = 3
# cfg["TAG::K_hops8"] = 3
# cfg["TAG::K_hops9"] = 3
# cfg["TAG::K_hops10"] = 3


## MLP
cfg["MLP::use_MLP"] = False
cfg["MLP::scikit_dataset_path"] = "/home/chris/work/pik/projects/tmp/dataset_zenodo/scikit_dataset/"
# cfg["MLP::4Pytorch_path"] = "/home/chris/work/pik/projects/tmp/dataset_zenodo/4Pytorch/"
cfg["MLP::num_classes"] = 1
cfg["MLP::num_input_features"] = 6
cfg["MLP::num_hidden_layers"] = 1
cfg["MLP::num_hidden_unit_per_layer"] = 30
cfg["MLP::selected_measures"] = [
    "P",
    'degree',
    'average_neighbor_degree',
    'clustering',
    'current_flow_betweenness_centrality',
    'closeness_centrality']
cfg["MLP::final_sigmoid_layer"] = False


# training settings
cfg["cuda"] = False
#cfg["num_workers"] = 1
#cfg["num_threads"] = 2
cfg["manual_seed"] = 1
# cfg["manual_seed"] = tune.choice([1,2,3,4,5])
cfg["epochs"] = 5
cfg["optim::optimizer"] = "SGD"
cfg["optim::LR"] = .3
# cfg["optim::LR"] = tune.loguniform(1e-4, 2e1)
# cfg["optim::LR"] = tune.choice([1e-4, 1e-2])
cfg["optim::momentum"] = .9
cfg["optim::weight_decay"] = 1e-9
cfg["optim::scheduler"] = None
cfg["optim::ReducePlat_patience"] = 20
cfg["optim::LR_reduce_factor"] = .7
cfg["optim::stepLR_step_size"] = 30
# cfg["optim::scheduler"] = "stepLR"
cfg["criterion"] = "MSELoss"
# cfg["search_alg"] = "Optuna"
cfg["search_alg"] = None

# evaluation
cfg["eval::threshold"] = .1

if cfg["MLP::use_MLP"]:
    metric = "valid_fbeta"
else:
    metric = "valid_loss"

asha_scheduler = AsyncHyperBandScheduler(
    time_attr="training_iteration",
    metric=metric,
    mode="min",
    max_t=cfg["epochs"],
    grace_period=150,
    reduction_factor=3,
    brackets=5,
)

optuna_search = OptunaSearch(
    metric="valid_loss",
    mode="min",
    # points_to_evaluate=[{"manual_seed": 1}, {"manual_seed": 2}, {"manual_seed": 3}, {"manual_seed": 4}, {"manual_seed": 5}])

# tune_stop = CombinedStopper(MaximumIterationStopper(max_iter=cfg["epochs"]))
    tune_stop = CombinedStopper(MaximumIterationStopper(max_iter=cfg["epochs"]), TrialPlateauStopper(metric=metric, num_results=100, std=0.005, grace_period=150))


checkpoint_freq = cfg["checkpoint_freq"]
num_samples = cfg["num_samples"]
name = cfg["ray_name"]


if cfg["search_alg"] == "Optuna":
    analysis = tune.run(
        NN_tune_trainable,
        name=name,
        stop=tune_stop,
        config=cfg,
        num_samples=num_samples,
        local_dir=result_path,
        search_alg=optuna_search,
        # checkpoint_freq=checkpoint_freq,
        keep_checkpoints_num=1,
        checkpoint_score_attr=metric,
        checkpoint_freq=1,
        checkpoint_at_end=True,
        resources_per_trial={'cpu': 2. ,'gpu': .5},
        max_failures=1,
        # scheduler=asha_scheduler,
    )
elif cfg["search_alg"] == None:
    analysis = tune.run(
        NN_tune_trainable,
        name=name,
        stop=tune_stop,
        config=cfg,
        num_samples=num_samples,
        local_dir=result_path,
        checkpoint_freq=checkpoint_freq,
        checkpoint_at_end=True,
        resources_per_trial={'cpu': 4. ,'gpu': .5},
        max_failures=3,
    )


# analysis = tune.run(
#     NN_tune_trainable,
#     stop={"training_iteration": cfg["epochs"]},
#     config=cfg,
#     num_samples=2,
#     local_dir=result_path,
# )

print('best config: ', analysis.get_best_config(metric=metric, mode="max"))


# ray.shutdown()
print("finished")
