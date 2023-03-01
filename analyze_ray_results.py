import ray
import ray.tune as tune
import training.engine
import utils.utils
import training.training
#ray.init()
path = 'C:/Users/tobia/OneDrive/Dokumente/Master/Semester4/Masterarbeit/results/cluster/GAT/10_subset/ray_study'
tuner = tune.Tuner.restore(path)
result_grid = tuner.get_results()
for i in range(len(result_grid)):
    print(result_grid[i])
