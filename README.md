# AC-CFM-GNN
 
 Code for Training GNNs on AC cascading failure model data [[1]](#1), for the prediction of nodal power outages.
 
 # Data Preperation
 Before getting started the data in form of .mat files created with AC-CFM must be provided in the folder 'raw/'. 
 Then running the main once will create the processed but unnormalized data in the folder 'processed/'.
 The data must then be normalized using the 'normalize.py' script. 

# Getting Started
Once the data is provided we can use the 'configuration.json' in the folder configurations to configure the model we want to train and how to train it.
In "cfg_path" the path of the code must be given and in "dataset::path" the path where 'raw/' and 'processed/' are located. If everything is in the same folder leaving these empty should also work.
To get started we train a single model once. For this we can choose the "model"  between 'GINE', 'TAG' and 'GAT' or one of the baselines 'NodeMean', 'Ridge', 'MLP' or 'Node2Vec'.
We can ajust a lot of model parameters like number of layers, number of features, dropout and so on.
Additionally we can adjust training parameters like the ratios of the train and test set or if we want to split by storm (stormsplit!=0), the learning rate, the weight decay etc.
After adjusting everything to our liking we can simply run the main.py.
Some metrics like train and test MSE and R2 are saved in 'results/regression.log', as well as the model, the train and test losses per epoch and the labels and outputs.
Supported optimizers are 'Adam' and 'SGD'.

# Stormsplit
To use the train and test set splitting by storm the data must be named correctly. The index giving the scenario number in the file name (the first of the two integers) must be preceded by an integer indicating the storm. I used alphabetical order. For example if a file of storm Hanna (alphabetically the 3rd) is called data_45_12.pt it must be renamed to data_345_12.pt. Then the stormsplit parameter will give the storm which should be used as the test set. Thus if we want to use storm Hanna as the test set we set stormsplit to 3. If stormsplit is 0 the sets will be split by ratio defined by 'train_size'.

# Running Hyperparameter Studies
To run hyperparameter studies we must set 'study::run' to true. Then we configure the searchspace with the study::*lower and study::*upper parameters. If the lower and upper limits are the same the parameter will not be studied. 
The studies are executed with ray. The results are return in 'objective' folders in 'results/'. To evaluate some of the metrics I used the analyze_ray.py script. For further analysis refer to the ray documentation (https://docs.ray.io/en/latest/tune/index.html)

# Running Cross-Validation
To run cross-validation we must provide 7 datasets in the 7 folders processed and processed2-7. In each folder the train test split is done by storm going from 1 to 7 (f.e. in the 3 fold the set in processed3 will be used and split so that storm 3 (in my case Hanna) will be the test set).
The crossvalidation returns the output and labels of every fold as .pt files as well as the MSE and R2 of every fold in the file crossval_results.pt. Additionally the MSE through the epochs (learning curve) is returned for the first fold and the MSE and R2 of the epochs (as defined by 'output_freq') are printed in 'results/regression.log' for every fold.


## References
<a id="1">[1]</a> 
Matthias Noebels, Robin Preece, and Mathaios Panteli. 
Ac cascading failure model
for resilience analysis in power networks.
 IEEE Systems Journal, pages 1–12, 2020.