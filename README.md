# AC-CFM-GNN
 Code for training hybrid deep learning models which combine Graph Neural Networks (GNNs) and Long Short-Term Memory (LSTM) networks to predict the post-event stable state of power grids based on sequences of transmission line failures.
 
# Running and Controlling the Code
The conda environment is stored in configurations/env.yml
The codes behaviour and all hyperparameters are controlled from the configurations/configuration.json
# Data Preperation
Before getting started the data in form of .mat files created with AC-CFM [1] must be provided in the folder 'raw/'. Then running the main once with "data": "LSTM" and "model": "TAG" will create the processed but unnormalized data in the folder 'processed/'.
The data must then be normalized using the 'normalize_GTSF.py' script. To train on the normalized data the folder must then be renamed to processed/ .

# Getting Started
Once the data is provided we can use the 'configuration.json' in the folder configurations to configure the model we want to train and how to train it.
In "cfg_path" the path of the code must be given and in "dataset::path" the path where 'raw/' and 'processed/' are located.
To get started we train a single model once. For this we can choose the "model"  between "MLPLSTM" (baseline), "TAGLSTM" and "GATLSTM".
We can ajust a lot of model parameters like number of layers, number of features, dropout and so on.
Additionally we can adjust training parameters like the ratios of the train and test set or if we want to split by storm (stormsplit!=0), the learning rate, the weight decay etc.
After adjusting everything to our liking we can simply run the main.py.
The performance metrics of every epoch are saved in results/results_.pkl additionally plots of the metrics evolution through the epochs are plotted at the end of training and saved in results/plots/ and the trained model is saved in results/ as well.

# Stormsplit
To use the train and test set splitting by storm the data must be named correctly. The index giving the scenario number in the folder name must be preceded by an integer indicating the storm. Then the stormsplit parameter will give the storm which should be used as the test set (i.e. all the files for the testset should be named scenario_\{stormsplit\}* and no other file should start with the same digit). If stormsplit is 0 the sets will be split by ratio defined by 'train_size'.

# Running Hyperparameter Studies
To run hyperparameter studies we must set 'study::run' to true. Then we configure the searchspace with the study::*lower and study::*upper parameters. If the lower and upper limits are the same the parameter will not be studied. 
The studies are executed with ray. The results are return in 'objective' folders in 'results/'. For further analysis refer to the ray documentation (https://docs.ray.io/en/latest/tune/index.html)

# Reproducing Paper Results
The configuration.json contains the configuration used for the paper. According to the different experiments the values which need to be changed are "model" and "max_seq_length"
## References
<a id="1">[1]</a> 
Matthias Noebels, Robin Preece, and Mathaios Panteli. 
Ac cascading failure model
for resilience analysis in power networks.
 IEEE Systems Journal, pages 1–12, 2020.

## License 
This project is licensed under the MIT License. See the LICENSE file for details