# AC-CFM-GNN

Code for training hybrid deep learning models combining Graph Neural Networks (GNNs) and Long Short-Term Memory (LSTM) networks to predict the post-event stable state of power grids from sequences of transmission line failures.
The code supplies all functiononalities (data processing, training, evaluation, crossvalidation and hyperparameter studies) for two tasks. 
    - *Graph Time Series Forecasting* processes scenarios as time series' and uses models combining GNNs and LSTMs for time series forecasting predicting node and edge targets that define the full state of the grid.
    - *Direct Prediction* processes and treats every scenario step as single instance to perform Node Regression in order to predict the post event complex power at each node.

---

## Installation / Setup

### Conda environment

The conda environment is provided in:

    configurations/local_env.yml

Create and activate it with:

    conda env create -f configurations/local_env.yml
    conda activate AC-CFM-GNN   

### Git LFS (required)

This project uses Git LFS for large data files. Install Git LFS: https://git-lfs.github.com/

After installing, run:

    git lfs install
    git lfs pull

> **Note:** Without Git LFS the repository contains pointer files instead of the actual data files. So without LFS you will have no data to train the models on.

---

## Full end-to-end pipeline (GTSF + DP)

From raw data to final training, the project now supports a clear and reproducible pipeline controlled via configuration keywords.

### 1) Pipeline control keywords

In `configurations/configuration.json`, use these keywords to control the pipeline flow:

- `"process": true/false` — whether to run processing (raw → `processed/`)
- `"normalize": true/false` — whether to run normalization (`processed/` → `normalized/`)
- `"recalculate_data_stats": true/false` — whether to recalculate stats or load cached version
- `"use_unnormalized_data": true/false` — if `true` uses data from `processed/`; if `false` uses `normalized/`

### 2) Core directories

- `processed_dir` (implicit, default: `processed/`): initial processed data
- `normalized_dir` (default: `./normalized/`): normalized dataset
- `dataset::path`: root dataset folder

### 3) Pipeline examples

**Full end-to-end: process + normalize + train**

Set in `configuration.json`:

```json
"process":                  true,
"normalize":                true,
"recalculate_data_stats":   true,
"use_unnormalized_data":    false,
"data":                     "LSTM",
"model":                    "TAGLSTM"
```

Then run:

    python main.py 1 1 1 8887

**Processing only (no normalization)**

Set:

```json
"process":              true,
"normalize":            false,
"data":                 "LSTM",
"model":                "TAGLSTM"
```

Then run:

    python main.py 1 1 1 8887

Result: processed data in `processed/` (unnormalized)

**Normalization only (requires processed data)**

Set:

```json
"process":                  false,
"normalize":                true,
"recalculate_data_stats":   true,
"data":                     "LSTM"
```

Then run:

    python main.py 1 1 1 8887

Result: normalized data written to `normalized/`

**Training only (assumes normalized data exists)**

Set:

```json
"process":                  false,
"normalize":                false,
"use_unnormalized_data":    false,
"model":                    "TAGLSTM"
```

Then run:

    python main.py 1 1 1 8887

Result: trains on data from `normalized/`

**Use unnormalized data for training**

Set:

```json
"use_unnormalized_data":    true,
"model":                    "TAGLSTM"
```

Then run:

    python main.py 1 1 1 8887

Result: trains on data from `processed/` (no normalization)

### 4) Train-test split & data leakage prevention

- Train-test split occurs during `create_datasets()`.
- Normalization stats are computed **from trainset only** in `normalize_GTSF()` and `normalize_DP()`.
- Splits are seeded by `manual_seed` and `train_size` / `stormsplit` (if set).
- Recommend: persist splits in `splits.pt` for reproducibility.

### 5) Key normalization settings

- `data_stats_filename`: file to store min/max/mean/std (e.g. `data_stats_Zhu.pt`)
- `log_normalize`: if `true` applies log normalization (feature-dependent)
- `gen_feature_index`: index of first generator feature (for skip during normalization)

### 6) Data types supported

- **GTSF (time series)**: `"LSTM"`
- **DP (direct prediction)**: `"AC"`, `"Zhu"`, `"ANGF_Vcf"`, `"Zhu_nobustype"`, `"n-k"`, `"Zhu_n_minus_k"`

---

## Running the Code (Graph Time Series Forecasting)

All runtime behaviour and hyperparameters are controlled via:

    configurations/configuration.json

Important fields:
- `cfg_path` — path to the repository code.
- `dataset::path` — path containing the `raw/`, `processed/`, and `normalized/` folders.
- `model` — choose between `"MLPLSTM"` (baseline), `"TAGLSTM"`, and `"GATLSTM"`.
- Additional fields control model architecture and training (layers, hidden sizes, dropout, learning rate, weight decay, train/test split, etc).

### Train a single model

Once data is prepared (processed and/or normalized) and `configuration.json` is configured, start training with:

    python main.py 1 1 1 8887

The arguments are in order:
 - Number of tasks (>1 for parallel trials in hyperparameter studies)
 - Number of CPUs per task
 - Number of GPUs (total)
 - Port for ray dashboard (hyperparameter studies)

Outputs:
- Epoch-wise metrics: `results/results_.pkl`
- Training curves: `results/plots/`
- Trained model: `results/`


Once `processed/` exists and `configuration.json` is configured (remember to switch to one of the LSTM models), start training with:

    python main.py 1 1 1 8887

The arguments are in order:
 - Number of tasks (>1 for parallel trials in hyperparameter studies)
 - Number of CPUs per task
 - Number of GPUs (total)
 - Port for ray dashboard (hyperparameter studies)

Outputs:
- Epoch-wise metrics: `results/results_.pkl`
- Training curves: `results/plots/`
- Trained model: `results/`


---



## Running the Code (Direct Prediction)

The process is generally the same as with time series forecasting:

All runtime behaviour and hyperparameters are controlled via:

    configurations/configuration.json

Important fields:
- `cfg_path` — path to the repository code.
- `dataset::path` — path containing the `raw/` and `processed/` folders.
- `model` — choose between `"GAT"`, `"TAG"` and `"GINE"`.
- Additional fields control model architecture and training (layers, hidden sizes, dropout, learning rate, weight decay, train/test split, etc).

### Data preparation (part of running)

1. **Provide raw data**  
   Place AC-CFM `.mat` files in:

       raw/

2. **Initial processing (creates unnormalized processed data)**  
    You can choose any of the defined data types (except `"LSTM"`, `"LDTSF"` and `"LDTSF_DC"`).
    The *Direct Prediction* data types are explained in *Thesis.pdf*. It is recommended to use `"ZHU"` for *Direct Prediction*.

   This will produce processed but **unnormalized** data in a folder (e.g. `processed_unnormalized/` or similar) inside the dataset path.

3. **Normalization**  
   Normalize the processed data with:

       python src/scripts/normalize.py

   After normalization, rename the normalized output folder to:

       processed/

   The `processed/` folder is then used for training.

### Train a single model

Once `processed/` exists and `configuration.json` is configured, start training with:

    python main.py 1 1 1 8887

The arguments are in order:
 - Number of tasks (>1 for parallel trials in hyperparameter studies)
 - Number of CPUs per task
 - Number of GPUs (total)
 - Port for ray dashboard (hyperparameter studies)

Outputs:
- Epoch-wise metrics: `results/results_.pkl`
- Training curves: `results/plots/`
- Trained model: `results/`


---

## Running Hyperparameter Studies

To run Ray Tune hyperparameter sweeps:

1. Set:

       "study::run": true

2. Configure search ranges using `study::*lower` and `study::*upper`.  
   If lower == upper for a parameter it stays fixed.

Results are saved under:

    results/objective/

See Ray Tune docs for details: https://docs.ray.io/en/latest/tune/index.html


---

## References

[1] Matthias Noebels, Robin Preece, and Mathaios Panteli.  
**AC cascading failure model for resilience analysis in power networks.**  
*IEEE Systems Journal*, 2020.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
