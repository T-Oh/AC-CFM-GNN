# AC-CFM-GNN

Code for training hybrid deep learning models combining Graph Neural Networks (GNNs) and Long Short-Term Memory (LSTM) networks to predict the post-event stable state of power grids from sequences of transmission line failures.

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

## Running the Code

All runtime behaviour and hyperparameters are controlled via:

    configurations/configuration.json

Important fields:
- `cfg_path` — path to the repository code.
- `dataset::path` — path containing the `raw/` and `processed/` folders.
- `model` — choose between `"MLPLSTM"` (baseline), `"TAGLSTM"`, and `"GATLSTM"`.
- Additional fields control model architecture and training (layers, hidden sizes, dropout, learning rate, weight decay, train/test split, etc).

### Data preparation (part of running)

1. **Provide raw data**  
   Place AC-CFM `.mat` files in:

       raw/

2. **Initial processing (creates unnormalized processed data)**  
   Run `main.py` once with the configuration fields set to:

   - `"data": "LSTM"`
   - `"model": "TAG"`

   This will produce processed but **unnormalized** data in a folder (e.g. `processed_unnormalized/` or similar) inside the dataset path.

3. **Normalization**  
   Normalize the processed data with:

       python normalize_GTSF.py

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
