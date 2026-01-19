# MUDiff: Unified Diffusion for Complete Molecule Generation

This repository contains the code for the paper "MUDiff: Unified Diffusion for Complete Molecule Generation".

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Compile Cython extension:**
    The `algos.pyx` file needs to be compiled. A `setup_algos.py` script is provided for this.
    ```bash
    python3 setup_algos.py build_ext --inplace
    ```

## Running the code

To train the model on the QM9 dataset, run the following command:

```bash
python3 main_qm9.py --exp_name qm9_train
```

**Note on Memory Usage:**
The default hyperparameters for the model are quite large and may lead to out-of-memory errors on machines with limited RAM. If you encounter such issues, you can run the script with a smaller model configuration. The `main_qm9.py` file has been updated to use smaller default values.

Example with smaller hyperparameters (already set as default in the provided `main_qm9.py`):
```bash
python3 main_qm9.py --exp_name qm9_train_small --n_epochs 10 --batch_size 16 --num_encoder_layers 3 --embedding_dim 64 --edge_embedding_dim 64 --ffn_embedding_dim 128
```

## Corrections and Fixes

This version of the code includes several fixes to make it runnable:

-   **Added `requirements.txt`:** All necessary dependencies are listed for easy installation.
-   **Recreated `configs/datasets_config.py`:** This file was missing from the original repository and has been recreated with necessary data for the QM9 dataset.
-   **Fixed `algos.pyx`:** The Cython code was updated to be compatible with modern versions of Python and NumPy.
-   **Fixed `qm9/models.py`:** A bug in the `DistributionNodes` class was fixed.
-   **Updated `main_qm9.py`:** The default hyperparameters have been reduced to allow the model to be trained on machines with less memory.



edited by: Mohammad Babaei https://adschi.com
