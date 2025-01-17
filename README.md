# Advancing Brainwave-Based Biometrics: A Large-Scale, Multi-Session Evaluation
This repository provides a collection of Jupyter notebooks designed to support reproducibility and facilitate further research in brainwave-based biometric authentication. The toolkit reproduces key experiments and findings from the paper *[Advancing Brainwave-Based Biometrics: A Large-Scale, Multi-Session Evaluation](arxiv.org)*. 

## Citation

If you use this repository in your research, please cite the following paper:

```bash
@inproceedings{brainwave2024, title = {Advancing Brainwave-Based Biometrics: A Large-Scale, Multi-Session Evaluation},
author = {},
booktitle = {},
year = {2025},
organization = {},
doi = {10.1145/XXXXXXX.XXXXXXX},
note = {Available at https://doi.org/XXXXXXX.XXXXXXX} }
```

## How to Use
### 1. Clone the Repository
To start using this project, clone the repository to your local machine:
```bash
git clone https://github.com/kit-ps/NeuroShield.git
```
### 2. Install Dependencies
You can install the required dependencies using one of the following methods:

#### Option 1: Install all dependencies in your conda environment:
Activate your conda environment and run:
```bash
```
#### Option 2: Install dependencies manually:
Install the following dependencies individually and resolve any additional requirements as needed.
```bash
pip install git+https://github.com/matinpf/pyeer.git
pip install h5py
pip install mne
```
### 3. Download Data
Navigate to the ./0_Data_Preparation/ directory to download the subject data. You have two options:

#### Option 1: Download from the main source
Run the following files to download and preprocess the data:
```bash
python 01_Download_and_Preprocessing.py
python 02_Merge_and_Split.py
```

For cluster usage, you can execute the provided script:
```bash
00_Download_via_Cluster_Job.sh
```
#### Option 2: Download preprocessed data
You can directly download the preprocessed dataset (150 GB) from *[this link](https://bwsyncandshare.kit.edu/s/BLrtm8A2oFFrtYR)* and put it in following directory
```bash
./Data/
```
### 4. Generate Additional Sub-Datasets and Extract Features
To prepare additional sub-datasets and extract PSD (Power Spectral Density) features, follow these steps:

1. Run the data-splitting script:
    ```bash
    jupyter notebook 03_Further_Data_Split.ipynb
    ```
2. Extract PSD features:
    ```bash
    jupyter notebook 04_Create_PSD_Dataset.ipynb
    ```

### 5. Reproduce Results
The repository is structured to align with the results section of the associated paper. You can replicate various results using the provided scripts.

- Pretrained models are stored in the ./PreTrained_Models/ directory for experiments requiring them.

- Alternatively, you can train new models using the training scripts available in the respective experiment folders. 








