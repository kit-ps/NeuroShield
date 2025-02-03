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
You can directly download the preprocessed dataset (150 GB) from *[this link](https://zenodo.org/records/14753435)* and put it in following directory
```bash
./Data/
```
