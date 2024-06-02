# SpatialGAE-AD

In order to obtain informative spatiotemporal embeddings, this study proposes a multi-view graph learning framework named SpatialGAE-AD to predict pathological scores from RNA-seq data.For transcriptomics data, interactions between individuals have been extracted as sample similarity networks, which act as graph-structured data in this study. Spatial and temporal patterns have been integrated by the canonical correlation fusion(CCF) strategy. Validation experiments about multiple groups of AMP-AD datasets have been conducted to validate the effectiveness of multi-view SpatialGAE-AD method.

## Architecture

![architecture](./architecture)

## Install

### Requirements

Python --- 3.6.4

Tensorflow --- 1.12.0

Keras --- 2.1.0

Numpy --- 1.19.5

Scipy --- 1.5.4

Pandas --- 1.1.5

Sklearn --- 0.24.2

## Data availability

| Dataset |     ID     |
| :-----: | :--------: |
| ROSMAP  | syn3219045 |
|  MSBB   | syn3159438 |
|   ACT   | syn5759376 |
|  Mayo   | syn5550404 |
|  ANM1   |  GSE63060  |
|  ANM2   |  GSE63061  |

The IDs starting with syn are data from the Synapse platform, which can be obtained from https://www.synapse.org/; the IDs starting with GSE are data from the GEO platform, which can be obtained from https://www.ncbi.nlm.nih.gov/.

## Usage

### 1 run graph.py

The graph.py file is used to obtain the graphmatrix of different data sets. The command statements are as follows:

```
python graph.py --dataset_str ACT
```

Running this command will generate a graphmatrix.csv file in the dataset folder under the corresponding data folder.

### 2 run CCF_strategy.py

The CCF_Stategy.py file is used to obtain different CCA feature fusion matrices. The command statements are as follows:

```
python CCF_strategy.py --dataset_str ACT
```

Running this command will generate four csv files in the dataset folder under the corresponding data folder, namely:

- expression.csv——fusion matrix based on feature matrix
- graph.csv——fusion matrix based on graph features
- plus.csv——matrix after adding two fusion matrices
- concat.csv——matrix after concatenating two fusion matrices

The above four matrices can be put into the subsequent GAE model for training.

### 3 run SpatialGAE-AD.py

The SpatialGAE-AD.py file is used to obtain the final prediction results. The command statements are as follows:

```
python SpatialGAE-AD.py --dataset_str ACT --phenotype CERAD
```

The statement needs to specify the dataset and the predicted label.

- There are two labels, CERAD and Braak, in the ROSMAP and MSBB datasets.
- There are two labels, Braak and Thal, in the Mayo dataset.
- There are four labels, CERAD, Braak, Abeta_IHC, and Tau_IHC in the ACT dataset.

The final predicted labels are stored in the result folder.

If you do not want to use CCA fusion, you can use the following command:

```
python SpatialGAE-AD.py --dataset_str ACT --phenotype CERAD --isCCA False
```

## Sample similarity networks

![sample similarity networks](./sample_similarity_networks)