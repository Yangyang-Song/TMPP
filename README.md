# TMPP

TMPP: A Transformer-Based Spatiotemporal Model for High-Accuracy PM2.5 Concentration Prediction

## Dataset

The dataset used in this program has been saved in [Google Drive](https://drive.google.com/file/d/1uK8bBP6BJlS-kvdnTJBCfpQ7WpQdTtoN/view?usp=sharing), which is given by [Shuo Wang](https://github.com/shuowang-ai/PM2.5-GNN.git)

## Requirements

```
Python 3.7.3
PyTorch 1.13.1
PyG: https://github.com/rusty1s/pytorch_geometric#pytorch-170
```

```bash
pip install -r requirements.txt
```

## Experiment Setup

First, open `util.py`,do the following setups

* Set the current machine's name or remove the comments if you are using Linux or MacOS

```python
# Get file directory based on the current machine's nodename
# nodename = os.uname().nodename
nodename ="LAPTOP-UP2D1R34"
file_dir = config['filepath'][nodename]
```

Second, open `config.yaml`, do the following setups

- Set data path after your server name. Like mine

```python
filepath: # Define file paths for different machines
  LAPTOP-UP2D1R34::
    knowair_fp: C:\Users\Lenovo\Desktop\PM2.5-GNN-main\data\KnowAir.npy
    results_dir: C:\Users\Lenovo\Desktop\PM2.5-GNN-main\results

```

- Uncomment the model you want to run

```python
  model: MLP
  # model: LSTM
  # model: GRU
  # model: GC_LSTM
  # model: nodesFC_GRU
  # model: PM25_GNN
  # model: PM25_GNN_nosub
  # model: TMPP
  # model: Informer
  # model: patchTST
  # model: Non_AR
  # model: Fixed_Memory
```

- Choose the sub-datast number in [1,2,3]

```python
 dataset_num: 1
```

- Set weather variables you wish to use. Following is the default setting in the paper. You can uncomment specific variables. Variables in dataset **KnowAir** is defined in `metero_var`

```python
  metero_use: ['2m_temperature',
               'boundary_layer_height',
               'k_index',
               'relative_humidity+950',
               'surface_pressure',
               'total_precipitation',
               'u_component_of_wind+950',
               'v_component_of_wind+950',]

```

## Run

```bash
python train.py
```
