# README
This is the PyTorch implementation of paper “Convolutional Neural Network Assisted Transformer for Automatic Modulation Recognition under Large CFOs and SROs”.

# Requirements
```
pytorch
yacs
h5py
matplotlib
thop  
```

# Architecture
``` 
home
├── amr/
│   ├── dataloaders/
│   ├── models/
│   │   ├── losses/
│   │   ├── networks/
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── draw.py
│   │   ├── init.py
│   │   ├── logger.py
│   │   ├── solver.py
│   │   ├── static.py
│   ├── RML2016.10a_dict.pkl
│   ├── RML2018.01.hdf5
├── configs/  (hyperparameters)
│   ├── *.yaml
├── main.py
├── datasets/
├── results/
```

# Quick Start
If you want to train a network from scratch, please follow these steps:
1. preparing dataset: download the dataset with large-scale offsets [dataset.rar](https://drive.google.com/file/d/1xZa9GcZoIZXstkwNd4E68Wbq7DdFN-a5/view?usp=sharing), and form the file path like 'dataset/cfo.hdf5'

2. training and testing: run `python main.py --config xxx`. e.g.`python main.py --config configs/transgroupnet_cfo.yaml`

3. checking the results: check the well-trained models and the figures in `results/`

# Result Reproduction
1. preparing dataset: download the dataset with large-scale offsets [dataset.rar](https://drive.google.com/file/d/1xZa9GcZoIZXstkwNd4E68Wbq7DdFN-a5/view?usp=sharing), and form the file path like 'dataset/cfo.hdf5'

2. preparing models: download the prepared models and results in [results.rar](https://drive.google.com/file/d/1MiHnfB_F25c0yTIHt52JuXWQ27r4sFYH/view?usp=sharing), and extract to the current path

3. modifying settings: change the state of `train` From `True` to `False` in `configs/transgroupnet_cfo.yaml`, and run `python main.py --config configs/transgroupnet_cfo.yaml` to get the expermential results.

    e.g.
    ```yaml
    method: 'ours'
    train: False  # change the state: from True to False
    dataset: 'cfo'
    mod_type: ["BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM32", "QAM64", "QAM128", "QAM256", "GFSK", "WBFM", "AM-DSB", "AM-SSB", "OOK", "4ASK", "8ASK", "16PSK", "32PSK","8APSK","GMSK", "DQPSK","16APSK","32APSK","64APSK","128APSK"]
    workers: 8
    seed: 1
    gpu: 0
    cpu: False
    params:
        "network": "TransGroupNet"
        "loss": "loss_CE"
        "batch_size": 1024
        "epochs": 200
        "lr": 5e-3
        "lr_decay": 0.8
        "weight_decay": 5e-2
        "early_stop": False
        "Xmode": [{"type":"APF","options":{"IQ_norm":False, "zero_mask":False}}]
    ```





