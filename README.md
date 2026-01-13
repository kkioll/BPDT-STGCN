# BPDT-STGCN
Body Part-Based Dual-Task Spatio-Temporal Graph Convolutional Network for Action Quality Assessment

## [Rhythmic Gymnastics (RG) Dataset](https://github.com/qinghuannn/ACTION-NET)
- Download Link: [OneDrive](https://onedrive.live.com/?id=76593CF7B7FC849C%21173443&resid=76593CF7B7FC849C%21173443&e=fdd2eO&migratedtospo=true&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBcHlFX0xmM1BGbDJpc3NEYmFLOTlzaGZaUktjaGc%5FZT1mZGQyZU8&cid=76593cf7b7fc849c&v=validatepermission)


### Data Structure
Organize the data in your `$DATASET_ROOT` as follows:
```text
$DATASET_ROOT
├── Ball (RG)
│   ├── train/
│   │   └── Ball_084.npy
│   ├── test/
│   │   └── Ball_080.npy
│   ├── test.pkl
│   └── train.pkl
├── Clubs (RG)
    ├── train/
    │   └── Clubs_084.npy
    ├── test/
    │   └── Clubs_080.npy
    ├── test.pkl
    └── train.pkl
```

## Installation

### Step 1: Create and activate conda environment ###

```bash
conda create -n interaqa python=3.8 -y
conda activate interaqa
```
### Step 2: Install dependencies ###

```bash
pip install -r requirements.txt
```
### Step 3: Training ###
Configure the parameters in config.py, run:
```bash
python train.py
