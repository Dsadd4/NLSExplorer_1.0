# Nuclear Localization Signal(NLS) Recommendation System Environment Setup

This guide helps you set up the environment for NLSExplorer.

## Prerequisites
Ensure that Anaconda or Miniconda is installed.

## Python Environment Setup -step1

### 1. Create Python 3.7 Environment
```bash
conda create -n NLSExplorer-pyto python==3.7
```

### 2. Install PyTorch and Dependencies
```bash
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch -c nvidia
```
note that this command is a general install chocice, we recommend you to https://pytorch.org/get-started/locally/
 to get the specific instruction.
```bash
conda env update -f environment-pyto.yaml
```

### 3. DSSP Installation
- Install DSSP:
```bash
conda install -c ostrokach dssp
```
- Add DSSP to PATH:
    Check DSSP location:
    ```bash
    which mkdssp
    ```

    Edit `.bashrc`:
    ```bash
    vim ~/.bashrc
    ```
    Add to the end of the file:
    (the addres of my mkdssp is "/home/server/miniconda3/envs/NLSExplorer-pyto/bin/mkdssp" you can see it from the command "which mkdssp")
    ```bash
    export PATH=$PATH:/home/server/miniconda3/envs/NLSExplorer-pyto/bin/mkdssp
    ```
    Source `.bashrc`:
    ```bash
    source ~/.bashrc
    ```

For DSSP usage, please refer to the website:
(https://swift.cmbi.umcn.nl/gv/dssp/).

##  Python Environment Setup -step2
### 1. Create Python 3.7 Environment
```bash
conda create --name NLSExplorer-progres --clone NLSExplorer-pyto
```
### 2. Install PyTorch and Additional Libraries
```bash
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch -c nvidia
pip install scikit-learn pandas tqdm egnn_pytorch torch-geometric
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.13.0+cu117.html(if your cant correctly install
you can install from local file ./progres/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl)
pip install fair-esm==2.0.0
```

## File Downloads
1. Download two files from [NLSExplorer Code](http://www.csbio.sjtu.edu.cn/bioinf/NLSExplorer/code.html):
   - **The NLSExplorer model**: Place it in the folder named "./Peptide_recomendation".
   - **loc_in_nucleus_with_pdb**: Put it in the folder named "./progres" and extract it using appropriate commands (e.g., `unzip loc_in_nucleus_with_pdb`).
   - **Dataset**: Put it in the folder named "./A2KA/Dataset" and extract it using appropriate commands (e.g., `unzip xxx`).
   - **databases**: Put it in the folder named "./progres/progres" and extract it using appropriate commands (e.g., `unzip xxx`).


## Running the System
1. Place input(fasta form and pdb file) in `./DANN_union/sequence_input`.
2. Run `python union_result.py` for computation. Results can be found in `./DANN_union/result`. Historical results are in `./DANN_union/result_his`.

## Attention To Key Area （A2KA） Module 
<!-- ![A2KA](./A2KA/A2KA.svg) -->

<img src="./A2KA/A2KA.svg" alt="A2KA" width="888"/>
1. Make sure pytorch is already installed 
2. you can direcly import AK2A module , and AK2A can be specified by your own config

```bash
from A2KA import A2KA
hidden_dimention = 512
config = [6,12,12,5]
model =A2KA( hidden_dimention,config)
```
3. the config means the structure of your A2KA , the length of config means the number of layers, and the value 
represents the number of basic units , for instance, the config = [6,12,12],means the structure has 3 layers,
and the the first layer includes 6 neurons , second layer includes 12 neurons, third layer includes 12 neurons. 

