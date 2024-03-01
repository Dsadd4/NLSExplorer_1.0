# Fragment Recommendation System Environment Setup

This guide helps you set up the environment for a Fragment Recommendation System.

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
conda env update -f environment-pyto.yaml
```

### 3. DSSP Installation
- Install DSSP:
```bash
conda install -c ostrokach dssp
```
- Add DSSP to PATH:
    - Check DSSP location:
    ```bash
    which mkdssp
    ```
    - Edit `.bashrc`:
    ```bash
    vim ~/.bashrc
    ```
    - Add to the end of the file:
    ```bash
    export PATH=$PATH:/home/server/miniconda3/envs/NLSExplorer-pyto/bin/mkdssp
    ```
    - Source `.bashrc`:
    ```bash
    source ~/.bashrc
    ```

For DSSP usage, please refer to the website:
(https://swift.cmbi.umcn.nl/gv/dssp/).

##  Python Environment Setup -step2
### 1. Create Python 3.9 Environment
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
   - **The NLSExplorer model**: Place it in the folder named "Peptide_recomendation".
   - **loc_in_nucleus_with_pdb**: Put it in the folder named "progres" and extract it using appropriate commands (e.g., `unzip loc_in_nucleus_with_pdb`).
   - **Dataset**: Put it in the folder named ".\A2KA\Dataset" and extract it using appropriate commands (e.g., `unzip xxx`).
   - **databases**: Put it in the folder named ".\progres\progres" and extract it using appropriate commands (e.g., `unzip xxx`).


## Running the System
1. Place input(fasta form and pdb file) in `./DANN_union/sequence_input`.
2. Run `python union_result.py` for computation. Results can be found in `./DANN_union/result`. Historical results are in `./DANN_union/result_his`.
```
