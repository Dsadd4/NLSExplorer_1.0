# Discovering nuclear localization signal universe through a novel deep learning model with interpretable attention units

This guide helps you set up the environment for NLSExplorer.




<img src="./image/CandidateL.png" alt="Potential NLS universe" width="888"/>

see the detailed introduction at http://www.csbio.sjtu.edu.cn/bioinf/NLSExplorer/introduction_new.html

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

## File Downloads
1. Download two files from [NLSExplorer Code](http://www.csbio.sjtu.edu.cn/bioinf/NLSExplorer/code.html):
   - **The NLSExplorer model**: Place it in the folder named "./Recommendation_system".
  
   - **Dataset**: Put it in the folder named "./A2KA_train/Dataset" and extract it using appropriate commands (e.g., `unzip xxx`).



## Attention To Key Area （A2KA） Module 
<!-- ![A2KA](./A2KA/A2KA.svg) -->

<img src="./image/A2KA.svg" alt="A2KA" width="888"/>

1. Make sure pytorch is already installed.

2. You can direcly import AK2A module , and AK2A can be specified by your own config.

3. The config means the structure of your A2KA , the length of config means the number of layers, and the value 
represents the number of basic units , for instance, the config = [6,12,12],means the structure has 3 layers,
and the the first layer includes 6 BAUs , second layer includes 12 BAUs, third layer includes 12 BAUs. 

```python
from A2KA import A2KA
hidden_dimention = 512
config = [6,12,12,5]
model =A2KA( hidden_dimention,config)
```

4. The hidden_dimention means the input tensor of A2KA, and A2KA will output the enhanced representation ,and the 
attention distribution along full sequence.


