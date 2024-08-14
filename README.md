# Discovering nuclear localization signal universe through a novel deep learning model with interpretable attention units

This guide helps you set up the environment for NLSExplorer.




<img src="./image/NLSExplorer.png" alt="Potential NLS universe" width="888"/>

see the detailed introduction at http://www.csbio.sjtu.edu.cn/bioinf/NLSExplorer/introduction_new.html

## Prerequisites
Ensure that Anaconda or Miniconda is installed.

## Python Environment Setup 
If you have already installed pytorch with python version>=3.7 successfully, you can
skip this step. 

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
1. Download files from [NLSExplorer Code](http://www.csbio.sjtu.edu.cn/bioinf/NLSExplorer/code.html):
   - **The NLSExplorer model**: Place NLS_loc_modeltes in the folder named "./Recommendation_system".

2. Download files from [NLSExplorer dataset](http://www.csbio.sjtu.edu.cn/bioinf/NLSExplorer/dataset_new.html): 
   - **Dataset**: Put NLSExplorer-p in the folder named "./A2KA_train/Dataset" and extract it using appropriate commands (e.g., `unzip xxx`).



## Attention To Key Area （A2KA） Module 
<!-- ![A2KA](./A2KA/A2KA.svg) -->

<img src="./image/A2KA.svg" alt="A2KA" width="888"/>

1. Make sure pytorch is already installed.

2. Once Pytorch is installed， you can directly install A2KA by run the command:
```bash
pip install A2KA
```


3. You can direcly import AK2A module , and AK2A can be specified by your own config.

4. The config means the structure of your A2KA , the length of config means the number of layers, and the value 
represents the number of basic units , for instance, the config = [8,8,32],means the structure has 3 layers,
and the the first layer includes 8 BAUs , second layer includes 8 BAUs, third layer includes 32 BAUs. 

```python
from A2KA import A2KA
import torch
hidden_dimention = 512
#configure your A2KA sturcture
config = [8,8,32]
#If your datasize is significant large, extending the scale of the network may be a good choice.
#Such a config = 18*[64] means it has 18 layers and each layer has 64 basic attention units.
model =A2KA( hidden_dimention,config)
# tensor in a shape of (Batchsize,sequence_length, embedding dimension)
exampletensor = torch.randn(5,100,512)
prediction,layerattention = model(exampletensor)
print(prediction)
print(layerattention)

```
