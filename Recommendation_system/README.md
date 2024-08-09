#  Recommendation system
This folder include our recommendation system
in the folder you can:
1. test the NLSExplorer's accuracy at INSP dataset
2. traing the recommendation system on your own parameter
3. run  ablation experiment for NLSExplorer

## Calculate the accuracy on INSP-dataset

The INSP-dataset includes yeast and hybrid dataset as test set， besides that， a
training dataset is included.


You can get the result below diffrent cofactor on the two datasets, based on: 
python acc_calculation.py {cofactor} {datasetname}

```bash
conda activate progres
#test accuracy on hybrid-dataset
python acc_calculation.py 0.3 hybrid
#test accuracy on yeast-dataset
python acc_calculation.py 0.3 yeast
```

## Train the recommendation system on NLSExplorer-t

You can train your own recommendation system, based on: 
python acc_calculation.py {cofactor} {datasetname}
just need to uncomment      
torch.save(model.state_dict(), f'./resconn_{epoch}_nls_model_not_include') 
at acc_calculation.py  921 lines

but notethat, if you only want to train a new system based on the datasets we provide you can just run
the script we provide, or you must know the structure of our data, and construct your own data first. our
data is consist of very simple parts, you can know it by using the python code below and construct your own
dataset
```python
from utils import load_mydict
data = load_mydict('./for_recom/insp_train_0.6')
print(data)
```

## run  ablation-experiment for NLSExplorer



You can run ablation-experiment on the two datasets, based on: 
python acc_ablation.py {cofactor} {datasetname}

```bash
conda activate progres
#run ablation-experiment on hybrid-dataset
python acc_ablation.py 0.3 hybrid
#run ablation-experiment on yeast-dataset
python acc_ablation.py 0.3 yeast
```



