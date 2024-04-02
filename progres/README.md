#  Recommendation system
This folder include our recommendation system
in the folder you can:
1. test the NLSExplorer's accuracy at INSP dataset
2. traing the recommendation system on your own parameter
3. execute the ablation experiment for NLSExplorer

## Calculate the accuracy on INSP-dataset

The INSP-dataset includes yeast and hybrid dataset as test set， besides that， a
training dataset is included.


You can get the result below diffrent cofactor on the two datasets, based on: 
python {cofactor} {datasetname}

```bash
conda activate progres
#test accuracy on hybrid-dataset
python acc_calculation.py 0.3 hybrid
#test accuracy on yeast-dataset
python acc_calculation.py 0.3 yeast
```






