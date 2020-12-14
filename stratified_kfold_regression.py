# stratified-kfold for regression
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn import model_selection


def create_folds(data,num_splits):
    # we creat a new column called kfold and fill it with -1
    data["kfold"] = -1

    # randmoize the rows of the data
    data = data.sample(frac=1).reset_index(drop=True)

    # calculate the numbers of bins by Sturge's rule
    # Sturge's rule : # of bins = 1 + log_2(N)
    # Take the floor of the value or round it
    num_bins = np.round((1 + np.log2(len(data))), decimals=0)
    num_bins = int(num_bins)

    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"],bins=num_bins,labels=False
    )

    # initiate the kfold class 
    kf = model_selection.StratifiedKFold(n_splits=num_splits)

    # fill the new kfold column
    # instead of targets we use bins
    for fold, (trn_,val_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[val_,"kfold"] = fold
    
    # drop the bins column
    data = data.drop("bins", axis=1)
    
    # return dataframe with folds
    return data

if __name__ == "__main__":
    # we  create a sample dataset with 15000 samples
    # and 100 features and 1 target
    X,y = datasets.make_regression(
        n_samples=15000, n_features=100, n_targets=1
    )

    # create a dataframe
    df = pd.DataFrame(
        X,
        columns=[f"f_{i}" for i in range(X.shape[1])]
    )
    df.loc[:,"target"] = y

    # create folds
    df = create_folds(df,6)