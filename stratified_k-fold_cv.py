# import pandas and model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    # Training data is in a CSV file called train.csv
    df = pd.read_csv("train.csv")

    # create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # randomize the rows
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch the feature on whicg to stratify
    # here this feature's name "target"
    y = df.target.values

    # initiate the kfold class from model_selection
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for fold, (trn_, val_) in enumerate(kf.split(X=df,y=y)):
        df.loc[val_, "kfold"] = fold

    # save the new csv with kfold column
    df.to_csv("train_fold.csv",index=False)