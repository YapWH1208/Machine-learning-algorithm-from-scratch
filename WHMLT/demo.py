from .linear.linear import Lasso_Regressor, Ridge_Regressor, Linear_Regressor
from .linear.logistic import Logistic_Regressor
from .tree.tree import Decision_Tree, Random_Forest
from .metrics.metrics import Metrics

import pandas as pd

def main(train, test):
    data = pd.read_csv(train)
    test = pd.read_csv(test)

    X = data.drop(["Price"])
    y = data.Price

    lr_model = Linear_Regressor(X, y, test)
    lr_model.train()

    predictions = lr_model.predict
    return predictions

if __name__ == "__init__":
    train_filepath = "INSERT HERE"
    test_filepath = "INSERT HERE"
    main(train_filepath, test_filepath)