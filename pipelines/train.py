from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.core.dataset import Dataset
from azureml.core.workspace import Workspace
from azureml.data.dataset_factory import TabularDatasetFactory



def setBinary(x_df, columnName, columnOne):
    x_df[columnName] = x_df[columnName].apply(lambda s: 1 if s == columnOne else 0)



def setHotEncoding(x_df, columnName):
    column = pd.get_dummies(x_df[columnName], prefix=columnName)
    x_df.drop(columnName, inplace=True, axis=1)
    x_df = x_df.join(column)



def clean_data(data):

    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    
    columns_to_dummify = ["Airline", "AirportFrom", "AirportTo"]
    for col in columns_to_dummify :
        setHotEncoding(x_df, col)

    y_df = x_df.pop("Delay")
    return x_df, y_df



def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    ws = Workspace.from_config()
    dataset = Dataset.get_by_name(ws, name='airline-ws')
    df = dataset.to_pandas_dataframe()
    df = dataset.to_pandas_dataframe()
    
    x, y = clean_data(df)

    # Split data into train and test sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7,test_size=0.3, random_state=101)

    ### YOUR CODE HERE ###

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/model.pkl')

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))




if __name__ == '__main__':
    main()