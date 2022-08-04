# Airlines Dataset to predict a delay

## Overview
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.


## Summary
This dataset is from [Kaggle](https://www.kaggle.com/datasets/jimschacko/airlines-dataset-to-predict-a-delay?select=Airlines.csv). 
Airlines dataset has 539383 instances and 8 different features. The task is to predict whether a given flight will be delayed, given the information of the scheduled departure.

Different Feature Names

1.Airline
2.Flight
3.Airport From
4.Airport To
5.DayOfWeek
6.Time

For the classification, two approaches have been used:

- Scikit-learn **Logistic Regression** using a custom coded model where the hyperparameters are fine tuned using AzureML's **Hyperdrive**, which is able to achieve an accuracy of 0.908%
- Azure **AutoML** (the 'no code/low code' solution), which uses a variety of machine learning models to arrive at the best performing acccuracy of xx%, tad better than the Hyperdrive method, with the **VotingEnsemble** model.


## Initial steps :

1. Run **scripts\Workspace_create.ps1** in order to create a workspace called **airline-ws**.
<img src="/pictures/workspace.png" title="workspace"  width="700">
2. Create a dataset, called **airline-ds** and upload **data\Airlines.csv**
<img src="/pictures/dataset.png" title="dataset"  width="700">
3. Create a compute instance and open a Jupyter.
4. Upload Data.
5. Run the notebooks.


## Scikit-learn Pipeline



## AutoML