from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np

#===================================================================================================================================================#

def venitian_blinds_split(X, Y, n):
    """
    Split feature and target data using the Venetian Blinds method. 

    Parameters:
        - X: Dataframe containing the feature data (n_samples by n_features)
        - Y: Dataframe containing the target data (n_features)
        - n: integer, every nth row goes into the test set

    Returns: 
        - Xtrain, Xtest, Ytrain, Ytest: Dataframes containing the training and testing data
    """
    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of rows (same number of samples)")
    if (n <= 1) or (n > len(X)):
        raise ValueError("Parameter 'n' must be a positive integer of at least 2 and cannot be larger than the number of samples.")
    try:
        X = pd.DataFrame(X, copy=True)
        Y = pd.DataFrame(Y, copy=True)
    except:
        pass
    
    test_indices = X.index[(X.index+1) % n == 0]
    train_indices = X.index[(X.index+1) % n != 0]

    Xtrain = X.loc[train_indices].reset_index(drop=True)
    Ytrain = Y.loc[train_indices].reset_index(drop=True)
    Xtest = X.loc[test_indices].reset_index(drop=True)
    Ytest = Y.loc[test_indices].reset_index(drop=True)

    return np.asarray(Xtrain).squeeze(), np.asarray(Xtest).squeeze(), np.asarray(Ytrain).squeeze(), np.asarray(Ytest).squeeze()

#===================================================================================================================================================#

def random_split(X, y, test_size=0.2, stratify=False, shuffle=True, random_state=None):
    """
    Split feature and target data using a random split.

    Parameters:
        - X: Dataframe containing the feature data (n_samples by n_features)
        - y: Dataframe containing the target data (n_features)
        - test_size: float, should be between 0.0 and 1.0 and represents the fraction of samples in the test set, default is 0.2
        - stratify: bool, whether the data should be split in a stratified fashion, if set to True the distribution of classes in the original data will be kept the same in the train and test sets, default is False
        - shuffle: bool, whether to shuffle the data before splitting, if stratify is set to True, shuffle will also automatically be True, default is True
        - random_state: int, can be set to a integer value for a reproducable output, the random_state will be used as the seed for the random number generation. If set to None the split will not be reproducable, default is None
    
    Returns:
        - Xtrain, Xtest, Ytrain, Ytest: Dataframes containing the training and testing data
    """
    if stratify:
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=test_size, stratify=y, shuffle=True, random_state=random_state)
    else:
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, y, test_size=test_size, stratify=None, shuffle=shuffle, random_state=random_state)
    return Xtrain, Xtest, Ytrain, Ytest

#===================================================================================================================================================#

def last_n_samples_split(X, y, n):
    """
    Split feature and target data by putting the last n samples into the testing set. 

    Parameters:
        - X: Dataframe containing the feature data (n_samples by n_features)
        - y: Dataframe containing the target data (n_features)
        - n: int, the number of samples to put into the test set (the last n samples will be put in the test set)

    Returns:
        - Xtrain, Xtest, Ytrain, Ytest: Dataframes containing the training and testing data
    """
    if len(X) != len(y):
        raise ValueError("X and Y must have the same number of rows (same number of samples)")
    if (n < 1) or (n > len(X)):
        raise ValueError("Parameter 'n' must be a positive integer of at least 1 and cannot be larger than the number of samples.")
    
    try:
        X = pd.DataFrame(X, copy=True)
        Y = pd.DataFrame(Y, copy=True)
    except:
        pass

    Xtrain = X.iloc[:-n].reset_index(drop=True)
    Ytrain = y.iloc[:-n].reset_index(drop=True)
    Xtest = X.iloc[-n:].reset_index(drop=True)
    Ytest = y.iloc[-n:].reset_index(drop=True)

    return np.asarray(Xtrain).squeeze(), np.asarray(Xtest).squeeze(), np.asarray(Ytrain).squeeze(), np.asarray(Ytest).squeeze()