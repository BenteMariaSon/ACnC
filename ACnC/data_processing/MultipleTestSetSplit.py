from sklearn.model_selection import BaseCrossValidator
import numpy as np 

class VenetianBlinds(BaseCrossValidator):
    """
    Venetian-blinds cross-validation splitter.
    Samples are assigned to folds by taking every `n_splits`-th sample.
    For n_splits=3:
        Fold 0: 0, 3, 6, 9, ...
        Fold 1: 1, 4, 7, 10, ...
        Fold 2: 2, 5, 8, 11, ...

    Parameters:
        - n_splits (int, default=5): Number of folds.
    """

    def __init__(self, n_splits=5):
        if not isinstance(n_splits, int):
            raise TypeError("n_splits must be an integer")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splitting iterations."""
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate train/test indices.
        Args:
            - X (array-like of shape (n_samples, ...)): Training data.
            - y (array-like, default=None): Ignored. Present for sklearn compatibility.
            - groups (array-like, default=None): Ignored. Present for sklearn compatibility.

        Returns:
            - train (ndarray): Training set indices.
            - test (ndarray): Test set indices.
        """
        n_samples = len(X)

        if self.n_splits > n_samples:
            raise ValueError(
                f"Cannot have n_splits={self.n_splits} greater than "
                f"the number of samples: {n_samples}"
            )

        indices = np.arange(n_samples)

        for fold in range(self.n_splits):
            test_indices = indices[fold::self.n_splits]
            train_indices = np.setdiff1d(indices, test_indices)

            yield train_indices, test_indices