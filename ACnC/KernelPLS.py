
import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix

def rbf_kernel(X1, X2, sigma):
    """Calculate the RBF kernel matrix."""
    if sigma <= 0:
        raise ValueError("RBF sigma must be > 0.")
    
    X1_sq = np.sum(X1 ** 2, axis=1)[:, None]
    X2_sq = np.sum(X2 ** 2, axis=1)[None, :]
    D = X1_sq + X2_sq - 2 * X1 @ X2.T

    # Protect against small negative floating-point errors.
    D = np.maximum(D, 0)
    return np.exp(-D / (2 * sigma ** 2))



def polynomial_kernel(X1, X2, degree):
    """Calculate the polynomial kernel."""
    return (X1 @ X2.T + 1.0) ** degree



class KernelPLS(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    This class does Kernel PLS with cross-validation to determine the optimal number of components and optimal kernel parameters. 

    Parameters:
        - kernel (str, optional): which kernel type to use, currently implementations: "rbf" or "poly", defaults to "rbf"
        - kernel_param (array, optional): The parameter setting for kernel calculation (degree when kernel="polynomial" and sigma when kernel="RBF"), defaults to 1 for sigma if kernel="RBF" and 2 for degree if kernel="polynomial"  
        - n_components (int, optional): The number of Latent Variables to use (defaults to 5)
        - classification (bool, optional): Set to true when doing classification, can only do binary classification and classes must be encoded as -1 and 1
        
    Attributes:
        - X_fit: The X data used to fit the model
        - K_fit: The non-scaled Kernel obtained from X_fit and bestKernelPrm
        - K_sc_fit: The scaled Kernel obtained from X_fit and bestKernelPrm (scaled with H_fit)
        - model: The PLS fitted PLS model 

    Methods:
        - fit(): Find the optimal number of LVs and optimal kernel parameter and fit a PLS model with given data
        - transform(): find the kernel of new X data
        - fit_transform(): Combines fit and transform (computes the PLS model and returns the kernel)
        - predict_proba(): Predicts the probabilities of the y-data for a new X matrix
        - predict(): Predicts the y-data for a new X matrix
    """

    def __init__(
        self,
        kernel="rbf", # can be "rbf" or "poly"
        kernel_param=None,
        n_components=5,
        classification=False, # Only works for binary classes encoded as -1 and 1
    ):
        """Initialize the kernel PLS model
        Args:
            - kernel (str, optional): which kernel type to use, currently implementations: "RBF" or "polynomial", defaults to "RBF"
            - kernel_param (array, optional): The parameter setting for kernel calculation (degree when kernel="polynomial" and sigma when kernel="RBF"), defaults to 1 for sigma if kernel="RBF" and 2 for degree if kernel="polynomial"  
            - n_components (int, optional): The number of Latent Variables to use (defaults to 5)
            - classification (bool, optional): Set to true when doing classification, can only do binary classification and classes must be encoded as -1 and 1
        """
        self.kernel = kernel
        self.n_components = n_components
        self.classification = classification
        
        if kernel != 'rbf' and kernel != 'poly':
            raise ValueError("kernel must be 'rbf' or 'poly'")
        
        # Get the kernel parameters 
        if kernel_param is not None:
            self.kernel_param = kernel_param
        elif self.kernel == "rbf":
            self.kernel_param = 1
        elif self.kernel == "poly":
            self.kernel_param = 2


    def fit(self, X, y):
        """Fit the kernel PLS model with given X and y data and number of LVs / kernel parameter
        Args:
            - X (ndarray): Data matrix of shape (n_samples, n_features)
            - y (ndarray): Data matrix of shape (n_samples,)
        """
        self.is_fitted_ = True
        X = np.array(X, copy=True)
        y = np.array(y, copy=True)
        self.X_fit = X
        
        # Input check
        n_samples, n_features = X.shape
        if n_samples != y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        self.n_components = min(self.n_components, n_samples, n_features)
        
        # check classes
        if self.classification:
            self.classes_ = np.unique(y)
            if not np.array_equal(np.sort(self.classes_), np.array([-1, 1])):
                raise ValueError("KernelPLSDA_CV expects binary classes encoded as -1 and +1.")
        
        # fit the model
        self.model = PLSRegression(n_components=self.n_components, scale=False)
        if self.kernel == 'poly':
            self.K_fit = polynomial_kernel(X, X, self.kernel_param)
        elif self.kernel == 'rbf':
            self.K_fit = rbf_kernel(X, X, self.kernel_param)
        self.K_sc_fit = (self.K_fit - np.mean(self.K_fit, axis=0)[None,:] - np.mean(self.K_fit, axis=1)[:,None] + np.mean(self.K_fit))
        self.model.fit(self.K_sc_fit, y)
       
        return self
        
    def transform(self, X):
        """Transform new X data to a kernel which can be used as input into the PLS model
        Args:
            - X (ndarray): Data matrix of shape (n_samples, n_features)
        """
        X = np.array(X, copy=True)
        
        # Determine the kernel of the new X input
        if self.kernel == 'poly':
            K_new = polynomial_kernel(X, self.X_fit, self.kernel_param)
        elif self.kernel == 'rbf':
            K_new = rbf_kernel(X, self.X_fit, self.kernel_param)

        # Center this new kernel with the training set statistics 
        mean_fit = np.mean(self.K_fit)
        mean_cols_fit = np.mean(self.K_fit, axis=0)
        # mean_rows_fit = np.mean(self.K_fit, axis=1)
        mean_rows_fit = np.mean(K_new, axis=1)
        K_new_mc = K_new - mean_rows_fit[:, None] - mean_cols_fit[None,:] + mean_fit
        
        return K_new_mc
        
    def fit_transform(self, X, y):
        """Combines fit and transform (computes the kernel PLS model and returns the kernel)
        Args:
            - X (ndarray): Data matrix of shape (n_samples, n_features)
            - y (ndarray): Data matrix of shape (n_samples,) 
        """
        self.fit(X, y)
        return self.transform(X)
    
    def predict_proba(self, X):
        """Predict the probabilities of the y-data for a new X matrix. In case of regression (classification=False), predict() and predict_proba() return the same values
        Args:
            - X (ndarray): New data matrix of shape (n_new_samples, n_features)
        """
        K_new_mc = self.transform(X)
        return self.model.predict(K_new_mc).ravel()
        
    def predict(self, X):
        """Predict the y-data for a new X matrix
        Args:
            - X (ndarray): New data matrix of shape (n_new_samples, n_features)
        """
        K_new_mc = self.transform(X)
        
        y_pred = self.model.predict(K_new_mc).ravel()
        if self.classification:
            y_pred = np.sign(y_pred)
            # In case a prediction is exactly zero.
            y_pred[y_pred == 0] = 1
        return y_pred

class KernelPLS_CV(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    This class does Kernel PLS with cross-validation to determine the optimal number of components and optimal kernel parameters. 

    Parameters:
        - kernel (str, optional): which kernel type to use, currently implementations: "rbf" or "poly", defaults to "rbf"
        - kernel_params (array, optional): The parameter setting for the kernel for which to find the optimum, defaults to [0.1, 0.5, 1, 2, 5, 10] for sigma if kernel="RBF" and [1, 2, 3] for degree if kernel="polynomial"  
        - max_LV (int, optional): The maximum number of LVs allowed to be used by the model, defaults to 20
        - classification (bool, optional): Set to true when doing classification, can only do binary classification and classes must be encoded as -1 and 1
        - CV_scheme (sklearn CV iterator, optional): The sklearn CV iterator to use when splitting the data in the cross validation loop. Defaults to KFold(n_splits=5, shuffle=True, random_state=37)

    Attributes:
        - bestLV: The number of optimal LVs found and used to create the PLS model
        - bestKernelPrm: The optimal kernel parameter found and used to create the PLS model
        - bestError: The CV error found when using the best number of LVs and best kernel parameter
        - X_fit: The X data used to fit the model
        - K_fit: The non-scaled Kernel obtained from X_fit and bestKernelPrm
        - K_sc_fit: The scaled Kernel obtained from X_fit and bestKernelPrm (scaled with H_fit)
        - model: The PLS fitted PLS model 

    Methods:
        - fit(): Find the optimal number of LVs and optimal kernel parameter and fit a PLS model with given data
        - transform(): find the kernel of new X data
        - fit_transform(): Combines fit and transform (computes the PLS model and returns the kernel)
        - predict_proba(): Predicts the probabilities of the y-data for a new X matrix
        - predict(): Predicts the y-data for a new X matrix
    """

    def __init__(
        self,
        kernel="rbf", # can be "rbf" or "poly"
        kernel_params=None,
        max_LV=20,
        classification=False, # Only works for binary classes encoded as -1 and 1
        CV_scheme=KFold(n_splits=5, shuffle=True, random_state=37),
    ):
        """Initialize the kernel PLS model
        Args:
            - kernel (str, optional): which kernel type to use, currently implementations: "RBF" or "polynomial", defaults to "RBF"
            - kernel_params (array, optional): The parameter setting for the kernel for which to find the optimum, defaults to [0.1, 0.5, 1, 2, 5, 10] for sigma if kernel="RBF" and [1, 2, 3] for degree if kernel="polynomial"  
            - max_LV (int, optional): The maximum number of LVs allowed to be used by the model, defaults to 20
            - classification (bool, optional): Set to true when doing classification, can only do binary classification and classes must be encoded as -1 and 1
            - CV_scheme (sklearn CV iterator, optional): The sklearn CV iterator to use when splitting the data in the cross validation loop. Defaults to KFold(n_splits=5, shuffle=True, random_state=37)
        """
        self.kernel = kernel
        self.max_LV = max_LV
        self.classification = classification
        self.CV_scheme = CV_scheme
        
        # Get the kernel parameters 
        if kernel_params is not None:
            self.kernel_params = kernel_params
        elif self.kernel == "rbf":
            self.kernel_params = [0.1, 0.5, 1, 2, 5, 10]
        elif self.kernel == "ploy":
            self.kernel_params = [1, 2, 3]
        else:
            raise ValueError("kernel must be either 'rbf' or 'poly'.")


    def fit(self, X, y, print_results=False):
        """Fit the kernel PLS model with given X and y data and optimal number of LVs / kernel parameter
        Args:
            - X (ndarray): Data matrix of shape (n_samples, n_features)
            - y (ndarray): Data matrix of shape (n_samples,)
            - print_results (bool, optional): Whether to print the found number of LVS and best kernel parameter. Defaults to False
        """
        
        # Check if the inputted max_LV does not exceed mathematical limitations. 
        self.max_LV = min(self.max_LV, X.shape[0], X.shape[1])
        for train_idx, _ in self.CV_scheme.split(X, y): # ensure the number of LVs is never larger than the number of samples in training data. 
            self.max_LV = min(self.max_LV, X.shape[1], len(train_idx))
    
        # Initialize the loop
        self.bestScore = -np.inf
        self.bestKernelPrm = np.nan
        self.bestLV = np.nan
        # Try all possible kernel parameters
        for Kprm in self.kernel_params:
            # Try all possible number of latent variables 
            for nComp in range(1, self.max_LV+1):
                inner_score = []
                for innerTrainIdx, innerTestIdx in self.CV_scheme.split(X, y):
                    # Fit the model 
                    kernelPLS = KernelPLS(kernel=self.kernel, kernel_param=Kprm, n_components=nComp, classification=self.classification)
                    kernelPLS.fit(X[innerTrainIdx,:], y[innerTrainIdx])
                    # predict the test data
                    YpredInner = kernelPLS.predict(X[innerTestIdx,:])
                    if self.classification:
                        score = np.sum((YpredInner==y[innerTestIdx]))/len(y[innerTestIdx])
                    else:
                        score = -np.mean((YpredInner-y[innerTestIdx])**2)
                    inner_score.append(score)
                avg_inner_score = np.mean(inner_score)
                # keep the best combination of LVs and kernel parameter 
                if avg_inner_score > self.bestScore:
                    self.bestScore = avg_inner_score
                    self.bestKernelPrm = Kprm
                    self.bestLV = nComp
            
        # print results 
        if print_results:
            print(f"Best kernel parameter = {self.bestKernelPrm}")
            print(f"Best number of LVs = {int(self.bestLV)}")
            
        # fit the best model for continuation
        self.kernelPLS = KernelPLS(kernel=self.kernel, kernel_param=self.bestKernelPrm, n_components=self.bestLV, classification=self.classification)
        self.kernelPLS.fit(X, y)
       
        return self
        
    def transform(self, X):
        """Transform new X data to a kernel which can be used as input into the PLS model
        Args:
            - X (ndarray): Data matrix of shape (n_samples, n_features)
        """
        return self.kernelPLS.transform(X)
        
    def fit_transform(self, X, y):
        """Combines fit and transform (computes the kernel PLS model and returns the kernel)
        Args:
            - X (ndarray): Data matrix of shape (n_samples, n_features)
            - y (ndarray): Data matrix of shape (n_samples,) 
        """
        self.fit(X, y)
        return self.transform(X)
    
    def predict_proba(self, X):
        """Predict the probabilities of the y-data for a new X matrix. In case of regression (classification=False), predict() and predict_proba() return the same values
        Args:
            - X (ndarray): New data matrix of shape (n_new_samples, n_features)
        """
        return self.kernelPLS.predict_proba(X)
        
    def predict(self, X):
        """Predict the y-data for a new X matrix
        Args:
            - X (ndarray): New data matrix of shape (n_new_samples, n_features)
        """
        return self.kernelPLS.predict(X)