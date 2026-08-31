import numpy as np
from .data_processing.PreProcessing import MeanCentering
from sklearn.preprocessing import StandardScaler
from scipy.stats import f

def VIP(t, q, w): 
    """Calculate and return the Variable Importance Projection (VIP) scores of the fitted PLS model. 
    Implemented as discribed in Mehmood, et al. (2012). A review of variable selection methods in Partial Least Squares Regression 
    
    Args:
        - t (ndarray of shape (n_samples, n_components)): the X-scores of a fitted PLS model 
        - q (ndarray of shape (n_targets, n_components)): the y-loadings of a fitted PLS model 
        - w (ndarray of shape (n_features, n_components)): the X-weigths of a fitted PLS model 

    Returns:
        - VIP (ndarray): The Variable Importance Projection values for each feature
    """
    n_features, n_components = w.shape
    
    s = np.diag(t.T @ t @ q.T @ q) # sum of squares explained by each component (SS_a in article)
    total_s = np.sum(s)
    
    vip = np.zeros(n_features)
    for i in range(n_features):
        weight = np.array([(w[i,a]**2)/np.sum(w[:,a]**2) for a in range(n_components)])
        vip[i] = np.sqrt(n_features * np.sum(s * weight) / total_s)
        
    return vip
    
def sMC(B, X, scale=False):
    """Calculate and return the Significance Multivariate Correlation (sMC) values of the fitted PLS model. 
    Implemented as discribed in Tran, et al (2014) Interpretation of variable importance in Partial Least Squares with Significance Multivariate Correlation (sMC).
    
    Args:
        - B (ndarray of shape (n_targets, n_features)): The coefficients of a fitted PLS model
        - X (ndarray of shape (n_samples, n_features)): The data matrix that was used for fitting the PLS model from which B was obtained
        - Scale (bool): Whether autoscaling was applied within the PLS model (meancentering is assumed to always be applied)
    
    Returns:
        - sMC (ndarray): Significance Multivariate Correlation f-values for each feature
        - sMC_p (ndarray): sMC transformed to p-values
    """
    if scale:
        X = np.asarray(StandardScaler().fit_transform(X))
    else:
        X = np.asarray(MeanCentering().fit_transform(X))
    n_samples, n_features = X.shape
    
    B = np.asarray(B) # shape (n_target, n_features)
    # if B is only one dimension, we convert it to two dimensional
    if B.ndim == 1:
        B = B.reshape(1, n_features) 
    n_targets = B.shape[0]
    
    smc_values = np.zeros((n_features, n_targets))
    p_values = np.zeros((n_features, n_targets))
    for target in range(n_targets): # if PLS2 we run this loop multiple times, for PLS1 just once
        b = B[target, :] # shapeL (n_features)
        
        y_hat = X @ b # predicted y vector (eq 15) shape: (n_samples)
        X_hat = np.outer(y_hat, b) / np.linalg.norm(b)**2 # predicted X (eq 16) shape (n_samples, n_features)
        resid = X - X_hat # (eq 16) shape (n_samples, n_features)
        
        SS_model = np.sum(X_hat**2, axis=0) # eq 18 shape: (n_features,) 
        SS_resid = np.sum(resid**2, axis=0) # eq 19 shape: (n_features,)
        
        smc_values[:, target] = (SS_model / (SS_resid/(n_samples-2))) # eq 22 shape: (n_features,)
        p_values[:, target] = (1 - f.cdf(smc_values[:,target], 1, n_samples-2)) # f-test conversion to p-values
                    
    return np.squeeze(smc_values), np.squeeze(p_values) # use squeeze to remove added dimensions
