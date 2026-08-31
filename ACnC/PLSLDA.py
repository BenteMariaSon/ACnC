# Import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
from .Plotting.ComponentPlots import *
from .Plotting.ValidationPlots import *
from .data_processing.PreProcessing import MeanCentering
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

class PLSLDA(BaseEstimator):
    """
    An implementation of Partial Least Squares Discriminant Analysis (PLSDA), meancentering of the X data is always automatically done, but the user can choose wheter to perform autoscaling or not

    Parameters:
        - n_components (int, optional): The number of LVs to use when creating the PLS model. Defaults to 10
        - scale (bool, optional): Whether to autoscale the data or not (Mean centering is always done). Defaults to False
        - calculate_explained_var (bool, optional): Whether to calculate the explained variance of the X and y data. Defaults to True

    Attributes:
        - x_scores: Projected X data in the latent variable space, ndarray of shape (n_samples, n_components)
        - y_scores: Projected y data in the latent variable space, ndarray of shape (n_samples, n_components)
        - x_loadings: Loadings used to project the X data from the original space to the latent variable space, ndarray of shape (n_features, n_components)
        - y_loadings: Loadings used to project the y data from the original space to the latent variable space, ndarray of shape (n_targets, n_components)
        - x_weights: The left singular vectors of the cross-covariance matrices of each iteration, ndarray of shape (n_features, n_components)
        - y_weights: The right singular vectors of the cross-covariance matrices of each iteration, ndarray of shape (n_targets, n_components)
        - coef: The coefficients of the linear model such that y is approximated as y = X @ coef.T + intercept, ndarray of shape (n_targets, n_features)
        - intercept: The intercept of the linear model such that y is approximated as y = X @ coef.T + intercept, ndarray of shape (n_targets,)
        - explained_variance_ratio_X: Proportions of the X data's total variance that is captured by each latent variable, ndarray of shape (n_components,)
        - explained_variance_ratio_Y: Proportions of the y data's total variance that is captured by each latent variable, ndarray of shape (n_components,)

    Methods:
        - fit(): Fit a PLS model with given data
        - transform(): projects the given data onto the existing PLS space and returns the scores
        - fit_transform(): Combines fit and transform (computes the PLS model and returns the scores)
        - inverse_transform(): reconstructs the original data from the reduced PLS representation
        - predict(): Predicts the y-data for a new X matrix
        - predict_proba(): Predicts the probabilities of the y-data for a new X matrix
        - plot_scree(): Creates a scree plot for the PLS model for both the X and y data
        - plot_cumulative_explained_variance(): Creates a cumulative explained variance plot for both the X and y data
        - plot_scores(): Creates a scores plot for the PLS model
        - plot_biplot(): Creates a biplot for the PLS model
        - plot_loadings(): Makes a loadings line plot of the chosen components
        - plot_multiple_scores(): Plots score plot combination of the LVs against each other with the distribution of each LV on the diagonal. 
    """  

    def __init__(self, n_components=10, scale=False, calculate_explained_var=True):
        """Initialize the PLS model
        Args:
            - n_components (int, optional): The number of LVs to use when creating the PLS model. Defaults to 10
            - scale (bool, optional): Whether to autoscale the data or not (Mean centering is always done). Defaults to False
            - calculate_explained_var (bool, optional): Whether to calculate the explained variance of the X and y data. Defaults to True
        """
        self.n_components = n_components
        self.scale = scale
        self.calculate_explained_var = calculate_explained_var
        self.is_fitted_ = False

    def fit(self, X, y):
        """Fit a PLS model with given X and y data
        Args:
            - X (ndarray): Data matrix of shape (n_samples, n_features)
            - y (ndarray): Data matrix of shape (n_samples,). This will be converted to a dummy matrix internally
        """
        self.is_fitted_ = True
        X = np.asarray(X, copy=True)
        y_original = np.asarray(y, copy=True)

        n_samples, n_features = X.shape

        # If y is not yet a dummy matrix we want to transform it into one. 
        if y_original.ndim < 2 or y_original.shape[1]==1: # Check if there is a second dimention or if the second dimention only has a length of 1
            self.classes_, self.y_encoded = np.unique(y_original, return_inverse=True)
            y = np.eye(len(self.classes_))[self.y_encoded]

        # Input check
        if n_samples != y.shape[0]:
            raise ValueError("X and Y must have the same number of samples")
        self.n_components = min(self.n_components, n_samples, n_features)

        # Make the model
        self.pls = PLSRegression(n_components=self.n_components, scale=self.scale)
        self.lda = LinearDiscriminantAnalysis()
        self.pls.fit(X,y)
        X_scores = self.pls.transform(X)
        self.lda.fit(X_scores, y_original)

        self.x_weights = self.pls.x_weights_
        self.y_weights = self.pls.y_weights_
        self.x_loadings = self.pls.x_loadings_
        self.y_loadings = self.pls.y_loadings_
        self.x_scores = self.pls.x_scores_
        self.y_scores = self.pls.y_scores_

        if self.calculate_explained_var:
            # EXPLAINED VARIANCE CALCULATION
            # Total variance of original data 
            if self.scale:
                Xs = StandardScaler().fit_transform(X)
                Ys = StandardScaler().fit_transform(y)
            else:
                Xs = X
                Ys = y
            total_var_X = np.var(Xs, axis=0, ddof=1).sum()
            total_var_Y = np.var(Ys, axis=0, ddof=1).sum()
            
            self.explained_variance_ratio_X = np.zeros(self.n_components)
            self.explained_variance_ratio_y = np.zeros(self.n_components)
            for i in range(self.n_components):
                # reconstruct X and y
                x_rec1 = self.x_scores[:,i].reshape(-1,1) @ self.x_loadings[:,i].reshape(-1,1).T
                y_rec1 = self.x_scores[:,i].reshape(-1,1) @ self.y_loadings[:,i].reshape(-1,1).T
                # calculate explained variance ratio from variance of reconstruction and total variance
                self.explained_variance_ratio_X[i] = np.var(x_rec1, axis=0, ddof=1).sum() / total_var_X
                self.explained_variance_ratio_y[i] = np.var(y_rec1, axis=0, ddof=1).sum() / total_var_Y
        
        return self

    # def transform(self, X, y=None):
    #     """projects the given data (X) onto the existing PLS space and returns the scores
    #     Args:
    #         - X (ndarray): Data matrix of shape (n_samples, n_features)
    #     """
    #     if self.is_fitted_==False:
    #         raise ValueError("Model has not been fitted yet, call fit() first.")
        
    #     X = np.asarray(X, copy=True)
    #     return self.model.transform(X)

    # def fit_transform(self, X, y):
    #     """Combines fit and transform (computes the PLS model and returns the scores)
    #     Args:
    #         - X (ndarray): Data matrix of shape (n_samples, n_features)
    #         - y (ndarray): Data matrix of shape (n_samples,) or (n_samples, n_targets) if there is more than one target variable 
    #     """
    #     X = np.asarray(X, copy=True)
    #     y = np.asarray(y, copy=True)
    #     self.fit(X, y, scale=self.scale)
    #     return self.transform(X)

    # def inverse_transform(self, x_scores, y_scores):
    #     """reconstructs the original data from the reduced PLS representation (x_scores and x_scores)
    #     Args:
    #         - x_scores (ndarray): scores matrix of shape (n_samples, n_components) obtained by calling .transform or .fit_transform
    #         - y_scores (ndarray): scores matrix of shape (n_samples, n_components) obtained by calling .transform or .fit_transform
    #     """
    #     if self.is_fitted_==False:
    #         raise ValueError("Model has not been fitted yet, call fit() first.")
        
    #     return self.model.inverse_transform(x_scores, y_scores)

    def predict_proba(self, X):
        """Predict the probabilities of the y-data for a new X matrix
        Args:
            - X (ndarray): New data matrix of shape (n_new_samples, n_features)
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call fit() first.")
        
        X = np.asarray(X, copy=True)
        x_scores = self.pls.transform(X)
        return self.lda.predict_proba(x_scores)
    
    def predict(self, X):
        """Predict the y-data for a new X matrix
        Args:
            - X (ndarray): New data matrix of shape (n_new_samples, n_features)
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call fit() first.")
        
        X = np.asarray(X, copy=True)
        x_scores = self.pls.transform(X)
        return self.lda.predict(x_scores)

    # def plot_scree(self, n_components=None, color=rucolors.red):
    #     """Plots a scree plot to visualize the explained variance of each latent variable
    #     Args:
    #         - n_components (int, optional): The number of components to use in the plot, can be used when we require the plot to show less components than the full model, if set to None, the maximum number of components will be selected
    #         - color (str, optional): A string containing the hex-code of the desired color.
    #     """
    #     if self.is_fitted_==False:
    #         raise ValueError("Model has not been fitted yet, call fit() first.")
    #     elif self.explained_variance_ratio_X is None:
    #         raise ValueError("A scree plot can only be made when the argument 'calculate_explained_variance' is set to True at model initialization.")
        
    #     fig, axes = plt.subplots(1,2, figsize=(12,6))

    #     scree_plot(self.explained_variance_ratio_X, n_components=n_components, title="Explained variance of X", color=color, method='pls', ax_in=axes[0])
    #     scree_plot(self.explained_variance_ratio_y, n_components=n_components, title="Explained variance of y", color=color, method='pls', ax_in=axes[1])

    #     plt.show()

    # def plot_cumulative_explained_variance(self, n_components=None, color=rucolors.red):
    #     """Plots the cummulative explained variance across latent variables
    #     Args:
    #         - n_components (int, optional): The number of components to use in the plot, can be used when we require the plot to show less components than the full model, if set to None, the maximum number of components will be selected
    #         - color (str, optional): A string containing the hex-code of the desired color.
    #     """
    #     if self.is_fitted_==False:
    #         raise ValueError("Model has not been fitted yet, call fit() first.")
    #     elif self.explained_variance_ratio_X is None:
    #         raise ValueError("A cumulative explained variance plot can only be made when the argument 'calculate_explained_variance' is set to True at model initialization.")
        
    #     fig, axes = plt.subplots(1,2,figsize=(12,6))

    #     cumulative_explained_variance_plot(self.explained_variance_ratio_X, n_components=n_components, title="Cumulative explained variance of X", color=color, method='pls', ax_in=axes[0])
    #     cumulative_explained_variance_plot(self.explained_variance_ratio_y, n_components=n_components, title="Cumulative explained variance of y", color=color, method='pls', ax_in=axes[1])

    #     plt.show()

    # def plot_scores(self, reference=None, txt_labels=None, LV_x=1, LV_y=2, cmap=rucolors.secondary_colormap, overwrite_cmap=None, title=None, ref_label=None, marker_size=100, alpha=0.8):
    #     """Plots the PLS scores for selected Latent Variables (LV_x and LV_y). 
    #     Args:
    #         - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
    #         - txt_labels (1d array, optional): An array that contains the labels of each sample, if not None, the text labels will be plotted next to each point in the scores plot. Defaults to None
    #         - LV_x (int, optional): the number of the LV to plot on the x-axis. Defaults to 1
    #         - LV_y (int, optional): The number of the LV to plot on the y-axis. Defaults to 2
    #         - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
    #         - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
    #         - title (str, optional): A string containing the title of the plot. Defaults to None
    #         - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). Defaults to None
    #         - marker_size (int, optional): The size of the markers in the plot. Defaults to 100
    #         - alpha (float, optional): Transparancy of the markers where 1 is opague and 0 is fully transparant. Defaults to 0.8
    #     """
    #     if self.is_fitted_==False:
    #         raise ValueError("Model has not been fitted yet, call .fit() first.")
        
    #     scores_plot(self.x_scores, np.vstack((self.explained_variance_ratio_X, self.explained_variance_ratio_y)), reference=reference, regression=False, txt_labels=txt_labels, Comp_x=LV_x, Comp_y=LV_y, cmap=cmap, overwrite_cmap=overwrite_cmap, title=title, ref_label=ref_label, marker_size=marker_size, alpha=alpha, method='pls')

    # def plot_biplot(self, reference=None, feature_names=False, LV_x=1, LV_y=2, scale_arrows=1.0, cmap=rucolors.secondary_colormap, overwrite_cmap=None, title=None, ref_label=None, marker_size=100, alpha=0.8):
    #     """Plots a PLS biplot for the selected latent variables (LV_x and LV_y)
    #     Args:
    #         - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
    #         - feature_names (1d array, optional): An array or list containing the feature names. If None, the feature_names will be listed as numbers from 1 to n_features. If False, feature_names will not be listed. Defaults to False
    #         - LV_x (int, optional): the number of the LV to plot on the x-axis. Defaults to 1
    #         - LV_y (int, optional): The number of the LV to plot on the y-axis. Defaults to 2
    #         - scale_arrows (float, optional): The length of the arrows will be multiplied by this number to scale them to the desired length, defaults to 1.0
    #         - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). Defaults to None
    #         - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
    #         - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
    #         - title (str, optional): A string containing the title of the plot. Defaults to None
    #         - marker_size (int, optional): The size of the markers in the plot. Defaults to 100
    #         - alpha (float, optional): Transparancy of the markers where 1 is opague and 0 is fully transparant. Defaults to 0.8
    #     """ 
    #     if self.is_fitted_==False:
    #         raise ValueError("Model has not been fitted yet, call .fit() first.")
        
    #     biplot(self.x_loadings, self.x_scores, np.vstack((self.explained_variance_ratio_X, self.explained_variance_ratio_y)), reference=reference, regression=False, feature_names=feature_names, Comp_x=LV_x, Comp_y=LV_y, scale_arrows=scale_arrows, cmap=cmap, overwrite_cmap=overwrite_cmap, title=title, ref_label=ref_label, marker_size=marker_size, alpha=alpha, method='pls')

    # def plot_loadings(self, LVs=[1], xaxis=None, xlabel=None, ylabel=None, title=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None):
    #     """Plots the PLS loadings as a line plot.
    #     Args:
    #         - LVs (list, optional): list of the components numbers that should be plotted (e.g. [1,2,3] will plot the first 3 components). Defaults to [1] - only plotting the loadings of the first component
    #         - xaxis (str, optional): Grid on which data lies, e.g. wavelength. Defaults to None.
    #         - xlabel (str, optional): Describes x axis. Defaults to None.
    #         - ylabel (str, optional): Describes y axis. Defaults to None.
    #         - title (str, optional): title of the plot
    #         - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
    #         - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
    #     """
    #     if self.is_fitted_==False:
    #         raise ValueError("Model has not been fitted yet, call .fit() first.")
        
    #     loadings_plot(self.x_loadings, Comps=LVs, xaxis=xaxis, xlabel=xlabel, ylabel=ylabel, title=title, cmap=cmap, overwrite_cmap=overwrite_cmap, method='pls')

    # def plot_multiple_scores(self, n_components=None, reference=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None):
    #     """Plots the scores plots of all possible combinations of LVs of the first n_components. This method will plot the distribution of the scores within the LV on the diagonal. 
    #     Args: 
    #         - n_components (int, optional): The number of components to use in the plot, can be used when we require the plot to show less components than the full model, if set to None, the maximum number of components will be selected
    #         - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
    #         - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). Defaults to None
    #         - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
    #         - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
    #     """
    #     if self.is_fitted_==False:
    #         raise ValueError("Model has not been fitted yet, call .fit() first.")
        
    #     multiple_scoreplot(self.x_scores, np.vstack((self.explained_variance_ratio_X, self.explained_variance_ratio_y)), reference=reference, regression=False, n_components=n_components, cmap=cmap, overwrite_cmap=overwrite_cmap, method='pls')


class PLSLDA_CV(PLSLDA):
    """
    This class does PLS-DA with cross-validation to determine the optimal number of components on given data. 

    Parameters:
        - max_LV (int, optional): The maximum number of LVs allowed to be used by the model, defaults to 20
        - scale (bool, optional): Whether to autoscale the data or not (Mean centering is always done). Defaults to False
        - CV_scheme (sklearn CV iterator, optional): The sklearn CV iterator to use when splitting the data in the cross validation loop. Defaults to KFold(n_splits=5, shuffle=True, random_state=37)
        - n_jobs (int, optional): The number of jobs to use for running the code in parallel, when -1 the maximum number of jobs will be used, when 1 the code will not run in parallel. Defaults to None
        - enforce_nLVs (int, optional): When set to an integer, the crossvalidation will be skipped and a model with be made with the input number of LVs. if None, the number of LVs will be determined with cross validation. Defaults to None
        - enforce_min_2_LVs (bool, optional): When set to True the model will always use a minimum of 2 LVs (1 LV models will be impossible), This helps when wanting to make 2-dimentional plots. Defaults to False

    Attributes:
        - opt_LV: The number of optimal LVs found and used to create the PLS model
        - AccuracyCs: The Accuracy of Calibration for each number of LVs (max_LV,)
        - AccuracyCVs: The Accuracy of Cross Validation for each number of LVs (max_LV,)
        - RMSECs: The Root Mean Square Error of Calibration for each number of LVs (max_LV,)
        - RMSECVs: The Root Mean Square Error of Cross Validation for each number of LVs (max_LV,)
        - x_scores: Projected X data in the latent variable space, ndarray of shape (n_samples, n_components)
        - y_scores: Projected y data in the latent variable space, ndarray of shape (n_samples, n_components)
        - x_loadings: Loadings used to project the X data from the original space to the latent variable space, ndarray of shape (n_features, n_components)
        - y_loadings: Loadings used to project the y data from the original space to the latent variable space, ndarray of shape (n_targets, n_components)
        - x_weights: The left singular vectors of the cross-covariance matrices of each iteration, ndarray of shape (n_features, n_components)
        - y_weights: The right singular vectors of the cross-covariance matrices of each iteration, ndarray of shape (n_targets, n_components)
        - coef: The coefficients of the linear model such that y is approximated as y = X @ coef.T + intercept, ndarray of shape (n_targets, n_features)
        - intercept: The intercept of the linear model such that y is approximated as y = X @ coef.T + intercept, ndarray of shape (n_targets,)
        - explained_variance_ratio_X: Proportions of the X data's total variance that is captured by each latent variable, ndarray of shape (n_components,)
        - explained_variance_ratio_Y: Proportions of the y data's total variance that is captured by each latent variable, ndarray of shape (n_components,)


    Methods:
        - fit(): Fit a PLS model with given data
        - transform(): projects the given data onto the existing PLS space and returns the scores
        - fit_transform(): Combines fit and transform (computes the PLS model and returns the scores)
        - inverse_transform(): reconstructs the original data from the reduced PLS representation
        - predict_proba(): Predicts the probabilities of the y-data for a new X matrix
        - predict(): Predicts the y-data for a new X matrix
        - plot_scree(): Creates a scree plot for the PLS model for both the X and y data
        - plot_cumulative_explained_variance(): Creates a cumulative explained variance plot for both the X and y data
        - plot_scores(): Creates a scores plot for the PLS model
        - plot_biplot(): Creates a biplot for the PLS model
        - plot_loadings(): Makes a loadings line plot of the chosen components
        - plot_multiple_scores(): Plots score plot combination of the LVs against each other with the distribution of each LV on the diagonal. 
        - plot_Accuracies_vs_components(): Creates a lineplot of AccuracyC and AccuracyCV vs the number of components, the optimal number of components is indecated with a vertical line. 
        - plot_RMSEs_vs_components(): Creates a lineplot of RMSEC and RMSECV vs the number of components, the optimal number of components is indecated with a vertical line. 
        - double_CV(): Run a double cross-validation loop to validate the model with fully independent test samples and return the predicted values with the RMSEP
        - plot_double_CV_PredVSRef(): create a predicted vs reference plot with the fully independent test sample performance
        - plot_double_CV_ConfusionMatrix(): create a confusion matrix with the fully independent test sample performance 
    """

    def __init__(self, max_LV=30, scale=False, CV_scheme=KFold(n_splits=5, shuffle=True, random_state=37), n_jobs=1, enforce_nLVs=None, enforce_min_2_LVs=False):
        """Initialize the PLS-DA model
        Args:
            - max_LV (int, optional): The maximum number of LVs allowed to be used by the model, defaults to 20
            - scale (bool, optional): Whether to autoscale the data or not (Mean centering is always done). Defaults to False
            - CV_scheme (sklearn CV iterator, optional): The sklearn CV iterator to use when splitting the data in the cross validation loop. Defaults to KFold(n_splits=5, shuffle=True, random_state=37)
            - n_jobs (int, optional): The number of jobs to use for running the code in parallel, when -1 the maximum number of jobs will be used, when 1 the code will not run in parallel. Defaults to None
            - enforce_nLVs (int, optional): When set to an integer, the crossvalidation will be skipped and a model with be made with the input number of LVs. if None, the number of LVs will be determined with cross validation. Defaults to None
            - enforce_min_2_LVs (bool, optional): When set to True the model will always use a minimum of 2 LVs (1 LV models will be impossible), This helps when wanting to make 2-dimentional plots. Defaults to False
        """
        self.max_LV = max_LV
        self.scale = scale
        self.CV_scheme = CV_scheme
        self.n_jobs = n_jobs
        self.enforce_nLVs = enforce_nLVs
        self.enforce_min_2_LVs = enforce_min_2_LVs
        self.is_fitted_ = False

        self.y_pred_P = None

    def fit(self, X, y, groups=None, print_results=False):
        """Fit a PLS model with given X and y data
        Args:
            - X (ndarray): Data matrix of shape (n_samples, n_features)
            - y (ndarray): Data matrix of shape (n_samples,) or (n_samples, n_targets) if there is more than one target variable 
            - print_results (bool, optional): Whether to print the found number of LVS with the corresponding accuracy and RMSE values. Defaults to False
        """
        self.is_fitted_ = True
        X = np.asarray(X, copy=True)
        y = np.asarray(y, copy=True)

        if y.ndim < 2 or y.shape[1]==1: # Check if there is a second dimention or if the second dimention only has a length of 1
            self.classes_, self.y_encoded = np.unique(y, return_inverse=True)
            y_dummy = np.eye(len(self.classes_))[self.y_encoded]
        else:
            y_dummy = y

        self.AccuracyCs = []
        self.AccuracyCVs = []
        self.RMSECs = []
        self.RMSECVs = []

        if self.enforce_nLVs is None:
            if self.max_LV is None:
                self.max_LV = min(X.shape[0], X.shape[1])
            else:
                self.max_LV = min(self.max_LV, X.shape[0], X.shape[1])
            for train_idx, _ in self.CV_scheme.split(X, y, groups=groups): # ensure the number of LVs is never larger than the number of samples in training data. 
                self.max_LV = min(self.max_LV, X.shape[1], len(train_idx))

            for i in range(0, self.max_LV):
                model = PLSLDA(n_components=i+1, scale=self.scale, calculate_explained_var=False)
                
                model.fit(X, y)
                y_cal = model.predict(X)
                y_CV = cross_val_predict(model, X, y, cv=self.CV_scheme, groups=groups)
                y_cal_proba = model.predict_proba(X)
                y_CV_proba = cross_val_predict(model, X, y, cv=self.CV_scheme, method='predict_proba', groups=groups)

                self.AccuracyCs.append(accuracy_score(y, y_cal))
                self.AccuracyCVs.append(accuracy_score(y, y_CV))
                self.RMSECs.append(root_mean_squared_error(y_dummy, y_cal_proba))
                self.RMSECVs.append(root_mean_squared_error(y_dummy, y_CV_proba))

                # if i>2 and abs(self.AccuracyCs[-2] - self.AccuracyCVs[-1]) < self.epsilon:
                #     break 
        
            # Find the optimal number of LVs
            # self.opt_LV = np.argmin(self.RMSECVs)+1 # lower RMSE is better
            self.opt_LV = np.argmax(self.AccuracyCVs)+1 # higher accuracy is better
        else: 
            self.opt_LV = self.enforce_nLVs

        if self.opt_LV==1 and self.enforce_min_2_LVs: 
            self.opt_LV = np.argmax(self.AccuracyCVs[1:])+2

        # Create the optimal model
        self.model = PLSLDA(n_components=self.opt_LV, scale=self.scale, calculate_explained_var=True)
        self.model.fit(X,y)
        self.n_components = self.opt_LV
        self.x_weights = self.model.x_weights
        self.y_weights = self.model.y_weights
        self.x_loadings = self.model.x_loadings
        self.y_loadings = self.model.y_loadings
        self.x_scores = self.model.x_scores
        self.y_scores = self.model.y_scores
        self.explained_variance_ratio_X = self.model.explained_variance_ratio_X
        self.explained_variance_ratio_y = self.model.explained_variance_ratio_y

        if print_results and self.enforce_nLVs is None:
            print(f"Best number of Latent Variables: {self.n_components}")
            print(f"RMSECV at {self.n_components} LVs: {self.RMSECVs[self.n_components-1]:.3f}")
            print(f"Accuracy of CV at {self.n_components} LVs: {self.AccuracyCVs[self.n_components-1]:.3f}")
            print(f"RMSEC at {self.n_components} LVs: {self.RMSECs[self.n_components-1]:.3f}")
            print(f"Accuracy of C at {self.n_components} LVs: {self.AccuracyCs[self.n_components-1]:.3f}")

        return self

    # def transform(self, X, y=None):
    #     """projects the given data (X) onto the existing PLS space and returns the scores
    #     Args:
    #         - X (ndarray): Data matrix of shape (n_samples, n_features)
    #     """
    #     if self.is_fitted_==False:
    #         raise ValueError("Model has not been fitted yet, call fit() first.")
        
    #     X = np.asarray(X, copy=True)
    #     return self.model.transform(X)

    # def fit_transform(self, X, y):
    #     """Combines fit and transform (computes the PLS model and returns the scores)
    #     Args:
    #         - X (ndarray): Data matrix of shape (n_samples, n_features)
    #         - y (ndarray): Data matrix of shape (n_samples,) or (n_samples, n_targets) if there is more than one target variable 
    #     """
    #     X = np.asarray(X, copy=True)
    #     y = np.asarray(y, copy=True)
    #     self.fit(X, y, scale=self.scale)
    #     return self.transform(X)

    # def inverse_transform(self, x_scores, y_scores):
    #     """reconstructs the original data from the reduced PLS representation (x_scores and x_scores)
    #     Args:
    #         - x_scores (ndarray): scores matrix of shape (n_samples, n_components) obtained by calling .transform or .fit_transform
    #         - y_scores (ndarray): scores matrix of shape (n_samples, n_components) obtained by calling .transform or .fit_transform
    #     """
    #     if self.is_fitted_==False:
    #         raise ValueError("Model has not been fitted yet, call fit() first.")
        
    #     return self.model.inverse_transform(x_scores, y_scores)

    def predict_proba(self, X):
        """Predict the probabilities of the y-data for a new X matrix
        Args:
            - X (ndarray): New data matrix of shape (n_new_samples, n_features)
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call fit() first.")
        
        X = np.asarray(X, copy=True)
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict the y-data for a new X matrix
        Args:
            - X (ndarray): New data matrix of shape (n_new_samples, n_features)
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call fit() first.")
        
        X = np.asarray(X, copy=True)
        return self.model.predict(X)
    
    def plot_Accuracies_vs_components(self, title=None):
        """Creates a lineplot of the AccuracyC (accuracy of calibration) and AccuracyCV (accuracy of cross validation) values vs the number of components in the model.
        The selected number of components for the optimal model is indicaded with a vertical dotted line. 
        Args:
            - title: (str, optional), string containing the desired title of the plot
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call fit() first.")
        
        if self.enforce_nLVs is not None:
            raise ValueError("It is not possible to make a Accuracy vs Components plot when you manually enforce a number of LVs through the enforce_nLVs attribute")

        if len(self.AccuracyCs) != len(self.AccuracyCVs):
            raise ValueError("both RMSE arrays must be of the same length (same number of components tested)")
        
        component_range = np.arange(1, len(self.AccuracyCs)+1)
        plt.figure(figsize=(6, 4))
        plt.plot(component_range, self.AccuracyCs, label='AccuracyC', marker='o', color=rucolors.secondary_colors["Blue"], zorder=1)
        plt.plot(component_range, self.AccuracyCVs, label='AccuracyCV', marker='o', color=rucolors.red, zorder=1)
        plt.axvline(self.opt_LV, color="k", linestyle=":", zorder=0)

        plt.xlabel("Number of components")
        plt.ylabel("Accuracy")
        plt.title(title if title else "Accuracy vs number of components plot")
        # plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_RMSEs_vs_components(self, title=None):
        """Creates a lineplot of the RMSEC (RMSE of calibration) and RMSECV (RMSE of cross validation) values vs the number of components in the model.
        The selected number of components for the optimal model is indicaded with a vertical dotted line. 
        Args:
            - title: (str, optional), string containing the desired title of the plot
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call fit() first.")
        
        if self.enforce_nLVs is not None:
            raise ValueError("It is not possible to make a RMSE vs Components plot when you manually enforce a number of LVs through the enforce_nLVs attribute")

        if len(self.RMSECs) != len(self.RMSECVs):
            raise ValueError("both RMSE arrays must be of the same length (same number of components tested)")
        
        component_range = np.arange(1, len(self.RMSECs)+1)
        plt.figure(figsize=(6, 4))
        plt.plot(component_range, self.RMSECs, label='RMSEC', marker='o', color=rucolors.secondary_colors["Blue"], zorder=1)
        plt.plot(component_range, self.RMSECVs, label='RMSECV', marker='o', color=rucolors.red, zorder=1)
        plt.axvline(self.opt_LV, color="k", linestyle=":", zorder=0)

        plt.xlabel("Number of components")
        plt.ylabel("RMSE")
        plt.title(title if title else "RMSE vs number of components plot")
        # plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    def double_CV(self, X, y, CV_scheme=KFold(n_splits=5, shuffle=True, random_state=37), groups=None, pp_pipe=None, print_results=False):
        """Run a double cross-validation loop to validate the model with fully indepentent test samples
        Args:
            - X (ndarray): Data matrix of shape (n_samples, n_features)
            - y (ndarray): Data matrix of shape (n_samples,) or (n_samples, n_targets) if there is more than one target variable
            - CV_scheme (sklearn CV iterator, optional): The sklearn CV iterator to use when splitting the data in the cross validation loop. Defaults to KFold(n_splits=5, shuffle=True, random_state=37)
            - groups (ndarray, optional): when the used CV_scheme takes groups as an input they can be passed here
            - pp_pipe (sklearn pipeline, optional): A sklearn pipeline with desired preprocessing methods can be passed here to check the performance of preprocessed data
            - print_results (bool, optional): whether to print the found number of LVS with the corresponding RMSEC and RMSECV values. Defaults to False
        Returns:
            - y_pred_P (ndarray): the independent test set predictions of each sample
            - AccuracyP (float): the accuracy of Prediction
        """
        X = np.asarray(X, copy=True)
        y = np.asarray(y, copy=True)

        if pp_pipe is None:
            pp_pipe = Pipeline([('mc', MeanCentering())])

        self.y_pred_P = np.zeros_like(y, dtype=object)
        self.y_pred_P_proba = np.zeros((len(y), len(self.classes_)), dtype=float)

        for train, test in CV_scheme.split(X, y, groups=groups):
            X_train_pp = np.asarray(pp_pipe.fit_transform(X[train,:]))
            X_test_pp = np.asarray(pp_pipe.transform(X[test,:]))
            self.fit(X_train_pp, y[train], print_results=print_results, groups=(groups[train] if groups is not None else None))
            self.y_pred_P[test] = self.predict(X_test_pp)
            self.y_pred_P_proba[test] = self.predict_proba(np.asarray(pp_pipe.transform(X[test,:])))
        
        self.AccuracyP = accuracy_score(y, self.y_pred_P)

        # Restore the model to again use all the data 
        self.fit(pp_pipe.fit_transform(X), y, groups=groups, print_results=print_results)

        return self.y_pred_P, self.AccuracyP


    def plot_double_CV_PredVSRef(self, y_true, x_label=None, y_label=None, color=rucolors.red):
        """create a predicted vs reference plot with the fully independent test sample performance
        Args:
            - y_true (ndarray): The reference y-values
            - x_label (str, optional): Describes x axis. 
            - y_label (str, optional): Describes y axis. 
            - color (str, optional): A string containing the hex-code of the desired color.
        """
        if self.y_pred_P_proba is None:
            raise ValueError("Double cross-validation has not been done yet, call .double_CV() first.")
        
        y_true = np.asarray(y_true, copy=True)
        n_classes = len(self.classes_)
        if y_true.ndim < 2 or y_true.shape[1]==1:
            y_dummy = (np.asarray(y_true)[:,None]==np.asarray(self.classes_)).astype(int)
        else:
            y_dummy = y_true

        fig, axes = plt.subplots(1, n_classes, figsize=(12,4))
        for i in range(n_classes):
            predvsref_plot(self.y_pred_P_proba[:,i], y_dummy[:,i], xlabel=x_label, ylabel=y_label, title=f"{self.classes_[i]} predictions", color=color, ax_in=axes[i])
        plt.show()


    def plot_double_CV_ConfusionMatrix(self, y_true, title=None, cmap=rucolors.RU_colormap):
        """create a confusion matrix with the fully independent test sample performance
        Args:
            - y_true (ndarray): The reference y-values
            - title (str, optional): title of the plot.
            - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.RU_colormap (this is a custom colormap)
        """
        if self.y_pred_P is None:
            raise ValueError("Double cross-validation has not been done yet, call .double_CV() first.")
        
        y_true = np.asarray(y_true, copy=True)
        confusion_matrix_plot(self.y_pred_P, y_true, labels=self.classes_, title=title, cmap=cmap)
