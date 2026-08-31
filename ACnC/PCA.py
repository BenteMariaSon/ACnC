# Import relevant libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.base import BaseEstimator
from .Plotting import rucolors
from .Plotting.ComponentPlots import *
# comment

def mean_centering(X):
    """
    Function to perform mean centering on the input data X.

    Parameters:
       -  X: The input data to be mean-centered (array-like or pandas DataFrame) 
    Returns:
        - pandas DataFrame: The mean-centered data.
    """
    try:
        X = pd.DataFrame(X)
    except Exception as e:
        raise ValueError(f"Unable to convert input to DataFrame: {e}")
    
    column_means = X.mean(axis=0)
    centered_data = X - column_means
    return centered_data

#===================================================================================================================================================#

class PCA(BaseEstimator):
    """
    An implementation of Principal Component Analysis (PCA) with built in plotting methods. The user can choose to do mean-centering, autoscaling or neither.  

    Parameters:
        - n_components (int or None, optional): The number of PCs to use when creating the PCA model. If None, the maximum number of components is used, Defaults to None

    Attributes:
        - scores_: Projected data in the principal component space, ndarray of shape (n_samples, n_components)
        - loadings_: PCA loadings used to project data from original space to principal component space, ndarray of shape (n_features, n_components)
        - explained_variance_ratio_: Proportion of the dataset's total variance that is captured by each component, ndarray of shape (n_components,) 
        - T2_: The Hotellings T^2 value of each sample, ndarray of shape (n_samples,)
        - Q: The Q residuals of each sample, ndarray of shape (n_samples,)

    Methods:
        - fit(): Fit the PCA model with given data
        - transform(): projects the given data (X) onto the existing PCA space and returns the scores
        - fit_transform(): Combines fit and transform (computes the PCA model and returns the scores)
        - inverse_transform(): Reconstructs the original data from the reduced PCA representation
        - plot_scree(): creates a scree plot for the PCA model
        - plot_cumulative_explained_variance(): plots the cumulative explained variance against the number of components
        - plot_scores(): Creates a scores plot for the PCA model
        - plot_biplot(): Creates a biplot for the PCA model
        - plot_loadings(): Makes a loadings line plot of the chosen components
        - plot_influence(): Plots the influence plot of the Q-residuals and Hotelling's T^2
        - plot_multiple_scores(): Plots score plot combinations of PCs against each other with the distribution of each PC on the diagonal. 
    """    

    def __init__(self, n_components=None):
        """Initialize the PCA model
        Args:
            - n_components (int or None, optional): The number of PCs to use when creating the PCA model. If None, the maximum number of components is used, Defaults to None
        """
        self.n_components_ = n_components
        self.loadings_ = []
        self.scores_ = []
        self.explained_variance_ratio_ = []
        self.is_fitted_ = False

    def fit(self, X, y=None, mean_center=True, scale=False):
        """Fit the PCA model with the given X data
        Args:
            - X (ndarray): Data matrix of shape (n_samples, n_features)
            - mean_center (bool, optional): Wheter to meancenter the data (True) or not (False). Defaults to True
            - scale (bool, optional): Wheter to scale the data (True) or not (False). If scale is set to True, mean_center will also always be True. Defaults to False
            
        """
        self.is_fitted_ = True
        X = np.asarray(X, copy=True)
        self.n_samples = X.shape[0]
        self.n_vars = X.shape[1]

        # set n_components
        if self.n_components_ is None:
            self.n_components_ = np.min([self.n_samples, self.n_vars])
        else: # If n_components is too high we want to set it to the max possible number of components
            self.n_components_ = np.min([self.n_samples, self.n_vars, self.n_components_])

        # mean center and scale
        if scale:
            self.means = np.mean(X, axis=0)
            self.stds = np.std(X, axis=0)
        elif mean_center:
            self.means = np.mean(X, axis=0)
            self.stds = np.ones(self.n_vars)
        else:
            self.means = np.zeros(self.n_vars)
            self.stds = np.ones(self.n_vars)

        X = (X - self.means) / self.stds

        # Decompose the data 
        U, D, V = np.linalg.svd(X, full_matrices=False)
        PC_var = D**2 / np.sum(D**2)
        U = U[:, :self.n_components_]
        self.D = D[:self.n_components_]
        self.loadings_ = V.T[:, :self.n_components_]
        self.explained_variance_ratio_ = PC_var[:self.n_components_]
        self.scores_ = U @ np.diag(self.D)

        self.errors_ = X - self.scores_@(self.loadings_.T)
        self.T2_ = np.sum((self.scores_ - np.mean(self.scores_, axis=0))**2 / (self.D**2 / self.n_samples), axis=1)
        self.Q_ = np.sum(self.errors_**2, axis=1)

    def transform(self, X, y=None, returnT2andQ=False):
        """Transforms a new datamatrix (X) into the PCA space and returns the scores
        Args:
            - X (ndarray): Data to fit into the PCA model
            - returnT2andQ (bool, optional): Whether the T2 and Q values of the newly transformed data should be returned. Defaults to False
        """
        X = np.asarray(X, copy=True)

        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first.")
        
        scores = ((X-self.means)/self.stds) @ self.loadings_
        if returnT2andQ:
            T2 = np.sum((scores - np.mean(scores, axis=0))**2 / (self.D**2 / self.n_samples), axis=1)
            Q = np.sum((X - scores@self.loadings_.T)**2, axis=1)  
            return scores, T2, Q
        else:
            return scores
    
    def fit_transform(self, X, y=None, mean_center=True, scale=False, returnT2andQ=False):
        """Fit a PCA model and return the scores of the data fitted to it. 
        Args:
            - X (ndarray): Data to fit into the PCA model
            - mean_center (bool, optional): Wheter to meancenter the data (True) or not (False). Defaults to True
            - scale (bool, optional): Wheter to scale the data (True) or not (False). If scale is set to True, mean_center will also always be True. Defaults to False
            - returnT2andQ (bool, optional): Whether the T2 and Q values of the newly transformed data should be returned    
        """
        X = np.asarray(X, copy=True)
        
        self.fit(X, y, mean_center, scale)
        return self.transform(X, y, returnT2andQ)
    
    def inverse_transform(self, scores):
        """Take some scores and back-tranform them to the original X data
        Args:
            - scores (ndarray): scores matrix of shape (n_samples x n_components) obtained by calling .transform or .fit_transform
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first.")
        
        x = scores @ self.loadings_.T
        return x*self.stds + self.means

    def plot_scree(self, n_components=None, title=None, color=rucolors.red):
        """Plots a scree plot to visualize the explained variance of each principal component
        Args:
            - n_components (int, optional): The number of components to use in the plot, can be used when we require the plot to show less components than the full model, if set to None, the maximum number of components will be selected
            - title (str, optional): A string containing the title of the plot. Defaults to None.
            - color (str, optional): A string containing the hex-code of the desired color.
        """

        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first.")

        scree_plot(self.explained_variance_ratio_, n_components=n_components, title=title, color=color, method='pca')
        

    def plot_cummulative_explained_variance(self, n_components=None, title=None, color=rucolors.red):
        """Plots the cummulative explained variance across principal components
        Args:
            - n_components (int, optional): The number of components to use in the plot, can be used when we require the plot to show less components than the full model, if set to None, the maximum number of components will be selected
            - title (str, optional): A string containing the title of the plot. Defaults to None.
            - color (str, optional): A string containing the hex-code of the desired color.
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first.")
        
        cumulative_explained_variance_plot(self.explained_variance_ratio_, n_components=n_components, title=title, color=color, method='pca')


    def plot_scores(self, reference=None, regression=False, txt_labels=None, PC_x=1, PC_y=2, cmap=rucolors.secondary_colormap, overwrite_cmap=None, title=None, ref_label=None, marker_size=100, alpha=0.8):
        """Plots the PCA scores for selected principal components (PC_x and PC_y).
        
        Args:
            - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
            - regression (bool, optional): When the reference data is continuous this must be set to True, if the reference data is categorical it must be set to False. Defaults to False
            - txt_labels (1d array, optional): An array that contains the labels of each sample, if not None, the text labels will be plotted next to each point in the scores plot. Defaults to None
            - PC_x (int, optional): the number of the PC to plot on the x-axis. Defaults to 1
            - PC_y (int, optional): The number of the PC to plot on the y-axis. Defaults to 2
            - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
            - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
            - title (str, optional): A string containing the title of the plot. Defaults to None
            - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). Defaults to None
            - marker_size (int, optional): The size of the markers in the plot. Defaults to 100
            - alpha (float, optional): Transparancy of the markers where 1 is opague and 0 is fully transparant. Defaults to 0.8
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first.")
        
        scores_plot(self.scores_, self.explained_variance_ratio_, reference=reference, regression=regression, txt_labels=txt_labels, Comp_x=PC_x, Comp_y=PC_y, cmap=cmap, overwrite_cmap=overwrite_cmap, title=title, ref_label=ref_label, marker_size=marker_size, alpha=alpha, method='pca')



    def plot_biplot(self, reference=None, regression=False, feature_names=False, PC_x=1, PC_y=2, scale_arrows=1.0, ref_label=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None, title=None ):
        """Plots a PCA biplot for the selected principal components (PC_x and PC_y).
        
        Args:
            - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
            - regression (bool, optional): When the reference data is continuous this must be set to True, if the reference data is categorical it must be set to False. Defaults to False
            - feature_names (1d array, optional): An array or list containing the feature names. If None, the feature_names will be listed as numbers from 1 to n_features. If False, feature_names will not be listed. Defaults to False
            - PC_x (int, optional): the number of the PC to plot on the x-axis. Defaults to 1
            - PC_y (int, optional): The number of the PC to plot on the y-axis. Defaults to 2
            - scale_arrows (float, optional): The length of the arrows will be multiplied by this number to scale them to the desired length, defaults to 1.0
            - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). Defaults to None
            - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
            - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
            - title (str, optional): A string containing the title of the plot. Defaults to None
        """ 
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first.")
        
        biplot(self.loadings_, self.scores_, self.explained_variance_ratio_, reference=reference, regression=regression, feature_names=feature_names, Comp_x=PC_x, Comp_y=PC_y, scale_arrows=scale_arrows, ref_label=ref_label, cmap=cmap, overwrite_cmap=overwrite_cmap, title=title)


    def plot_loadings(self, PCs=[1], xaxis=None, xlabel=None, ylabel=None, title=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None): 
        """Plots the PCA loadings as a line plot.
        
        Args:
            - PCs (list, optional): list of the components numbers that should be plotted (e.g. [1,2,3] will plot the first 3 components). Defaults to [1] - only plotting the loadings of the first component
            - xaxis (str, optional): Grid on which data lies, e.g. wavelength. Defaults to None.
            - xlabel (str, optional): Describes x axis. Defaults to None.
            - ylabel (str, optional): Describes y axis. Defaults to None.
            - title (str, optional): title of the plot
            - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
            - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first.")
        
        loadings_plot(self.loadings_, Comps=PCs, xaxis=xaxis, xlabel=xlabel, ylabel=ylabel, title=title, cmap=cmap, overwrite_cmap=overwrite_cmap, method='pca')



    def plot_influence(self, reference=None, regression=False, txt_labels=None, a=0.95, title=None, ref_label=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None):
        """Plots the influence plot of the Q residuals and the Hotellings T^2.
        
        Args:
            - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
            - regression (bool, optional): When the reference data is continuous this must be set to True, if the reference data is categorical it must be set to False. Defaults to False
            - txt_labels (1d array, optional): String array containing the labels of each plotted point, if set to None labels will not be plotted. Defaults to None
            - a (float, optional): significance level of the plotted threshold lines, must be between 0 and 1. Defaults to 0.95
            - title (str, optional): title of the plot
            - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). Defaults to None
            - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
            - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first.")
        
        influence_plot(self.Q_, self.T2_, self.n_components_, regression=regression, reference=reference, txt_labels=txt_labels, a=a, title=title, ref_label=ref_label, cmap=cmap, overwrite_cmap=overwrite_cmap)

    def plot_multiple_scores(self, n_components=None, reference=None, regression=False, ref_label=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None):
        """Plots the scores plots of all possible combinations of PCs of the first n_components. This method will plot the distribution of the scores within the PC on the diagonal. 

        Args: 
            - n_components (int, optional): The number of components to use in the plot, can be used when we require the plot to show less components than the full model, if set to None, the maximum number of components will be selected
            - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
            - regression (bool, optional): When the reference data is continuous this must be set to True, if the reference data is categorical it must be set to False. Defaults to False
            - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). Defaults to None
            - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
            - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first.")
        
        multiple_scoreplot(self.scores_, explained_variance_ratio=self.explained_variance_ratio_, reference=reference, regression=regression, n_components=n_components, ref_label=ref_label, cmap=cmap, overwrite_cmap=overwrite_cmap, method='pca')

    def plot_scores_MarkerShapeAndColor(self, color_reference, marker_reference, regression=False, txt_labels=None, PC_x=1, PC_y=2, cmap=rucolors.secondary_colormap, overwrite_cmap=None, title=None, ref_label=None, marker_size=100, alpha=0.8):
        """Plots the PCA scores for selected principal components (PC_x and PC_y).
        
        Args:
            - color_reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot.
            - marker_reference (1d array, optional): An array that contains reference values that can be used to give different marker shapes the points in the plot.
            - regression (bool, optional): When the reference data is continuous this must be set to True, if the reference data is categorical it must be set to False. Defaults to False
            - PC_x (int, optional): the number of the PC to plot on the x-axis. Defaults to 1
            - PC_y (int, optional): The number of the PC to plot on the y-axis. Defaults to 2
            - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
            - title (str, optional): A string containing the title of the plot. Defaults to None
            - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). If set to None
            - marker_size (int, optional): The size of the markers in the plot. Defaults to 100
            - alpha (float, optional): Transparancy of the markers where 1 is opague and 0 is fully transparant. Defaults to 0.8
        """
        if self.is_fitted_==False:
            raise ValueError("Model has not been fitted yet, call .fit() first.")
        
        scores_plot_MarkerShapeAndColor(self.scores_, color_reference, marker_reference, self.explained_variance_ratio_, regression=regression, txt_labels=txt_labels, Comp_x=PC_x, Comp_y=PC_y, cmap=cmap, overwrite_cmap=overwrite_cmap, title=title, ref_label=ref_label, marker_size=marker_size, alpha=alpha, method='pca')



# X = np.random.rand(50, 20)
# pca = PCA(n_components=10)
# pca.fit(X, mean_center=True, scale=False)
# pca.plot_scree(n_components=10)
# pca.plot_influence()
