import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns
from scipy import stats
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
# from Codebase import rucolors
from . import rucolors

#===================================================================================================================================================#

def scree_plot(explained_variance_ratio, n_components=None, title=None, color=rucolors.red, method='pca', ax_in=None):
    """Plots a scree plot to visualize the explained variance for each component in the decomposed space

    Parameters:
    - explained_variance_ratio (ndarray): Proportion of the dataset's total variance that is captured by each component, ndarray of shape (n_components,)
    - n_components (int, optional): The number of components to use in the plot, can be used when we require the plot to show less components than the full model, if set to None, the maximum number of components will be selected
    - title (str, optional): A string containing the title of the plot. Defaults to None.
    - color (str, optional): A string containing the hex-code of the desired color.
    - method (str, optiona): A string that can be set to 'pca' or 'pls', this will put the labels as 'PC' or 'LV' respectivly
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output:
    - Displays the scree plot
    """

    if n_components is None or n_components > len(explained_variance_ratio):
        n_components = len(explained_variance_ratio)

    explained_var = explained_variance_ratio[:n_components]

    x_pos = np.arange(1, n_components + 1)
    if method == 'pca':
        labels = [f'PC{i}' for i in x_pos]
    elif method == 'pls':
        labels = [f'LV{i}' for i in x_pos]
    else:
        raise ValueError("given method is invalid, this should either be 'pca' or 'pls'.")
    
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        ax = ax_in
        fig = ax_in.figure

    bars = ax.bar(x_pos, explained_var * 100, color=color)
    ax.plot(x_pos, explained_var * 100, marker='o', color='black', linestyle='-', linewidth=2)
    
    ax.set_xticks(ticks=x_pos, labels=labels)
    ax.set_xlabel('Principal Components' if method=='pca' else 'Latent Variables')
    ax.set_ylabel('Percentage of Explained Variance')
    ax.set_title(title if title else "")
    
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.1f}%', ha='center', va='bottom', fontsize=10)
    
    if ax_in is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax

#===================================================================================================================================================#

def cumulative_explained_variance_plot(explained_variance_ratio, n_components=None, title=None, color=rucolors.red, method='pca', ax_in=None):
    """Plots the cumulative explained variance for each component in the decomposed space

    Parameters:
    - explained_variance_ratio (ndarray): Proportion of the dataset's total variance that is captured by each component, ndarray of shape (n_components,)
    - n_components (int, optional): The number of components to use in the plot, can be used when we require the plot to show less components than the full model, if set to None, the maximum number of components will be selected
    - title (str, optional): A string containing the title of the plot. Defaults to None.
    - color (str, optional): A string containing the hex-code of the desired color.
    - method (str, optiona): A string that can be set to 'pca' or 'pls', this will put the labels as 'PC' or 'LV' respectivly
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output:
    - Displays the cumulative explained variance plot
    """

    if n_components is None or n_components > len(explained_variance_ratio):
        n_components = len(explained_variance_ratio)

    explained_var = explained_variance_ratio[:n_components]

    x_pos = np.arange(1, len(explained_var) + 1)
    cumulative_variance = np.cumsum(explained_var)
    cumulative_variance = np.insert(cumulative_variance, 0, 0)

    if method == 'pca':
        labels = [f'PC{i}' for i in x_pos]
    elif method == 'pls':
        labels = [f'LV{i}' for i in x_pos]
    else:
        raise ValueError("given method is invalid, this should either be 'pca' or 'pls'.")
    
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        ax = ax_in
        fig = ax_in.figure

    ax.plot(np.arange(0, len(cumulative_variance)), cumulative_variance * 100, marker='o', linestyle='-', color=color)
    ax.axhline(y=0, c='k', linestyle=':')
    ax.set_xticks(ticks=x_pos, labels=labels)
    ax.set_xlabel('Principal Component' if method=='pca' else 'Latent Variables')
    ax.set_ylabel('Cumulative Explained Variance (%)')
    ax.set_title(title if title else "")
    
    if ax_in is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax

#===================================================================================================================================================#

def scores_plot(scores, explained_variance_ratio=None, reference=None, regression=False, txt_labels=None, Comp_x=1, Comp_y=2, cmap=rucolors.secondary_colormap, overwrite_cmap=None, title=None, ref_label=None, marker_size=100, alpha=0.8, _biplot=False, method='pca', ax_in=None):
    """plots a scores plot for selected principal components.

    Parameters:
    - scores (ndarray): Projected data in the principal component space, ndarray of shape (n_samples, n_components)
    - explained_variance_ratio (ndarray): Proportion of the dataset's total variance that is captured by each component, ndarray of shape (n_components,)
    - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
    - regression (bool, optional): When the reference data is continuous this must be set to True, if the reference data is categorical it must be set to False. Defaults to False
    - Comp_x (int, optional): the number of the component to plot on the x-axis. Defaults to 1
    - Comp_y (int, optional): The number of the component to plot on the y-axis. Defaults to 2
    - cmap (cmap or str, optional): A cmap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap
    - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
    - title (str, optional): A string containing the title of the plot. Defaults to None
    - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). If set to None
    - marker_size (int, optional): The size of the markers in the plot. Defaults to 100
    - alpha (float, optional): Transparancy of the markers where 1 is opague and 0 is fully transparant. Defaults to 0.8
    - method (str, optional): A string that can be set to 'pca' or 'pls', this will put the labels as 'PC' or 'LV' respectivly
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output:
    - Displays the scores plot
    """
    
    if Comp_x > scores.shape[1] or Comp_y > scores.shape[1]:
        raise ValueError("The number of the components to plot in Comp_x and Comp_y cannot be larger than the number of components included in the model")

    i = Comp_x - 1
    j = Comp_y - 1
    x_vals = scores[:, i]
    y_vals = scores[:, j]

    if not _biplot and ax_in is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        ax = ax_in
        fig = ax_in.figure

    if regression and overwrite_cmap is not None:
        raise ValueError("Cannot overwrite colormap when regression is set to True. Overwriting the cmap can only be done in case of classification")

    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    if reference is None:
        color = overwrite_cmap if overwrite_cmap else rucolors.red
        scatter = ax.scatter(x_vals, y_vals, color=color, edgecolor='black', s=marker_size, alpha=alpha)
    elif regression:
        scatter = ax.scatter(x_vals, y_vals, c=reference, cmap=cmap, edgecolor='black', s=marker_size, alpha=alpha)
        cbar = fig.colorbar(scatter)
        cbar.set_label(ref_label if ref_label else 'Target value')
    else:
        unique_classes, reference_encoded = np.unique(reference, return_inverse=True)
        if overwrite_cmap is None:
            if isinstance(cmap, str):
                cmap = mpl.colormaps[cmap]
            cmap_dicrete = cmap.resampled(len(unique_classes))
            colors = [cmap_dicrete(idx) for idx in reference_encoded]
        else:
            colors = [overwrite_cmap[idx] for idx in reference_encoded]

        scatter = ax.scatter(x_vals, y_vals, color=colors, edgecolor='black', s=marker_size, alpha=alpha)
        # Legend
        handles = [
            Line2D([], [], marker='o', color='w',
                    markerfacecolor=overwrite_cmap[n] if overwrite_cmap else cmap_dicrete(n), 
                    markeredgecolor='black',
                    markersize=8, 
                    label=cls)
            for n, cls in enumerate(unique_classes)
        ]
        ax.legend(handles=handles, title=ref_label if ref_label else 'Class', loc='best')

    if txt_labels is not None:
        for n, txt in enumerate(txt_labels):
            ax.text(x_vals[n], y_vals[n], txt)

    # --- Axes, Labels, Title ---
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    if explained_variance_ratio is not None:
        ax.set_xlabel(f'PC{Comp_x} ({explained_variance_ratio[i]*100:.1f}%)' if method=='pca' else f'LV{Comp_x} ({explained_variance_ratio[0, i]*100:.1f}% in X, {explained_variance_ratio[1, i]*100:.1f}% in y)')
        ax.set_ylabel(f'PC{Comp_y} ({explained_variance_ratio[j]*100:.1f}%)' if method=='pca' else f'LV{Comp_y} ({explained_variance_ratio[0, j]*100:.1f}% in X, {explained_variance_ratio[1, j]*100:.1f}% in y)')
    else:
        ax.set_xlabel(f'PC{Comp_x}' if method=='pca' else f'LV{Comp_x}')
        ax.set_ylabel(f'PC{Comp_y}' if method=='pca' else f'LV{Comp_y}')

    if title:
        ax.set_title(title)
    elif method=='pca':
        ax.set_title(f'PCA Score Plot (PC{Comp_x} vs PC{Comp_y})')
    elif method=='pls':
        ax.set_title(f'PLS Score Plot (LV{Comp_x} vs LV{Comp_y})')

    if not _biplot and ax_in is None:
        fig.tight_layout()
        plt.show()
        return fig, ax
    elif _biplot:
        return scatter

    return fig, ax

#===================================================================================================================================================#

def scores_plot_MarkerShapeAndColor(scores, color_reference, marker_reference, explained_variance_ratio=None, regression=False, txt_labels=None, Comp_x=1, Comp_y=2, cmap=rucolors.secondary_colormap, overwrite_cmap=None, title=None, ref_label=None, marker_size=100, alpha=0.8, method='pca', ax_in=None):
    """plots a scores plot for selected principal components where points are colored according to one reference and shaped according to another.

    Parameters:
    - scores (ndarray): Projected data in the principal component space, ndarray of shape (n_samples, n_components)
    - color_reference (ndarray): An array that contains reference values that can be used to color the points in the plot. 
    - marker_reference (ndarray): An array that contains reference values that can be used to give a shape to the markers in the plot. 
    - explained_variance_ratio (ndarray, optional): Proportion of the dataset's total variance that is captured by each component, ndarray of shape (n_components,)
    - regression (bool, optional): When the reference data is continuous this must be set to True, if the reference data is categorical it must be set to False. Defaults to False
    - txt_labels (ndarray, optional): An array containing a text label for each samples, if given, the text will be shown next to the point. Defaults to None
    - Comp_x (int, optional): the number of the component to plot on the x-axis. Defaults to 1
    - Comp_y (int, optional): The number of the component to plot on the y-axis. Defaults to 2
    - cmap (cmap or str, optional): A cmap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap
    - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
    - title (str, optional): A string containing the title of the plot. Defaults to None
    - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). If set to None
    - marker_size (int, optional): The size of the markers in the plot. Defaults to 100
    - alpha (float, optional): Transparancy of the markers where 1 is opague and 0 is fully transparant. Defaults to 0.8
    - method (str, optional): A string that can be set to 'pca' or 'pls', this will put the labels as 'PC' or 'LV' respectivly
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output:
    - Displays the scores plot with markers colored and shaped according to their own reference
    """

    marker_options = ["o", "^", "s", "P", "*", "h", "X", "D", "p", "d", "<", ">", "v", "H", "+", "x"]
    if len(np.unique(marker_reference)) > len(marker_options):
        raise ValueError(f"Function currently has {len(marker_options)} possible markers, your input for y_marker has too many classes (number of classes {len(np.unique(marker_reference))})")
    unique_marker_classes, y_marker_numeric = np.unique(marker_reference, return_inverse=True)
    markers = np.array([marker_options[idx] for idx in y_marker_numeric])
    
    if Comp_x > scores.shape[1] or Comp_y > scores.shape[1]:
        raise ValueError("The number of the components to plot in Comp_x and Comp_y cannot be larger than the number of components included in the model")

    i = Comp_x - 1
    j = Comp_y - 1
    x_vals = scores[:, i]
    y_vals = scores[:, j]

    if ax_in is None:
        fig, ax = plt.subplots(figsize=(9,6))
    else:
        ax = ax_in
        fig = ax_in.figure

    if regression and overwrite_cmap is not None:
        raise ValueError("Cannot overwrite colormap when regression is set to True. Overwriting the cmap can only be done in case of classification")

    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    if regression:
        norm = mcolors.Normalize(vmin=color_reference.min(), vmax=color_reference.max())
        for m in np.unique(markers):
            mask = markers==m
            scatter = ax.scatter(x_vals[mask], y_vals[mask], c=color_reference[mask], marker=m, cmap=cmap, norm=norm, edgecolor='black', s=marker_size, alpha=alpha)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(ref_label if ref_label else 'Target value')
    else:
        unique_classes, reference_encoded = np.unique(color_reference, return_inverse=True)
        if overwrite_cmap is None:
            if isinstance(cmap, str):
                cmap = mpl.colormaps[cmap]
            cmap_dicrete = cmap.resampled(len(unique_classes))
            colors = np.asarray([cmap_dicrete(idx) for idx in reference_encoded])
        else:
            colors = np.asarray([overwrite_cmap[idx] for idx in reference_encoded])

        for m in np.unique(markers):
            mask = markers==m
            scatter = ax.scatter(x_vals[mask], y_vals[mask], color=colors[mask], marker=m, edgecolor='black', s=marker_size, alpha=alpha)

        color_handles = [plt.Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=overwrite_cmap[idx] if overwrite_cmap else cmap_dicrete(idx), markeredgecolor='k', markersize=10, alpha=alpha)
            for idx, label in enumerate(unique_classes)]
        

    if txt_labels is not None:
        for n, txt in enumerate(txt_labels):
            ax.text(x_vals[n], y_vals[n], txt)

    # --- Axes, Labels, Title ---
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    if explained_variance_ratio is not None:
        ax.set_xlabel(f'PC{Comp_x} ({explained_variance_ratio[i]*100:.1f}%)' if method=='pca' else f'LV{Comp_x} ({explained_variance_ratio[0, i]*100:.1f}% in X, {explained_variance_ratio[1, i]*100:.1f}% in y)')
        ax.set_ylabel(f'PC{Comp_y} ({explained_variance_ratio[j]*100:.1f}%)' if method=='pca' else f'LV{Comp_y} ({explained_variance_ratio[0, j]*100:.1f}% in X, {explained_variance_ratio[1, j]*100:.1f}% in y)')
    else:
        ax.set_xlabel(f'PC{Comp_x}' if method=='pca' else f'LV{Comp_x}')
        ax.set_ylabel(f'PC{Comp_y}' if method=='pca' else f'LV{Comp_y}')

    if title:
        ax.set_title(title)
    elif method=='pca':
        ax.set_title(f'PCA Score Plot (PC{Comp_x} vs PC{Comp_y})')
    elif method=='pls':
        ax.set_title(f'PLS Score Plot (LV{Comp_x} vs LV{Comp_y})')

    marker_handles = [plt.Line2D([0], [0], marker=marker_options[idx], color='w', label=class_label, markerfacecolor='k', markersize=15, alpha=1)
        for idx, class_label in enumerate(unique_marker_classes)]
    
    first_legend = ax.legend(handles=marker_handles, title="", loc='upper left', bbox_to_anchor=(1,1))
    if regression == False:
        # plt.gca().add_artist(first_legend) # Ensures the first legend is not overwritten by the second but is kept. 
        ax.add_artist(first_legend)
        ax.legend(handles=color_handles, title="", loc='lower left', bbox_to_anchor=(1,0))
    

    if ax_in is None:
        fig.tight_layout()
        plt.show()

    return fig, ax

#===================================================================================================================================================#

def biplot(loadings, scores, explained_variance_ratio, reference=None, regression=False, feature_names=False, Comp_x=1, Comp_y=2, scale_arrows=1.0, cmap=rucolors.secondary_colormap, overwrite_cmap=None, title=None, ref_label=None, marker_size=100, alpha=0.8, method='pca', ax_in=None):
    """plots a biplot for selected principal components.

    Parameters:
    - loadings (ndarray): loadings used to project data from original space to principal component space, ndarray of shape (n_features, n_components)
    - scores (ndarray): Projected data in the principal component space, ndarray of shape (n_samples, n_components)
    - explained_variance_ratio (ndarray): Proportion of the dataset's total variance that is captured by each component, ndarray of shape (n_components,)
    - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
    - regression (bool, optional): When the reference data is continuous this must be set to True, if the reference data is categorical it must be set to False. Defaults to False
    - feature_names (str-array, optional): A string array containing the names of all features, if given, the names will be displayed next to the arrows. Defaults to None
    - Comp_x (int, optional): the number of the component to plot on the x-axis. Defaults to 1
    - Comp_y (int, optional): The number of the component to plot on the y-axis. Defaults to 2
    - scale_arrows (float, optional): The length of the arrows will be multiplied by this number to scale them to the desired length, defaults to 1.0
    - cmap (cmap or str, optional): A cmap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap
    - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
    - title (str, optional): A string containing the title of the plot. Defaults to None
    - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). If set to None
    - marker_size (int, optional): The size of the markers in the plot. Defaults to 100
    - alpha (float, optional): Transparancy of the markers where 1 is opague and 0 is fully transparant. Defaults to 0.8
    - method (str, optional): A string that can be set to 'pca' or 'pls', this will put the labels as 'PC' or 'LV' respectivly
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output:
    - Displays the biplot
    """
    
    if Comp_x > loadings.shape[1] or Comp_y > loadings.shape[1]:
        raise ValueError("The number of the components to plot in Comp_x and Comp_y cannot be larger than the number of components included in the model")
    if feature_names is None:
        feature_names = np.arange(1, loadings.shape[0]+1)

    i = Comp_x - 1
    j = Comp_y - 1    

    x_loadings = loadings[:,i] 
    y_loadings = loadings[:,j]

    if ax_in is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        ax = ax_in
        fig = ax_in.figure

    # Make the scores plot first
    scatter = scores_plot(scores, explained_variance_ratio, reference=reference, regression=regression, Comp_x=Comp_x, Comp_y=Comp_y, cmap=cmap, overwrite_cmap=overwrite_cmap, title=title, ref_label=ref_label, marker_size=marker_size, alpha=alpha, _biplot=True, method=method, ax_in=ax)

    # Draw loading vectors (arrows) for features
    for k in range(len(x_loadings)):
        ax.arrow(0, 0, scale_arrows*x_loadings[k], scale_arrows*y_loadings[k], color='black', alpha=0.8, head_width=0.1, length_includes_head=True)
        if feature_names is not False:
            ax.text(scale_arrows*x_loadings[k], scale_arrows*y_loadings[k], feature_names[k], color='black', fontsize=10)

    if ax is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax

#===================================================================================================================================================#

def loadings_plot(loadings, Comps=[1], xaxis=None, xlabel=None, ylabel=None, title=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None, method='pca', ax_in=None):
    """Plots the loadings as a line plot.
    
    Parameters:
    - loadings (ndarray): loadings used to project data from original space to principal component space, ndarray of shape (n_features, n_components)
    - Comps (list, optional): list of the components numbers that should be plotted (e.g. [1,2,3] will plot the first 3 components). Defaults to [1] - only plotting the loadings of the first component
    - xaxis (str, optional): Grid on which data lies, e.g. wavelength. Defaults to None.
    - title (str, optional): title of the plot
    - xlabel (str, optional): Describes x axis. Defaults to None.
    - ylabel (str, optional): Describes y axis.
    - cmap (cmap or str, optional): A cmap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap
    - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
    - method (str, optional): A string that can be set to 'pca' or 'pls', this will put the labels as 'PC' or 'LV' respectivly
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output:
    - Displays the loadings plot
    """

    if any(c > loadings.shape[1] for c in Comps):
        raise ValueError("Comps can only contain values that are lower than the number of components included in the model")

    if xaxis is None:
        xaxis = np.arange(len(loadings))

    if overwrite_cmap is None:
        if isinstance(cmap, str):
            cmap = mpl.colormaps[cmap]
        cmap_dicrete = cmap.resampled(len(Comps))
    else:
        cmap_dicrete = overwrite_cmap

    if ax_in is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        ax = ax_in
        fig = ax_in.figure

    for Comp in Comps:
        loading = loadings[:,Comp-1]
        label = f"PC{Comp}" if method=='pca' else f"LV{Comp}"
        ax.plot(xaxis, loading, label=label, color=cmap_dicrete[Comp-1] if overwrite_cmap else cmap_dicrete(Comp-1))
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xlabel(xlabel if xlabel else "Wavelengths")
    ax.set_ylabel(ylabel if ylabel else "Loading Value")
    if method == 'pca':
        ax.set_title(title if title else f"Loadings plot of PCs: {Comps}")
    elif method == 'pls':
        ax.set_title(title if title else f"Loadings plot of LVs: {Comps}")
    ax.legend()

    if ax_in is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax

#===================================================================================================================================================#

def influence_plot(q_residuals, t2_scores, n_components, regression=False, reference=None, txt_labels=None, a=0.95, title=None, ref_label=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None, ax_in=None):
    """Plots the loadings as a line plot.
    
    Parameters:
    - q_residuals (ndarray): The Q residuals of each sample, ndarray of shape (n_samples,)
    - t2_scores (ndarray): The Hotellings T^2 value of each sample, ndarray of shape (n_samples,)
    - n_components (int): The number of components in the model
    - regression (bool, optional): When the reference data is continuous this must be set to True, if the reference data is categorical it must be set to False. Defaults to False
    - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
    - txt_labels (1d array, optional): String array containing the labels of each plotted point, if set to None labels will not be plotted. Defaults to None
    - a (float, optional): significance level of the plotted threshold lines, must be between 0 and 1. Defaults to 0.95
    - title (str, optional): title of the plot
    - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). Defaults to None
    - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
    - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output:
    - Displays the influence plot
    """
    if regression and overwrite_cmap is not None:
        raise ValueError("Cannot overwrite colormap when regression is set to True. Overwriting the cmap can only be done in case of classification")

    q_thresh = np.percentile(q_residuals, 100*a)
    t2_thresh = stats.chi2.ppf(a, df=n_components)

    if ax_in is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        ax = ax_in
        fig = ax_in.figure

    handles = []

    if reference is None:
        color = overwrite_cmap if overwrite_cmap is not None else rucolors.red
        scatter = ax.scatter(t2_scores, q_residuals, color=color, edgecolor='black', s=100, alpha=0.8)
    elif regression:
        scatter = ax.scatter(t2_scores, q_residuals, c=reference, cmap=cmap, edgecolor='black', s=100, alpha=0.8)
        cbar = fig.colorbar(scatter)
        cbar.set_label(ref_label if ref_label else 'Target value')
    else:
        unique_classes, reference_encoded = np.unique(reference, return_inverse=True)
        if overwrite_cmap is None:
            if isinstance(cmap, str):
                cmap = mpl.colormaps[cmap]
            cmap_dicrete = cmap.resampled(len(unique_classes))
            colors = [cmap_dicrete(idx) for idx in reference_encoded]
        else:
            colors = [overwrite_cmap[idx] for idx in reference_encoded]
        scatter = ax.scatter(t2_scores, q_residuals, color=colors, edgecolor='black', s=100, alpha=0.8)

        # Legend
        handles = [
            Line2D([], [], marker='o', color='w',
                    markerfacecolor=cmap_dicrete(n) if overwrite_cmap is None else overwrite_cmap[n], 
                    markeredgecolor='black',
                    markersize=8, label=cls)
            for n, cls in enumerate(unique_classes)
        ]
        # ax.legend(handles=handles, title=ref_label if ref_label else 'Class', loc='best')

    if txt_labels is not None:
        for n, txt in enumerate(txt_labels):
            ax.text(t2_scores[n], q_residuals[n], txt)

    # --- Axes, Labels, Title ---
    h_q = ax.axhline(q_thresh, color='blue', linestyle='--', label=f'Q Residual limit ({a:.0%})', alpha=0.5, zorder=0)
    h_t2 = ax.axvline(t2_thresh, color='red', linestyle='--', label=f'$T^2$ Limit ({a:.0%})', alpha=0.5, zorder=0)
    ax.set_xlabel("Hotelling's $T^2$")
    ax.set_ylabel("Q residual")
    ax.set_title(title if title else "PCA influence plot (Q vs $T^2$)")

    handles.extend([h_q, h_t2])
    ax.legend(handles=handles, loc='best')

    if ax_in is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax

#===================================================================================================================================================#

def multiple_scoreplot(scores, explained_variance_ratio=None, reference=None, regression=False, n_components=None, ref_label=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None, method='pca'):
    """Plots the scores plots of all possible combinations of components of the first n_components. This method will plot the distribution of the scores within the PC on the diagonal. 
    
    Parameters:
    - scores (ndarray): Projected data in the principal component space, ndarray of shape (n_samples, n_components)
    - explained_variance_ratio (ndarray): Proportion of the dataset's total variance that is captured by each component, ndarray of shape (n_components,)
    - reference (1d array, optional): An array that contains reference values that can be used to color the points in the plot, when equal to None all points will be the same color. Defaults to None
    - regression (bool, optional): When the reference data is continuous this must be set to True, if the reference data is categorical it must be set to False. Defaults to False
    - n_components (int): The number of components to use in the plot, can be used when we require the plot to show less components than the full model, if set to None, the maximum number of components will be selected
    - ref_label (str, optional): A string containing an overarching name of your references (legend/colorbar title). Defaults to None
    - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
    - overwrite_cmap (1d array, optional): An array containing the specified colors to use for plotting, can only be used when regression is False. The number of colors given cannot be lower than the number of classes
    - method (str, optional): A string that can be set to 'pca' or 'pls', this will put the labels as 'PC' or 'LV' respectivly

    Output:
    - Displays the scores plots of all possible combinations of components
    """
    if regression and overwrite_cmap is not None:
        raise ValueError("Cannot overwrite colormap when regression is set to True. Overwriting the cmap can only be done in case of classification")
    
    if n_components is None:
        n_components = scores.shape[1]
    
    if explained_variance_ratio is not None:
        columns = [f'PC{i+1} ({explained_variance_ratio[i]*100:.1f}%)' for i in range(n_components)] if method=='pca' else [f'LV{i+1} ({explained_variance_ratio[0, i]*100:.1f}% in X, {explained_variance_ratio[1, i]*100:.1f}% in y)' for i in range(n_components)]
    else:
        columns = [f'PC{i+1}' for i in range(n_components)] if method=='pca' else [f'LV{i+1}' for i in range(n_components)]
    df_scores = pd.DataFrame(scores[:,:n_components], columns=columns)
    if reference is not None:
        df_scores['reference'] = np.asarray(reference)
    else:
        df_scores['reference'] = np.ones(scores.shape[0])

    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    
    # If doing regression (continuous)
    if regression:
        norm = mpl.colors.Normalize(vmin=reference.min(), vmax=reference.max())
        g = sns.PairGrid(df_scores, vars=columns)
        def scatter_continuous(x,y, **kwargs):
            plt.scatter(x, y, c=reference, cmap=cmap, norm=norm, s=50, edgecolor='k', alpha=0.8)
            plt.axhline(0, color='gray', linestyle='--', zorder=0)
            plt.axvline(0, color='gray', linestyle='--', zorder=0)
        g.map_offdiag(scatter_continuous)
        c = cmap.resampled(3)(1) # Take the center color of the colormap
        g.map_diag(sns.kdeplot, fill=True, color=c)

        #colorbar
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = g.figure.add_axes([1.01, 0.15, 0.02, 0.7])
        cb = g.figure.colorbar(sm, cax=cax)
        cb.set_label(ref_label if ref_label else "Target Value")
    
    # If doing classification (discrete)
    else:
        unique_classes, reference_encoded = np.unique(reference, return_inverse=True)
        if overwrite_cmap is None:
            cmap_dicrete = cmap.resampled(len(unique_classes))
            palette = [cmap_dicrete(i) for i in range(len(unique_classes))]
        else:
            palette = [overwrite_cmap[i] for i in range(len(unique_classes))]
        g = sns.PairGrid(df_scores, vars=columns, hue='reference', palette=palette)
        g.map_diag(sns.kdeplot, fill=True)
        g.map_offdiag(plt.scatter, s=50, edgecolors='black', alpha=0.8)
        g.add_legend(title=ref_label if ref_label is None else "Classes")

        # add the y=0 and x=0 reference lines (but only to the off-diagonal)
        for i, row in enumerate(g.axes):
            for j, ax in enumerate(row):
                if i != j:
                    ax.axhline(0, color="gray", linestyle="--", zorder=0)
                    ax.axvline(0, color="gray", linestyle="--", zorder=0)

    plt.show()
    
