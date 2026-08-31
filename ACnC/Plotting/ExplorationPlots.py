# import relevant libraries 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
from scipy.stats import linregress
# from Codebase import rucolors
from . import rucolors

#===================================================================================================================================================#

def scatter_plot(data, x_feature, y_feature, target_class=None, title=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None, ax_in=None):
    """Plots a scatter plot of two selected features

    Parameters:
    - data (Dataframe): containing feature columns and a 'Class' column.
    - x_feature (str): Name of the colomn that should be plotted on the x-axis
    - y_feature (str): Name of the colomn that should be plotted on the y-axis
    - target_class (str, optional): If specified, only plots samples belonging to that class. If None, plots samples belonging to all classes. Defaults to None
    - title (str, optional): give the plot a title
    - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
    - overwrite_cmap (array-like, optional): array containing the new colors to use, there should be a color for each class, if a target class is set only one color is needed. Defaults to None
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output: 
    - A scatter plot of two chosen variables
    """
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        ax = ax_in
        fig = ax_in.figure

    if target_class:
        subset = data[data["Class"] == target_class]
        ax.scatter(
            subset[x_feature],
            subset[y_feature],
            label=target_class,
            color = overwrite_cmap if overwrite_cmap is not None else rucolors.red
        )
    else:
        for i, cls in enumerate(data["Class"].unique()):
            subset = data[data["Class"] == cls]
            ax.scatter(
                subset[x_feature],
                subset[y_feature],
                label=cls,
                color = overwrite_cmap[i] if overwrite_cmap is not None else cmap.resampled(len(data["Class"].unique()))(i)
            )

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.set_title(title)
    ax.legend()

    if ax_in is None:
        plt.show()
    else:
        return fig, ax
    

#===================================================================================================================================================#

def multiple_scatter_plots(data, features, class_column="Class", target_class=None, title=None, point_color=rucolors.red, line_color='k'):
    """Creates a scatterplot matrix for all combinations of selected features.
    
    Parameters:
    - data (Dataframe): dataframe with features and class column.
    - features (array-like, str): List of feature names to include in the matrix.
    - class_column (str, optional): Name of the column containing the class information. Defaults to "Class"
    - target_class (str, optional): If specified, only plot the samples in this class. If None, plot the samples of all classes. Defaults to None
    - title (str, optional): give the plot a title
    - point_color (color, optional): the color to give to the points (can be rgb values or string conaining hex code). Defaults to rucolors.red (radboud red)
    - line_color (color, optional): the color to give to the line (can be rgb values or string conaining hex code). Defaults to black

    Output: 
    - An image containing matrix of all possible scatter plots combination
    """
    n = len(features)
    
    # Filter by class if specified
    if target_class:
        data_to_plot = data[data[class_column] == target_class]
        classes = [target_class]
    else:
        data_to_plot = data
        classes = data_to_plot[class_column].unique()

    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n), squeeze=False)

    for i, x_feature in enumerate(features):
        for j, y_feature in enumerate(features):
            ax = axes[i, j]
            if i == j:
                ax.text(0.5, 0.5, x_feature, fontsize=12, ha='center', va='center')
                ax.set_xticks([])
                ax.set_yticks([])
            elif i > j:
                x = data_to_plot[x_feature]
                y = data_to_plot[y_feature]

                # Plot scatter + regression line
                sns.regplot(x=x, y=y, ax=ax, scatter_kws={"s": 15, "alpha": 0.6, "color": point_color}, line_kws={"color": line_color, "lw": 2})

                # Calculate R²
                r2 = linregress(x, y).rvalue ** 2

                ax.annotate(
                    f"$R^2$ = {r2:.2f}",
                    xy=(0.05, 0.9),
                    xycoords='axes fraction',
                    fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray")
                )
            else:
                ax.axis("off")

            if i < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(x_feature)

            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(y_feature)

    plt.title(title)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

#===================================================================================================================================================#

def plot_feature_bars(data, features, class_column="Class", by_class=False, scaling="none", title=None, ylabel=None, cmap=rucolors.secondary_colormap, overwrite_cmap=None, ax_in=None):
    """Plots a bar chart showing mean and standard deviation for selected features.
    
    Parameters:
    - data (Dataframe): DataFrame with features and class column.
    - features (array-like, str): List of feature names to include.
    - class_column (str, optional): Name of the column containing the class information. Defaults to "Class"
    - by_class (bool, optional): If True, plot per class + overall; if False, plot overall only. Defaults to False
    - scaling (str, optional): Which scaling method to use, can be "none", "autoscale", "mean", "minmax". Defaults to "none"
    - title (str, optional): title to give to the plot
    - ylabel (str, optional): label to give to the y-axis
    - cmap (cmap or str, optional): A colormap or string containing the name of the colormap that should be used to color the points. Defaults to rucolors.secondary_colormap (this is a custom colormap)
    - overwrite_cmap (array-like, optional): array containing the new colors to use, there should be a color for each class, if a target class is set only one color is needed. Defaults to None
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output: 
    - A bar chart showing the mean and standard deviations of selected features. 
    """
    df = data.copy()

    # Scaling options
    if scaling == "autoscale":
        df[features] = (df[features] - df[features].mean()) / df[features].std()
    elif scaling == "mean":
        df[features] = df[features] - df[features].mean()
    elif scaling == "minmax":
        df[features] = (df[features] - df[features].min()) / (df[features].max() - df[features].min())
    elif scaling == "none":
        pass  # use raw data
    else:
        raise ValueError("Invalid value for 'scaling'. Use 'none', 'autoscale', 'mean', or 'minmax'.")
    
    overall_mean = df[features].mean()
    overall_std = df[features].std()

    x = np.arange(len(features))
    width = 0.15

    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    if ax_in is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = ax_in
        fig = ax_in.figure

    if by_class:
        mean_per_class = df.groupby(class_column)[features].mean()
        std_per_class = df.groupby(class_column)[features].std()

        mean_per_class.loc["Overall"] = overall_mean
        std_per_class.loc["Overall"] = overall_std

        classes = mean_per_class.index

        for i, class_name in enumerate(classes):
            ax.bar(
                x + i * width,
                mean_per_class.loc[class_name],
                yerr=std_per_class.loc[class_name],
                width=width,
                capsize=4,
                label=class_name,
                color = overwrite_cmap[i] if overwrite_cmap is not None else cmap.resampled(len(classes.unique()))(i),
                edgecolor="black"
            )

        ax.set_xticks(x + width * (len(classes) - 1) / 2, features, rotation=90)
        ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.bar(
            x,
            overall_mean,
            yerr=overall_std,
            width=width * 2,
            capsize=4,
            color=overwrite_cmap if overwrite_cmap is not None else rucolors.red,
            edgecolor="black"
        )
        ax.set_xticks(x, features, rotation=90)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y')

    if ax_in is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax
    

#===================================================================================================================================================#

def boxplot(data, color=rucolors.red, title=None, ax_in=None):
    """Create a boxplot for each column in the input dataframe (data).

    Parameters:
    - data (Dataframe): containing feature columns and a class column.
    - color (color, optional): the color to give to the plot (can be rgb values or string conaining hex code). Defaults to rucolors.red (radboud red)
    - title (str, optional): give the plot a title
    - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output: 
    - Displays the box plot
    """

    # reshape for boxplot
    data_melted = data.melt(var_name="Variables", value_name="Values")
    
    if ax_in is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    else:
        ax = ax_in
        fig = ax_in.figure

    sns.boxplot(x="Variables", y="Values", data=data_melted, color=color, fliersize=5, linewidth=1.5, ax=ax)
    ax.set_title(title if title else "Boxplot of Variables")
    ax.set_ylabel("Quantiles")

    if ax_in is None:
        fig.tight_layout()
        plt.show()
    else:
        return fig, ax


#===================================================================================================================================================#

def plot_spectra(data, xaxis=None, reference=None, regression=True, title=None, xlabel=None, ylabel=None, ref_label='Target value', cmap=rucolors.secondary_colormap, alpha=0.6, LegendOutside=False, overwrite_cmap=None, ax_in=None):
    """ Plots spectral data

    Parameters:
        - data (ndarray): maxtrix of [n_samples x n_variables]
        - xaxis (array-like, optional): Grid on which data lies, e.g. wavelength. Defaults to None
        - reference (array-like, optional): Reference data to use for colors, in the case of classification the y contains the class labels, for regression y is continuous. 
        - regression (bool, optional): Set to True when doing regression (makes colormap) and to False when doing classification (uses individual colors)
        - title (str, optional): Title for plot.
        - xlabel (str, optional): Label for x-axis.
        - ylabel (str, optional): Label for y-axis.
        - ref_label (str, optional): Labal of the reference data used when doing regression. Defaults to 'Target value'. 
        - cmap (str, optional): Name of the colormap to use. Defaults to rucolors.secondary_colormap (secondary radboud colors)
        - alpha (float, optional): The transparancy applied to the lines (0 is fully transparent and 1 is fully opaque)
        - LegendOutside (bool, optional): Can be set to True if the legend should be plot on the outside of the plot (benificial for busy plots). Defaults to False
        - overwrite_cmap (array-like or None, optional): Can be used if you want to use custom colors (only for classification), the input should then be an array-like containing the hex codes or rgb values of the wanted colors. Defaults to None
        - ax_in (figure axis, optional): Axis can be passed here to add the scatter plot to an existing axis system. Defaults to None

    Output:
        Plot of the spectral data within X
    """
    data = np.asarray(data, copy=True)
    # Checks:
    if reference is not None and data.shape[0] != len(reference):
        raise ValueError("Number of rows in 'data' must match length of 'reference'")
    if xaxis is not None and data.shape[1] != len(xaxis):
        raise ValueError("Number of columns in 'data' must match length of 'wavelengths'")

    if xaxis is None:
        xaxis = np.linspace(0, data.shape[1] - 1, data.shape[1])

    if ax_in is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        ax = ax_in
        fig = ax_in.figure

    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    if reference is None:
        color = overwrite_cmap if overwrite_cmap else rucolors.red
        ax.plot(xaxis, data.T, c=color, alpha=alpha)
    else:
        y = np.array(reference)
        if regression:  # continuous colormap
            norm = plt.Normalize(vmin=y.min(), vmax=y.max())
            for i in range(data.shape[0]):
                ax.plot(xaxis, data[i, :], color=cmap(norm(y[i])), alpha=alpha)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(ref_label if ref_label else 'Target value')
        else:  # categorical coloring
            unique_classes = np.unique(y)
            n_classes = len(unique_classes)
            if overwrite_cmap is None:
                cmap_dicrete = cmap.resampled(len(unique_classes))
            
            for i, class_label in enumerate(unique_classes):
                class_data = data[y == class_label]
                for j, row in enumerate(class_data):
                    if overwrite_cmap is None:
                        color = cmap_dicrete(i)
                    else:
                        color = overwrite_cmap[i]
                    ax.plot(xaxis.astype(float), row.astype(float), color=color, alpha=alpha, label=f"{str(class_label)}" if j==0 else "")

            if LegendOutside:
                ax.legend(loc='upper left', bbox_to_anchor=(1,1), title=ref_label if ref_label else 'Class') #Place the upper left corner of the legend in location (1,1)
            else:
                ax.legend(title=ref_label if ref_label else 'Class')

    ax.set_xlabel(xlabel if xlabel else "wavelengths (nm)")
    ax.set_ylabel(ylabel if ylabel else "intensity (a.u.)")
    ax.set_title(title if title else "Spectra")

    if ax_in is None:
        plt.show()
    else:
        return fig, ax
