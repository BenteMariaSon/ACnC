#======================================================================
# RUCOLORS - QUALITY OF LIFE LIBRARY FOR WORKING WITH RU COLOR PALLETTE
#======================================================================
# Mickey Lukkien
# 10-09-2025
# Radboud University Nijmegen
#======================================================================

#Use .help() for explainer!
#You can get the radboud colors through rucolors.red (for red impact), and rucolors.main_colors['colorname'] or rucolors.secondary_colors['colorname'] for the other colors.
#You can get the radboud colormaps through rucolors.RU_colormap (for white to red impact), rucolors.main_colormap (for all shades or radboud red) or rucolors.secondary_colormap (for a 'rainbow' of the secondary colors)
#You can create custom colormaps (matplotlib style) using rucolors.make_colormap(name, colors), where 'name' is the name of your colormap (string) and 'colors' is a list of colors.
#The list of colors can be those from this library, or custom colors specified as a list of [r,g,b,a], [r,g,b] where each rgb value ranges between 0-1, or as a list of hexcodes.
#You can specify more than two colors to make a 'rainbow' colormap

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def make_colormap(name, colors):
    """Makes a custom colormap

    Args:
        name (string): name of the colormap
        colors (list): a list of colors. Individual colors must be passed as [r,g,b], [r,g,b,a] or hexcode (#xxxxxx)

    Returns:
        matplotlib colormap: colormap following matplotlib format
    """
    cmap = mcolors.LinearSegmentedColormap.from_list(name, colors)
    return cmap

red = [227/255, 0/255, 11/255, 1]

main_colors = {
    "Poppy": [255/255, 66/255, 75/255, 1],
    "Lady Bug": [190/255, 49/255, 26/255, 1],
    "Berry": [143/255, 32/255, 17/255, 1],
    "Maroon": [115/255, 14/255, 4/255, 1],
    "Mahogany": [74/255, 0, 4/255, 1]
}

secondary_colors = {
    "Gray": [121/255, 119/255, 119/255, 1],
    "Orange": [207/255, 82/255, 8/255, 1],
    "Blue": [0, 128/255, 203/255, 1],
    "Petrol": [0, 143/255, 127/255, 1],
    "Green": [74/255, 169/255, 67/255, 1],
    "Yellow": [204/255, 175/255, 0, 1]
}

bigger_secondary_colors = {
    "DarkGray": [48/255, 49/255, 49/255, 1],
    "Orange": [207/255, 82/255, 8/255, 1],
    "Blue": [0, 128/255, 203/255, 1],
    "Petrol": [0, 109/255, 107/255, 1],
    "Green": [74/255, 169/255, 67/255, 1],
    "Yellow": [204/255, 175/255, 0, 1],
    "LightGray": [185/255, 185/255, 184/255, 1]
}

RU_colormap = make_colormap("RU",
                            [[255/255, 198/255, 201/255, 1], # 8% lighter
                             [227/255, 0/255, 11/255, 1], #radboud red
                             [45.4/255, 0/255, 2.2/255]]) # 80% darker

main_colormap = make_colormap("RU Main",
                              [main_colors["Poppy"],
                               main_colors["Lady Bug"],
                               main_colors["Berry"],
                               main_colors["Maroon"],
                               main_colors["Mahogany"]])

secondary_colormap = make_colormap("RU Secondary", 
                                   [secondary_colors["Gray"], 
                                    secondary_colors["Orange"], 
                                    secondary_colors["Blue"], 
                                    secondary_colors["Petrol"], 
                                    secondary_colors["Green"], 
                                    secondary_colors["Yellow"]])

bigger_secondary_colormap = make_colormap("RU Secondary", 
                                   [bigger_secondary_colors["DarkGray"], 
                                    bigger_secondary_colors["Orange"], 
                                    bigger_secondary_colors["Blue"], 
                                    bigger_secondary_colors["Petrol"], 
                                    bigger_secondary_colors["Green"], 
                                    bigger_secondary_colors["Yellow"], 
                                    bigger_secondary_colors["LightGray"]])

def help():
    print("You can get the radboud colors through rucolors.red (for red impact), and rucolors.main_colors['colorname'] or rucolors.secondary_colors['colorname'] for the other colors.")
    print("You can get the radboud colormaps through rucolors.RU_colormap (for white to red impact), rucolors.main_colormap (for all shades or radboud red) or rucolors.secondary_colormap (for a 'rainbow' of the secondary colors)")
    print("You can create custom colormaps (matplotlib style) using rucolors.make_colormap(name, colors), where 'name' is the name of your colormap (string) and 'colors' is a list of colors.")
    print("The list of colors can be those from this library, or custom colors specified as a list of [r,g,b,a], [r,g,b] where each rgb value ranges between 0-1, or as a list of hexcodes")
    print("You can specify more than two colors to make a 'rainbow' colormap")
    fig, ax = plt.subplots(1,1, figsize=(2,2))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0,0),1,1, transform=ax.transAxes, color=red))
    ax.set_title("Red Impact (.red)")
    
    fig, axs = plt.subplots(1,5, figsize=(10,2))
    for ax, color in zip(axs, main_colors):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0,0),1,1, transform=ax.transAxes, color=main_colors[color]))
        ax.set_title(color)
    fig.suptitle("Main colors (.main_colors['name'])", y=1.1)

    fig, axs = plt.subplots(1,5, figsize=(10,2))
    for ax, color in zip(axs, secondary_colors):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0,0),1,1, transform=ax.transAxes, color=secondary_colors[color]))
        ax.set_title(color)
    fig.suptitle("Secondary colors (.secondary_colors['name'])", y=1.1)
    plt.show()