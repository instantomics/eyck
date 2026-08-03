import polyptich
import matplotlib as mpl
import pandas as pd
from .data import gene_id
from .data import Transcriptome
import scanpy as sc
import numpy as np
from typing import List
from types import SimpleNamespace

try:
    import datashader as ds
    from datashader.mpl_ext import dsshow
except ModuleNotFoundError:
    ds = None
    dsshow = None


def get_colors(n):
    if n <= 10:
        colors = mpl.colormaps["tab10"].colors[:n]
    elif n <= 20:
        colors = mpl.colormaps["tab20"].colors[:n]
    else:
        colors = mpl.colormaps["nipy_spectral"](np.linspace(0, 1, n))
    return [mpl.colors.to_hex(c) for c in colors]

def find_emptiest_corner(df, x_col='x', y_col='y', corner_fraction=0.2):
    """
    Identifies the emptiest corner in a 2D dataset defined by x and y columns.

    Args:
        df (pd.DataFrame): The input dataframe with x and y coordinates.
        x_col (str): The name of the column containing x coordinates.
        y_col (str): The name of the column containing y coordinates.
        corner_fraction (float): The fraction (0 to 0.5) of the total x and y
                                 range to define the corner area. E.g., 0.1
                                 means 10% from each edge.

    Returns:
        tuple: A tuple containing:
               - str: The name of the emptiest corner ('bottom-left',
                      'bottom-right', 'top-left', 'top-right').
               - int: The number of points found in that corner.
               Returns (None, 0) if the dataframe is empty or has no range.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns '{x_col}' and/or '{y_col}' not found in DataFrame.")
    if not (0 < corner_fraction <= 0.5):
        raise ValueError("corner_fraction must be between 0 (exclusive) and 0.5 (inclusive).")

    if df.empty:
        print("Warning: Input DataFrame is empty.")
        return None, 0

    # Ensure columns are numeric
    df[x_col] = pd.to_numeric(df[x_col], errors='coerce')
    df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
    df = df.dropna(subset=[x_col, y_col]) # Remove rows where conversion failed

    if df.empty:
        print("Warning: DataFrame is empty after removing non-numeric rows.")
        return None, 0

    # Calculate boundaries
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    # Handle cases with zero range (all points are the same)
    if x_range == 0 or y_range == 0:
         print(f"Warning: Data has zero range in {'x' if x_range == 0 else 'y'} dimension.")
         # In this degenerate case, all points are technically in all "corners"
         # We can arbitrarily return one or indicate the situation.
         # Let's count points matching the definition, though it might be the whole dataset.

    x_cutoff = x_range * corner_fraction
    y_cutoff = y_range * corner_fraction

    # Define corner boundaries
    # Note: Using <= and >= to include points exactly on the boundary line.
    corners = {
        'bottom-left': (df[x_col] <= x_min + x_cutoff) & (df[y_col] <= y_min + y_cutoff),
        'bottom-right':(df[x_col] >= x_max - x_cutoff) & (df[y_col] <= y_min + y_cutoff),
        'top-left':    (df[x_col] <= x_min + x_cutoff) & (df[y_col] >= y_max - y_cutoff),
        'top-right':   (df[x_col] >= x_max - x_cutoff) & (df[y_col] >= y_max - y_cutoff),
    }

    # Count points in each corner
    corner_counts = {name: df[condition].shape[0] for name, condition in corners.items()}

    # Find the corner with the minimum count
    # The key=lambda item: item[1] tells min to look at the second element (the count)
    emptiest_corner, min_count = min(corner_counts.items(), key=lambda item: item[1])

    return emptiest_corner, min_count

def annotate_corner(ax: mpl.axes.Axes,
                    corner: str,
                    text: str,
                    offset_points: float = 1.0,
                    **kwargs):
    """
    Places text annotation near a specified corner of a Matplotlib Axes.

    Args:
        ax (matplotlib.axes.Axes): The Axes object to annotate.
        corner (str): The corner to place the text near. Must be one of
                      'bottom-left', 'bottom-right', 'top-left', 'top-right'.
        text (str): The text string to display.
        offset_points (float): The offset distance in points from the exact
                               corner, applied equally in x and y directions
                               towards the center of the plot. Defaults to 10.0.
        **kwargs: Additional keyword arguments passed directly to ax.annotate().
                  Useful for setting fontsize, color, etc.
                  Default horizontalalignment (ha) and verticalalignment (va)
                  are set based on the corner but can be overridden.

    Returns:
        matplotlib.text.Annotation: The created annotation object.

    Raises:
        ValueError: If the provided corner string is invalid.
        TypeError: If ax is not a Matplotlib Axes object.
    """
    if not isinstance(ax, mpl.axes.Axes):
        raise TypeError(f"Input 'ax' must be a Matplotlib Axes object, not {type(ax)}")

    valid_corners = {'bottom-left', 'bottom-right', 'top-left', 'top-right'}
    if corner not in valid_corners:
        raise ValueError(f"Invalid corner '{corner}'. Must be one of {valid_corners}")

    # Get axis limits in data coordinates
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Determine the anchor point (xy) in data coords and offset (xytext) in points
    # Also set default text alignment to keep text inside the plot area
    default_alignments = {}
    if corner == 'bottom-left':
        xy = (xlim[0], ylim[0])
        xytext = (offset_points, offset_points)
        default_alignments = {'ha': 'left', 'va': 'bottom'}
    elif corner == 'bottom-right':
        xy = (xlim[1], ylim[0])
        xytext = (-offset_points, offset_points)
        default_alignments = {'ha': 'right', 'va': 'bottom'}
    elif corner == 'top-left':
        xy = (xlim[0], ylim[1])
        xytext = (offset_points, -offset_points)
        default_alignments = {'ha': 'left', 'va': 'top'}
    elif corner == 'top-right':
        xy = (xlim[1], ylim[1])
        xytext = (-offset_points, -offset_points)
        default_alignments = {'ha': 'right', 'va': 'top'}

    # Combine default alignments with user-provided kwargs
    # User kwargs take precedence
    annotate_kwargs = default_alignments.copy()
    annotate_kwargs.update(kwargs)

    # Create the annotation
    annotation = ax.annotate(
        text=text,
        xy=xy,                    # Point to annotate (the corner)
        xycoords='data',          # Use data coordinates for xy
        xytext=xytext,            # Offset in points for text placement
        textcoords='offset points', # Specify offset is in points
        **annotate_kwargs         # Pass combined alignment and other user args
    )
    annotation.set_path_effects(
        [
            mpl.patheffects.Stroke(linewidth=2, foreground="#FFFFFFAA"),
            mpl.patheffects.Normal(),
        ]
    )

    return annotation


def _cmap_with_nan_color(cmap, nan_color):
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    cmap = cmap.copy()
    cmap.set_bad(nan_color)
    return cmap


def _nanquantile(values, q):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan
    return np.quantile(values, q)


def _get_gene_values(adata, cells_oi, gene, layer=None):
    matrix = adata.layers[layer] if layer is not None else adata.X
    gene_ix = adata.var_names.get_loc(gene)
    cells_oi = np.asarray(cells_oi)
    values = matrix[cells_oi, gene_ix]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values).ravel()


def _as_transcriptome_view(transcriptome):
    if isinstance(transcriptome, sc.AnnData):
        return SimpleNamespace(
            adata=transcriptome,
            obs=transcriptome.obs,
            var=transcriptome.var,
        )
    return transcriptome


def plot_embedding(
    transcriptome: Transcriptome,
    color: List[str] = None,
    panel_size: float = 1.5,
    cmap="magma",
    cmaps=None,
    palettes=None,
    norms=(0, "q.98"),
    colors="red",
    transforms=None,
    fig=None,
    grid=None,
    layer=None,
    embedding="X_umap",
    ax=None,
    cells_oi=None,
    datashader=None,
    ncol=4,
    annotations=None,
    show_norm=True,
    size=None,
    title=None,
    title_position = "top",
    legend="on data",
    rasterized=False,
    sort=True,
    active=None,
    inactive_color="#DDDDDD",
    inactive_size=None,
    inactive_size_scale=0.5,
    facecolor=None,
    nan_color="#F2F2F2",
):
    """
    Plot cell-based features on the UMAP of the transcriptome.

    Parameters
    ----------
    transcriptome
        The transcriptome to plot. Can be a Transcriptome object or an AnnData object.
    color
        The features to plot.
    panel_size
        The size of the panel.
    cmap
        The colormap to use.
    palettes
        The palettes to use for categorical features. If None, a default palette will be used.
        If a dictionary, the keys should be the feature names and the values should be the palettes.
    colors
        The color to use for boolean features.
    grid
        The grid to add the panels to.
    embedding
        The embedding to plot.
    layer
        The layer to use.
    ax
        The axis to plot on.
    norms
        The normalization to use. If None, the default normalization will be used.
    legend
        Where to put the legend in case of categorical features. Can be "on data" or "under panel" or False/None. Can be a single value, list (same order as color) or dictionary (key is feature name).
    sort
        Whether to sort the cells by the feature value, putting the highest value on top.
    title
        The title of the panel. If None, the feature name will be used.
    title_position
        The position of the title. Can be "top" or "on data".
    active
        Column name in transcriptome.obs (or boolean array-like) indicating active cells to draw in foreground.
        Inactive cells are drawn in the background.
    inactive_color
        Color for inactive (background) cells.
    inactive_size
        Marker size for inactive (background) cells. If None, uses size * inactive_size_scale.
    inactive_size_scale
        Relative marker size for inactive (background) cells when inactive_size is None.
    nan_color
        Color used for NaN values in continuous plots. NaNs are excluded when determining norms.
    """
    transcriptome = _as_transcriptome_view(transcriptome)

    # if not isinstance(transcriptome, Transcriptome):
    #     raise ValueError("transcriptome must be a Transcriptome object")

    if ax is None:
        if grid is None:
            fig = polyptich.grid.Figure(
                polyptich.grid.Wrap(padding_width=0.1, ncol=ncol, padding_height=0.3)
            )

            grid = fig.main
        else:
            fig = None

    if embedding not in transcriptome.adata.obsm:
        raise ValueError(f"Could not find embedding {embedding}")

    if cells_oi is None:
        cells_oi = np.ones(len(transcriptome.adata), dtype=bool)

    plotdata = plotdata_raw = pd.DataFrame(
        {
            "x": transcriptome.adata.obsm[embedding][cells_oi, 0],
            "y": transcriptome.adata.obsm[embedding][cells_oi, 1],
        }
    )

    if active is None:
        active_mask = np.ones(len(plotdata_raw), dtype=bool)
    elif isinstance(active, str):
        if active not in transcriptome.obs.columns:
            raise ValueError(f"Could not find active column {active}")
        active_mask = transcriptome.obs.loc[cells_oi, active].astype(bool).values
    else:
        active_mask = np.asarray(active).astype(bool)
        if len(active_mask) != len(plotdata_raw):
            raise ValueError("active must be None, a column name, or a boolean array of length len(cells_oi)")

    if isinstance(color, str):
        color = [color]
    elif color is None:
        color = [transcriptome.obs.columns[0]]

    # check legend
    if isinstance(legend, str):
        legend = {feature: legend for feature in color}
    elif legend is False:
        legend = {feature: False for feature in color}
    elif legend is None:
        legend = {feature: "on data" for feature in color}
    elif isinstance(legend, list):
        legend = {feature: legend[i] for i, feature in enumerate(color)}

    if annotations is None:
        annotations = {}

    if palettes is None:
        palettes = {}
    if norms is None:
        norms = {feature: (0, "q.95") for feature in color}
    elif isinstance(norms, tuple):
        norms = {feature: norms for feature in color}
    elif isinstance(norms, str):
        norms = {feature: norms for feature in color}
    elif not isinstance(norms, dict):
        norms = {feature: norms for feature in color}
    if transforms is None:
        transforms = {}
    if cmaps is None:
        cmaps = {}
    if isinstance(colors, str):
        colors = {g: colors for g in color}

    # determine width and height based on aspect ratio
    aspect = (plotdata["y"].max() - plotdata["y"].min()) / (
        plotdata["x"].max() - plotdata["x"].min()
    )
    panel_width = panel_size
    panel_height = panel_size * aspect

    cmap_default = cmap

    for feature in color:
        plotdata = plotdata_raw.copy(deep=False)

        # determine feature and version
        version = "continuous"

        # expression
        if (feature in transcriptome.var.index) or (
            ("symbol" in transcriptome.var.columns)
            and (feature in transcriptome.var["symbol"].values)
        ):
            if feature not in transcriptome.var.index:
                label = feature
                gene = gene_id(transcriptome.var, feature, column="symbol")
            elif feature in transcriptome.var.index:
                gene = feature
                if "symbol" in transcriptome.var.columns:
                    label = transcriptome.var.loc[feature]["symbol"]
                else:
                    label = feature
            else:
                raise ValueError(f"Could not find gene {feature}")

            plotdata["z"] = _get_gene_values(transcriptome.adata, cells_oi, gene, layer=layer)

            if feature not in norms:
                q99 = plotdata["z"].quantile(0.999)
                if q99 == 0:
                    q99 = plotdata["z"].max()
                norms[feature] = mpl.colors.Normalize(0.0, q99)
            elif norms[feature] == "minmax":
                norms[feature] = mpl.colors.Normalize(
                    plotdata["z"].min(), plotdata["z"].max()
                )

        # obs feature
        elif feature in transcriptome.obs.columns:
            plotdata["z"] = transcriptome.obs.loc[cells_oi, feature].values

            if plotdata["z"].dtype == "category":
                plotdata["value"] = plotdata["z"]
                plotdata["z"] = plotdata["z"]
                version = "category"
            elif plotdata["z"].dtype in ["object", "str"]:
                plotdata["value"] = plotdata["z"]
                plotdata["z"] = plotdata["z"].astype("category")
                version = "category"
            elif plotdata["z"].dtype == "bool":
                version = "bool"
                plotdata["z"] = plotdata["z"].astype(int)
            else:
                plotdata["z"] = plotdata["z"]

            label = feature
        else:
            raise ValueError(f"Could not find feature {feature}")

        if not isinstance(norms, dict):
            norms = {feature: norms for feature in color}

        if feature not in norms:
            norms[feature] = (0, "q.98")

        if ax is None:
            current_ax = polyptich.grid.Panel((panel_width, panel_height))
        else:
            current_ax = ax
        if size is None:
            s = min(5, 10000 / len(plotdata))
        else:
            s = size
        if inactive_size is None:
            inactive_s = s * inactive_size_scale
        else:
            inactive_s = inactive_size

        plotdata_active = plotdata.loc[active_mask]
        plotdata_inactive = plotdata.loc[~active_mask]

        if len(plotdata_inactive) > 0:
            scatter_bg = current_ax.scatter(
                plotdata_inactive["x"],
                plotdata_inactive["y"],
                c=inactive_color,
                s=inactive_s,
                linewidths=0,
                clip_on=False,
            )
            if rasterized:
                scatter_bg.set_rasterized(True)
        scale = 20

        # check if datashader
        if datashader is True and ds is None:
            raise ModuleNotFoundError("datashader=True requires the optional datashader package")
        do_datashader = (
            ds is not None and len(plotdata_active) >= 10000 and datashader is not False
        )

        # actual plotting depending on version
        if version == "category":
            if feature in palettes:
                palette = palettes[feature]
                if not isinstance(palette, pd.Series):
                    palette = pd.Series(palette)
                missing = set(plotdata["z"].cat.categories) - set(palette.index)
                if missing:
                    palette = palette.reindex(
                        plotdata["z"].cat.categories, fill_value="None"
                    )

                    palette[palette == "None"] = get_colors(len(missing))
            else:
                palette = pd.Series(
                    get_colors(len(plotdata["z"].cat.categories)),
                    index=plotdata["z"].cat.categories,
                )

            if do_datashader:
                dsshow(
                    plotdata_active,
                    ds.Point("x", "y"),
                    ds.count_cat("z"),
                    color_key=palette.to_dict(),
                    aspect="equal",
                    ax=current_ax,
                    plot_width=int(panel_width * 100),
                    plot_height=int(panel_height * 100),
                    height_scale=scale,
                    width_scale=scale,
                    alpha_range=[200, 255],
                )
            else:
                scatter = current_ax.scatter(
                    plotdata_active["x"],
                    plotdata_active["y"],
                    c=palette[plotdata_active["z"]],
                    s=s,
                    linewidths=0,
                    clip_on=False,
                )
                if rasterized:
                    scatter.set_rasterized(True)

        elif version == "continuous":
            # get cmap
            if feature in cmaps:
                cmap = cmaps[feature]
            else:
                cmap = cmap_default
            cmap = _cmap_with_nan_color(cmap, nan_color)

            # get norm
            if feature not in norms:
                norm = mpl.colors.Normalize()
            elif norms[feature] == "minmax":
                norm = mpl.colors.Normalize(
                    vmin=plotdata["z"].min(), vmax=plotdata["z"].max()
                )
            elif norms[feature] == "0max":
                norm = mpl.colors.Normalize(vmin=0, vmax=plotdata["z"].max())
            elif isinstance(norms[feature], str) and "q" in norms[feature]:
                qmin, qmax = norms[feature].split("q")[0]
            elif isinstance(norms[feature], tuple):
                if norms[feature][0] == "0":
                    zmin = 0
                elif isinstance(norms[feature][0], (int, float)):
                    zmin = norms[feature][0]
                elif norms[feature][0].startswith("q"):
                    zmin = _nanquantile(plotdata["z"], float(norms[feature][0][1:]))
                elif norms[feature][0] == "min":
                    zmin = plotdata["z"].min()
                else:
                    zmin = norms[feature][0]

                if norms[feature][1] == "0":
                    zmax = 0
                elif isinstance(norms[feature][1], (int, float)):
                    zmax = norms[feature][1]
                elif norms[feature][1].startswith("q"):
                    zmax = _nanquantile(plotdata["z"], float(norms[feature][1][1:]))
                elif norms[feature][1] == "max":
                    zmax = plotdata["z"].max()
                else:
                    zmax = norms[feature][1]
                norm = mpl.colors.Normalize(zmin, zmax + 1e-8)
            elif isinstance(norms[feature], mpl.colors.Normalize):
                norm = norms[feature]
            else:
                raise ValueError(f"Unknown normalization {norms[feature]}")

            # decide datashader or not
            if do_datashader:
                dsshow(
                    plotdata_active,
                    ds.Point("x", "y"),
                    ds.mean("z"),
                    cmap=cmap,
                    norm=norm,
                    aspect="equal",
                    ax=current_ax,
                    plot_width=int(panel_width * 100),
                    plot_height=int(panel_height * 100),
                    height_scale=scale,
                    width_scale=scale,
                    alpha_range=[200, 255],
                )
            else:
                if sort:
                    plotdata_active = plotdata_active.sort_values("z", na_position="first")
                scatter = current_ax.scatter(
                    plotdata_active["x"],
                    plotdata_active["y"],
                    c=plotdata_active["z"].values,
                    s=s,
                    cmap=cmap,
                    norm=norm,
                    plotnonfinite=True,
                    linewidths=0,
                    clip_on=False,
                )

                if rasterized:
                    scatter.set_rasterized(True)

        elif version == "bool":
            if do_datashader:
                dsshow(
                    plotdata_active,
                    ds.Point("x", "y"),
                    ds.mean("z"),
                    cmap=["grey", colors[feature]],
                    norm=mpl.colors.Normalize(vmin=0, vmax=1),
                    aspect="equal",
                    ax=current_ax,
                    plot_width=int(panel_width * 100),
                    plot_height=int(panel_height * 100),
                    height_scale=scale,
                    width_scale=scale,
                    alpha_range=[200, 255],
                )
            else:
                plotdata_active = plotdata_active.sort_values("z")
                plotdata_1 = plotdata_active[plotdata_active["z"] == 1]
                plotdata_0 = plotdata_active[plotdata_active["z"] == 0]
                scatter = current_ax.scatter(
                    plotdata_0["x"],
                    plotdata_0["y"],
                    c="grey",
                    s=s,
                    linewidths=0,
                    clip_on=False,
                )
                if rasterized:
                    scatter.set_rasterized(True)
                scatter = current_ax.scatter(
                    plotdata_1["x"],
                    plotdata_1["y"],
                    c=colors[feature],
                    s=s,
                    linewidths=0,
                    clip_on=False,
                )
                if rasterized:
                    scatter.set_rasterized(True)

        current_ax.set_xlim(plotdata["x"].min(), plotdata["x"].max())
        current_ax.set_ylim(plotdata["y"].min(), plotdata["y"].max())
        current_ax.spines["top"].set_visible(False)
        current_ax.spines["right"].set_visible(False)
        current_ax.spines["bottom"].set_visible(False)
        current_ax.spines["left"].set_visible(False)
        current_ax.set_xticks([])
        current_ax.set_yticks([])
        if facecolor is not None:
            current_ax.set_facecolor(facecolor)

        if title is None:
            current_title = label
        else:
            if isinstance(title, str):
                current_title = title
            elif isinstance(title, dict):
                current_title = title.get(feature, label)
            # check if function
            elif callable(title):
                current_title = title(feature)
            elif title is False:
                current_title = False
            else:
                raise ValueError("title must be a string or a dictionary")
        if feature in annotations:
            current_title = label + "\n" + annotations[feature]

        if current_title is False:
            pass
        elif title_position == "top":
            current_ax.set_title(current_title, fontsize=10)
        else:
            if title_position == "on data":
            # get empty corner
                corner, _ = find_emptiest_corner(plotdata)
            else:
                corner = title_position
            # annotate corner
            annotate_corner(
                current_ax,
                corner=corner,
                text=current_title,
                offset_points=1,
                fontsize=8,
                color="black",
            )
            # raise ValueError("title_position must be 'top' or 'on data'")

        if version == "category":
            if legend[feature] is False:
                continue
            elif legend[feature] == "on data":
                texts = []
                plotdata_grouped = plotdata_active.groupby("value", observed=True)[
                    ["x", "y"]
                ].median()
                for i, row in plotdata_grouped.iterrows():
                    text = current_ax.text(
                        row["x"],
                        row["y"],
                        i,
                        fontsize=8,
                        color=palette[i],
                        fontweight="bold",
                    )
                    text.set_path_effects(
                        [
                            mpl.patheffects.Stroke(linewidth=3, foreground="#FFFFFFAA"),
                            mpl.patheffects.Stroke(linewidth=1, foreground="#333333"),
                            mpl.patheffects.Normal(),
                        ]
                    )
                    texts.append(text)

                try:
                    import adjustText

                    adjustText.adjust_text(
                        texts,
                        arrowprops=dict(arrowstyle="->", color="black"),
                        ax=current_ax,
                    )

                    # fig.plot_hooks.append(
                    #     lambda:
                    # )
                except ImportError:
                    pass

        if version == "continuous" and show_norm:
            if (norm.vmin is not None) and (norm.vmax is not None):
                text = current_ax.text(
                    0.0,
                    0.0,
                    f"[{norm.vmin:.1f}, {norm.vmax:.1f}]",
                    fontsize=6,
                    color="grey",
                    transform=current_ax.transAxes,
                )
                text.set_path_effects(
                    [
                        mpl.patheffects.Stroke(linewidth=3, foreground="#FFFFFF"),
                        mpl.patheffects.Normal(),
                    ]
                )

        # add legend optionally
        if version == "category" and legend[feature] in ["under panel", "above panel"]:
            height = np.ceil(len(palette) / 2) * 0.15
            legend_ax = polyptich.grid.Panel((panel_width, height))
            legend_ax.set_xlim(0, 1)
            legend_ax.set_ylim(0, 1)

            handles = []
            for i, (category, color) in enumerate(palette.items()):
                handles.append(
                    mpl.patches.Circle(
                        (0, 0),
                        facecolor=color,
                        edgecolor=None,
                        label=category,
                    )
                )
            legend_ax.legend(
                handles=handles,
                loc="upper center",
                ncol=2,
                fontsize=6,
                frameon=False,
            )
            legend_ax.axis("off")

            if legend[feature] == "under panel":
                current_ax = polyptich.grid.Grid(
                    [[current_ax], [legend_ax]],
                    padding_width=0,
                    padding_height=0,
                    margin_top=0,
                    margin_bottom=0,
                )
            else:
                current_ax = polyptich.grid.Grid(
                    [[legend_ax], [current_ax]],
                    padding_width=0,
                    padding_height=0,
                    margin_top=0,
                    margin_bottom=0,
                )

        # if ax is None:
        if ax is None:
            grid.add(current_ax)

    return fig


def plot_umap(
    transcriptome: Transcriptome,
    color: List[str] = None,
    panel_size: float = 1.5,
    cmap="magma",
    cmaps=None,
    palettes=None,
    norms=(0, "q.98"),
    colors="tomato",
    transforms=None,
    fig=None,
    grid=None,
    layer=None,
    ax=None,
    cells_oi=None,
    datashader=None,
    ncol=4,
    annotations=None,
    show_norm=False,
    title=None,
    title_position = "top",
    legend: str ="on data",
    active=None,
    inactive_color="#DDDDDD",
    inactive_size=None,
    inactive_size_scale=0.5,
    facecolor=None,
    nan_color="#F2F2F2",
    **kwargs,
) -> polyptich.grid.Figure:
    return plot_embedding(
        transcriptome=transcriptome,
        color=color,
        panel_size=panel_size,
        cmap=cmap,
        cmaps=cmaps,
        palettes=palettes,
        norms=norms,
        colors=colors,
        transforms=transforms,
        fig=fig,
        grid=grid,
        layer=layer,
        embedding="X_umap",
        ax=ax,
        cells_oi=cells_oi,
        datashader=datashader,
        ncol=ncol,
        annotations=annotations,
        show_norm=show_norm,
        title=title,
        title_position=title_position,
        legend=legend,
        active=active,
        inactive_color=inactive_color,
        inactive_size=inactive_size,
        inactive_size_scale=inactive_size_scale,
        facecolor=facecolor,
        nan_color=nan_color,
        **kwargs,
    )


plot_umap.__doc__ = plot_embedding.__doc__


def plot_umap_categories(
    transcriptome: Transcriptome,
    feature: str,
    panel_size: float = 2.0,
    colors="red",
    fig=None,
    grid=None,
    labels=None,
    **kwargs,
) -> polyptich.grid.Figure:
    if grid is None:
        fig = polyptich.grid.Figure(polyptich.grid.Wrap())
        grid = fig.main
    for category in transcriptome.obs[feature].cat.categories:
        category_feature = f"__{feature}_{category}_active"
        while category_feature in transcriptome.obs.columns:
            category_feature = "_" + category_feature
        if labels is None:
            title = category
        else:
            title = labels[category]
        transcriptome.obs[category_feature] = (transcriptome.obs[feature] == category).values
        try:
            plot_umap(
                transcriptome=transcriptome,
                color=[category_feature],
                panel_size=panel_size,
                colors=colors,
                grid=grid,
                title=title,
                **kwargs,
            )
        finally:
            del transcriptome.obs[category_feature]
    return fig


def plot_umap_categorized(
    transcriptome: Transcriptome,
    feature: str,
    color,
    panel_size: float = 2.0,
    fig=None,
    grid=None,
    **kwargs,
) -> polyptich.grid.Figure:
    if grid is None:
        fig = polyptich.grid.Figure(polyptich.grid.Wrap())
        grid = fig.main
    for category in transcriptome.obs[feature].cat.categories:
        ax = grid.add(polyptich.grid.Panel((panel_size, panel_size)))
        plot_umap(
            transcriptome=transcriptome,
            cells_oi=(transcriptome.obs[feature] == category).values,
            color=color,
            panel_size=panel_size,
            ax=ax,
            title=category,
            **kwargs,
        )
    return fig


def plot_umap_categorized(
    transcriptome: Transcriptome,
    feature: str,
    color: str,
    categories: List[str] = None,
    panel_size: float = 1.5,
    cmap="magma",
    cmaps=None,
    norms=(0, "q.98"),
    transforms=None,
    fig=None,
    grid=None,
    layer=None,
    embedding="X_umap",
    ax=None,
    cells_oi=None,
    ncol=4,
    annotations=None,
    show_norm=True,
    size=None,
    title=None,
    rasterized=False,
    sort=True,
    background_cells_color = "#DDDDDD",
    sort_categories = "prevalence",
):
    """
    Plot something (`color`) on several panels, one for each category of a categorical feature (`feature`).

    Parameters
    ----------
    transcriptome
        The transcriptome to plot. Can be a Transcriptome object or an AnnData object.
    color
        The feature to plot.
    feature
        The categorical feature to split the panels by.
    categories
        The categories of the feature to plot. If None, all categories will be plotted.
    panel_size
        The size of the panel.
    cmap
        The colormap to use.
    grid
        The grid to add the panels to.
    embedding
        The embedding to plot.
    layer
        The layer to use.
    ax
        The axis to plot on.
    norms
        The normalization to use. If None, the default normalization will be used.
    legend
        Where to put the legend in case of categorical features. Can be "on data" or "under panel" or False/None. Can be a single value, list (same order as color) or dictionary (key is feature name).
    sort
        Whether to sort the cells by the feature value, putting the highest value on top.
    """
    transcriptome = _as_transcriptome_view(transcriptome)

    if ax is None:
        if grid is None:
            fig = polyptich.grid.Figure(
                polyptich.grid.Wrap(padding_width=0.1, ncol=ncol, padding_height=0.3)
            )

            grid = fig.main
        else:
            fig = None

    if embedding not in transcriptome.adata.obsm:
        raise ValueError(f"Could not find embedding {embedding}")

    if cells_oi is None:
        cells_oi = np.ones(len(transcriptome.adata), dtype=bool)

    plotdata = pd.DataFrame(
        {
            "x": transcriptome.adata.obsm[embedding][cells_oi, 0],
            "y": transcriptome.adata.obsm[embedding][cells_oi, 1],
        }
    )
    print(1)

    if annotations is None:
        annotations = None
    if norms is None:
        norms = (0, "q.95")
    elif isinstance(norms, tuple):
        norms = {color: norms}
    elif isinstance(norms, str):
        norms = {color: norms}
    elif not isinstance(norms, dict):
        norms = {color: norms}
    if transforms is None:
        transforms = None
    if cmaps is None:
        cmaps = None

    # determine width and height based on aspect ratio
    aspect = (plotdata["y"].max() - plotdata["y"].min()) / (
        plotdata["x"].max() - plotdata["x"].min()
    )
    panel_width = panel_size
    panel_height = panel_size * aspect

    cmap_default = cmap


    # color
    if (color in transcriptome.var.index) or (
        ("symbol" in transcriptome.var.columns)
        and (color in transcriptome.var["symbol"].values)
    ):
        if color not in transcriptome.var.index:
            label = color
            gene = gene_id(transcriptome.var, color, column="symbol")
        elif color in transcriptome.var.index:
            gene = color
            if "symbol" in transcriptome.var.columns:
                label = transcriptome.var.loc[color]["symbol"]
            else:
                label = color
        else:
            raise ValueError(f"Could not find gene {color}")

        plotdata["z"] = _get_gene_values(transcriptome.adata, cells_oi, gene, layer=layer)

        if color not in norms:
            q99 = plotdata["z"].quantile(0.999)
            if q99 == 0:
                q99 = plotdata["z"].max()
            norms[color] = mpl.colors.Normalize(0.0, q99)
        elif norms[color] == "minmax":
            norms[color] = mpl.colors.Normalize(
                plotdata["z"].min(), plotdata["z"].max()
            )
    elif color in transcriptome.obs.columns:
        plotdata["z"] = transcriptome.obs.loc[cells_oi, color].values

        print(plotdata["z"].dtype)
        if plotdata["z"].dtype == "category":
            plotdata["value"] = plotdata["z"]
            plotdata["z"] = plotdata["z"]
            version = "category"
        elif plotdata["z"].dtype in ["object", "str"]:
            plotdata["value"] = plotdata["z"]
            plotdata["z"] = plotdata["z"].astype("category")
            version = "category"
        elif plotdata["z"].dtype == "bool":
            version = "bool"
            plotdata["z"] = plotdata["z"].astype(int)
        else:
            plotdata["z"] = plotdata["z"]
        label = color
    else:
        raise ValueError(f"Could not find color {color}")
    
    # feature
    plotdata["feature"] = transcriptome.obs.loc[cells_oi, feature].values

    if size is None:
        s = min(5, 10000 / len(plotdata))
    else:
        s = size

    # get cmap
    if cmap is None:
        cmap = cmap_default
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    # get norm
    if color not in norms:
        norm = mpl.colors.Normalize()
    elif norms[color] == "minmax":
        norm = mpl.colors.Normalize(
            vmin=plotdata["z"].min(), vmax=plotdata["z"].max()
        )
    elif norms[color] == "0max":
        norm = mpl.colors.Normalize(vmin=0, vmax=plotdata["z"].max())
    elif isinstance(norms[color], str) and "q" in norms[color]:
        qmin, qmax = norms[color].split("q")[0]
    elif isinstance(norms[color], tuple):
        if norms[color][0] == "0":
            zmin = 0
        elif isinstance(norms[color][0], (int, float)):
            zmin = norms[color][0]
        elif norms[color][0].startswith("q"):
            zmin = np.quantile(plotdata["z"], float(norms[color][0][1:]))
        elif norms[color][0] == "min":
            zmin = plotdata["z"].min()
        else:
            zmin = norms[color][0]

        if norms[color][1] == "0":
            zmax = 0
        elif isinstance(norms[color][1], (int, float)):
            zmax = norms[color][1]
        elif norms[color][1].startswith("q"):
            zmax = np.quantile(plotdata["z"], float(norms[color][1][1:]))
        elif norms[color][1] == "max":
            zmax = plotdata["z"].max()
        else:
            zmax = norms[color][1]
        norm = mpl.colors.Normalize(zmin, zmax + 1e-8)
    elif isinstance(norms[color], mpl.colors.Normalize):
        norm = norms[color]
    else:
        raise ValueError(f"Unknown normalization {norms[color]}")

    if sort_categories == "prevalence":
        category_counts = plotdata["feature"].value_counts()
        category_order = category_counts.sort_values(ascending=False).index
        plotdata["feature"] = pd.Categorical(plotdata["feature"], categories=category_order, ordered=True)

    for feature, plotdata_feature in plotdata.groupby("feature", observed=True):
        if categories is not None:
            if feature not in categories:
                continue
        current_ax = polyptich.grid.Panel((panel_width, panel_height))

        if sort:
            plotdata_feature = plotdata_feature.sort_values("z")
        scatter = current_ax.scatter(
            plotdata["x"],
            plotdata["y"],
            s=s*0.5,
            color = background_cells_color,
            linewidths=0,
            clip_on=False,
        )  
        scatter = current_ax.scatter(
            plotdata_feature["x"],
            plotdata_feature["y"],
            c=plotdata_feature["z"].values,
            s=s,
            cmap=cmap,
            norm=norm,
            linewidths=0,
            clip_on=False,
        )

        if rasterized:
            scatter.set_rasterized(True)

        current_ax.set_xlim(plotdata["x"].min(), plotdata["x"].max())
        current_ax.set_ylim(plotdata["y"].min(), plotdata["y"].max())
        current_ax.spines["top"].set_visible(False)
        current_ax.spines["right"].set_visible(False)
        current_ax.spines["bottom"].set_visible(False)
        current_ax.spines["left"].set_visible(False)
        current_ax.set_xticks([])
        current_ax.set_yticks([])

        if title is None:
            current_title = feature
        else:
            if isinstance(title, str):
                current_title = title
            elif isinstance(title, dict):
                current_title = title.get(feature, label)
            # check if function
            elif callable(title):
                current_title = title(feature)
            elif title is False:
                current_title = ""
            else:
                raise ValueError("title must be a string or a dictionary")

        current_ax.set_title(current_title, fontsize=10)

        if show_norm:
            if (norm.vmin is not None) and (norm.vmax is not None):
                text = current_ax.text(
                    0.0,
                    0.0,
                    f"[{norm.vmin:.1f}, {norm.vmax:.1f}]",
                    fontsize=6,
                    color="grey",
                    transform=current_ax.transAxes,
                )
                text.set_path_effects(
                    [
                        mpl.patheffects.Stroke(linewidth=3, foreground="#FFFFFF"),
                        mpl.patheffects.Normal(),
                    ]
                )

        # if ax is None:
        if ax is None:
            grid.add(current_ax)

    return fig


def create_continuous_colorbar(ax, norm=mpl.colors.Normalize(0, 1), cmap=None):
    if cmap is None:
        cmap = mpl.cm.magma
    mappable = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=cmap,
    )
    import matplotlib.pyplot as plt

    colorbar = plt.colorbar(mappable, cax=ax, orientation="vertical", extend="max")
    colorbar.set_label("Expression")
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["0", "Q95"])
