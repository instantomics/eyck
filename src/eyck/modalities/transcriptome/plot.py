import numbers
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


def _get_gene_values(adata, positions, gene, layer=None):
    matrix = adata.layers[layer] if layer is not None else adata.X
    gene_ix = adata.var_names.get_loc(gene)
    values = matrix[positions, gene_ix]
    if hasattr(values, "toarray"):
        values = values.toarray()
    return np.asarray(values, dtype=float).ravel()


def _as_transcriptome_view(transcriptome):
    if isinstance(transcriptome, sc.AnnData):
        return SimpleNamespace(
            adata=transcriptome,
            obs=transcriptome.obs,
            var=transcriptome.var,
        )
    return transcriptome


def _cell_positions(obs_names, cells_oi):
    """Integer positions of the selected cells; accepts a boolean mask, positions or cell names."""
    if cells_oi is None:
        return np.arange(len(obs_names))
    cells_oi = np.asarray(cells_oi)
    if cells_oi.dtype == bool:
        if len(cells_oi) != len(obs_names):
            raise ValueError("A boolean cells_oi must have one value per cell")
        return np.flatnonzero(cells_oi)
    if np.issubdtype(cells_oi.dtype, np.integer):
        return cells_oi
    positions = obs_names.get_indexer(cells_oi)
    if (positions < 0).any():
        raise ValueError(f"Unknown cells in cells_oi: {list(cells_oi[positions < 0][:5])}")
    return positions


def _feature_values(transcriptome, feature, positions, layer):
    """Return the values, label, kind ("continuous", "bool" or "category") and whether it is a gene."""
    var = transcriptome.var
    has_symbol = "symbol" in var.columns
    if feature in var.index:
        gene = feature
        label = var.loc[feature, "symbol"] if has_symbol else feature
    elif has_symbol and feature in var["symbol"].values:
        gene = gene_id(var, feature, column="symbol")
        label = feature
    elif feature in transcriptome.obs.columns:
        values = transcriptome.obs[feature].iloc[positions].reset_index(drop=True)
        if pd.api.types.is_bool_dtype(values.dtype):
            return values.astype(float), feature, "bool", False
        if pd.api.types.is_numeric_dtype(values.dtype):
            return values.astype(float), feature, "continuous", False
        return values.astype("category"), feature, "category", False
    else:
        raise ValueError(f"Could not find feature {feature}")
    values = _get_gene_values(transcriptome.adata, positions, gene, layer=layer)
    return pd.Series(values), label, "continuous", True


def _norm_bound(bound, values):
    if bound is None or isinstance(bound, numbers.Real):
        return bound
    if bound == "0":
        return 0
    if bound == "min":
        return np.nanmin(values)
    if bound == "max":
        return np.nanmax(values)
    if isinstance(bound, str) and bound.startswith("q"):
        return _nanquantile(values, float(bound[1:]))
    raise ValueError(f"Unknown normalization bound {bound!r}")


def _resolve_norm(spec, values):
    """Turn a norm specification into a Normalize; NaNs are ignored when computing bounds."""
    if isinstance(spec, mpl.colors.Normalize):
        return spec
    if isinstance(spec, str):
        if spec == "minmax":
            spec = ("min", "max")
        elif spec == "0max":
            spec = (0, "max")
        elif spec.startswith("q"):
            spec = (0, spec)
    if not (isinstance(spec, tuple) and len(spec) == 2):
        raise ValueError(f"Unknown normalization {spec!r}")
    values = np.asarray(values, dtype=float)
    vmin, vmax = (_norm_bound(bound, values) for bound in spec)
    if vmax is not None:
        vmax = vmax + 1e-8
    return mpl.colors.Normalize(vmin, vmax)


def _default_gene_norm(values):
    vmax = _nanquantile(values, 0.999)
    if vmax == 0:
        vmax = np.nanmax(values)
    return mpl.colors.Normalize(0.0, vmax)


def _category_palette(categories, palette=None):
    """One color per category; categories missing from a given palette get unused default colors."""
    if palette is None:
        return pd.Series(get_colors(len(categories)), index=categories)
    palette = pd.Series(palette).reindex(categories)
    missing = palette.isna()
    if missing.any():
        used = set(palette.dropna())
        defaults = get_colors(len(categories) + len(used))
        palette[missing] = [c for c in defaults if c not in used][: missing.sum()]
    return palette


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
        The features to plot: genes (by id or symbol) or obs columns.
    panel_size
        The size of the panel.
    cmap
        The colormap to use.
    cmaps
        Per-feature colormaps (dictionary), overriding cmap.
    palettes
        The palettes to use for categorical features. If None, a default palette will be used.
        If a dictionary, the keys should be the feature names and the values should be the palettes.
        Categories missing from a palette get default colors not used by it.
    colors
        The color to use for boolean features.
    transforms
        A function, or a dictionary of functions per feature, applied to continuous values before
        normalization, e.g. np.log1p.
    grid
        The grid to add the panels to.
    embedding
        The embedding to plot.
    layer
        The layer to use.
    ax
        The axis to plot on.
    cells_oi
        The cells to plot, as a boolean mask, integer positions or cell names.
    norms
        The normalization to use, as a single value or a dictionary per feature. A value can be a
        Normalize, "minmax", "0max", a quantile string such as "q.99" (from 0 to that quantile), or a
        (min, max) tuple whose bounds are numbers, "0", "min", "max", or quantile strings.
        If None, (0, "q.95") is used. Genes missing from a dictionary are scaled from 0 to their 99.9%
        quantile, other features to their 98% quantile.
    legend
        Where to put the legend in case of categorical features. Can be "on data", "under panel",
        "above panel", or False to only color the cells. Can be a single value, list (same order as
        color) or dictionary (key is feature name); features missing from a dictionary use "on data".
    sort
        Whether to sort the cells by the feature value, putting the highest value on top.
    title
        The title of the panel. If None, the feature name will be used.
    title_position
        The position of the title. Can be "top", "on data", or a corner such as "top-left".
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
        Color used for missing values, drawn below the other cells. NaNs are excluded when
        determining norms.
    """
    transcriptome = _as_transcriptome_view(transcriptome)
    adata = transcriptome.adata

    if ax is None and grid is None:
        fig = polyptich.grid.Figure(
            polyptich.grid.Wrap(padding_width=0.1, ncol=ncol, padding_height=0.3)
        )
        grid = fig.main

    if embedding not in adata.obsm:
        raise ValueError(f"Could not find embedding {embedding}")

    positions = _cell_positions(adata.obs_names, cells_oi)
    coordinates = np.asarray(adata.obsm[embedding])[positions]
    plotdata_raw = pd.DataFrame({"x": coordinates[:, 0], "y": coordinates[:, 1]})

    if active is None:
        active_mask = np.ones(len(plotdata_raw), dtype=bool)
    elif isinstance(active, str):
        if active not in transcriptome.obs.columns:
            raise ValueError(f"Could not find active column {active}")
        active_mask = transcriptome.obs[active].iloc[positions].astype(bool).values
    else:
        active_mask = np.asarray(active).astype(bool)
        if len(active_mask) != len(plotdata_raw):
            raise ValueError("active must be None, a column name, or a boolean array of length len(cells_oi)")

    if isinstance(color, str):
        color = [color]
    elif color is None:
        color = [transcriptome.obs.columns[0]]

    if legend is None:
        legend = "on data"
    if isinstance(legend, list):
        legend = dict(zip(color, legend))
    elif not isinstance(legend, dict):
        legend = {feature: legend for feature in color}

    if annotations is None:
        annotations = {}
    if palettes is None:
        palettes = {}
    if norms is None:
        norms = (0, "q.95")
    if not isinstance(norms, dict):
        norms = {feature: norms for feature in color}
    if transforms is None:
        transforms = {}
    elif callable(transforms):
        transforms = {feature: transforms for feature in color}
    if cmaps is None:
        cmaps = {}
    if not isinstance(colors, dict):
        colors = {feature: colors for feature in color}

    # determine width and height based on aspect ratio
    aspect = (plotdata_raw["y"].max() - plotdata_raw["y"].min()) / (
        plotdata_raw["x"].max() - plotdata_raw["x"].min()
    )
    panel_width = panel_size
    panel_height = panel_size * aspect

    if size is None:
        s = min(5, 10000 / len(plotdata_raw))
    else:
        s = size
    if inactive_size is None:
        inactive_s = s * inactive_size_scale
    else:
        inactive_s = inactive_size
    scale = 20

    if datashader is True and ds is None:
        raise ModuleNotFoundError("datashader=True requires the optional datashader package")
    do_datashader = (
        ds is not None and active_mask.sum() >= 10000 and datashader is not False
    )

    def scatter(current_ax, data, **kwargs):
        kwargs.setdefault("s", s)
        artist = current_ax.scatter(
            data["x"], data["y"], linewidths=0, clip_on=False, **kwargs
        )
        if rasterized:
            artist.set_rasterized(True)

    def datashade(current_ax, data, aggregation, **kwargs):
        dsshow(
            data,
            ds.Point("x", "y"),
            aggregation,
            aspect="equal",
            ax=current_ax,
            plot_width=int(panel_width * 100),
            plot_height=int(panel_height * 100),
            height_scale=scale,
            width_scale=scale,
            alpha_range=[200, 255],
            **kwargs,
        )

    for feature in color:
        plotdata = plotdata_raw.copy(deep=False)
        plotdata["z"], label, version, is_gene = _feature_values(
            transcriptome, feature, positions, layer
        )
        if version == "continuous" and feature in transforms:
            plotdata["z"] = transforms[feature](plotdata["z"])

        if ax is None:
            current_ax = polyptich.grid.Panel((panel_width, panel_height))
        else:
            current_ax = ax

        plotdata_active = plotdata.loc[active_mask]
        plotdata_inactive = plotdata.loc[~active_mask]
        if len(plotdata_inactive) > 0:
            scatter(current_ax, plotdata_inactive, c=inactive_color, s=inactive_s)

        # missing values go below the other active cells
        missing = plotdata_active["z"].isna()
        if missing.any():
            scatter(current_ax, plotdata_active.loc[missing], c=nan_color)
            plotdata_active = plotdata_active.loc[~missing]

        # actual plotting depending on version
        if version == "category":
            palette = _category_palette(
                plotdata["z"].cat.categories, palettes.get(feature)
            )
            if do_datashader:
                datashade(
                    current_ax,
                    plotdata_active,
                    ds.count_cat("z"),
                    color_key=palette.to_dict(),
                )
            else:
                scatter(
                    current_ax,
                    plotdata_active,
                    c=palette.loc[plotdata_active["z"].astype(object)].values,
                )

        elif version == "continuous":
            cmap_feature = _cmap_with_nan_color(cmaps.get(feature, cmap), nan_color)
            if feature in norms:
                norm = _resolve_norm(norms[feature], plotdata["z"])
            elif is_gene:
                norm = _default_gene_norm(plotdata["z"])
            else:
                norm = _resolve_norm((0, "q.98"), plotdata["z"])

            if do_datashader:
                datashade(
                    current_ax,
                    plotdata_active,
                    ds.mean("z"),
                    cmap=cmap_feature,
                    norm=norm,
                )
            else:
                if sort:
                    plotdata_active = plotdata_active.sort_values("z")
                scatter(
                    current_ax,
                    plotdata_active,
                    c=plotdata_active["z"].values,
                    cmap=cmap_feature,
                    norm=norm,
                    plotnonfinite=True,
                )

        elif version == "bool":
            if do_datashader:
                datashade(
                    current_ax,
                    plotdata_active,
                    ds.mean("z"),
                    cmap=["grey", colors.get(feature, "red")],
                    norm=mpl.colors.Normalize(vmin=0, vmax=1),
                )
            else:
                scatter(current_ax, plotdata_active[plotdata_active["z"] == 0], c="grey")
                scatter(
                    current_ax,
                    plotdata_active[plotdata_active["z"] == 1],
                    c=colors.get(feature, "red"),
                )

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
        elif isinstance(title, str) or title is False:
            current_title = title
        elif isinstance(title, dict):
            current_title = title.get(feature, label)
        elif callable(title):
            current_title = title(feature)
        else:
            raise ValueError("title must be a string, dictionary, function or False")
        if feature in annotations:
            if current_title is False:
                current_title = annotations[feature]
            else:
                current_title = f"{current_title}\n{annotations[feature]}"

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

        feature_legend = legend.get(feature, "on data")
        if version == "category" and feature_legend == "on data":
            texts = []
            plotdata_grouped = plotdata_active.groupby("z", observed=True)[
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
        if version == "category" and feature_legend in ["under panel", "above panel"]:
            height = np.ceil(len(palette) / 2) * 0.15
            legend_ax = polyptich.grid.Panel((panel_width, height))
            legend_ax.set_xlim(0, 1)
            legend_ax.set_ylim(0, 1)

            handles = []
            for i, (category, category_color) in enumerate(palette.items()):
                handles.append(
                    mpl.patches.Circle(
                        (0, 0),
                        facecolor=category_color,
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

            if feature_legend == "under panel":
                rows = [[current_ax], [legend_ax]]
            else:
                rows = [[legend_ax], [current_ax]]
            current_ax = polyptich.grid.Grid(
                rows,
                padding_width=0,
                padding_height=0,
                margin_top=0,
                margin_bottom=0,
            )

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
    color: str,
    categories: List[str] = None,
    panel_size: float = 1.5,
    fig=None,
    grid=None,
    ax=None,
    cells_oi=None,
    ncol=4,
    title=None,
    background_cells_color="#DDDDDD",
    sort_categories=None,
    **kwargs,
) -> polyptich.grid.Figure:
    """
    Plot something (`color`) on several panels, one for each category of a categorical feature (`feature`).
    The cells of the category are drawn on top of all other cells, and colors are shared across panels.

    Parameters
    ----------
    transcriptome
        The transcriptome to plot. Can be a Transcriptome object or an AnnData object.
    feature
        The categorical feature to split the panels by.
    color
        The feature to plot: a gene or an obs column (continuous, boolean or categorical).
    categories
        The categories of the feature to plot. If None, all categories with cells will be plotted.
    panel_size
        The size of the panel.
    grid
        The grid to add the panels to.
    cells_oi
        The cells to plot, as a boolean mask, integer positions or cell names.
    title
        The panel titles: None for the category names, a string, a dictionary per category, a
        function of the category, or False.
    background_cells_color
        Color of the cells outside the category.
    sort_categories
        "prevalence" orders panels from the largest to the smallest category; None keeps the category
        order of `feature`.
    **kwargs
        Passed to plot_embedding, e.g. embedding, norms, cmap, layer, size, datashader or show_norm.
    """
    view = _as_transcriptome_view(transcriptome)
    positions = _cell_positions(view.adata.obs_names, cells_oi)
    groups = pd.Categorical(view.obs[feature].iloc[positions])
    counts = pd.Series(groups).value_counts(sort=False)
    counts = counts[counts > 0]

    if sort_categories == "prevalence":
        counts = counts.sort_values(ascending=False, kind="stable")
    elif sort_categories is not None:
        raise ValueError("sort_categories must be 'prevalence' or None")
    order = [
        category
        for category in counts.index
        if categories is None or category in categories
    ]

    if ax is None and grid is None:
        fig = polyptich.grid.Figure(
            polyptich.grid.Wrap(padding_width=0.1, ncol=ncol, padding_height=0.3)
        )
        grid = fig.main

    for category in order:
        if title is None:
            category_title = category
        elif isinstance(title, dict):
            category_title = title.get(category, category)
        elif callable(title):
            category_title = title(category)
        else:
            category_title = title
        plot_embedding(
            transcriptome,
            color=[color],
            panel_size=panel_size,
            grid=grid,
            ax=ax,
            cells_oi=positions,
            active=np.asarray(groups == category),
            inactive_color=background_cells_color,
            title=category_title,
            **kwargs,
        )
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
    colorbar.set_ticks([norm.vmin, norm.vmax])
    colorbar.set_ticklabels(["0", "Q95"])
