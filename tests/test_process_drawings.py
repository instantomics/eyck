import numpy as np
import pytest
from eyck.models import ProcessFeature
from eyck.process_drawings import process_values


def feature(identifier, role, coordinates, **properties):
    geometry = (
        {"type": "LineString", "coordinates": coordinates}
        if role == "curve"
        else {"type": "Polygon", "coordinates": [coordinates + [coordinates[0]]]}
    )
    return ProcessFeature.model_validate(
        {
            "id": identifier,
            "geometry": geometry,
            "properties": {
                "process_id": "p",
                "role": role,
                "embedding_id": properties.pop("embedding_id", "umap"),
                "annotator": "a",
                "annotation_set": "a",
                **properties,
            },
        }
    )


def test_cells_take_the_position_on_their_nearest_curve_within_scope():
    cells = np.array([[0.0, 0.1], [10.0, -0.1], [5.0, 9.0], [5.0, 20.0], [2.0, 0.0]])
    other = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [105.0, 95.0]])
    features = [
        # Drawn right to left: the start of the drawing is 0.
        feature("main", "curve", [[10.0, 0.0], [0.0, 0.0]]),
        feature(
            "branch",
            "curve",
            [[0.0, 10.0], [10.0, 10.0]],
            value_start=0.5,
            value_end=1.0,
        ),
        feature(
            "scope", "region", [[-1.0, -1.0], [11.0, -1.0], [11.0, 11.0], [-1.0, 11.0]]
        ),
        feature(
            "drop",
            "region",
            [[90.0, 90.0], [110.0, 90.0], [110.0, 110.0]],
            region_mode="exclude",
            embedding_id="other",
        ),
    ]
    values = process_values(features, {"umap": cells, "other": other}, len(cells))
    assert values[:3] == pytest.approx([1.0, 0.0, 0.75])
    assert np.isnan(values[3]) and np.isnan(values[4])


def test_subset_views_and_region_only_processes():
    # The second cell is absent from the subset view (NaN coordinates).
    subset = np.array([[0.0, 0.0], [np.nan, np.nan], [10.0, 0.0]])
    box = [[-1.0, -1.0], [11.0, -1.0], [11.0, 1.0], [-1.0, 1.0]]
    curve = feature("c", "curve", [[0.0, 0.0], [10.0, 0.0]], embedding_id="subset")
    values = process_values([curve], {"subset": subset}, 3)
    assert values[[0, 2]] == pytest.approx([0.0, 1.0]) and np.isnan(values[1])
    marked = process_values(
        [feature("r", "region", box, embedding_id="subset")], {"subset": subset}, 3
    )
    assert marked[[0, 2]] == pytest.approx([1.0, 1.0]) and np.isnan(marked[1])
