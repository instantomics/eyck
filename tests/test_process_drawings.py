import numpy as np
import pytest
from eyck.models import ProcessFeature, ProcessPart
from eyck.process_drawings import process_values


def part(identifier, status="value"):
    return ProcessPart(id=identifier, process_id="p", status=status)


def feature(identifier, role, coordinates, part_id="main", holes=(), **properties):
    geometry = (
        {"type": "LineString", "coordinates": coordinates}
        if role == "curve"
        else {
            "type": "Polygon",
            "coordinates": [ring + [ring[0]] for ring in [coordinates, *holes]],
        }
    )
    return ProcessFeature.model_validate(
        {
            "id": identifier,
            "geometry": geometry,
            "properties": {
                "process_id": "p",
                "part_id": part_id,
                "role": role,
                "embedding_id": properties.pop("embedding_id", "umap"),
                "annotator": "a",
                "annotation_set": "a",
                **properties,
            },
        }
    )


def box(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


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
        feature("scope", "region", box(-1.0, -1.0, 11.0, 11.0)),
        feature(
            "drop",
            "region",
            [[90.0, 90.0], [110.0, 90.0], [110.0, 110.0]],
            region_mode="exclude",
            embedding_id="other",
        ),
    ]
    values, statuses = process_values(
        [part("main")], features, {"umap": cells, "other": other}, len(cells)
    )
    assert values[:3] == pytest.approx([1.0, 0.0, 0.75])
    assert np.isnan(values[3]) and np.isnan(values[4])
    assert statuses.tolist() == ["value"] * 3 + ["absent"] * 2


def test_parts_project_onto_their_own_curves_and_the_first_part_wins():
    # Two separate branches: the second cell is nearer the lower curve but lies
    # in the upper part, so it projects onto the upper curve. The unknown and
    # absent parts listed first override the upper part where they overlap it.
    cells = np.array([[1.0, 4.0], [9.0, 5.5], [5.0, 8.0], [3.0, 8.0], [50.0, 50.0]])
    parts = [
        part("carve", "absent"),
        part("unsure", "unknown"),
        part("upper"),
        part("lower"),
    ]
    features = [
        feature("carve", "region", box(2.0, 7.0, 4.0, 9.0), part_id="carve"),
        feature("unsure", "region", box(4.0, 7.0, 6.0, 9.0), part_id="unsure"),
        feature("upper-area", "region", box(0.0, 5.0, 10.0, 10.0), part_id="upper"),
        feature(
            "upper-curve",
            "curve",
            [[0.0, 10.0], [10.0, 10.0]],
            part_id="upper",
            value_start=0.5,
            value_end=1.0,
        ),
        feature("lower-area", "region", box(0.0, 0.0, 10.0, 5.0), part_id="lower"),
        feature(
            "lower-curve",
            "curve",
            [[0.0, 5.0], [10.0, 5.0]],
            part_id="lower",
            value_end=0.5,
        ),
    ]
    values, statuses = process_values(parts, features, {"umap": cells}, len(cells))
    assert statuses.tolist() == ["value", "value", "unknown", "absent", "absent"]
    assert values[:2] == pytest.approx([0.05, 0.95])
    assert np.isnan(values[2:]).all()


def test_region_holes_and_subset_parts():
    # The second cell is absent from the subset view (NaN coordinates); the
    # third lies in the hole of the region.
    subset = np.array([[0.0, 0.0], [np.nan, np.nan], [5.0, 0.0], [10.0, 0.0]])
    ring = box(-1.0, -1.0, 11.0, 1.0)
    hole = box(4.0, -0.5, 6.0, 0.5)
    marked, statuses = process_values(
        [part("main")],
        [feature("r", "region", ring, holes=[hole], embedding_id="subset")],
        {"subset": subset},
        4,
    )
    assert marked[[0, 3]] == pytest.approx([1.0, 1.0])
    assert np.isnan(marked[[1, 2]]).all()
    assert statuses.tolist() == ["value", "absent", "absent", "value"]
