import { describe, expect, it } from "vitest";
import { layoutClusterLabels, type ClusterLabelInput } from "../src/layout";

const labels: ClusterLabelInput[] = [
  { key: "b", label: "B", color: "blue", x: 20, y: 19, count: 2 },
  { key: "a", label: "A", color: "red", x: 10, y: 20, count: 3 },
  { key: "c", label: "C", color: "green", x: 30, y: 21, count: 4 }
];

describe("cluster label layout", () => {
  it("keeps on-cluster labels at centroids in deterministic key order", () => {
    const layout = layoutClusterLabels(labels, { placement: "on-cluster", width: 100, height: 80 });
    expect(layout.map((label) => [label.key, label.labelX, label.labelY])).toEqual([
      ["a", 10, 20],
      ["b", 20, 19],
      ["c", 30, 21]
    ]);
  });

  it("separates beside labels and preserves leader origins", () => {
    const layout = layoutClusterLabels(labels, {
      placement: "beside",
      width: 100,
      height: 80,
      padding: 8,
      gap: 16
    });
    expect(layout.map((label) => label.key)).toEqual(["b", "a", "c"]);
    expect(layout.map((label) => label.labelY)).toEqual([19, 35, 51]);
    expect(layout.map((label) => label.labelX)).toEqual([92, 92, 92]);
    expect(layout[0]).toMatchObject({ x: 20, y: 19, lineX: 86, lineY: 19 });
  });
});
