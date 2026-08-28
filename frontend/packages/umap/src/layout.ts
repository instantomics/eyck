export interface ClusterLabelInput {
  key: string;
  label: string;
  color: string;
  x: number;
  y: number;
  count: number;
}

export interface ClusterLabelLayout extends ClusterLabelInput {
  labelX: number;
  labelY: number;
  textAnchor: "middle" | "end";
  lineX: number;
  lineY: number;
}

export interface ClusterLabelLayoutOptions {
  placement: "on-cluster" | "beside";
  width: number;
  height: number;
  padding?: number;
  gap?: number;
}

export function layoutClusterLabels(
  labels: readonly ClusterLabelInput[],
  options: ClusterLabelLayoutOptions
): ClusterLabelLayout[] {
  const padding = Math.max(0, options.padding ?? 8);
  if (options.placement === "on-cluster") {
    return [...labels]
      .sort((left, right) => left.key.localeCompare(right.key, "en"))
      .map((label) => ({
        ...label,
        labelX: label.x,
        labelY: label.y,
        textAnchor: "middle",
        lineX: label.x,
        lineY: label.y
      }));
  }

  const sorted = [...labels].sort((left, right) => left.y - right.y || left.key.localeCompare(right.key, "en"));
  if (sorted.length === 0) return [];
  const available = Math.max(0, options.height - padding * 2);
  const requestedGap = Math.max(0, options.gap ?? 16);
  const gap = sorted.length === 1 ? 0 : Math.min(requestedGap, available / (sorted.length - 1));
  const positions = sorted.map((label) => Math.max(padding, Math.min(options.height - padding, label.y)));
  for (let index = 1; index < positions.length; index += 1) {
    positions[index] = Math.max(positions[index], positions[index - 1] + gap);
  }
  const overflow = positions[positions.length - 1] - (options.height - padding);
  if (overflow > 0) positions.forEach((_position, index) => { positions[index] -= overflow; });
  for (let index = positions.length - 2; index >= 0; index -= 1) {
    positions[index] = Math.min(positions[index], positions[index + 1] - gap);
  }
  const labelX = Math.max(padding, options.width - padding);
  return sorted.map((label, index) => ({
    ...label,
    labelX,
    labelY: positions[index],
    textAnchor: "end",
    lineX: Math.max(padding, labelX - 6),
    lineY: positions[index]
  }));
}
