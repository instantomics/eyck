import type { ProcessFeature } from "./types";

export type XY = [number, number];

export function isClosed(feature: ProcessFeature): boolean {
  return feature.geometry.type === "Polygon";
}

/** Editable vertices: a curve's points, or a region's ring without its closing point. */
export function openVertices(feature: ProcessFeature): XY[] {
  return feature.geometry.type === "LineString" ? feature.geometry.coordinates : feature.geometry.coordinates[0].slice(0, -1);
}

export function withVertices(feature: ProcessFeature, vertices: XY[]): ProcessFeature {
  return {
    ...feature,
    geometry: feature.geometry.type === "LineString"
      ? { type: "LineString", coordinates: vertices }
      : { type: "Polygon", coordinates: [[...vertices, vertices[0]]] }
  };
}

export function minVertices(feature: ProcessFeature): number {
  return isClosed(feature) ? 3 : 2;
}

/**
 * Weights for dragging vertex `from`: 1 there, falling smoothly to 0 at `radius`
 * pixels of path length, so a dense hand-drawn line bends instead of spiking.
 */
export function dragWeights(screen: XY[], from: number, closed: boolean, radius: number): number[] {
  const step = screen.map((point, index) => {
    const next = screen[(index + 1) % screen.length];
    return Math.hypot(next[0] - point[0], next[1] - point[1]);
  });
  const total = closed ? step.reduce((sum, value) => sum + value, 0) : 0;
  const forward = new Array<number>(screen.length).fill(Infinity);
  forward[from] = 0;
  for (let offset = 1; offset < screen.length; offset += 1) {
    const index = from + offset;
    if (!closed && index >= screen.length) break;
    forward[index % screen.length] = forward[(index - 1) % screen.length] + step[(index - 1) % screen.length];
  }
  const backward = new Array<number>(screen.length).fill(Infinity);
  backward[from] = 0;
  for (let offset = 1; offset < screen.length; offset += 1) {
    const index = from - offset;
    if (!closed && index < 0) break;
    const wrapped = (index + screen.length) % screen.length;
    backward[wrapped] = backward[(wrapped + 1) % screen.length] + step[wrapped];
  }
  return screen.map((_point, index) => {
    const distance = closed ? Math.min(forward[index], backward[index], total) : Math.min(forward[index], backward[index]);
    if (index === from) return 1;
    if (radius <= 0 || distance >= radius) return 0;
    const t = distance / radius;
    return (1 - t * t) ** 2;
  });
}

export function moved(original: XY[], weights: number[], delta: XY): XY[] {
  return original.map((point, index) => [point[0] + weights[index] * delta[0], point[1] + weights[index] * delta[1]]);
}

/** Insert a vertex on the segment nearest to a screen position. */
export function insertVertex(vertices: XY[], screen: XY[], at: XY, atEmbedding: XY, closed: boolean): XY[] {
  let best = 0;
  let bestDistance = Infinity;
  const segments = closed ? screen.length : screen.length - 1;
  for (let index = 0; index < segments; index += 1) {
    const a = screen[index];
    const b = screen[(index + 1) % screen.length];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const t = Math.max(0, Math.min(1, ((at[0] - a[0]) * dx + (at[1] - a[1]) * dy) / (dx * dx + dy * dy || 1)));
    const distance = Math.hypot(a[0] + t * dx - at[0], a[1] + t * dy - at[1]);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  }
  return [...vertices.slice(0, best + 1), atEmbedding, ...vertices.slice(best + 1)];
}

/** One pass of neighbour averaging; a curve keeps its end points. */
export function smoothed(vertices: XY[], closed: boolean): XY[] {
  if (vertices.length < 3) return vertices;
  return vertices.map((point, index) => {
    if (!closed && (index === 0 || index === vertices.length - 1)) return point;
    const before = vertices[(index - 1 + vertices.length) % vertices.length];
    const after = vertices[(index + 1) % vertices.length];
    return [0.5 * point[0] + 0.25 * (before[0] + after[0]), 0.5 * point[1] + 0.25 * (before[1] + after[1])];
  });
}
