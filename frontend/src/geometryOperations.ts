import polygonClipping, { type MultiPolygon } from "polygon-clipping";
import type { XY } from "./shapes";

/** Polygons as rings of [x, y] with a closing point, outer ring first. */
export type Rings = XY[][];

function closedRings(polygon: XY[][]): Rings {
  return polygon.map((ring) => {
    const [first, last] = [ring[0], ring[ring.length - 1]];
    return first[0] === last[0] && first[1] === last[1] ? ring : [...ring, first];
  });
}

/** The union of polygons, as separate polygons when they do not touch. */
export function unionPolygons(polygons: Rings[]): Rings[] {
  const [first, ...rest] = polygons as MultiPolygon;
  return polygonClipping.union(first, ...rest).map((polygon) => closedRings(polygon as XY[][]));
}

/** A polygon minus others: none, one or several polygons, possibly with holes. */
export function subtractPolygons(subject: Rings, others: Rings[]): Rings[] {
  return polygonClipping.difference(subject as MultiPolygon[number], ...(others as MultiPolygon)).map((polygon) => closedRings(polygon as XY[][]));
}
