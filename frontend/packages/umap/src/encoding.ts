import { interpolateRgbBasis } from "d3-interpolate";
import { scaleOrdinal, scaleSequential } from "d3-scale";
import type {
  Accessor,
  CategoricalEncoding,
  CategoryValue,
  ContinuousEncoding,
  LegendItem,
  NumericDomain,
  ResolvedEncoding,
  UmapBounds
} from "./types";

export const DEFAULT_CATEGORICAL_PALETTE = [
  "#2a6097",
  "#da6148",
  "#2c8766",
  "#915c9c",
  "#c98f2d",
  "#5485aa",
  "#bb5571",
  "#629247",
  "#776c5c",
  "#4595a1",
  "#8f6dce",
  "#d0772d"
] as const;

export const DEFAULT_CONTINUOUS_COLORS = ["#eef2f4", "#7eb0bf", "#1e7b9e"] as const;
export const DEFAULT_MISSING_COLOR = "#c4c8cc";
export const DEFAULT_MASKED_COLOR = "#d8dadd";

function isPresentCategory(value: CategoryValue | null | undefined): value is CategoryValue {
  return value !== null && value !== undefined && !(typeof value === "number" && !Number.isFinite(value));
}

export function categoryKey(value: CategoryValue): string {
  if (typeof value === "string") return `string:${value}`;
  if (typeof value === "boolean") return `boolean:${value ? "1" : "0"}`;
  return `number:${Object.is(value, -0) ? "0" : String(value)}`;
}

function compareCategories(left: CategoryValue, right: CategoryValue): number {
  return categoryKey(left).localeCompare(categoryKey(right), "en");
}

function uniqueCategories(values: readonly CategoryValue[]): CategoryValue[] {
  const found = new Map<string, CategoryValue>();
  values.forEach((value) => found.set(categoryKey(value), value));
  return [...found.values()];
}

function selected<T>(datum: T, index: number, mask?: Accessor<T, boolean>): boolean {
  return mask?.(datum, index) ?? true;
}

export function inferContinuousDomain(values: readonly (number | null | undefined)[]): NumericDomain {
  let minimum = Infinity;
  let maximum = -Infinity;
  values.forEach((value) => {
    if (value === null || value === undefined || !Number.isFinite(value)) return;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  });
  if (!Number.isFinite(minimum)) return [0, 1];
  if (minimum !== maximum) return [minimum, maximum];
  const halfSpan = Math.max(1, Math.abs(minimum) * 0.01);
  return [minimum - halfSpan, maximum + halfSpan];
}

function checkedDomain(domain: NumericDomain, name: string): NumericDomain {
  if (!Number.isFinite(domain[0]) || !Number.isFinite(domain[1]) || domain[0] >= domain[1]) {
    throw new Error(`${name} must contain two finite, increasing values`);
  }
  return domain;
}

export function inferUmapBounds<T>(
  data: readonly T[],
  x: Accessor<T, number>,
  y: Accessor<T, number>
): UmapBounds {
  return {
    x: inferContinuousDomain(data.map(x)),
    y: inferContinuousDomain(data.map(y))
  };
}

export function validateUmapBounds(bounds: UmapBounds): UmapBounds {
  return {
    x: checkedDomain(bounds.x, "bounds.x"),
    y: checkedDomain(bounds.y, "bounds.y")
  };
}

export function resolveCategoricalEncoding<T>(
  data: readonly T[],
  encoding: CategoricalEncoding<T>,
  mask?: Accessor<T, boolean>,
  maskedColor = DEFAULT_MASKED_COLOR
): ResolvedEncoding {
  const values = data.map(encoding.value);
  const observed = uniqueCategories(values.filter(isPresentCategory)).sort(compareCategories);
  const explicit = uniqueCategories((encoding.categories ?? []).filter(isPresentCategory));
  const explicitKeys = new Set(explicit.map(categoryKey));
  const categories = encoding.categories
    ? [...explicit, ...observed.filter((value) => !explicitKeys.has(categoryKey(value)))]
    : observed;

  const palette = encoding.palette?.length ? encoding.palette : DEFAULT_CATEGORICAL_PALETTE;
  const categoryColors = categories.map((value, index) => (
    encoding.color?.(value, index) ?? palette[index % palette.length]
  ));
  const scale = scaleOrdinal<string, string>()
    .domain(categories.map(categoryKey))
    .range(categoryColors)
    .unknown(encoding.missingColor ?? DEFAULT_MISSING_COLOR);
  const missingColor = encoding.missingColor ?? DEFAULT_MISSING_COLOR;
  const activeKeys = new Set<string>();
  let hasMissing = false;
  let hasMasked = false;
  const colors = values.map((value, index) => {
    if (!selected(data[index], index, mask)) {
      hasMasked = true;
      return maskedColor;
    }
    if (!isPresentCategory(value)) {
      hasMissing = true;
      return missingColor;
    }
    const key = categoryKey(value);
    activeKeys.add(key);
    return scale(key);
  });
  const legend: LegendItem[] = categories.flatMap((value, index) => {
    const key = categoryKey(value);
    return activeKeys.has(key) ? [{
      key,
      label: encoding.formatCategory?.(value) ?? String(value),
      color: categoryColors[index],
      kind: "value" as const
    }] : [];
  });
  if (hasMissing) legend.push({
    key: "missing",
    label: encoding.missingLabel ?? "Missing",
    color: missingColor,
    kind: "missing"
  });
  if (hasMasked) legend.push({
    key: "masked",
    label: "Not included",
    color: maskedColor,
    kind: "masked"
  });
  return { colors, legend };
}

function defaultValueFormat(value: number): string {
  return new Intl.NumberFormat("en", { maximumSignificantDigits: 4 }).format(value);
}

export function resolveContinuousEncoding<T>(
  data: readonly T[],
  encoding: ContinuousEncoding<T>,
  mask?: Accessor<T, boolean>,
  maskedColor = DEFAULT_MASKED_COLOR
): ResolvedEncoding {
  const values = data.map(encoding.value);
  const domain = encoding.domain
    ? checkedDomain(encoding.domain, "encoding.domain")
    : inferContinuousDomain(values);
  const gradient = encoding.colors && encoding.colors.length >= 2
    ? encoding.colors
    : DEFAULT_CONTINUOUS_COLORS;
  const colorScale = scaleSequential(interpolateRgbBasis([...gradient]))
    .domain(domain)
    .clamp(encoding.clamp ?? true);
  const missingColor = encoding.missingColor ?? DEFAULT_MISSING_COLOR;
  let hasMissing = false;
  let hasMasked = false;
  const colors = values.map((value, index) => {
    if (!selected(data[index], index, mask)) {
      hasMasked = true;
      return maskedColor;
    }
    if (value === null || value === undefined || !Number.isFinite(value)) {
      hasMissing = true;
      return missingColor;
    }
    return colorScale(value);
  });
  const format = encoding.formatValue ?? defaultValueFormat;
  const legend: LegendItem[] = [
    { key: "minimum", label: format(domain[0]), color: colorScale(domain[0]), kind: "value" },
    { key: "maximum", label: format(domain[1]), color: colorScale(domain[1]), kind: "value" }
  ];
  if (hasMissing) legend.push({
    key: "missing",
    label: encoding.missingLabel ?? "Missing",
    color: missingColor,
    kind: "missing"
  });
  if (hasMasked) legend.push({
    key: "masked",
    label: "Not included",
    color: maskedColor,
    kind: "masked"
  });
  return { colors, legend, domain, gradient };
}
