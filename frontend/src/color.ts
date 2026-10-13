import type { Scalar } from "./types";

const CATEGORICAL = [
  [42, 96, 151],
  [218, 97, 72],
  [44, 135, 102],
  [145, 92, 156],
  [201, 143, 45],
  [84, 133, 170],
  [187, 85, 113],
  [98, 146, 71],
  [119, 108, 92],
  [69, 149, 161],
  [140, 182, 226],
  [240, 166, 140],
  [137, 199, 157],
  [196, 166, 214],
  [234, 200, 112],
  [30, 60, 90],
  [128, 45, 30],
  [25, 85, 60],
  [80, 40, 95],
  [160, 160, 160]
] as const;

export interface ColorLegendItem {
  label: string;
  color: string;
}

export interface ColorResult {
  colors: Float32Array;
  legend: ColorLegendItem[];
}

function cssColor(rgb: readonly number[]): string {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function writeColor(target: Float32Array, index: number, rgb: readonly number[]): void {
  target[index * 4] = rgb[0] / 255;
  target[index * 4 + 1] = rgb[1] / 255;
  target[index * 4 + 2] = rgb[2] / 255;
  target[index * 4 + 3] = 0.82;
}

export function uniformColors(length: number): ColorResult {
  const colors = new Float32Array(length * 4);
  for (let index = 0; index < length; index += 1) writeColor(colors, index, [70, 112, 146]);
  return { colors, legend: [{ label: "Observations", color: "rgb(70, 112, 146)" }] };
}

const VIRIDIS = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]] as const;

function viridis(t: number): number[] {
  const position = t * (VIRIDIS.length - 1);
  const low = Math.min(Math.floor(position), VIRIDIS.length - 2);
  const fraction = position - low;
  return VIRIDIS[low].map((value, channel) => Math.round(value + (VIRIDIS[low + 1][channel] - value) * fraction));
}

export interface ColorOptions {
  /** High-contrast sequential palette with the 1st-99th percentile range. */
  contrast?: boolean;
}

export function colorValues(values: Scalar[] | undefined, length: number, options: ColorOptions = {}): ColorResult {
  if (!values || values.length !== length) return uniformColors(length);
  const present = values.filter((value) => value !== null);
  const numeric = present.length > 0 && present.every((value) => typeof value === "number");
  const colors = new Float32Array(length * 4);
  if (numeric) {
    const numbers = present as number[];
    let min = Infinity;
    let max = -Infinity;
    numbers.forEach((value) => {
      min = Math.min(min, value);
      max = Math.max(max, value);
    });
    if (options.contrast && numbers.length > 100) {
      const sorted = [...numbers].sort((a, b) => a - b);
      min = sorted[Math.floor(sorted.length * 0.01)];
      max = sorted[Math.ceil(sorted.length * 0.99) - 1];
    }
    const span = max - min || 1;
    values.forEach((value, index) => {
      if (typeof value !== "number") {
        writeColor(colors, index, [190, 194, 198]);
        return;
      }
      const t = Math.max(0, Math.min(1, (value - min) / span));
      if (options.contrast) {
        writeColor(colors, index, viridis(t));
        return;
      }
      writeColor(colors, index, [
        Math.round(232 - t * 202),
        Math.round(238 - t * 115),
        Math.round(241 - t * 83)
      ]);
    });
    if (options.contrast) {
      return {
        colors,
        legend: [
          { label: `${min.toPrecision(4)} or less`, color: cssColor(viridis(0)) },
          { label: `${max.toPrecision(4)} or more`, color: cssColor(viridis(1)) }
        ]
      };
    }
    return {
      colors,
      legend: [
        { label: min.toPrecision(4), color: "rgb(232, 238, 241)" },
        { label: max.toPrecision(4), color: "rgb(30, 123, 158)" }
      ]
    };
  }

  const categories = [...new Set(present.map(String))].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  const categoryColors = new Map(categories.map((category, index) => [
    category,
    CATEGORICAL[index % CATEGORICAL.length]
  ]));
  values.forEach((value, index) => {
    writeColor(colors, index, value === null
      ? [190, 194, 198]
      : categoryColors.get(String(value)) ?? CATEGORICAL[0]);
  });
  const legend = categories.map((category) => ({
    label: category,
    color: cssColor(categoryColors.get(category) ?? CATEGORICAL[0])
  }));
  return { colors, legend };
}

export type Palette = "viridis" | "grey_blue" | "grey_red" | "grey_purple";

export const PALETTES: { id: Palette; label: string }[] = [
  { id: "grey_blue", label: "Grey to blue" },
  { id: "grey_red", label: "Grey to red" },
  { id: "grey_purple", label: "Grey to purple" },
  { id: "viridis", label: "Viridis" }
];

const GREY = [205, 208, 211] as const;
const BLUE = [31, 88, 214] as const;
const RED = [214, 39, 40] as const;
const PURPLE = [128, 24, 156] as const;
const MISSING = [190, 194, 198] as const;

function mix(from: readonly number[], to: readonly number[], t: number): number[] {
  return from.map((value, channel) => Math.round(value + (to[channel] - value) * t));
}

function paletteColor(palette: Palette, t: number): number[] {
  if (palette === "viridis") return viridis(t);
  return mix(GREY, palette === "grey_blue" ? BLUE : palette === "grey_red" ? RED : PURPLE, t);
}

/** Grey with no signal, blue for the first value, red for the second, purple for both. */
function blendColor(a: number, b: number): number[] {
  return GREY.map((grey, channel) => Math.round(
    grey * (1 - a) * (1 - b) + BLUE[channel] * a * (1 - b) + RED[channel] * (1 - a) * b + PURPLE[channel] * a * b
  ));
}

export interface DisplayOptions {
  palette: Palette;
  /** Values at or above this percentile take the strongest colour. */
  upperPercentile: number;
  opacity: number;
}

/** What a legend needs to explain one plot's colours. */
export type LegendScale =
  | { kind: "uniform" }
  | { kind: "categorical"; items: ColorLegendItem[] }
  | { kind: "continuous"; min: number; max: number; stops: string[] }
  | { kind: "blend"; first: { min: number; max: number }; second: { min: number; max: number } };

export interface DisplayResult extends ColorResult {
  /** Draw order: missing values first, then increasing value, so high values are on top. */
  order: Uint32Array;
  scale: LegendScale;
  /** Whether any cell has no value; such cells are drawn faint. */
  hasMissing: boolean;
  /** Whether any cell has an unknown process value. */
  hasUnknown?: boolean;
}

/** Cells without a value are pale and mostly transparent, so a value of 0 stays distinguishable. */
export const NO_VALUE_COLOR = "rgba(200, 204, 208, 0.35)";
const NO_VALUE = [214, 217, 221] as const;
const NO_VALUE_ALPHA = 0.22;

function numericRange(values: Scalar[], upperPercentile: number): { min: number; max: number } | null {
  const numbers = values.filter((value): value is number => typeof value === "number");
  if (numbers.length === 0 || numbers.length !== values.filter((value) => value !== null).length) return null;
  const sorted = Float64Array.from(numbers).sort();
  const min = sorted[Math.floor((sorted.length - 1) * 0.01)];
  const max = sorted[Math.round((sorted.length - 1) * upperPercentile / 100)];
  return { min, max: max > min ? max : sorted[sorted.length - 1] };
}

function scaled(value: Scalar, range: { min: number; max: number }): number | null {
  if (typeof value !== "number") return null;
  // A constant channel, such as a subset marked 1, takes the full colour.
  if (range.max <= range.min) return 1;
  return Math.max(0, Math.min(1, (value - range.min) / (range.max - range.min)));
}

function orderBy(keys: Float64Array): Uint32Array {
  return Uint32Array.from(keys.keys()).sort((a, b) => keys[a] - keys[b]);
}

function setOpacity(colors: Float32Array, opacity: number, missing: Uint8Array): void {
  for (let index = 0; index < missing.length; index += 1) {
    colors[index * 4 + 3] = missing[index] ? opacity * NO_VALUE_ALPHA : opacity;
  }
}

/** Colour one or two value channels for display; the second is blended in red. */
export function displayColors(
  primary: Scalar[] | undefined,
  secondary: Scalar[] | undefined,
  length: number,
  options: DisplayOptions
): DisplayResult {
  const first = primary && primary.length === length ? primary : undefined;
  const second = secondary && secondary.length === length ? secondary : undefined;
  const firstRange = first ? numericRange(first, options.upperPercentile) : null;
  const secondRange = second ? numericRange(second, options.upperPercentile) : null;
  const missing = new Uint8Array(length);
  const colors = new Float32Array(length * 4);
  const keys = new Float64Array(length);
  if (!first || !firstRange) {
    // Uniform or categorical colours.
    const result = colorValues(first, length);
    colors.set(result.colors);
    first?.forEach((value, index) => {
      if (value === null) {
        missing[index] = 1;
        keys[index] = -1;
        writeColor(colors, index, NO_VALUE);
      }
    });
    setOpacity(colors, options.opacity, missing);
    const items = result.legend.filter((item) => item.label !== "Observations");
    return {
      colors,
      legend: result.legend,
      order: orderBy(keys),
      scale: first ? { kind: "categorical", items } : { kind: "uniform" },
      hasMissing: missing.some((value) => value === 1)
    };
  }
  for (let index = 0; index < length; index += 1) {
    const a = scaled(first[index], firstRange);
    const b = second && secondRange ? scaled(second[index], secondRange) : null;
    let rgb: readonly number[];
    if (second && secondRange) {
      rgb = a === null && b === null ? NO_VALUE : blendColor(a ?? 0, b ?? 0);
      keys[index] = a === null && b === null ? -1 : Math.max(a ?? 0, b ?? 0);
      missing[index] = a === null && b === null ? 1 : 0;
    } else {
      rgb = a === null ? NO_VALUE : paletteColor(options.palette, a);
      keys[index] = a === null ? -1 : a;
      missing[index] = a === null ? 1 : 0;
    }
    writeColor(colors, index, rgb);
  }
  setOpacity(colors, options.opacity, missing);
  const label = (range: { min: number; max: number }, end: "min" | "max") =>
    end === "min" ? `${range.min.toPrecision(3)} or less` : `${range.max.toPrecision(3)} or more`;
  const legend = second && secondRange
    ? [
      { label: "neither", color: cssColor(GREY) },
      { label: `first ${label(firstRange, "max")}`, color: cssColor(BLUE) },
      { label: `second ${label(secondRange, "max")}`, color: cssColor(RED) },
      { label: "both", color: cssColor(PURPLE) }
    ]
    : firstRange.max <= firstRange.min
      ? [{ label: firstRange.max.toPrecision(3), color: cssColor(paletteColor(options.palette, 1)) }]
      : [
        { label: label(firstRange, "min"), color: cssColor(paletteColor(options.palette, 0)) },
        { label: label(firstRange, "max"), color: cssColor(paletteColor(options.palette, 1)) }
      ];
  const scale: LegendScale = second && secondRange
    ? { kind: "blend", first: firstRange, second: secondRange }
    : { kind: "continuous", min: firstRange.min, max: firstRange.max, stops: [0, 0.25, 0.5, 0.75, 1].map((t) => cssColor(paletteColor(options.palette, firstRange.max <= firstRange.min ? 1 : t))) };
  return { colors, legend, order: orderBy(keys), scale, hasMissing: missing.some((value) => value === 1) };
}

/** Cells where a process is active but its value is unknown. */
const UNKNOWN = [214, 140, 40] as const;
export const UNKNOWN_COLOR = cssColor(UNKNOWN);

/** Draw cells with an unknown process value in their own colour, on top of cells without a value. */
export function markUnknown(result: DisplayResult, unknown: Uint8Array | undefined, opacity: number): DisplayResult {
  if (!unknown || unknown.length * 4 !== result.colors.length || !unknown.some((value) => value === 1)) return result;
  const colors = result.colors.slice();
  unknown.forEach((flag, index) => {
    if (!flag) return;
    writeColor(colors, index, UNKNOWN);
    colors[index * 4 + 3] = opacity;
  });
  const order = Uint32Array.from([...result.order].sort((a, b) => unknown[a] - unknown[b]));
  return { ...result, colors, order, hasUnknown: true };
}

/** Corner colours of the two-channel blend legend. */
export const BLEND_CORNERS = { neither: cssColor(GREY), first: cssColor(BLUE), second: cssColor(RED), both: cssColor(PURPLE) };
