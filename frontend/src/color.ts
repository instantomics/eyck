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
  [69, 149, 161]
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

export function colorValues(values: Scalar[] | undefined, length: number): ColorResult {
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
    const span = max - min || 1;
    values.forEach((value, index) => {
      if (typeof value !== "number") {
        writeColor(colors, index, [190, 194, 198]);
        return;
      }
      const t = Math.max(0, Math.min(1, (value - min) / span));
      writeColor(colors, index, [
        Math.round(232 - t * 202),
        Math.round(238 - t * 115),
        Math.round(241 - t * 83)
      ]);
    });
    return {
      colors,
      legend: [
        { label: min.toPrecision(4), color: "rgb(232, 238, 241)" },
        { label: max.toPrecision(4), color: "rgb(30, 123, 158)" }
      ]
    };
  }

  const categories = [...new Set(present.map(String))].sort((a, b) => a.localeCompare(b));
  const categoryColors = new Map(categories.map((category, index) => [
    category,
    CATEGORICAL[index % CATEGORICAL.length]
  ]));
  values.forEach((value, index) => {
    writeColor(colors, index, value === null
      ? [190, 194, 198]
      : categoryColors.get(String(value)) ?? CATEGORICAL[0]);
  });
  const legend = categories.slice(0, 12).map((category) => ({
    label: category,
    color: cssColor(categoryColors.get(category) ?? CATEGORICAL[0])
  }));
  if (categories.length > 12) legend.push({ label: `+${categories.length - 12} more`, color: "#bec2c6" });
  return { colors, legend };
}
