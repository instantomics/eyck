import { describe, expect, it } from "vitest";
import {
  inferContinuousDomain,
  resolveCategoricalEncoding,
  resolveContinuousEncoding
} from "../src/encoding";

interface Datum {
  category: string | null;
  score: number | null;
  included: boolean;
}

const data: Datum[] = [
  { category: "zeta", score: 10, included: true },
  { category: "alpha", score: -5, included: false },
  { category: null, score: null, included: true },
  { category: "alpha", score: 2, included: true }
];

describe("categorical encoding", () => {
  it("assigns deterministic category colors while applying values only through the mask", () => {
    const encoding = { kind: "categorical" as const, value: (datum: Datum) => datum.category };
    const first = resolveCategoricalEncoding(data, encoding, (datum) => datum.included, "#eeeeee");
    const reordered = resolveCategoricalEncoding([...data].reverse(), encoding);

    expect(first.colors).toEqual([reordered.colors[3], "#eeeeee", reordered.colors[1], reordered.colors[0]]);
    expect(first.legend.map((item) => [item.label, item.kind])).toEqual([
      ["alpha", "value"],
      ["zeta", "value"],
      ["Missing", "missing"],
      ["Not included", "masked"]
    ]);
  });
});

describe("continuous encoding", () => {
  it("infers an order-independent domain from all values, independent of the mask", () => {
    const resolved = resolveContinuousEncoding(
      data,
      { kind: "continuous", value: (datum) => datum.score },
      (datum) => datum.included,
      "#eeeeee"
    );

    expect(resolved.domain).toEqual([-5, 10]);
    expect(resolved.colors[1]).toBe("#eeeeee");
    expect(resolved.legend.at(-1)).toMatchObject({ label: "Not included", kind: "masked" });
  });

  it("creates a finite stable span for constant and entirely missing values", () => {
    expect(inferContinuousDomain([5, 5, null])).toEqual([4, 6]);
    expect(inferContinuousDomain([null, Number.NaN])).toEqual([0, 1]);
  });
});
