import type { CSSProperties } from "react";

export type Accessor<T, V> = (datum: T, index: number) => V;
export type NumericDomain = readonly [number, number];

export interface UmapBounds {
  x: NumericDomain;
  y: NumericDomain;
}

export type CategoryValue = string | number | boolean;

export interface ClusterLabelOptions {
  placement?: "on-cluster" | "beside";
  minCount?: number;
  fontSize?: number;
  gap?: number;
}

export interface CategoricalEncoding<T> {
  kind: "categorical";
  value: Accessor<T, CategoryValue | null | undefined>;
  categories?: readonly CategoryValue[];
  palette?: readonly string[];
  color?: (value: CategoryValue, index: number) => string;
  formatCategory?: (value: CategoryValue) => string;
  missingLabel?: string;
  missingColor?: string;
  labels?: ClusterLabelOptions;
}

export interface ContinuousEncoding<T> {
  kind: "continuous";
  value: Accessor<T, number | null | undefined>;
  domain?: NumericDomain;
  colors?: readonly string[];
  clamp?: boolean;
  formatValue?: (value: number) => string;
  missingLabel?: string;
  missingColor?: string;
}

export type UmapEncoding<T> = CategoricalEncoding<T> | ContinuousEncoding<T>;

export interface UmapPadding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface UmapProps<T> {
  data: readonly T[];
  x: Accessor<T, number>;
  y: Accessor<T, number>;
  encoding: UmapEncoding<T>;
  bounds?: UmapBounds;
  mask?: Accessor<T, boolean>;
  maskedColor?: string;
  pointRadius?: number | Accessor<T, number>;
  pointOpacity?: number;
  padding?: number | Partial<UmapPadding>;
  height?: number | string;
  ariaLabel?: string;
  legendLabel?: string;
  showLegend?: boolean;
  className?: string;
  style?: CSSProperties;
}

export interface LegendItem {
  key: string;
  label: string;
  color: string;
  kind: "value" | "missing" | "masked";
}

export interface ResolvedEncoding {
  colors: readonly string[];
  legend: readonly LegendItem[];
  domain?: NumericDomain;
  gradient?: readonly string[];
}
