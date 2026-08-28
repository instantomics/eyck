import "./styles.css";

export { Umap } from "./Umap";
export {
  categoryKey,
  DEFAULT_CATEGORICAL_PALETTE,
  DEFAULT_CONTINUOUS_COLORS,
  DEFAULT_MASKED_COLOR,
  DEFAULT_MISSING_COLOR,
  inferContinuousDomain,
  inferUmapBounds,
  resolveCategoricalEncoding,
  resolveContinuousEncoding,
  validateUmapBounds
} from "./encoding";
export { layoutClusterLabels } from "./layout";
export type {
  ClusterLabelInput,
  ClusterLabelLayout,
  ClusterLabelLayoutOptions
} from "./layout";
export type {
  Accessor,
  CategoricalEncoding,
  CategoryValue,
  ClusterLabelOptions,
  ContinuousEncoding,
  LegendItem,
  NumericDomain,
  ResolvedEncoding,
  UmapBounds,
  UmapEncoding,
  UmapPadding,
  UmapProps
} from "./types";
