import { scaleLinear } from "d3-scale";
import { useEffect, useId, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import {
  categoryKey,
  DEFAULT_MASKED_COLOR,
  inferUmapBounds,
  resolveCategoricalEncoding,
  resolveContinuousEncoding,
  validateUmapBounds
} from "./encoding";
import { layoutClusterLabels, type ClusterLabelInput } from "./layout";
import type { CategoryValue, UmapPadding, UmapProps } from "./types";

interface Dimensions {
  width: number;
  height: number;
}

const DEFAULT_PADDING: UmapPadding = { top: 14, right: 14, bottom: 14, left: 14 };

function resolvePadding(padding: UmapProps<unknown>["padding"]): UmapPadding {
  if (typeof padding === "number") {
    return { top: padding, right: padding, bottom: padding, left: padding };
  }
  return { ...DEFAULT_PADDING, ...padding };
}

function radiusFor<T>(radius: UmapProps<T>["pointRadius"], datum: T, index: number): number {
  const value = typeof radius === "function" ? radius(datum, index) : (radius ?? 2.2);
  return Number.isFinite(value) ? Math.max(0, value) : 0;
}

export function Umap<T>({
  data,
  x,
  y,
  encoding,
  bounds,
  mask,
  maskedColor = DEFAULT_MASKED_COLOR,
  pointRadius,
  pointOpacity = 0.82,
  padding,
  height = 480,
  ariaLabel = "UMAP scatter plot",
  legendLabel = "Color legend",
  showLegend = true,
  className,
  style
}: UmapProps<T>) {
  const stageRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dimensions, setDimensions] = useState<Dimensions>({ width: 0, height: 0 });
  const legendId = useId();
  const resolvedPadding = useMemo(() => resolvePadding(padding), [padding]);
  const resolvedBounds = useMemo(
    () => bounds ? validateUmapBounds(bounds) : inferUmapBounds(data, x, y),
    [bounds, data, x, y]
  );
  const resolvedEncoding = useMemo(() => encoding.kind === "categorical"
    ? resolveCategoricalEncoding(data, encoding, mask, maskedColor)
    : resolveContinuousEncoding(data, encoding, mask, maskedColor),
  [data, encoding, mask, maskedColor]);

  const scales = useMemo(() => ({
    x: scaleLinear()
      .domain(resolvedBounds.x)
      .range([resolvedPadding.left, Math.max(resolvedPadding.left, dimensions.width - resolvedPadding.right)]),
    y: scaleLinear()
      .domain(resolvedBounds.y)
      .range([Math.max(resolvedPadding.top, dimensions.height - resolvedPadding.bottom), resolvedPadding.top])
  }), [dimensions, resolvedBounds, resolvedPadding]);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const update = () => {
      const rectangle = stage.getBoundingClientRect();
      setDimensions({ width: rectangle.width, height: rectangle.height });
    };
    update();
    if (typeof ResizeObserver !== "function") {
      window.addEventListener("resize", update);
      return () => window.removeEventListener("resize", update);
    }
    const observer = new ResizeObserver(update);
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || dimensions.width <= 0 || dimensions.height <= 0) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.round(dimensions.width * ratio));
    canvas.height = Math.max(1, Math.round(dimensions.height * ratio));
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, dimensions.width, dimensions.height);
    context.globalAlpha = Math.max(0, Math.min(1, pointOpacity));

    const order = data.map((_datum, index) => index).sort((left, right) => {
      const leftActive = mask?.(data[left], left) ?? true;
      const rightActive = mask?.(data[right], right) ?? true;
      return Number(leftActive) - Number(rightActive);
    });
    order.forEach((index) => {
      const datum = data[index];
      const coordinateX = x(datum, index);
      const coordinateY = y(datum, index);
      if (!Number.isFinite(coordinateX) || !Number.isFinite(coordinateY)) return;
      const radius = radiusFor(pointRadius, datum, index);
      if (radius <= 0) return;
      context.beginPath();
      context.arc(scales.x(coordinateX), scales.y(coordinateY), radius, 0, Math.PI * 2);
      context.fillStyle = resolvedEncoding.colors[index];
      context.fill();
    });
  }, [data, dimensions, mask, pointOpacity, pointRadius, resolvedEncoding.colors, scales, x, y]);

  const clusterLabels = useMemo(() => {
    if (encoding.kind !== "categorical" || !encoding.labels) return [];
    const groups = new Map<string, ClusterLabelInput & { value: CategoryValue; sumX: number; sumY: number }>();
    data.forEach((datum, index) => {
      if (!(mask?.(datum, index) ?? true)) return;
      const value = encoding.value(datum, index);
      const coordinateX = x(datum, index);
      const coordinateY = y(datum, index);
      if (value === null || value === undefined || !Number.isFinite(coordinateX) || !Number.isFinite(coordinateY)) return;
      if (typeof value === "number" && !Number.isFinite(value)) return;
      const key = categoryKey(value);
      const existing = groups.get(key);
      if (existing) {
        existing.sumX += scales.x(coordinateX);
        existing.sumY += scales.y(coordinateY);
        existing.count += 1;
      } else {
        const legend = resolvedEncoding.legend.find((item) => item.key === key);
        groups.set(key, {
          key,
          value,
          label: encoding.formatCategory?.(value) ?? String(value),
          color: legend?.color ?? resolvedEncoding.colors[index],
          x: 0,
          y: 0,
          sumX: scales.x(coordinateX),
          sumY: scales.y(coordinateY),
          count: 1
        });
      }
    });
    const minimum = encoding.labels.minCount ?? 1;
    const inputs = [...groups.values()].flatMap((group) => group.count >= minimum ? [{
      key: group.key,
      label: group.label,
      color: group.color,
      x: group.sumX / group.count,
      y: group.sumY / group.count,
      count: group.count
    }] : []);
    return layoutClusterLabels(inputs, {
      placement: encoding.labels.placement ?? "on-cluster",
      width: dimensions.width,
      height: dimensions.height,
      padding: Math.max(resolvedPadding.right, 8),
      gap: encoding.labels.gap
    });
  }, [data, dimensions, encoding, mask, resolvedEncoding.colors, resolvedEncoding.legend, resolvedPadding.right, scales, x, y]);

  const rootStyle: CSSProperties = { ...style, height };
  const classNames = ["eyck-umap", className].filter(Boolean).join(" ");

  return (
    <figure className={classNames} style={rootStyle}>
      <div ref={stageRef} className="eyck-umap__stage">
        <canvas
          ref={canvasRef}
          className="eyck-umap__canvas"
          role="img"
          aria-label={ariaLabel}
          aria-describedby={showLegend ? legendId : undefined}
        />
        {clusterLabels.length > 0 && (
          <svg
            className="eyck-umap__labels"
            viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
            aria-hidden="true"
          >
            {clusterLabels.map((label) => (
              <g key={label.key}>
                {encoding.kind === "categorical" && encoding.labels
                  && (encoding.labels.placement ?? "on-cluster") === "beside" && (
                  <line
                    x1={label.x}
                    y1={label.y}
                    x2={label.lineX}
                    y2={label.lineY}
                    stroke={label.color}
                  />
                )}
                <text
                  x={label.labelX}
                  y={label.labelY}
                  textAnchor={label.textAnchor}
                  dominantBaseline="central"
                  style={{ fontSize: encoding.kind === "categorical" && encoding.labels
                    ? encoding.labels.fontSize
                    : undefined }}
                >
                  {label.label}
                </text>
              </g>
            ))}
          </svg>
        )}
      </div>
      {showLegend && (
        <figcaption id={legendId} className="eyck-umap__legend" aria-label={legendLabel}>
          {resolvedEncoding.gradient && (
            <span
              className="eyck-umap__gradient"
              style={{ background: `linear-gradient(to right, ${resolvedEncoding.gradient.join(", ")})` }}
              aria-hidden="true"
            />
          )}
          <ul className="eyck-umap__legend-items">
            {resolvedEncoding.legend.map((item) => (
              <li key={item.key}>
                <span className="eyck-umap__swatch" style={{ backgroundColor: item.color }} aria-hidden="true" />
                <span>{item.label}</span>
              </li>
            ))}
          </ul>
        </figcaption>
      )}
    </figure>
  );
}
