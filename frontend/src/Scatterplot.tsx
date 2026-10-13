import createREGL, { type Buffer, type Regl } from "regl";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Button } from "./ui";

export interface ViewState {
  scale: number;
  translate: [number, number];
}

export const DEFAULT_VIEW: ViewState = { scale: 1, translate: [0, 0] };

interface Point {
  x: number;
  y: number;
}

interface Gesture {
  kind: "pan" | "lasso";
  selectionMode: SelectionMode;
  start: Point;
  last: Point;
  points: Point[];
  startView: ViewState;
  moved: boolean;
}

export type SelectionMode = "replace" | "add" | "subtract";
/** "select" lasso-selects cells through onLasso; the others draw shapes through onDraw. */
export type DrawMode = "curve" | "region" | "select";
export type ToScreen = (x: number, y: number) => [number, number];
export interface Projection {
  toScreen: ToScreen;
  /** Overlay pixel position to embedding coordinates. */
  fromScreen: (x: number, y: number) => [number, number];
}

export interface ScatterplotProps {
  /** Embedding coordinates; null for cells outside a view that covers a subset. */
  coordinates: ([number, number] | null)[];
  colors: Float32Array;
  selectedIndices: Set<number>;
  focusedIndices: Set<number>;
  onSingleSelect: (index: number) => void;
  onLasso: (indices: number[], mode: SelectionMode, polygon: [number, number][]) => void;
  onClearSelection: () => void;
  /** Diameter of unselected points in pixels; defaults to 4. */
  pointSize?: number;
  /** Observation indices in drawing order, so later ones are drawn on top. */
  drawOrder?: Uint32Array;
  /** Freehand drawing replaces lasso selection; points are in embedding coordinates. */
  drawing?: {
    mode: DrawMode | null;
    onDraw: (points: [number, number][]) => void;
    renderOverlay: (projection: Projection) => ReactNode;
    hint: string;
    /** Overlay shapes receive pointer events, for example to edit them. */
    interactive?: boolean;
  };
  /** Controlled pan and zoom, for example to link plots of one embedding. */
  view?: ViewState;
  onViewChange?: (view: ViewState) => void;
  title?: ReactNode;
  /** Called on any pointer press inside the plot. */
  onActivate?: () => void;
}

function pointInPolygon(point: Point, polygon: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    const crosses = (a.y > point.y) !== (b.y > point.y)
      && point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x;
    if (crosses) inside = !inside;
  }
  return inside;
}

function pointerPoint(event: React.PointerEvent<HTMLCanvasElement>): Point {
  const rect = event.currentTarget.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
}

export function Scatterplot({
  coordinates,
  colors,
  selectedIndices,
  focusedIndices,
  onSingleSelect,
  onLasso,
  onClearSelection,
  pointSize = 4,
  drawOrder,
  drawing,
  view: controlledView,
  onViewChange,
  title,
  onActivate
}: ScatterplotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderRef = useRef<(() => void) | null>(null);
  const colorBufferRef = useRef<Buffer | null>(null);
  const selectionBufferRef = useRef<{ subdata(data: Float32Array): void } | null>(null);
  const focusBufferRef = useRef<{ subdata(data: Float32Array): void } | null>(null);
  const elementsRef = useRef<((data: Uint32Array) => void) | null>(null);
  const pointSizeRef = useRef(pointSize);
  pointSizeRef.current = pointSize;
  const gestureRef = useRef<Gesture | null>(null);
  const [dimensions, setDimensions] = useState({ width: 1, height: 1 });
  const stageRef = useRef<HTMLDivElement>(null);
  const [innerView, setInnerView] = useState<ViewState>(DEFAULT_VIEW);
  const view = controlledView ?? innerView;
  const [lassoEnabled, setLassoEnabled] = useState(false);
  const [lasso, setLasso] = useState<Point[]>([]);
  const [plotError, setPlotError] = useState("");

  const coordinateBounds = useMemo(() => {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    coordinates.forEach((point) => {
      if (!point) return;
      minX = Math.min(minX, point[0]);
      maxX = Math.max(maxX, point[0]);
      minY = Math.min(minY, point[1]);
      maxY = Math.max(maxY, point[1]);
    });
    if (minX > maxX) return { centerX: 0, centerY: 0, range: 1 };
    return {
      centerX: (minX + maxX) / 2,
      centerY: (minY + maxY) / 2,
      range: Math.max(maxX - minX, maxY - minY) || 1
    };
  }, [coordinates]);
  // Absent cells are placed far outside the clip space and left out of the draw order.
  const normalized = useMemo(() => coordinates.map((point) => point
    ? [
      ((point[0] - coordinateBounds.centerX) / coordinateBounds.range) * 1.82,
      ((point[1] - coordinateBounds.centerY) / coordinateBounds.range) * 1.82
    ] as [number, number]
    : null), [coordinateBounds, coordinates]);
  const presentOrder = useMemo(() => Uint32Array.from(normalized.flatMap((point, index) => point ? [index] : [])), [normalized]);

  const safeColors = useMemo(() => {
    if (colors.length === normalized.length * 4) return colors;
    const fallback = new Float32Array(normalized.length * 4);
    for (let index = 0; index < normalized.length; index += 1) {
      fallback.set([0.38, 0.45, 0.53, 0.72], index * 4);
    }
    return fallback;
  }, [colors, normalized.length]);

  const viewRef = useRef(view);
  viewRef.current = view;
  const dimensionsRef = useRef(dimensions);
  dimensionsRef.current = dimensions;
  const setView = (next: ViewState | ((current: ViewState) => ViewState)) => {
    const resolved = typeof next === "function" ? next(viewRef.current) : next;
    viewRef.current = resolved;
    if (onViewChange) onViewChange(resolved);
    else setInnerView(resolved);
  };
  const setViewRef = useRef(setView);
  setViewRef.current = setView;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || normalized.length === 0) return;
    let regl: Regl;
    try {
      regl = createREGL({ canvas, attributes: { antialias: false, alpha: false }, optionalExtensions: ["OES_element_index_uint"] });
    } catch (error) {
      setPlotError(error instanceof Error ? error.message : "WebGL is unavailable");
      return;
    }
    const positions = new Float32Array(normalized.flatMap((point) => point ?? [1e4, 1e4]));
    const selection = new Float32Array(normalized.length);
    const focus = new Float32Array(normalized.length);
    const positionBuffer = regl.buffer(positions);
    const colorBuffer = regl.buffer({ data: safeColors, usage: "dynamic" });
    const selectionBuffer = regl.buffer(selection);
    const focusBuffer = regl.buffer(focus);
    colorBufferRef.current = colorBuffer;
    selectionBufferRef.current = selectionBuffer;
    focusBufferRef.current = focusBuffer;
    const ordered = regl.hasExtension("OES_element_index_uint")
      ? regl.elements({ data: presentOrder, type: "uint32", primitive: "points" })
      : null;
    elementsRef.current = ordered && ((data) => ordered({ data, type: "uint32", primitive: "points" }));

    interface DrawProps {
      scale: number;
      translate: [number, number];
      onlySelected: number;
      pointSize: number;
    }

    const drawPoints = regl({
      vert: `
        precision highp float;
        attribute vec2 position;
        attribute vec4 pointColor;
        attribute float selected;
        attribute float focused;
        uniform float scale;
        uniform vec2 translate;
        uniform float pointSize;
        varying vec4 color;
        varying float isSelected;
        varying float isFocused;
        void main() {
          gl_Position = vec4(position * scale + translate, 0.0, 1.0);
          gl_PointSize = selected > 0.5 ? max(10.0, pointSize + 4.0) : (focused > 0.5 ? pointSize + 3.0 : pointSize);
          color = pointColor;
          isSelected = selected;
          isFocused = focused;
        }
      `,
      frag: `
        precision mediump float;
        varying vec4 color;
        varying float isSelected;
        varying float isFocused;
        uniform float onlySelected;
        void main() {
          float radius = distance(gl_PointCoord, vec2(0.5));
          if (radius > 0.5) discard;
          if (onlySelected > 0.5 && isSelected < 0.5) discard;
          if (isSelected > 0.5 && radius > 0.34) {
            gl_FragColor = vec4(0.08, 0.10, 0.12, 1.0);
          } else if (isFocused > 0.5 && radius > 0.39) {
            gl_FragColor = vec4(1.0, 1.0, 1.0, 0.95);
          } else {
            gl_FragColor = color;
          }
        }
      `,
      attributes: {
        position: positionBuffer,
        pointColor: colorBuffer,
        selected: selectionBuffer,
        focused: focusBuffer
      },
      uniforms: {
        scale: regl.prop<DrawProps, "scale">("scale"),
        translate: regl.prop<DrawProps, "translate">("translate"),
        onlySelected: regl.prop<DrawProps, "onlySelected">("onlySelected"),
        pointSize: regl.prop<DrawProps, "pointSize">("pointSize")
      },
      ...(ordered ? { elements: ordered } : { count: normalized.length, primitive: "points" as const }),
      depth: { enable: false },
      blend: {
        enable: true,
        func: { src: "src alpha", dst: "one minus src alpha" }
      }
    });

    const render = () => {
      try {
        regl.poll();
        regl.clear({ color: [0.985, 0.988, 0.99, 1], depth: 1 });
        drawPoints({ ...viewRef.current, onlySelected: 0, pointSize: pointSizeRef.current });
        drawPoints({ ...viewRef.current, onlySelected: 1, pointSize: pointSizeRef.current });
      } catch (error) {
        setPlotError(error instanceof Error ? error.message : "The plot renderer failed");
      }
    };
    renderRef.current = render;
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(rect.width * ratio));
      canvas.height = Math.max(1, Math.round(rect.height * ratio));
      setDimensions({ width: rect.width, height: rect.height });
      render();
    };
    const observer = typeof ResizeObserver === "function" ? new ResizeObserver(resize) : null;
    if (observer) observer.observe(canvas);
    else window.addEventListener("resize", resize);
    resize();
    return () => {
      observer?.disconnect();
      if (!observer) window.removeEventListener("resize", resize);
      renderRef.current = null;
      colorBufferRef.current = null;
      selectionBufferRef.current = null;
      focusBufferRef.current = null;
      elementsRef.current = null;
      regl.destroy();
    };
  }, [normalized]);

  useEffect(() => {
    colorBufferRef.current?.(safeColors);
    renderRef.current?.();
  }, [safeColors]);

  useEffect(() => {
    if (drawOrder && drawOrder.length === normalized.length) {
      elementsRef.current?.(presentOrder.length === normalized.length ? drawOrder : drawOrder.filter((index) => normalized[index] !== null));
    } else {
      elementsRef.current?.(presentOrder);
    }
    renderRef.current?.();
  }, [drawOrder, normalized, presentOrder]);

  useEffect(() => {
    renderRef.current?.();
  }, [pointSize]);

  useEffect(() => {
    const selection = new Float32Array(normalized.length);
    selectedIndices.forEach((index) => {
      if (index >= 0 && index < selection.length) selection[index] = 1;
    });
    selectionBufferRef.current?.subdata(selection);
    renderRef.current?.();
  }, [normalized.length, selectedIndices]);

  useEffect(() => {
    const focus = new Float32Array(normalized.length);
    focusedIndices.forEach((index) => {
      if (index >= 0 && index < focus.length) focus[index] = 1;
    });
    focusBufferRef.current?.subdata(focus);
    renderRef.current?.();
  }, [focusedIndices, normalized.length]);

  useEffect(() => {
    renderRef.current?.();
  }, [view]);

  const project = (index: number): Point | null => {
    const point = normalized[index];
    if (!point) return null;
    return {
      x: ((point[0] * view.scale + view.translate[0] + 1) / 2) * dimensions.width,
      y: ((1 - point[1] * view.scale - view.translate[1]) / 2) * dimensions.height
    };
  };

  const toScreen: ToScreen = (x, y) => [
    ((((x - coordinateBounds.centerX) / coordinateBounds.range) * 1.82 * view.scale + view.translate[0] + 1) / 2) * dimensions.width,
    ((1 - ((y - coordinateBounds.centerY) / coordinateBounds.range) * 1.82 * view.scale - view.translate[1]) / 2) * dimensions.height
  ];
  const drawMode = drawing?.mode ?? null;
  const freehand = lassoEnabled || drawMode !== null;

  const unproject = (point: Point): [number, number] => {
    const clipX = (point.x / dimensions.width) * 2 - 1;
    const clipY = 1 - (point.y / dimensions.height) * 2;
    const normalizedX = (clipX - view.translate[0]) / view.scale;
    const normalizedY = (clipY - view.translate[1]) / view.scale;
    return [
      (normalizedX / 1.82) * coordinateBounds.range + coordinateBounds.centerX,
      (normalizedY / 1.82) * coordinateBounds.range + coordinateBounds.centerY
    ];
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const point = pointerPoint(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    gestureRef.current = {
      kind: freehand ? "lasso" : "pan",
      selectionMode: event.altKey ? "subtract" : event.shiftKey ? "add" : "replace",
      start: point,
      last: point,
      points: [point],
      startView: view,
      moved: false
    };
    if (freehand) setLasso([point]);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const gesture = gestureRef.current;
    if (!gesture) return;
    const point = pointerPoint(event);
    const distance = Math.hypot(point.x - gesture.start.x, point.y - gesture.start.y);
    if (distance > 3) gesture.moved = true;
    if (gesture.kind === "pan") {
      setView({
        scale: gesture.startView.scale,
        translate: [
          gesture.startView.translate[0] + ((point.x - gesture.start.x) * 2) / dimensions.width,
          gesture.startView.translate[1] - ((point.y - gesture.start.y) * 2) / dimensions.height
        ]
      });
    } else if (Math.hypot(point.x - gesture.last.x, point.y - gesture.last.y) > 3) {
      gesture.last = point;
      gesture.points.push(point);
      setLasso([...gesture.points]);
    }
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const gesture = gestureRef.current;
    gestureRef.current = null;
    if (!gesture) return;
    const point = pointerPoint(event);
    if (gesture.kind === "lasso" && drawing && drawMode && drawMode !== "select") {
      if (gesture.points.length >= (drawMode === "curve" ? 2 : 3)) drawing.onDraw(gesture.points.map(unproject));
      setLasso([]);
      return;
    }
    if (gesture.kind === "lasso") {
      if (gesture.points.length >= 3) {
        const indices = normalized.flatMap((_coordinate, index) => {
          const projected = project(index);
          return projected && pointInPolygon(projected, gesture.points) ? [index] : [];
        });
        onLasso(indices, gesture.selectionMode, gesture.points.map(unproject));
      }
      setLasso([]);
      return;
    }
    if (gesture.moved) return;
    let nearest = -1;
    let nearestDistance = 12;
    normalized.forEach((_coordinate, index) => {
      const candidate = project(index);
      if (!candidate) return;
      const distance = Math.hypot(candidate.x - point.x, candidate.y - point.y);
      if (distance < nearestDistance) {
        nearest = index;
        nearestDistance = distance;
      }
    });
    if (nearest >= 0) onSingleSelect(nearest);
    else onClearSelection();
  };

  // A native listener, so that zooming over overlay shapes cannot scroll the page.
  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const rect = stage.getBoundingClientRect();
      const cursor: [number, number] = [
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        1 - ((event.clientY - rect.top) / rect.height) * 2
      ];
      const current = viewRef.current;
      const nextScale = Math.max(0.35, Math.min(30, current.scale * Math.exp(-event.deltaY * 0.0015)));
      const ratio = nextScale / current.scale;
      setViewRef.current({
        scale: nextScale,
        translate: [
          cursor[0] - (cursor[0] - current.translate[0]) * ratio,
          cursor[1] - (cursor[1] - current.translate[1]) * ratio
        ]
      });
    };
    stage.addEventListener("wheel", handleWheel, { passive: false });
    return () => stage.removeEventListener("wheel", handleWheel);
  }, []);

  const resetView = () => setView(DEFAULT_VIEW);
  const zoomBy = (factor: number) => setView((current) => {
    const scale = Math.max(0.35, Math.min(30, current.scale * factor));
    const ratio = scale / current.scale;
    return {
      scale,
      translate: [current.translate[0] * ratio, current.translate[1] * ratio]
    };
  });

  return (
    <div className={`scatter-shell ${freehand ? "lasso-mode" : ""}`} onPointerDownCapture={onActivate}>
      <div className="plot-toolbar">
        {title && <div className="plot-title">{title}</div>}
        {!drawing && <Button
          active={lassoEnabled}
          aria-pressed={lassoEnabled}
          onClick={() => setLassoEnabled((enabled) => !enabled)}
        >
          Lasso
        </Button>}
        <Button onClick={resetView}>Fit</Button>
        <Button aria-label="Zoom in" title="Zoom in" onClick={() => zoomBy(1.35)}>+</Button>
        <Button aria-label="Zoom out" title="Zoom out" onClick={() => zoomBy(1 / 1.35)}>-</Button>
        {!drawing && <Button disabled={selectedIndices.size === 0} onClick={onClearSelection}>Clear</Button>}
        <span>{drawing ? drawing.hint : "Shift add / Alt subtract / Esc clear"}</span>
      </div>
      <div className="scatter-stage" ref={stageRef}>
        {plotError ? (
          <div className="plot-fallback" role="alert">
            <strong>Scatterplot unavailable</strong>
            <span>{plotError}</span>
            <Button onClick={() => window.location.reload()}>Retry</Button>
          </div>
        ) : (
          <canvas
            ref={canvasRef}
            aria-label="Observation scatterplot"
            tabIndex={0}
            onDoubleClick={resetView}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={() => {
              gestureRef.current = null;
              setLasso([]);
            }}
            onKeyDown={(event) => {
              if (event.key === "Escape") {
                event.preventDefault();
                onClearSelection();
              } else if (event.key === "+" || event.key === "=") {
                event.preventDefault();
                zoomBy(1.35);
              } else if (event.key === "-") {
                event.preventDefault();
                zoomBy(1 / 1.35);
              }
            }}
          />
        )}
        {drawing && (
          <svg className={`drawing-overlay ${drawing.interactive ? "interactive" : ""}`} viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}>
            {drawing.renderOverlay({ toScreen, fromScreen: (x, y) => unproject({ x, y }) })}
          </svg>
        )}
        {lasso.length > 1 && (
          <svg className={`lasso-overlay ${drawMode === "curve" ? "curve-stroke" : drawMode === "select" ? "select-stroke" : ""}`} viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}>
            <polyline points={lasso.map((point) => `${point.x},${point.y}`).join(" ")} />
          </svg>
        )}
      </div>
    </div>
  );
}
