import createREGL, { type Buffer, type Regl } from "regl";
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "./ui";

interface ViewState {
  scale: number;
  translate: [number, number];
}

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

export interface ScatterplotProps {
  coordinates: [number, number][];
  colors: Float32Array;
  selectedIndices: Set<number>;
  onSingleSelect: (index: number) => void;
  onLasso: (indices: number[], mode: SelectionMode) => void;
  onClearSelection: () => void;
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
  onSingleSelect,
  onLasso,
  onClearSelection
}: ScatterplotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const renderRef = useRef<(() => void) | null>(null);
  const colorBufferRef = useRef<Buffer | null>(null);
  const selectionBufferRef = useRef<{ subdata(data: Float32Array): void } | null>(null);
  const gestureRef = useRef<Gesture | null>(null);
  const [dimensions, setDimensions] = useState({ width: 1, height: 1 });
  const [view, setView] = useState<ViewState>({ scale: 1, translate: [0, 0] });
  const [lassoEnabled, setLassoEnabled] = useState(false);
  const [lasso, setLasso] = useState<Point[]>([]);
  const [plotError, setPlotError] = useState("");

  const normalized = useMemo(() => {
    if (coordinates.length === 0) return [];
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    coordinates.forEach(([x, y]) => {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    });
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const range = Math.max(maxX - minX, maxY - minY) || 1;
    return coordinates.map(([x, y]) => [
      ((x - centerX) / range) * 1.82,
      ((y - centerY) / range) * 1.82
    ] as [number, number]);
  }, [coordinates]);

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

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || normalized.length === 0) return;
    let regl: Regl;
    try {
      regl = createREGL({ canvas, attributes: { antialias: false, alpha: false } });
    } catch (error) {
      setPlotError(error instanceof Error ? error.message : "WebGL is unavailable");
      return;
    }
    const positions = new Float32Array(normalized.flat());
    const selection = new Float32Array(normalized.length);
    const positionBuffer = regl.buffer(positions);
    const colorBuffer = regl.buffer({ data: safeColors, usage: "dynamic" });
    const selectionBuffer = regl.buffer(selection);
    colorBufferRef.current = colorBuffer;
    selectionBufferRef.current = selectionBuffer;

    interface DrawProps {
      scale: number;
      translate: [number, number];
      onlySelected: number;
    }

    const drawPoints = regl({
      vert: `
        precision highp float;
        attribute vec2 position;
        attribute vec4 pointColor;
        attribute float selected;
        uniform float scale;
        uniform vec2 translate;
        varying vec4 color;
        varying float isSelected;
        void main() {
          gl_Position = vec4(position * scale + translate, 0.0, 1.0);
          gl_PointSize = selected > 0.5 ? 9.0 : 4.0;
          color = pointColor;
          isSelected = selected;
        }
      `,
      frag: `
        precision mediump float;
        varying vec4 color;
        varying float isSelected;
        uniform float onlySelected;
        void main() {
          float radius = distance(gl_PointCoord, vec2(0.5));
          if (radius > 0.5) discard;
          if (onlySelected > 0.5 && isSelected < 0.5) discard;
          if (isSelected > 0.5 && radius > 0.34) {
            gl_FragColor = vec4(0.08, 0.10, 0.12, 1.0);
          } else {
            gl_FragColor = color;
          }
        }
      `,
      attributes: {
        position: positionBuffer,
        pointColor: colorBuffer,
        selected: selectionBuffer
      },
      uniforms: {
        scale: regl.prop<DrawProps, "scale">("scale"),
        translate: regl.prop<DrawProps, "translate">("translate"),
        onlySelected: regl.prop<DrawProps, "onlySelected">("onlySelected")
      },
      count: normalized.length,
      primitive: "points",
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
        drawPoints({ ...viewRef.current, onlySelected: 0 });
        drawPoints({ ...viewRef.current, onlySelected: 1 });
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
      regl.destroy();
    };
  }, [normalized]);

  useEffect(() => {
    colorBufferRef.current?.(safeColors);
    renderRef.current?.();
  }, [safeColors]);

  useEffect(() => {
    const selection = new Float32Array(normalized.length);
    selectedIndices.forEach((index) => {
      if (index >= 0 && index < selection.length) selection[index] = 1;
    });
    selectionBufferRef.current?.subdata(selection);
    renderRef.current?.();
  }, [normalized.length, selectedIndices]);

  useEffect(() => {
    renderRef.current?.();
  }, [view]);

  const project = (index: number): Point => ({
    x: ((normalized[index][0] * view.scale + view.translate[0] + 1) / 2) * dimensions.width,
    y: ((1 - normalized[index][1] * view.scale - view.translate[1]) / 2) * dimensions.height
  });

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const point = pointerPoint(event);
    event.currentTarget.setPointerCapture(event.pointerId);
    gestureRef.current = {
      kind: lassoEnabled ? "lasso" : "pan",
      selectionMode: event.altKey ? "subtract" : event.shiftKey ? "add" : "replace",
      start: point,
      last: point,
      points: [point],
      startView: view,
      moved: false
    };
    if (lassoEnabled) setLasso([point]);
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
    if (gesture.kind === "lasso") {
      if (gesture.points.length >= 3) {
        const indices = normalized.flatMap((_coordinate, index) => (
          pointInPolygon(project(index), gesture.points) ? [index] : []
        ));
        onLasso(indices, gesture.selectionMode);
      }
      setLasso([]);
      return;
    }
    if (gesture.moved) return;
    let nearest = -1;
    let nearestDistance = 12;
    normalized.forEach((_coordinate, index) => {
      const candidate = project(index);
      const distance = Math.hypot(candidate.x - point.x, candidate.y - point.y);
      if (distance < nearestDistance) {
        nearest = index;
        nearestDistance = distance;
      }
    });
    if (nearest >= 0) onSingleSelect(nearest);
    else onClearSelection();
  };

  const handleWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const cursor: [number, number] = [
      ((event.clientX - rect.left) / rect.width) * 2 - 1,
      1 - ((event.clientY - rect.top) / rect.height) * 2
    ];
    const nextScale = Math.max(0.35, Math.min(30, view.scale * Math.exp(-event.deltaY * 0.0015)));
    const ratio = nextScale / view.scale;
    setView({
      scale: nextScale,
      translate: [
        cursor[0] - (cursor[0] - view.translate[0]) * ratio,
        cursor[1] - (cursor[1] - view.translate[1]) * ratio
      ]
    });
  };

  const resetView = () => setView({ scale: 1, translate: [0, 0] });

  return (
    <div className={`scatter-shell ${lassoEnabled ? "lasso-mode" : ""}`}>
      <div className="plot-toolbar">
        <Button
          active={lassoEnabled}
          aria-pressed={lassoEnabled}
          onClick={() => setLassoEnabled((enabled) => !enabled)}
        >
          Lasso
        </Button>
        <Button onClick={resetView}>Fit</Button>
        <Button disabled={selectedIndices.size === 0} onClick={onClearSelection}>Clear</Button>
        <span>Shift add / Alt subtract / Esc clear</span>
      </div>
      <div className="scatter-stage">
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
            onWheel={handleWheel}
            onKeyDown={(event) => {
              if (event.key !== "Escape") return;
              event.preventDefault();
              onClearSelection();
            }}
          />
        )}
        {lasso.length > 1 && (
          <svg className="lasso-overlay" viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}>
            <polyline points={lasso.map((point) => `${point.x},${point.y}`).join(" ")} />
          </svg>
        )}
      </div>
    </div>
  );
}
