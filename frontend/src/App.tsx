import { useCallback, useEffect, useMemo, useState } from "react";
import {
  basePath,
  exportAnnotations,
  getPoints,
  getProject,
  getProjects,
  restartServer,
  waitForRestart
} from "./api";
import { AnnotationPanel } from "./AnnotationPanel";
import { colorValues, uniformColors, type ColorResult } from "./color";
import { ColoringControls } from "./ColoringControls";
import { Scatterplot, type SelectionMode } from "./Scatterplot";
import type { PointsPayload, ProjectDetail, ProjectIndex, Scalar } from "./types";
import { Badge, Button, Loading, Notice, PanelHeader } from "./ui";
import { useMembershipDraft } from "./useMembershipDraft";

function projectIdFromPath(): string | null {
  const route = window.location.pathname.slice(basePath.length);
  const match = route.match(/\/annotations\/([^/]+)\/?$/);
  return match ? decodeURIComponent(match[1]) : null;
}

function formatCount(value: number | undefined): string {
  return value === undefined ? "-" : new Intl.NumberFormat().format(value);
}

function RestartButton({ csrfToken, available }: { csrfToken: string; available: boolean }) {
  const [restarting, setRestarting] = useState(false);
  const [error, setError] = useState("");
  if (!available) return null;

  const restart = async () => {
    setRestarting(true);
    setError("");
    try {
      await restartServer(csrfToken);
      await waitForRestart();
      window.location.reload();
    } catch (restartError) {
      setError(restartError instanceof Error ? restartError.message : "Restart failed");
      setRestarting(false);
    }
  };

  return (
    <div className="restart-control">
      <Button disabled={restarting} onClick={() => void restart()}>
        {restarting ? "Restarting..." : "Restart server"}
      </Button>
      {error && <span>{error}</span>}
    </div>
  );
}

function Dashboard() {
  const [index, setIndex] = useState<ProjectIndex | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    getProjects(controller.signal).then(setIndex).catch((loadError: unknown) => {
      if (!controller.signal.aborted) {
        setError(loadError instanceof Error ? loadError.message : "Could not load projects");
      }
    });
    return () => controller.abort();
  }, []);

  return (
    <main className="dashboard">
      <header className="dashboard-header">
        <strong>Eyck</strong>
        <span>Single-cell annotation</span>
      </header>
      <section className="project-section">
        <div className="project-list-header">
          <div>
            <h1>Annotation projects</h1>
            <p>Select a dataset to inspect and annotate.</p>
          </div>
          {index && <RestartButton csrfToken={index.csrf_token} available={index.restart_available} />}
        </div>
        {error && <Notice tone="danger">{error}</Notice>}
        {!index && !error && <Loading label="Loading projects" />}
        {index && index.projects.length === 0 && (
          <div className="empty-state">No Eyck annotation projects were discovered.</div>
        )}
        {index && index.projects.length > 0 && (
          <div className="project-list">
            {index.projects.map((project) => (
              <a className="project-row" href={`${basePath}/annotations/${encodeURIComponent(project.project_id)}`} key={project.project_id}>
                <div className="project-row-main">
                  <strong>{project.title}</strong>
                  <span>{project.description || project.project_id}</span>
                </div>
                <div className="project-row-meta">
                  <span>{formatCount(project.n_observations)} observations</span>
                  <span>{formatCount(project.n_features)} features</span>
                  <code>{project.project_id}</code>
                </div>
                <span className="project-open">Open</span>
              </a>
            ))}
          </div>
        )}
      </section>
    </main>
  );
}

function SaveIndicator({ status }: { status: ReturnType<typeof useMembershipDraft>["status"] }) {
  const labels = {
    loading: "Loading draft",
    saved: "Saved",
    dirty: "Unsaved",
    saving: "Saving",
    conflict: "Conflict",
    error: "Save failed"
  };
  const className = status === "saved"
    ? "saved"
    : status === "dirty" || status === "saving" || status === "loading" ? "working" : "problem";
  return <span className={`save-indicator ${className}`}><i />{labels[status]}</span>;
}

function Workbench({ projectId }: { projectId: string }) {
  const [project, setProject] = useState<ProjectDetail | null>(null);
  const [points, setPoints] = useState<PointsPayload | null>(null);
  const [loadError, setLoadError] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [activeIndex, setActiveIndex] = useState<number | null>(null);
  const [colorTitle, setColorTitle] = useState("Observations");
  const [colorResult, setColorResult] = useState<ColorResult>(() => uniformColors(0));

  useEffect(() => {
    const controller = new AbortController();
    getProject(projectId, controller.signal)
      .then(async (detail) => {
        setProject(detail);
        setPoints(await getPoints(detail.points_url, controller.signal));
      })
      .catch((error: unknown) => {
        if (!controller.signal.aborted) {
          setLoadError(error instanceof Error ? error.message : "Could not load the project");
        }
      });
    return () => controller.abort();
  }, [projectId]);

  const updateColor = useCallback((title: string, values: Scalar[] | undefined) => {
    if (!points) return;
    setColorTitle(title);
    setColorResult(colorValues(values, points.observation_ids.length));
  }, [points]);

  useEffect(() => {
    if (!points) return;
    setColorResult(uniformColors(points.observation_ids.length));
  }, [points]);

  if (loadError) {
    return (
      <main className="fatal-page">
        <a className="back-link" href={`${basePath}/annotations`}>&lt; All projects</a>
        <Notice tone="danger">{loadError}</Notice>
      </main>
    );
  }
  if (!project || !points) return <Loading fullPage label="Loading annotation workbench" />;

  return (
    <LoadedWorkbench
      project={project}
      points={points}
      activeIndex={activeIndex}
      selectedIds={selectedIds}
      colorTitle={colorTitle}
      colorResult={colorResult}
      onColor={updateColor}
      onSingleSelect={(index) => {
        setActiveIndex(index);
        setSelectedIds(new Set([points.observation_ids[index]]));
      }}
      onLasso={(indices, mode) => {
        const incoming = indices.map((index) => points.observation_ids[index]);
        const next = mode === "replace" ? new Set<string>() : new Set(selectedIds);
        incoming.forEach((id) => {
          if (mode === "subtract") next.delete(id);
          else next.add(id);
        });
        setSelectedIds(next);
        const firstId = next.values().next().value;
        setActiveIndex(firstId === undefined ? null : points.observation_ids.indexOf(firstId));
      }}
      onClearSelection={() => {
        setSelectedIds(new Set());
        setActiveIndex(null);
      }}
    />
  );
}

interface LoadedWorkbenchProps {
  project: ProjectDetail;
  points: PointsPayload;
  activeIndex: number | null;
  selectedIds: Set<string>;
  colorTitle: string;
  colorResult: ColorResult;
  onColor: (title: string, values: Scalar[] | undefined) => void;
  onSingleSelect: (index: number) => void;
  onLasso: (indices: number[], mode: SelectionMode) => void;
  onClearSelection: () => void;
}

function LoadedWorkbench({
  project,
  points,
  activeIndex,
  selectedIds,
  colorTitle,
  colorResult,
  onColor,
  onSingleSelect,
  onLasso,
  onClearSelection
}: LoadedWorkbenchProps) {
  const memberships = useMembershipDraft(project.memberships_url, project.csrf_token);
  const [exportState, setExportState] = useState<"idle" | "working" | "done" | "error">("idle");
  const [exportMessage, setExportMessage] = useState("");
  const indexById = useMemo(() => new Map(points.observation_ids.map((id, index) => [id, index])), [points]);
  const selectedIndices = useMemo(() => new Set([...selectedIds]
    .map((id) => indexById.get(id))
    .filter((index): index is number => index !== undefined)), [indexById, selectedIds]);
  const selectedObservationIds = [...selectedIds];

  const runExport = async () => {
    setExportState("working");
    setExportMessage("");
    try {
      await exportAnnotations(project.export_url, memberships.revision, project.csrf_token);
      setExportState("done");
      setExportMessage("Export completed");
    } catch (error) {
      setExportState("error");
      setExportMessage(error instanceof Error ? error.message : "Export failed");
    }
  };

  return (
    <main className="workbench">
      <header className="workbench-header">
        <a className="compact-brand" href={`${basePath}/annotations`}>Eyck</a>
        <div className="project-title">
          <h1>{project.title}</h1>
          <code>{project.project_id}</code>
          <Badge>{formatCount(project.n_observations)} cells</Badge>
        </div>
        <div className="header-actions">
          <SaveIndicator status={memberships.status} />
          <Button
            tone="primary"
            disabled={memberships.status !== "saved" || exportState === "working"}
            onClick={() => void runExport()}
          >
            {exportState === "working" ? "Exporting..." : "Export"}
          </Button>
          <RestartButton csrfToken={project.csrf_token} available={project.restart_available} />
        </div>
      </header>

      {(memberships.status === "conflict" || memberships.status === "error" || exportMessage) && (
        <div className="issue-strip">
          <div className={`issue ${memberships.status === "conflict" || memberships.status === "error" || exportState === "error" ? "issue-error" : ""}`}>
            <span>{memberships.message || exportMessage}</span>
            <div className="issue-actions">
              {memberships.status === "conflict" && (
                <>
                  <Button onClick={() => void memberships.mergeLatest()}>Merge local edits</Button>
                  <Button onClick={() => void memberships.discardLocal()}>Discard local</Button>
                </>
              )}
              {memberships.status === "error" && <Button onClick={memberships.retry}>Retry save</Button>}
              {exportMessage && <Button onClick={() => setExportMessage("")}>Dismiss</Button>}
            </div>
          </div>
        </div>
      )}

      <div className="workbench-grid">
        <aside className="controls-panel panel">
          <PanelHeader title="Display" meta={<Badge>{formatCount(project.n_features)} features</Badge>} />
          <ColoringControls
            metadataColumns={project.metadata_columns}
            modalities={project.modalities}
            points={points}
            featuresUrl={project.features_url}
            onColor={onColor}
          />
          <section className="legend-section">
            <div className="section-heading"><span>{colorTitle}</span></div>
            <div className="legend-list">
              {colorResult.legend.map((item, index) => (
                <div className="legend-item" key={`${item.label}-${index}`}>
                  <i style={{ backgroundColor: item.color }} />
                  <span title={item.label}>{item.label}</span>
                </div>
              ))}
            </div>
          </section>
          {project.description && (
            <section className="project-description">
              <div className="section-heading"><span>Project</span></div>
              <p>{project.description}</p>
            </section>
          )}
        </aside>

        <section className="plot-panel">
          <Scatterplot
            coordinates={points.coordinates}
            colors={colorResult.colors}
            selectedIndices={selectedIndices}
            onSingleSelect={onSingleSelect}
            onLasso={onLasso}
            onClearSelection={onClearSelection}
          />
        </section>

        <AnnotationPanel
          labels={project.labels}
          points={points}
          activeIndex={activeIndex}
          selectedObservationIds={selectedObservationIds}
          draft={memberships.draft}
          origin={memberships.origin}
          onApply={(entityId, labelId, state) => memberships.applyState(
            selectedObservationIds,
            entityId,
            labelId,
            state
          )}
        />
      </div>
    </main>
  );
}

export default function App() {
  const projectId = projectIdFromPath();
  return projectId ? <Workbench projectId={projectId} /> : <Dashboard />;
}
