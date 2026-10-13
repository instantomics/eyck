import { useEffect, useMemo, useState } from "react";
import { basePath, getProjects } from "./api";
import type { ProjectIndex, ProjectSummary } from "./types";
import { Loading, Notice } from "./ui";
import { formatCount } from "./workspace";

// The server rescans for projects every minute; the overview follows it.
const POLL_MS = 30_000;

function projectUrl(project: ProjectSummary, processes: boolean): string {
  return `${basePath}/annotations/${encodeURIComponent(project.project_id)}${processes ? "/processes" : ""}`;
}

function SetProgress({ project }: { project: ProjectSummary }) {
  if (project.n_processes === 0) return null;
  if (project.process_sets.length === 0) return <span className="overview-set overview-set-empty">not drawn yet</span>;
  return <>{project.process_sets.map((set) => {
    const done = set.drawn_processes >= project.n_processes;
    return <span
      key={set.id}
      className={`overview-set${done ? " overview-set-done" : ""}`}
      title={`Set ${set.id}: ${set.drawn_processes} of ${project.n_processes} processes drawn, ${set.reviewed_processes} reviewed${set.exported ? ", exported" : ""}`}
    >
      <b>{set.id}</b> {set.drawn_processes}/{project.n_processes} drawn
      {set.reviewed_processes > 0 && ` · ${set.reviewed_processes} reviewed`}
      {set.exported && " · exported"}
    </span>;
  })}</>;
}

function ProjectCard({ project }: { project: ProjectSummary }) {
  const drawable = project.n_processes > 0;
  return <article className="overview-card">
    <a className="overview-card-main" href={projectUrl(project, drawable)}>
      <strong>{project.title}</strong>
      <span className="overview-description">{project.description}</span>
      <span className="overview-meta">
        <span>{formatCount(project.n_observations)} cells</span>
        {drawable && <span>{project.n_processes} processes</span>}
        <code>{project.project_id}</code>
      </span>
    </a>
    <div className="overview-card-side">
      <div className="overview-sets"><SetProgress project={project} /></div>
      <div className="overview-actions">
        {drawable && <a className="pt-button overview-primary" href={projectUrl(project, true)}>Draw processes</a>}
        <a className="pt-button" href={projectUrl(project, false)}>Annotate cells</a>
      </div>
    </div>
  </article>;
}

export function ProjectOverview() {
  const [index, setIndex] = useState<ProjectIndex | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    let controller = new AbortController();
    const load = () => {
      controller.abort();
      controller = new AbortController();
      getProjects(controller.signal).then((loaded) => { setIndex(loaded); setError(null); }).catch((caught: unknown) => {
        if (!controller.signal.aborted) setError(caught instanceof Error ? caught.message : "Could not load projects");
      });
    };
    load();
    const timer = window.setInterval(load, POLL_MS);
    return () => { window.clearInterval(timer); controller.abort(); };
  }, []);

  const groups = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const grouped = new Map<string, ProjectSummary[]>();
    for (const project of index?.projects ?? []) {
      const text = [project.title, project.description, project.project_id, project.source_owner, project.dataset_id].join(" ").toLocaleLowerCase();
      if (needle && !text.includes(needle)) continue;
      const owner = project.source_owner || "other";
      grouped.set(owner, [...(grouped.get(owner) ?? []), project]);
    }
    return [...grouped].sort(([a], [b]) => a.localeCompare(b));
  }, [index, query]);

  const total = index?.projects.length ?? 0;
  const drawable = index?.projects.filter((project) => project.n_processes > 0).length ?? 0;

  return <main className="dashboard">
    <header className="dashboard-header"><strong>Eyck</strong><span>Single-cell annotation</span></header>
    <section className="project-section">
      <div className="project-list-header">
        <div>
          <h1>Projects</h1>
          <p>{index ? `${total} projects from ${new Set(index.projects.map((p) => p.source_owner)).size} datasets, ${drawable} with processes to draw. New projects appear here automatically.` : "Loading the project list."}</p>
        </div>
        <input className="overview-search" type="search" placeholder="Search projects" value={query} onChange={(event) => setQuery(event.target.value)} />
      </div>
      {error && <Notice tone="danger">{error}</Notice>}
      {index && index.discovery_errors.length > 0 && <Notice tone="warning">
        <strong>Some project declarations could not be loaded; the other projects are unaffected.</strong>
        <ul className="overview-problems">{index.discovery_errors.map((problem) => <li key={problem.manifest}><code>{problem.manifest}</code>: {problem.message}</li>)}</ul>
      </Notice>}
      {!index && !error && <Loading label="Loading projects" />}
      {index && total === 0 && <div className="empty-state">No Eyck projects were found yet.</div>}
      {index && total > 0 && groups.length === 0 && <div className="empty-state">No project matches “{query}”.</div>}
      {groups.map(([owner, projects]) => <section className="overview-group" key={owner}>
        <h2>{owner}<span>{projects.length}</span></h2>
        <div className="project-list">{projects.map((project) => <ProjectCard project={project} key={project.project_id} />)}</div>
      </section>)}
    </section>
  </main>;
}
