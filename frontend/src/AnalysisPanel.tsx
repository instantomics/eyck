import { useState } from "react";
import type { ClusteringImport, LocalAnalysisCreate, ProjectDetail, WorkspaceDocument } from "./types";
import { Badge, Button, Field, Notice, Select } from "./ui";

interface AnalysisPanelProps {
  project: ProjectDetail;
  workspace: WorkspaceDocument;
  zoomId: string;
  onImport: (request: ClusteringImport) => Promise<void>;
  onCompute: (request: LocalAnalysisCreate) => Promise<void>;
  onEmbeddingCreated: (embeddingId: string) => void;
}

export function AnalysisPanel({ project, workspace, zoomId, onImport, onCompute, onEmbeddingCreated }: AnalysisPanelProps) {
  const [importId, setImportId] = useState("");
  const [importName, setImportName] = useState("");
  const [column, setColumn] = useState(project.metadata_columns[0] ?? "");
  const [clusteringId, setClusteringId] = useState("");
  const [clusteringName, setClusteringName] = useState("");
  const [embeddingId, setEmbeddingId] = useState("");
  const [embeddingName, setEmbeddingName] = useState("");
  const [profile, setProfile] = useState<LocalAnalysisCreate["profile"]>("scanpy_standard");
  const [useCounts, setUseCounts] = useState(false);
  const [neighbors, setNeighbors] = useState(15);
  const [pcs, setPcs] = useState("");
  const [resolutionsText, setResolutionsText] = useState("1");
  const [topGenes, setTopGenes] = useState("");
  const [components, setComponents] = useState("");
  const [seed, setSeed] = useState(0);
  const [working, setWorking] = useState<"import" | "local" | null>(null);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const resolutions = resolutionsText.split(",").map((value) => Number(value.trim())).filter((value) => Number.isFinite(value) && value > 0);
  const clusteringOutputs = resolutions.map((resolution) => ({
    id: resolutions.length === 1 ? clusteringId : `${clusteringId}-${resolution}`,
    name: resolutions.length === 1 ? clusteringName : `${clusteringName} ${resolution}`,
    resolution
  }));
  const validOutputs = resolutions.length > 0 && new Set(clusteringOutputs.map((item) => item.id)).size === clusteringOutputs.length;
  const validFastDimensions = profile !== "scanpy_fast" || !pcs || !components || Number(pcs) <= Number(components);

  const run = async (kind: "import" | "local", action: () => Promise<void>) => {
    setWorking(kind);
    setMessage("");
    setError("");
    try {
      await action();
      setMessage(kind === "local" ? "Local embedding and clusterings completed." : "Declared metadata clustering imported.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Analysis failed");
    } finally {
      setWorking(null);
    }
  };

  return (
    <section className="tool-card analysis-tools">
      <header className="section-title-row">
        <div><span>Explicit computation</span><h2>Analysis controls</h2></div>
        <Badge>{workspace.clusterings.filter((item) => item.zoom_id === zoomId).length} clusterings</Badge>
      </header>
      {error && <Notice tone="danger" role="alert"><strong>Analysis failed.</strong> {error}</Notice>}
      {working && <Notice role="status">Working: {working === "local" ? "neighbors, embedding, and clusterings" : "metadata import"}. Keep this page open.</Notice>}
      {message && <Notice>{message}</Notice>}
      <details className="tool-section" open>
        <summary>Import declared metadata as clustering</summary>
        <div className="form-grid">
          <Field label="Clustering ID"><input className="pt-input" value={importId} onChange={(event) => setImportId(event.target.value)} /></Field>
          <Field label="Name"><input className="pt-input" value={importName} onChange={(event) => setImportName(event.target.value)} /></Field>
          <Field label="Declared column"><Select value={column} onChange={(event) => setColumn(event.target.value)}>{project.metadata_columns.map((item) => <option key={item}>{item}</option>)}</Select></Field>
          <div className="form-actions"><Button tone="primary" disabled={!importId || !importName || !column || working !== null} onClick={() => void run("import", () => onImport({ id: importId, name: importName, zoom_id: zoomId, metadata_column: column }))}>Import clustering</Button></div>
        </div>
      </details>
      <details className="tool-section" open>
        <summary>Compute local structure</summary>
        <div className="form-grid analysis-grid">
          <Field label="Clustering ID"><input className="pt-input" value={clusteringId} onChange={(event) => setClusteringId(event.target.value)} /></Field>
          <Field label="Clustering name"><input className="pt-input" value={clusteringName} onChange={(event) => setClusteringName(event.target.value)} /></Field>
          <Field label="Embedding ID"><input className="pt-input" value={embeddingId} onChange={(event) => setEmbeddingId(event.target.value)} /></Field>
          <Field label="Embedding name"><input className="pt-input" value={embeddingName} onChange={(event) => setEmbeddingName(event.target.value)} /></Field>
          <Field label="Profile"><Select value={profile} onChange={(event) => setProfile(event.target.value as LocalAnalysisCreate["profile"])}><option value="scanpy_default">Scanpy default</option><option value="scanpy_standard">Scanpy standard</option><option value="scanpy_fast">Scanpy fast</option></Select></Field>
          <Field label="Use counts layer"><input type="checkbox" checked={useCounts} onChange={(event) => setUseCounts(event.target.checked)} /></Field>
          <Field label="n_neighbors"><input className="pt-input" type="number" min={1} value={neighbors} onChange={(event) => setNeighbors(Number(event.target.value))} /></Field>
          <Field label="n_pcs" hint="Blank uses backend default."><input className="pt-input" type="number" min={1} value={pcs} onChange={(event) => setPcs(event.target.value)} /></Field>
          <Field label="Resolutions" hint="Comma-separated; shared graph and embedding."><input className="pt-input" value={resolutionsText} onChange={(event) => setResolutionsText(event.target.value)} /></Field>
          {profile === "scanpy_fast" && <Field label="Top genes" hint="Blank uses 3000."><input className="pt-input" type="number" min={1} value={topGenes} onChange={(event) => setTopGenes(event.target.value)} /></Field>}
          {profile === "scanpy_fast" && <Field label="PCA components" hint="Blank uses 50."><input className="pt-input" type="number" min={1} value={components} onChange={(event) => setComponents(event.target.value)} /></Field>}
          <Field label="Random seed"><input className="pt-input" type="number" value={seed} onChange={(event) => setSeed(Number(event.target.value))} /></Field>
          <div className="wide-field resolved-parameters"><span>Resolved request</span><code>{JSON.stringify({ profile, use_counts: useCounts, n_neighbors: neighbors, n_pcs: pcs ? Number(pcs) : null, clustering_outputs: clusteringOutputs, random_state: seed })}</code></div>
          <div className="form-actions wide-field"><Button tone="primary" disabled={!clusteringId || !clusteringName || !embeddingId || !embeddingName || neighbors < 1 || !validOutputs || !validFastDimensions || working !== null} onClick={() => void run("local", async () => {
            const request: LocalAnalysisCreate = {
              embedding_id: embeddingId,
              embedding_name: embeddingName,
              zoom_id: zoomId,
              clustering_outputs: clusteringOutputs,
              profile,
              use_counts: useCounts,
              n_neighbors: neighbors,
              n_pcs: pcs ? Number(pcs) : null,
              random_state: seed,
              ...(profile === "scanpy_fast" ? {
                n_top_genes: topGenes ? Number(topGenes) : null,
                n_comps: components ? Number(components) : null
              } : {})
            };
            await onCompute(request);
            onEmbeddingCreated(embeddingId);
          })}>{working === "local" ? "Computing..." : "Run local analysis"}</Button></div>
        </div>
      </details>
    </section>
  );
}
