import { useEffect, useRef, useState } from "react";
import type { DeletionImpact, WorkspaceObjectRef } from "./types";
import { Badge, Button, Notice } from "./ui";
import { formatCount } from "./workspace";

interface DeleteDialogProps {
  target: WorkspaceObjectRef | null;
  onPreview: (target: WorkspaceObjectRef) => Promise<DeletionImpact>;
  onConfirm: (impact: DeletionImpact) => Promise<void>;
  onClose: () => void;
}

export function DeleteDialog({ target, onPreview, onConfirm, onClose }: DeleteDialogProps) {
  const [impact, setImpact] = useState<DeletionImpact | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const requestGeneration = useRef(0);
  const previousFocus = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!target) return;
    previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    window.setTimeout(() => document.getElementById("delete-dialog")?.focus(), 0);
    const generation = ++requestGeneration.current;
    const targetKey = JSON.stringify(target);
    setImpact(null);
    setConfirmed(false);
    setWorking(true);
    setError("");
    onPreview(target).then((loaded) => {
      if (requestGeneration.current !== generation || JSON.stringify(loaded.requested) !== targetKey) return;
      setImpact(loaded);
    }).catch((caught: unknown) => {
      if (requestGeneration.current === generation) setError(caught instanceof Error ? caught.message : "Could not preview deletion");
    }).finally(() => {
      if (requestGeneration.current === generation) setWorking(false);
    });
    return () => {
      requestGeneration.current += 1;
      const restore = previousFocus.current;
      window.setTimeout(() => restore?.focus(), 0);
    };
  }, [onPreview, target]);

  useEffect(() => {
    if (!target) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || working) return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose, target, working]);

  if (!target) return null;
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !working) onClose(); }}>
      <section id="delete-dialog" className="modal-card delete-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-title" tabIndex={-1}>
        <header className="modal-header"><div><span>Dependency-aware operation</span><h2 id="delete-title">Delete {target.kind}: {target.id}</h2></div><Button id="delete-dialog-close" disabled={working} onClick={onClose}>Close</Button></header>
        <div className="modal-body">
          {working && !impact && <Notice role="status">Asking the server for the exact dependency cascade...</Notice>}
          {error && <Notice tone="danger" role="alert">{error}</Notice>}
          {impact && (
            <>
              <Notice tone="warning">The server resolved this cascade at workspace revision <code>{impact.workspace_revision.slice(0, 12)}</code> and membership revision <code>{impact.membership_revision.slice(0, 12)}</code>. Confirmation sends this exact list and impact hash.</Notice>
              <div className="impact-counts">{Object.entries(impact.counts).map(([kind, count]) => <Badge key={kind}>{kind}: {count}</Badge>)}</div>
              <div className="impact-list">{impact.removed.map((item) => <div key={`${item.kind}-${item.id}`}><span className="tree-type">{item.kind[0].toUpperCase()}</span><strong>{item.kind}</strong><code>{item.id}</code></div>)}</div>
              <div className="membership-impact"><strong>{formatCount(impact.membership_decision_count)} membership decisions affected</strong><span>{formatCount(impact.membership_observation_ids.length)} unique observations</span>{impact.membership_observation_ids.length > 0 && <code>{impact.membership_observation_ids.slice(0, 24).join(", ")}{impact.membership_observation_ids.length > 24 ? " ..." : ""}</code>}</div>
              <label className="confirm-check"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /><span>I reviewed and confirm this exact server-provided cascade.</span></label>
            </>
          )}
        </div>
        <footer className="modal-footer"><Button onClick={onClose} disabled={working}>Cancel</Button><Button tone="danger" disabled={!impact || !confirmed || working} onClick={() => {
          if (!impact) return;
          setWorking(true);
          setError("");
          void onConfirm(impact).then(onClose).catch((caught: unknown) => setError(caught instanceof Error ? caught.message : "Deletion failed")).finally(() => setWorking(false));
        }}>{working && impact ? "Deleting..." : "Delete exact cascade"}</Button></footer>
      </section>
    </div>
  );
}
