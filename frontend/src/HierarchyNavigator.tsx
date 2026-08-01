import { useEffect, useMemo, useState } from "react";
import type {
  MembershipDraft,
  SavedSelection,
  WorkspaceDocument,
  WorkspaceObjectKind,
  Zoom
} from "./types";
import { Button, Notice, PanelHeader } from "./ui";
import { formatCount } from "./workspace";

interface HierarchyNavigatorProps {
  workspace: WorkspaceDocument;
  currentZoomId: string;
  focusedSelectionId: string | null;
  visibleSelectionIds: Set<string>;
  selectionColors: Map<string, string>;
  draftCount: number;
  memberships: MembershipDraft | null;
  destructiveDisabled: boolean;
  destructiveDisabledReason: string;
  onNavigate: (zoomId: string) => void;
  onFocusSelection: (selection: SavedSelection) => void;
  onToggleSelection: (selectionId: string) => void;
  onDelete: (kind: WorkspaceObjectKind, id: string) => void;
  onOpenMarker: (programId: string) => void;
}

function TypeIcon({ children, title }: { children: string; title: string }) {
  return <span className="tree-type" title={title} aria-label={title}>{children}</span>;
}

export function HierarchyNavigator({
  workspace,
  currentZoomId,
  focusedSelectionId,
  visibleSelectionIds,
  selectionColors,
  draftCount,
  memberships,
  destructiveDisabled,
  destructiveDisabledReason,
  onNavigate,
  onFocusSelection,
  onToggleSelection,
  onDelete,
  onOpenMarker
}: HierarchyNavigatorProps) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const children = useMemo(() => {
    const result = new Map<string, Zoom[]>();
    workspace.zooms.forEach((zoom) => {
      if (!zoom.parent_id) return;
      result.set(zoom.parent_id, [...(result.get(zoom.parent_id) ?? []), zoom]);
    });
    return result;
  }, [workspace.zooms]);
  const annotations = useMemo(() => {
    const result = new Map<string, Array<{ key: string; label: string; count: number }>>();
    if (!memberships) return result;
    const grouped = new Map<string, Set<string>>();
    memberships.rows.forEach((row) => {
      if (!row.selection_id || !row.zoom_id || row.state === "unreviewed") return;
      const key = `${row.zoom_id}\u0000${row.selection_id}\u0000${row.entity_id}\u0000${row.label_id}\u0000${row.state}`;
      if (!grouped.has(key)) grouped.set(key, new Set());
      grouped.get(key)?.add(row.observation_id);
    });
    grouped.forEach((ids, key) => {
      const [zoomId, selectionId, entityId, labelId, state] = key.split("\u0000");
      const items = result.get(zoomId) ?? [];
      items.push({ key, label: `${entityId}: ${labelId} ${state} [${selectionId}]`, count: ids.size });
      result.set(zoomId, items);
    });
    return result;
  }, [memberships]);

  useEffect(() => {
    setCollapsed((current) => {
      const next = new Set(current);
      let zoom = workspace.zooms.find((item) => item.id === currentZoomId);
      while (zoom) {
        next.delete(zoom.id);
        zoom = zoom.parent_id ? workspace.zooms.find((item) => item.id === zoom?.parent_id) : undefined;
      }
      return next;
    });
  }, [currentZoomId, workspace.zooms]);

  const renderZoom = (zoom: Zoom, depth: number): React.ReactNode => {
    const zoomChildren = children.get(zoom.id) ?? [];
    const isCollapsed = collapsed.has(zoom.id);
    const selections = workspace.selections.filter((item) => item.zoom_id === zoom.id);
    const clusterings = workspace.clusterings.filter((item) => item.zoom_id === zoom.id);
    const embeddings = workspace.embeddings.filter((item) => item.zoom_id === zoom.id);
    const programs = workspace.marker_programs.filter((item) => item.zoom_id === zoom.id);
    const summaries = annotations.get(zoom.id) ?? [];
    const hasContents = zoomChildren.length + selections.length + clusterings.length + embeddings.length + programs.length + summaries.length > 0
      || (zoom.id === currentZoomId && draftCount > 0);
    return (
      <div className="tree-branch" key={zoom.id}>
        <div className={`tree-row zoom-row ${zoom.id === currentZoomId ? "current" : ""}`} style={{ paddingLeft: 6 + depth * 14 }}>
          <button
            className="tree-collapse"
            type="button"
            disabled={!hasContents}
            aria-label={`${isCollapsed ? "Expand" : "Collapse"} ${zoom.name}`}
            onClick={() => setCollapsed((current) => {
              const next = new Set(current);
              if (next.has(zoom.id)) next.delete(zoom.id); else next.add(zoom.id);
              return next;
            })}
          >{hasContents ? (isCollapsed ? "+" : "-") : ""}</button>
          <TypeIcon title="Zoom" children="Z" />
          <button className="tree-main" type="button" onClick={() => onNavigate(zoom.id)}>
            <strong>{zoom.name}</strong><small>{formatCount(zoom.observation_count)} cells</small>
          </button>
          {zoom.parent_id && <Button className="tree-action danger-link" disabled={destructiveDisabled} title={destructiveDisabled ? destructiveDisabledReason : `Delete ${zoom.name}`} onClick={() => onDelete("zoom", zoom.id)}>x</Button>}
        </div>
        {!isCollapsed && (
          <div>
            {zoom.id === currentZoomId && draftCount > 0 && (
              <div className="tree-row draft-row" style={{ paddingLeft: 34 + depth * 14 }}>
                <TypeIcon title="Draft selection" children="D" />
                <span className="tree-main"><strong>Unsaved selection</strong><small>{formatCount(draftCount)} cells</small></span>
              </div>
            )}
            {selections.map((selection) => (
              <div className={`tree-row ${focusedSelectionId === selection.id ? "focused" : ""}`} key={selection.id} style={{ paddingLeft: 34 + depth * 14 }}>
                <button
                  className={`visibility-toggle ${visibleSelectionIds.has(selection.id) ? "visible" : ""}`}
                  type="button"
                  aria-label={`${visibleSelectionIds.has(selection.id) ? "Hide" : "Show"} ${selection.name}`}
                  aria-pressed={visibleSelectionIds.has(selection.id)}
                  onClick={() => onToggleSelection(selection.id)}
                ><i style={{ background: selectionColors.get(selection.id) }} /></button>
                <TypeIcon title={`Selection: ${selection.definition.kind}`} children="S" />
                <button className="tree-main" type="button" onClick={() => onFocusSelection(selection)}>
                  <strong>{selection.name}</strong><small>{selection.definition.kind} / {formatCount(selection.observation_count)}</small>
                </button>
                <Button className="tree-action danger-link" disabled={destructiveDisabled} title={destructiveDisabled ? destructiveDisabledReason : `Delete ${selection.name}`} onClick={() => onDelete("selection", selection.id)}>x</Button>
              </div>
            ))}
            {clusterings.map((item) => (
              <div className="tree-row" key={item.id} style={{ paddingLeft: 34 + depth * 14 }}>
                <TypeIcon title="Clustering" children="C" />
                <span className="tree-main"><strong>{item.name}</strong><small>{formatCount(item.cluster_count)} clusters</small></span>
                <Button className="tree-action danger-link" disabled={destructiveDisabled} title={destructiveDisabled ? destructiveDisabledReason : `Delete ${item.name}`} onClick={() => onDelete("clustering", item.id)}>x</Button>
              </div>
            ))}
            {embeddings.map((item) => (
              <div className="tree-row" key={item.id} style={{ paddingLeft: 34 + depth * 14 }}>
                <TypeIcon title="Embedding" children="E" />
                <span className="tree-main"><strong>{item.name}</strong><small>{item.cache_path ? "computed" : "source"}</small></span>
                {item.cache_path && <Button className="tree-action danger-link" disabled={destructiveDisabled} title={destructiveDisabled ? destructiveDisabledReason : `Delete ${item.name}`} onClick={() => onDelete("embedding", item.id)}>x</Button>}
              </div>
            ))}
            {programs.map((item) => (
              <div className="tree-row" key={item.id} style={{ paddingLeft: 34 + depth * 14 }}>
                <TypeIcon title="Marker program" children="M" />
                <button className="tree-main" type="button" onClick={() => onOpenMarker(item.id)}>
                  <strong>{item.name}</strong><small>{item.marker_ids.length} markers</small>
                </button>
                <Button className="tree-action danger-link" disabled={destructiveDisabled} title={destructiveDisabled ? destructiveDisabledReason : `Delete ${item.name}`} onClick={() => onDelete("marker_program", item.id)}>x</Button>
              </div>
            ))}
            {summaries.map((item) => (
              <div className="tree-row annotation-summary" key={item.key} style={{ paddingLeft: 34 + depth * 14 }}>
                <TypeIcon title="Annotation summary" children="A" />
                <button className="tree-main" type="button" onClick={() => {
                  const selectionId = item.key.split("\u0000")[1];
                  const selection = workspace.selections.find((candidate) => candidate.id === selectionId);
                  if (selection) onFocusSelection(selection);
                }}><strong>{item.label}</strong><small>{formatCount(item.count)} decisions</small></button>
              </div>
            ))}
            {zoomChildren.map((child) => renderZoom(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  const root = workspace.zooms.find((zoom) => zoom.id === workspace.root_zoom_id);
  return (
    <aside className="hierarchy-panel panel">
      <PanelHeader title="Hierarchy / layers" meta={<span className="revision-tag">rev {workspace.revision.slice(0, 7)}</span>} />
      <div className="tree-legend"><span>Z zoom</span><span>S selection</span><span>C clustering</span><span>E embedding</span><span>M marker</span><span>A annotation</span></div>
      {destructiveDisabled && <Notice className="destructive-lock">Deletion locked: {destructiveDisabledReason}</Notice>}
      <div className="hierarchy-tree">{root ? renderZoom(root, 0) : <p className="panel-empty">Root zoom is unavailable.</p>}</div>
    </aside>
  );
}
