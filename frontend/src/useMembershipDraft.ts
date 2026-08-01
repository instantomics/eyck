import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, getMemberships, putMemberships } from "./api";
import type {
  MembershipDraft,
  MembershipPayload,
  MembershipRow,
  MembershipState
} from "./types";

export type SaveStatus = "loading" | "saved" | "dirty" | "saving" | "conflict" | "error";

function asDraft(payload: MembershipPayload): MembershipDraft {
  return {
    rows: payload.rows,
    support_observation_ids: payload.support_observation_ids
  };
}

function rowKey(row: MembershipRow): string {
  return `${row.support_id}\u0000${row.observation_id}\u0000${row.entity_id}\u0000${row.label_id}`;
}

function rowSignature(row: MembershipRow | undefined): string {
  return row ? JSON.stringify(row) : "";
}

function sortRows(rows: MembershipRow[]): MembershipRow[] {
  return [...rows].sort((a, b) => rowKey(a).localeCompare(rowKey(b)));
}

export function useMembershipDraft(url: string, csrfToken: string) {
  const [draft, setDraft] = useState<MembershipDraft | null>(null);
  const [base, setBase] = useState<MembershipDraft | null>(null);
  const [revision, setRevision] = useState("");
  const [origin, setOrigin] = useState("");
  const [status, setStatus] = useState<SaveStatus>("loading");
  const [message, setMessage] = useState("");
  const [hasLocalChanges, setHasLocalChanges] = useState(false);
  const editVersion = useRef(0);
  const saveInFlight = useRef(false);

  const installPayload = useCallback((payload: MembershipPayload) => {
    const next = asDraft(payload);
    setDraft(next);
    setBase(next);
    setRevision(payload.revision);
    setOrigin(payload.origin);
    setStatus("saved");
    setMessage("");
    setHasLocalChanges(false);
    editVersion.current += 1;
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setStatus("loading");
    getMemberships(url, controller.signal).then(installPayload).catch((error: unknown) => {
      if (controller.signal.aborted) return;
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Could not load annotation draft");
    });
    return () => controller.abort();
  }, [installPayload, url]);

  const saveSnapshot = useCallback(async () => {
    if (!draft || !revision || status === "saving" || status === "conflict" || saveInFlight.current) return;
    const snapshot = draft;
    const version = editVersion.current;
    saveInFlight.current = true;
    setStatus("saving");
    setMessage("");
    try {
      const payload = await putMemberships(url, revision, csrfToken, snapshot);
      setRevision(payload.revision);
      setBase(snapshot);
      if (editVersion.current === version) {
        setDraft(asDraft(payload));
        setBase(asDraft(payload));
        setOrigin(payload.origin);
        setStatus("saved");
        setHasLocalChanges(false);
      } else {
        setStatus("dirty");
        setHasLocalChanges(true);
      }
    } catch (error) {
      if (error instanceof ApiError && (error.status === 409 || error.status === 412)) {
        setStatus("conflict");
        setMessage("The server draft changed since it was opened.");
      } else {
        setStatus("error");
        setMessage(error instanceof Error ? error.message : "Autosave failed");
      }
    } finally {
      saveInFlight.current = false;
    }
  }, [csrfToken, draft, revision, status, url]);

  useEffect(() => {
    if (status !== "dirty") return;
    const timer = window.setTimeout(() => void saveSnapshot(), 650);
    return () => window.clearTimeout(timer);
  }, [saveSnapshot, status]);

  const applyState = useCallback((
    observationIds: string[],
    entityId: string,
    labelId: string,
    state: MembershipState,
    context: { selectionId: string; zoomId: string }
  ) => {
    if (observationIds.length === 0) return;
    setDraft((current) => {
      if (!current) return current;
      const targetIds = new Set(observationIds);
      const remaining = current.rows.filter((row) => !(
        targetIds.has(row.observation_id)
        && row.support_id === context.selectionId
        && row.entity_id === entityId
        && row.label_id === labelId
      ));
      const additions = observationIds.map((observationId) => ({
        support_id: context.selectionId,
        observation_id: observationId,
        entity_id: entityId,
        label_id: labelId,
        state,
        decision_view_id: context.zoomId,
        provenance: "explicit",
        selection_id: context.selectionId,
        zoom_id: context.zoomId
      }));
      return {
        rows: sortRows([...remaining, ...additions]),
        support_observation_ids: [...new Set([
          ...current.support_observation_ids,
          ...observationIds
        ])].sort()
      };
    });
    editVersion.current += 1;
    setHasLocalChanges(true);
    setStatus("dirty");
    setMessage("");
  }, []);

  const loadLatest = useCallback(async () => {
    setStatus("loading");
    try {
      installPayload(await getMemberships(url));
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Could not reload the server draft");
      throw error;
    }
  }, [installPayload, url]);

  const reload = useCallback(async () => {
    if (status !== "saved" || hasLocalChanges) {
      throw new Error(`Membership reload is blocked while local state is ${status}`);
    }
    await loadLatest();
  }, [hasLocalChanges, loadLatest, status]);

  const discardLocal = useCallback(async () => {
    try {
      await loadLatest();
    } catch {
      // loadLatest already exposes the failure through hook status and message.
    }
  }, [loadLatest]);

  const mergeLatest = useCallback(async () => {
    if (!draft || !base) return;
    setStatus("loading");
    try {
      const latest = await getMemberships(url);
      const baseRows = new Map(base.rows.map((row) => [rowKey(row), row]));
      const localRows = new Map(draft.rows.map((row) => [rowKey(row), row]));
      const mergedRows = new Map(latest.rows.map((row) => [rowKey(row), row]));
      const keys = new Set([...baseRows.keys(), ...localRows.keys()]);
      keys.forEach((key) => {
        const before = baseRows.get(key);
        const local = localRows.get(key);
        if (rowSignature(before) === rowSignature(local)) return;
        if (local) mergedRows.set(key, local);
        else mergedRows.delete(key);
      });

      const baseSupport = new Set(base.support_observation_ids);
      const localSupport = new Set(draft.support_observation_ids);
      const mergedSupport = new Set(latest.support_observation_ids);
      new Set([...baseSupport, ...localSupport]).forEach((id) => {
        if (baseSupport.has(id) === localSupport.has(id)) return;
        if (localSupport.has(id)) mergedSupport.add(id);
        else mergedSupport.delete(id);
      });

      setBase(asDraft(latest));
      setDraft({
        rows: sortRows([...mergedRows.values()]),
        support_observation_ids: [...mergedSupport].sort()
      });
      setRevision(latest.revision);
      setOrigin(latest.origin);
      editVersion.current += 1;
      setHasLocalChanges(true);
      setStatus("dirty");
      setMessage("Local edits were merged onto the latest server draft.");
    } catch (error) {
      setStatus("error");
      setMessage(error instanceof Error ? error.message : "Could not merge the server draft");
    }
  }, [base, draft, url]);

  const retry = useCallback(() => {
    if (draft) setStatus("dirty");
    else void loadLatest().catch(() => undefined);
  }, [draft, loadLatest]);

  return {
    draft,
    revision,
    origin,
    status,
    message,
    hasLocalChanges,
    applyState,
    reload,
    reloadAfterCommit: loadLatest,
    discardLocal,
    mergeLatest,
    retry
  };
}
