import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { getFeatureValues, searchFeatures } from "./api";
import type {
  Descriptor,
  FeatureDescriptor,
  PointsPayload,
  Scalar
} from "./types";
import { descriptorKey, descriptorName } from "./types";
import { Field, Loading, Select } from "./ui";

type ColorMode = "metadata" | "gene" | "modality";

interface ColoringControlsProps {
  metadataColumns: Descriptor[];
  modalities: Descriptor[];
  points: PointsPayload;
  featuresUrl: string;
  onColor: (title: string, values: Scalar[] | undefined) => void;
}

function availableDescriptors(
  descriptors: Descriptor[],
  values: Record<string, Scalar[]>
): Descriptor[] {
  const declared = descriptors.filter((descriptor) => descriptorKey(descriptor) in values);
  if (declared.length > 0) return declared;
  return Object.keys(values);
}

export function ColoringControls({
  metadataColumns,
  modalities,
  points,
  featuresUrl,
  onColor
}: ColoringControlsProps) {
  const metadata = useMemo(
    () => availableDescriptors(metadataColumns, points.metadata),
    [metadataColumns, points.metadata]
  );
  const modalityOptions = useMemo(
    () => availableDescriptors(modalities, points.modalities),
    [modalities, points.modalities]
  );
  const initialMode: ColorMode = metadata.length > 0
    ? "metadata"
    : modalityOptions.length > 0 ? "modality" : "gene";
  const [mode, setMode] = useState<ColorMode>(initialMode);
  const [metadataKey, setMetadataKey] = useState(descriptorKey(metadata[0] ?? ""));
  const [modalityKey, setModalityKey] = useState(descriptorKey(modalityOptions[0] ?? ""));
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);
  const [features, setFeatures] = useState<FeatureDescriptor[]>([]);
  const [searching, setSearching] = useState(false);
  const [geneError, setGeneError] = useState("");
  const [selectedFeature, setSelectedFeature] = useState<FeatureDescriptor | null>(null);
  const [geneValues, setGeneValues] = useState<Scalar[] | null>(null);

  useEffect(() => {
    if (mode === "metadata" && metadataKey) {
      const descriptor = metadata.find((item) => descriptorKey(item) === metadataKey);
      onColor(`Metadata / ${descriptor ? descriptorName(descriptor) : metadataKey}`, points.metadata[metadataKey]);
    } else if (mode === "modality" && modalityKey) {
      const descriptor = modalityOptions.find((item) => descriptorKey(item) === modalityKey);
      onColor(`Modality / ${descriptor ? descriptorName(descriptor) : modalityKey}`, points.modalities[modalityKey]);
    } else if (mode === "gene") {
      onColor(
        selectedFeature ? `Gene / ${selectedFeature.feature_symbol ?? selectedFeature.feature_id}` : "Gene expression",
        geneValues ?? undefined
      );
    }
  }, [geneValues, metadata, metadataKey, modalityKey, modalityOptions, mode, onColor, points, selectedFeature]);

  useEffect(() => {
    if (mode !== "gene") return;
    const controller = new AbortController();
    setSearching(true);
    setGeneError("");
    const timer = window.setTimeout(() => {
      searchFeatures(featuresUrl, deferredQuery.trim(), controller.signal)
        .then((result) => setFeatures(result.slice(0, 30)))
        .catch((error: unknown) => {
          if (!controller.signal.aborted) {
            setGeneError(error instanceof Error ? error.message : "Feature search failed");
          }
        })
        .finally(() => {
          if (!controller.signal.aborted) setSearching(false);
        });
    }, 180);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [deferredQuery, featuresUrl, mode]);

  const chooseFeature = async (feature: FeatureDescriptor) => {
    setSelectedFeature(feature);
    setGeneValues(null);
    setSearching(true);
    setGeneError("");
    try {
      const values = await getFeatureValues(featuresUrl, feature.feature_index);
      setGeneValues(values);
    } catch (error) {
      setGeneError(error instanceof Error ? error.message : "Could not load expression values");
    } finally {
      setSearching(false);
    }
  };

  return (
    <div className="control-section">
      <Field label="Color by">
        <Select value={mode} onChange={(event) => setMode(event.target.value as ColorMode)}>
          <option value="metadata" disabled={metadata.length === 0}>Metadata</option>
          <option value="gene">Gene expression</option>
          <option value="modality" disabled={modalityOptions.length === 0}>Modality</option>
        </Select>
      </Field>
      {mode === "metadata" && (
        <Field label="Column">
          <Select value={metadataKey} onChange={(event) => setMetadataKey(event.target.value)}>
            {metadata.map((descriptor) => (
              <option key={descriptorKey(descriptor)} value={descriptorKey(descriptor)}>
                {descriptorName(descriptor)}
              </option>
            ))}
          </Select>
        </Field>
      )}
      {mode === "modality" && (
        <Field label="Modality">
          <Select value={modalityKey} onChange={(event) => setModalityKey(event.target.value)}>
            {modalityOptions.map((descriptor) => (
              <option key={descriptorKey(descriptor)} value={descriptorKey(descriptor)}>
                {descriptorName(descriptor)}
              </option>
            ))}
          </Select>
        </Field>
      )}
      {mode === "gene" && (
        <div className="gene-control">
          <Field label="Feature search">
            <input
              className="pt-input"
              value={query}
              placeholder="Symbol or feature ID"
              onChange={(event) => setQuery(event.target.value)}
            />
          </Field>
          {selectedFeature && (
            <div className="selected-feature">
              <span>Active</span>
              <strong>{selectedFeature.feature_symbol ?? selectedFeature.feature_id}</strong>
            </div>
          )}
          {searching && <Loading label="Loading features" />}
          {geneError && <p className="inline-error">{geneError}</p>}
          {!searching && (
            <div className="feature-results">
              {features.map((feature) => (
                <button key={feature.feature_index} type="button" onClick={() => void chooseFeature(feature)}>
                  <strong>{feature.feature_symbol ?? feature.feature_id}</strong>
                  {feature.feature_symbol && <span>{feature.feature_id}</span>}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
