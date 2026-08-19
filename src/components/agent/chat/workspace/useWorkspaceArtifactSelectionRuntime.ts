import { useMemo } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  artifactsAtom,
  selectedArtifactAtom,
  selectedArtifactIdAtom,
} from "@/lib/artifact/store";
import type { Artifact } from "@/lib/artifact/types";
import type { CanvasState as GeneralCanvasState } from "@/components/general-chat/bridge";
import { hasNamedGeneralCanvasFilePreview } from "./generalCanvasPreviewState";
import {
  useWorkspaceGeneralArtifactUpsert,
  type ArtifactStoreSetter,
} from "./useWorkspaceArtifactStoreRuntime";
import { resolveDefaultSelectedArtifact } from "./agentChatWorkspaceHelpers";

interface UseWorkspaceArtifactSelectionRuntimeParams {
  activeTheme: string;
  generalCanvasState: GeneralCanvasState;
}

export function useWorkspaceArtifactSelectionRuntime({
  activeTheme,
  generalCanvasState,
}: UseWorkspaceArtifactSelectionRuntimeParams) {
  const artifacts = useAtomValue(artifactsAtom);
  const selectedArtifactId = useAtomValue(selectedArtifactIdAtom);
  const selectedArtifact = useAtomValue(selectedArtifactAtom);
  const setArtifacts = useSetAtom(artifactsAtom) as ArtifactStoreSetter<
    Artifact[]
  >;
  const setSelectedArtifactId = useSetAtom(
    selectedArtifactIdAtom,
  ) as ArtifactStoreSetter<string | null>;
  const upsertGeneralArtifact = useWorkspaceGeneralArtifactUpsert({
    setArtifacts,
  });
  const defaultSelectedArtifact = useMemo(
    () => resolveDefaultSelectedArtifact(activeTheme, artifacts),
    [activeTheme, artifacts],
  );
  const defaultSelectedArtifactId = defaultSelectedArtifact?.id ?? null;
  const preferGeneralCanvasFilePreview = useMemo(
    () =>
      activeTheme === "general" &&
      hasNamedGeneralCanvasFilePreview(generalCanvasState),
    [activeTheme, generalCanvasState],
  );
  const liveArtifact = useMemo(
    () =>
      preferGeneralCanvasFilePreview
        ? null
        : selectedArtifact || defaultSelectedArtifact,
    [defaultSelectedArtifact, preferGeneralCanvasFilePreview, selectedArtifact],
  );

  return {
    artifacts,
    defaultSelectedArtifactId,
    liveArtifact,
    preferGeneralCanvasFilePreview,
    selectedArtifact,
    selectedArtifactId,
    setArtifacts,
    setSelectedArtifactId,
    upsertGeneralArtifact,
  };
}
