import type { Artifact } from "@/lib/artifact/types";

export interface WorkspaceSelectedArtifactIdCorrectionInput {
  activeTheme: string;
  artifacts: readonly Artifact[];
  selectedArtifact: Artifact | null | undefined;
  selectedArtifactId: string | null;
  defaultSelectedArtifactId: string | null;
  preferGeneralCanvasFilePreview: boolean;
}

export function resolveWorkspaceSelectedArtifactIdCorrection({
  activeTheme,
  artifacts,
  selectedArtifact,
  selectedArtifactId,
  defaultSelectedArtifactId,
  preferGeneralCanvasFilePreview,
}: WorkspaceSelectedArtifactIdCorrectionInput): string | null | undefined {
  if (activeTheme !== "general") {
    return selectedArtifactId === null ? undefined : null;
  }

  if (preferGeneralCanvasFilePreview) {
    return selectedArtifactId === null ? undefined : null;
  }

  if (artifacts.length === 0) {
    return selectedArtifactId === null ? undefined : null;
  }

  if (!selectedArtifact) {
    return selectedArtifactId === defaultSelectedArtifactId
      ? undefined
      : defaultSelectedArtifactId;
  }

  const selectedStillExists = artifacts.some(
    (artifact) => artifact.id === selectedArtifact.id,
  );
  if (
    !selectedStillExists &&
    selectedArtifactId !== defaultSelectedArtifactId
  ) {
    return defaultSelectedArtifactId;
  }

  return undefined;
}
