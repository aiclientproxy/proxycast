import type { Artifact } from "@/lib/artifact/types";
import type { Message } from "../types";
import { mergeArtifacts } from "../utils/messageArtifacts";

export interface ResolveWorkspaceArtifactsFromMessagesParams {
  activeTheme: string;
  messages: readonly Pick<Message, "artifacts">[];
  currentArtifacts: Artifact[];
}

function shouldPreserveGeneralArtifact(artifact: Artifact): boolean {
  return (
    artifact.meta.persistOutsideMessages === true ||
    artifact.meta.previewArtifact === true
  );
}

function stringifyArtifactForCompare(artifact: Artifact): string {
  return JSON.stringify(artifact);
}

export function areWorkspaceArtifactsEqual(
  left: readonly Artifact[],
  right: readonly Artifact[],
): boolean {
  if (left === right) {
    return true;
  }
  if (left.length !== right.length) {
    return false;
  }

  return left.every((artifact, index) => {
    const other = right[index];
    return (
      other !== undefined &&
      stringifyArtifactForCompare(artifact) ===
        stringifyArtifactForCompare(other)
    );
  });
}

export function resolveWorkspaceArtifactsFromMessages({
  activeTheme,
  messages,
  currentArtifacts,
}: ResolveWorkspaceArtifactsFromMessagesParams): Artifact[] {
  if (activeTheme !== "general") {
    return [];
  }

  const messageArtifacts = mergeArtifacts(
    messages.flatMap((message) => message.artifacts || []),
  );
  const preservedArtifacts = currentArtifacts.filter(
    shouldPreserveGeneralArtifact,
  );
  if (messageArtifacts.length === 0) {
    return mergeArtifacts(preservedArtifacts);
  }

  const currentArtifactsById = new Map(
    currentArtifacts.map((artifact) => [artifact.id, artifact]),
  );
  return mergeArtifacts([
    ...messageArtifacts.map((artifact) => {
      const existing = currentArtifactsById.get(artifact.id);
      if (!existing) {
        return artifact;
      }
      const shouldReuseExistingContent =
        existing.content.length > 0 &&
        (artifact.content.length === 0 ||
          (artifact.status === "streaming" &&
            artifact.content.length < existing.content.length &&
            existing.content.startsWith(artifact.content)));
      return {
        ...existing,
        ...artifact,
        content: shouldReuseExistingContent
          ? existing.content
          : artifact.content,
        meta: { ...existing.meta, ...artifact.meta },
        createdAt: Math.min(existing.createdAt, artifact.createdAt),
        updatedAt: Math.max(existing.updatedAt, artifact.updatedAt),
      };
    }),
    ...preservedArtifacts,
  ]);
}
