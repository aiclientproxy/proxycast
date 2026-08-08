interface ArticleWorkspaceActionOutputKindIntent {
  articleWorkspace: {
    appId: string;
    sourceArtifacts?: Record<string, unknown>[];
    workerEvidence?: Array<{ artifactKind?: string | null }>;
  };
  object: {
    source?: Record<string, unknown> | null;
  };
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function readString(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value !== "string") {
      continue;
    }
    const normalized = value.trim();
    if (normalized) {
      return normalized;
    }
  }
  return "";
}

function readOutputArtifactKindFromRecord(
  record: Record<string, unknown> | null | undefined,
): string {
  if (!record) {
    return "";
  }
  return readString(
    record.outputArtifactKind,
    record.output_artifact_kind,
    record.artifactKind,
    record.artifact_kind,
  );
}

export function resolveWorkspaceArticleWorkspaceActionOutputArtifactKind(
  intent: ArticleWorkspaceActionOutputKindIntent,
): string | null {
  const objectSourceOutput = readOutputArtifactKindFromRecord(
    asRecord(intent.object.source),
  );
  if (objectSourceOutput) {
    return objectSourceOutput;
  }

  for (const artifact of intent.articleWorkspace.sourceArtifacts ?? []) {
    const output = readOutputArtifactKindFromRecord(asRecord(artifact));
    if (output) {
      return output;
    }
  }

  for (const evidence of intent.articleWorkspace.workerEvidence ?? []) {
    const output = evidence.artifactKind?.trim();
    if (output) {
      return output;
    }
  }

  return null;
}
