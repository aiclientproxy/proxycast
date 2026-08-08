export type WorkspacePluginPaneActionRisk = "read" | "write";

export interface WorkspacePluginPaneActionIntent {
  action: {
    key: string;
    intent: string;
    risk: WorkspacePluginPaneActionRisk;
    taskKind?: string | null;
  };
  appId: string;
  object?: Record<string, unknown> | null;
  paneKind: string;
  prompt: string;
  sessionId: string;
  outputArtifactKind?: string | null;
  source?: string;
  sourceArtifactIds?: string[];
  surfaceKind: string;
  workspaceId?: string | null;
}

function normalizedString(value: string | null | undefined): string | null {
  const normalized = value?.trim();
  return normalized ? normalized : null;
}

function normalizedStringList(values: readonly string[] | undefined): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values ?? []) {
    const normalized = normalizedString(value);
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    result.push(normalized);
  }
  return result;
}

export function buildWorkspacePluginPaneActionRequestMetadata(
  intent: WorkspacePluginPaneActionIntent,
): Record<string, unknown> {
  const source = normalizedString(intent.source) ?? "right_surface_pane_action";
  const actionKey = normalizedString(intent.action.key);
  const paneKind = normalizedString(intent.paneKind);
  const surfaceKind = normalizedString(intent.surfaceKind);
  const outputArtifactKind = normalizedString(intent.outputArtifactKind);

  return {
    plugin: {
      source,
      app_id: intent.appId,
      session_id: intent.sessionId,
      workspace_id: intent.workspaceId ?? null,
      pane_action: {
        key: actionKey,
        intent: normalizedString(intent.action.intent),
        risk: intent.action.risk,
        task_kind: normalizedString(intent.action.taskKind),
        output_artifact_kind: outputArtifactKind,
        prompt: intent.prompt,
        pane_kind: paneKind,
        surface_kind: surfaceKind,
        object: intent.object ?? null,
        source_artifact_ids: normalizedStringList(intent.sourceArtifactIds),
      },
    },
    right_surface: {
      surface_kind: surfaceKind,
      pane_kind: paneKind,
      action_key: actionKey,
    },
  };
}
