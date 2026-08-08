import { describe, expect, it } from "vitest";
import { buildWorkspacePluginPaneActionRequestMetadata } from "./workspacePluginPaneAction";

describe("workspacePluginPaneAction", () => {
  it("应构造自定义 pane action 可透传的受控 metadata", () => {
    const metadata = buildWorkspacePluginPaneActionRequestMetadata({
      action: {
        key: "regenerate",
        intent: "regenerate",
        risk: "write",
        taskKind: "content.image.generate",
      },
      appId: "content-factory-app",
      sessionId: "session-main",
      workspaceId: "workspace-main",
      surfaceKind: "appSurface",
      paneKind: "imageGrid",
      prompt: "重新生成配图",
      outputArtifactKind: "workspace_patch",
      sourceArtifactIds: ["artifact-1", "artifact-1", "  ", "artifact-2"],
      object: {
        kind: "imageGenerationSet",
        id: "image-set-1",
      },
    });

    expect(metadata).toEqual({
      plugin: {
        source: "right_surface_pane_action",
        app_id: "content-factory-app",
        session_id: "session-main",
        workspace_id: "workspace-main",
        pane_action: {
          key: "regenerate",
          intent: "regenerate",
          risk: "write",
          task_kind: "content.image.generate",
          output_artifact_kind: "workspace_patch",
          prompt: "重新生成配图",
          pane_kind: "imageGrid",
          surface_kind: "appSurface",
          object: {
            kind: "imageGenerationSet",
            id: "image-set-1",
          },
          source_artifact_ids: ["artifact-1", "artifact-2"],
        },
      },
      right_surface: {
        surface_kind: "appSurface",
        pane_kind: "imageGrid",
        action_key: "regenerate",
      },
    });
  });
});
