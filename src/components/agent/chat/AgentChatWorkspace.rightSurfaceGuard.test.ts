import { readFileSync } from "node:fs";
import { join } from "node:path";
import process from "node:process";
import { describe, expect, it } from "vitest";

describe("AgentChatWorkspace right surface state boundary", () => {
  it("右侧 surface 本地状态必须由 current hook owner 提供，并先于 artifact 打开 runtime 捕获", () => {
    const workspaceSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/useAgentChatWorkspaceRuntime.tsx",
      ),
      "utf8",
    );
    const setupRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSetupRuntime.ts",
      ),
      "utf8",
    );
    const commandRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceCommandRuntime.ts",
      ),
      "utf8",
    );
    const ownerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceLocalStateRuntime.ts",
      ),
      "utf8",
    );
    const artifactOpenRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceArtifactOpenRuntime.tsx",
      ),
      "utf8",
    );
    const contextSurfaceSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceContextSurfaceRuntime.ts",
      ),
      "utf8",
    );
    const artifactActionRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceArtifactActionRuntime.ts",
      ),
      "utf8",
    );
    const artifactInteractionRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceArtifactInteractionRuntime.ts",
      ),
      "utf8",
    );

    expect(setupRuntimeSource).toContain("useWorkspaceContextSurfaceRuntime({");
    expect(commandRuntimeSource).toContain(
      "useAgentChatWorkspaceArtifactInteractionRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceArtifactActionRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfaceLocalStateRuntime()",
    );
    expect(workspaceSource).not.toContain("useWorkspaceArtifactOpenRuntime({");
    expect(contextSurfaceSource).toContain(
      "useWorkspaceRightSurfaceLocalStateRuntime()",
    );
    expect(artifactInteractionRuntimeSource).toContain(
      "useWorkspaceArtifactActionRuntime(action)",
    );
    expect(artifactActionRuntimeSource).toContain(
      "useWorkspaceArtifactOpenRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfaceArtifactOpenRuntime({",
    );
    expect(artifactOpenRuntimeSource).toContain(
      "useWorkspaceRightSurfaceArtifactOpenRuntime({",
    );
    expect(workspaceSource).not.toContain("const [manualRightSurface");
    expect(workspaceSource).not.toContain(
      "const [activeFilesRightSurfaceTarget",
    );
    expect(workspaceSource).not.toContain("const [activeArticleWorkspace");
    expect(ownerSource).toContain("const [manualRightSurface");
    expect(ownerSource).toContain("const [activeFilesRightSurfaceTarget");
    expect(ownerSource).toContain("const [activeArticleWorkspace");
    expect(ownerSource).toContain("openArticleWorkspaceRightSurface");
  });

  it("Artifact click 的 Article right surface 打开逻辑必须由 artifact open runtime 提供", () => {
    const workspaceSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/useAgentChatWorkspaceRuntime.tsx",
      ),
      "utf8",
    );
    const commandRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceCommandRuntime.ts",
      ),
      "utf8",
    );
    const artifactOpenRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceArtifactOpenRuntime.tsx",
      ),
      "utf8",
    );
    const ownerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceArtifactOpenRuntime.ts",
      ),
      "utf8",
    );

    const artifactActionRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceArtifactActionRuntime.ts",
      ),
      "utf8",
    );
    const artifactInteractionRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceArtifactInteractionRuntime.ts",
      ),
      "utf8",
    );

    expect(commandRuntimeSource).toContain(
      "useAgentChatWorkspaceArtifactInteractionRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceArtifactActionRuntime({",
    );
    expect(workspaceSource).not.toContain("useWorkspaceArtifactOpenRuntime({");
    expect(artifactInteractionRuntimeSource).toContain(
      "useWorkspaceArtifactActionRuntime(action)",
    );
    expect(artifactActionRuntimeSource).toContain(
      "useWorkspaceArtifactOpenRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfaceArtifactOpenRuntime({",
    );
    expect(artifactOpenRuntimeSource).toContain(
      "useWorkspaceRightSurfaceArtifactOpenRuntime({",
    );
    expect(artifactOpenRuntimeSource).toContain(
      "bindArticleEditorRightSurface",
    );
    expect(artifactOpenRuntimeSource).toContain(
      "bindRightSurfacePendingActions",
    );
    expect(workspaceSource).not.toContain(
      "buildArticleWorkspaceForArtifactOpen(",
    );
    expect(workspaceSource).not.toContain("articleEditorRightSurfaceRef");
    expect(workspaceSource).not.toContain("rightSurfacePendingActionsRef");
    expect(ownerSource).toContain("buildArticleWorkspaceForArtifactOpen(");
    expect(ownerSource).toContain("bindArticleEditorRightSurface");
    expect(ownerSource).toContain("bindRightSurfacePendingActions");
    expect(ownerSource).toContain(
      'consumePendingRequestsForSurface?.(\n          "articleWorkspace"',
    );
    expect(ownerSource).toContain(
      'consumePendingRequestsForSurface?.(\n          "objectCanvas"',
    );
  });

  it("右侧 surface pending App Server runtime 必须由 pending bridge runtime 绑定", () => {
    const workspaceSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/useAgentChatWorkspaceRuntime.tsx",
      ),
      "utf8",
    );
    const ownerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfacePendingBridgeRuntime.ts",
      ),
      "utf8",
    );
    const coordinatorSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceCoordinatorRuntime.ts",
      ),
      "utf8",
    );

    expect(coordinatorSource).toContain(
      "useWorkspaceRightSurfacePendingBridgeRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfacePendingBridgeRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfacePendingRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "shouldAutoRefreshWorkspaceRightSurfacePending({",
    );
    expect(workspaceSource).not.toContain("bindRightSurfacePendingActions({");
    expect(ownerSource).toContain("useWorkspaceRightSurfacePendingRuntime({");
    expect(ownerSource).toContain(
      "shouldAutoRefreshWorkspaceRightSurfacePending({",
    );
    expect(ownerSource).toContain("bindRightSurfacePendingActions({");
  });

  it("右侧 surface 动作编排必须由 action runtime 提供，不回流父级 Workspace", () => {
    const workspaceSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/useAgentChatWorkspaceRuntime.tsx",
      ),
      "utf8",
    );
    const sceneRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneRuntime.tsx",
      ),
      "utf8",
    );
    const ownerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceActionRuntime.ts",
      ),
      "utf8",
    );
    const coordinatorSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceCoordinatorRuntime.ts",
      ),
      "utf8",
    );
    const compositionSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceCompositionRuntime.ts",
      ),
      "utf8",
    );
    const sceneCompositionSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneComposition.tsx",
      ),
      "utf8",
    );

    expect(sceneRuntimeSource).toContain(
      "useAgentChatWorkspaceSceneComposition({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfaceCompositionRuntime({",
    );
    expect(sceneCompositionSource).toContain(
      "useWorkspaceRightSurfaceCompositionRuntime({",
    );
    expect(compositionSource).toContain(
      "useWorkspaceRightSurfaceCoordinatorRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfaceActionRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfaceDerivedRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfaceProjectionRuntime({",
    );
    expect(coordinatorSource).toContain(
      "useWorkspaceRightSurfaceActionRuntime({",
    );
    expect(coordinatorSource).toContain(
      "useWorkspaceRightSurfaceDerivedRuntime({",
    );
    expect(coordinatorSource).toContain(
      "useWorkspaceRightSurfaceProjectionRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "const handleToggleExpertInfoPanel = useCallback",
    );
    expect(ownerSource).toContain("const handleToggleExpertInfoPanel");
    expect(ownerSource).toContain(
      'consumePendingRequestsForSurface("expertInfo")',
    );
    expect(ownerSource).toContain(
      'dismissPendingRequestsForSurface(\n        "expertInfo"',
    );
  });

  it("Expert Panel 可见性和 props 组装必须由 right surface expert runtime 提供", () => {
    const workspaceSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/useAgentChatWorkspaceRuntime.tsx",
      ),
      "utf8",
    );
    const ownerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceExpertPanelRuntime.ts",
      ),
      "utf8",
    );
    const sceneCompositionSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneComposition.tsx",
      ),
      "utf8",
    );
    const compositionCallStart = sceneCompositionSource.indexOf(
      "useWorkspaceRightSurfaceCompositionRuntime({",
    );
    const compositionCallEnd = sceneCompositionSource.indexOf(
      "const { homeRecoverySession",
      compositionCallStart,
    );
    expect(compositionCallStart).toBeGreaterThanOrEqual(0);
    expect(compositionCallEnd).toBeGreaterThan(compositionCallStart);
    const compositionCallSource = sceneCompositionSource.slice(
      compositionCallStart,
      compositionCallEnd,
    );

    expect(workspaceSource).not.toContain(
      "resolveExpertInfoPanelCollapsedAfterLayoutChange({",
    );
    expect(workspaceSource).not.toContain(
      "previousExpertInfoPanelLayoutModeRef",
    );
    expect(compositionCallSource).toContain("expertInfoPanelProps,");
    expect(compositionCallSource).not.toContain(
      "requestMetadata: expertPanelRequestMetadata",
    );
    expect(ownerSource).toContain(
      "resolveExpertInfoPanelCollapsedAfterLayoutChange({",
    );
    expect(ownerSource).toContain("previousExpertInfoPanelLayoutModeRef");
    expect(ownerSource).toContain("const expertInfoPanelProps = useMemo");
    expect(ownerSource).toContain("skillRefsEdited:");
    expect(ownerSource).toContain("expertInfoPanelVisible");
  });

  it("右侧 surface host 渲染和 Article action 写回必须由 host runtime 提供", () => {
    const workspaceSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/useAgentChatWorkspaceRuntime.tsx",
      ),
      "utf8",
    );
    const ownerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceHostRuntime.ts",
      ),
      "utf8",
    );
    const compositionSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceCompositionRuntime.ts",
      ),
      "utf8",
    );
    const hostCallStart = compositionSource.indexOf(
      "useWorkspaceRightSurfaceHostRuntime({",
    );
    const hostCallEnd = compositionSource.indexOf(
      "buildWorkspaceConversationRightSurfaceChrome({",
      hostCallStart,
    );
    expect(hostCallStart).toBeGreaterThanOrEqual(0);
    expect(hostCallEnd).toBeGreaterThan(hostCallStart);
    const hostCallSource = compositionSource.slice(hostCallStart, hostCallEnd);

    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfaceHostRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "renderWorkspaceRightSurfaceHostRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "submitWorkspaceArticleEditorActionIntent",
    );
    expect(workspaceSource).not.toContain(
      "buildWorkspaceArticleWorkspaceSelectionUpdateRequest",
    );
    expect(ownerSource).toContain("renderWorkspaceRightSurfaceHostRuntime({");
    expect(ownerSource).toContain("submitWorkspaceArticleEditorActionIntent");
    expect(ownerSource).not.toContain(
      "buildWorkspaceArticleWorkspaceSelectionUpdateRequest",
    );
    expect(ownerSource).not.toContain("updateAgentRuntimeSession");
    expect(hostCallSource).toContain("rightSurfaceRuntime,");
    expect(hostCallSource).not.toContain("activePluginSurfaceContainerId,");
    expect(hostCallSource).not.toContain("browserRightSurfaceAvailable,");
    expect(hostCallSource).not.toContain("filesRightSurfaceAvailable,");
    expect(hostCallSource).not.toContain("objectCanvasRightSurfaceAvailable,");
    expect(hostCallSource).not.toContain("pluginSurfaceRightSurface,");
    expect(ownerSource).toContain("rightSurfaceRuntime:");
  });

  it("Article Editor 图片槽位命令组装必须由 current image slot runtime 提供", () => {
    const workspaceSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/useAgentChatWorkspaceRuntime.tsx",
      ),
      "utf8",
    );
    const sceneRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneRuntime.tsx",
      ),
      "utf8",
    );
    const ownerSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceArticleEditorImageSlotRuntime.ts",
      ),
      "utf8",
    );
    const compositionSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceCompositionRuntime.ts",
      ),
      "utf8",
    );
    const sceneCompositionSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneComposition.tsx",
      ),
      "utf8",
    );

    expect(sceneRuntimeSource).toContain(
      "useAgentChatWorkspaceSceneComposition({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceRightSurfaceCompositionRuntime({",
    );
    expect(sceneCompositionSource).toContain(
      "useWorkspaceRightSurfaceCompositionRuntime({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceArticleEditorImageSlotRuntime({",
    );
    expect(compositionSource).toContain(
      "useWorkspaceArticleEditorImageSlotRuntime(",
    );
    expect(workspaceSource).not.toContain(
      "buildWorkspaceArticleEditorImageSlotCommand",
    );
    expect(workspaceSource).not.toContain(
      "WorkspaceArticleWorkspaceImageSlotIntent",
    );
    expect(ownerSource).toContain(
      "buildWorkspaceArticleEditorImageSlotCommand",
    );
    expect(ownerSource).toContain("handleArticleWorkspaceImageSlotIntent");
  });
});

describe("AgentChatWorkspace conversation right surface boundary", () => {
  it("右侧 surface / Harness / Expert chrome 必须通过 scene runtime 窄契约传递", () => {
    const workspaceSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/useAgentChatWorkspaceRuntime.tsx",
      ),
      "utf8",
    );
    const sceneRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneRuntime.tsx",
      ),
      "utf8",
    );
    const conversationRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceConversationSceneRuntime.tsx",
      ),
      "utf8",
    );
    const chromeRuntimeSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/workspaceConversationRightSurfaceChrome.ts",
      ),
      "utf8",
    );
    const compositionSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceRightSurfaceCompositionRuntime.ts",
      ),
      "utf8",
    );
    const conversationCompositionSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useWorkspaceConversationCompositionRuntime.ts",
      ),
      "utf8",
    );
    const sceneCompositionSource = readFileSync(
      join(
        process.cwd(),
        "src/components/agent/chat/workspace/useAgentChatWorkspaceSceneComposition.tsx",
      ),
      "utf8",
    );
    const compositionCallStart = sceneCompositionSource.indexOf(
      "useWorkspaceConversationCompositionRuntime({",
    );
    const sceneCallStart = sceneCompositionSource.indexOf(
      "scene: {",
      compositionCallStart,
    );
    const sceneCallEnd = sceneCompositionSource.indexOf(
      "\n    },\n  });",
      sceneCallStart,
    );
    expect(compositionCallStart).toBeGreaterThanOrEqual(0);
    expect(sceneCallStart).toBeGreaterThan(compositionCallStart);
    expect(sceneCallEnd).toBeGreaterThan(sceneCallStart);
    const sceneCallSource = sceneCompositionSource.slice(
      sceneCallStart,
      sceneCallEnd,
    );

    expect(sceneRuntimeSource).toContain(
      "useAgentChatWorkspaceSceneComposition({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceConversationCompositionRuntime({",
    );
    expect(compositionSource).toContain(
      "buildWorkspaceConversationRightSurfaceChrome({",
    );
    expect(workspaceSource).not.toContain(
      "buildWorkspaceConversationRightSurfaceChrome({",
    );
    expect(workspaceSource).not.toContain(
      "useWorkspaceConversationSceneRuntime(",
    );
    expect(conversationCompositionSource).toContain(
      "useWorkspaceConversationSceneRuntime(",
    );
    expect(sceneCallSource).toContain("rightSurfaceChrome,");
    expect(sceneCallSource).not.toContain("rightSurfaceChrome: {");
    expect(workspaceSource).not.toContain(
      'rightSurfaceState.activeSurface === "objectCanvas"',
    );
    expect(workspaceSource).not.toContain(
      'rightSurfaceState.activeSurface === "browser"',
    );
    expect(workspaceSource).not.toContain(
      'rightSurfaceState.activeSurface === "files"',
    );
    expect(workspaceSource).not.toContain(
      'rightSurfaceState.activeSurface === "trace"',
    );
    expect(workspaceSource).not.toContain(
      'rightSurfaceState.activeSurface === "shell"',
    );
    for (const retiredSceneParam of [
      "rightSurfaceObjectCanvasOpen:",
      "onToggleRightSurfaceObjectCanvas:",
      "rightSurfaceBrowserOpen:",
      "onToggleRightSurfaceBrowser:",
      "rightSurfaceFilesOpen:",
      "onToggleRightSurfaceFiles:",
      "rightSurfaceTraceOpen:",
      "onToggleRightSurfaceTrace:",
      "rightSurfaceShellOpen:",
      "onToggleRightSurfaceShell:",
      "navbarHarnessPanelVisible:",
      "handleToggleHarnessPanel:",
    ]) {
      expect(sceneCallSource).not.toContain(retiredSceneParam);
    }
    for (const retiredTopLevelSceneParam of [
      "showHarnessToggle:",
      "expertInfoPanelVisible,",
      "handleToggleExpertInfoPanel,",
      "harnessPendingCount:",
      "harnessAttentionLevel:",
      "harnessToggleLabel:",
    ]) {
      expect(sceneCallSource).not.toMatch(
        new RegExp(`\\n {4}${retiredTopLevelSceneParam}`),
      );
    }
    expect(conversationRuntimeSource).toContain(
      "rightSurfaceChrome: WorkspaceConversationRightSurfaceChromeRuntime",
    );
    expect(conversationRuntimeSource).toContain(
      "buildWorkspaceConversationRightSurfaceSceneProps({",
    );
    expect(chromeRuntimeSource).toContain(
      "buildWorkspaceConversationRightSurfaceChrome({",
    );
    expect(chromeRuntimeSource).toContain(
      "buildWorkspaceConversationRightSurfaceSceneProps({",
    );
    for (const retiredRuntimeParam of [
      "navbarHarnessPanelVisible",
      "handleToggleHarnessPanel",
    ]) {
      expect(conversationRuntimeSource).not.toContain(retiredRuntimeParam);
    }
  });
});
