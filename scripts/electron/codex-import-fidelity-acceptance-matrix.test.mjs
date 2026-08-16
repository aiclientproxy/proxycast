import fs from "node:fs";
import { describe, expect, it } from "vitest";

const MATRIX_PATH =
  "internal/roadmap/codeximport/fidelity-acceptance-matrix.md";
const TRACKER_PATH = "internal/roadmap/codeximport/implementation-tracker.md";
const ARTIFACT_ROADMAP_PATH = "internal/roadmap/artifacts/roadmap.md";
const RUNTIME_EVENTS_TEST_PATH =
  "lime-rs/crates/app-server/src/runtime/conversation_import/tests/runtime_events.rs";
const SOURCE_SCAN_TEST_PATH =
  "lime-rs/crates/app-server/src/runtime/conversation_import/tests.rs";
const SOURCE_HEALTH_TEST_PATH =
  "lime-rs/crates/app-server/src/runtime/conversation_import/tests/health.rs";
const PATH_RESOLUTION_TEST_PATH =
  "lime-rs/crates/app-server/src/runtime/conversation_import/tests/path_resolution.rs";
const PERFORMANCE_TEST_PATH =
  "lime-rs/crates/app-server/src/runtime/conversation_import/tests/performance.rs";
const SECURITY_TEST_PATH =
  "lime-rs/crates/app-server/src/runtime/conversation_import/tests/security.rs";
const IMPORT_COMMIT_PATH =
  "lime-rs/crates/app-server/src/runtime/conversation_import/commit.rs";
const CLICK_THROUGH_SMOKE_TEST_PATH =
  "scripts/electron/codex-import-click-through-fixture-smoke.test.mjs";
const REAL_SAMPLE_SMOKE_TEST_PATH =
  "scripts/electron/local-history-import-real-sample-visual-audit-smoke.test.mjs";
const REAL_SAMPLE_SMOKE_PATH =
  "scripts/electron/local-history-import-real-sample-visual-audit-smoke.mjs";
const VISUAL_AUDIT_SMOKE_TEST_PATH =
  "scripts/electron/local-history-import-visual-audit-smoke.test.mjs";
const VISUAL_AUDIT_SMOKE_PATH =
  "scripts/electron/local-history-import-visual-audit-smoke.mjs";
const CLICK_THROUGH_SMOKE_PATH =
  "scripts/electron/codex-import-click-through-fixture-smoke.mjs";
const CLICK_THROUGH_GUI_HELPER_PATH =
  "scripts/electron/lib/local-history-import-click-through-gui.mjs";
const CLICK_THROUGH_FIXTURE_HELPER_PATH =
  "scripts/electron/lib/local-history-import-click-through-fixture.mjs";
const PREVIEW_ARTIFACT_TEST_PATH = "src/lib/artifact/previewArtifact.test.ts";
const ARTIFACT_RENDERER_TEST_PATH =
  "src/components/artifact/ArtifactRenderer.ui.test.tsx";
const ARTIFACT_TOOLBAR_TEST_PATH =
  "src/components/artifact/ArtifactToolbar.ui.test.tsx";
const WORKSPACE_PREVIEW_ACTIONS_TEST_PATH =
  "src/components/agent/chat/workspace/useWorkspaceArtifactPreviewActions.test.tsx";
const SESSION_CLIENT_PATH = "src/lib/api/agentRuntime/sessionClient.ts";
const SESSION_CLIENT_BOUNDARY_TEST_PATH =
  "src/lib/api/agentRuntime/sessionClient.current-boundary.test.ts";
const CONVERSATION_IMPORT_API_TEST_PATH =
  "src/lib/api/conversationImport.test.ts";
const CONVERSATION_IMPORT_PROGRESS_TEST_PATH =
  "src/components/app-sidebar/AppSidebarConversationImportProgress.test.tsx";
const CONVERSATION_IMPORT_DIALOG_PATH =
  "src/components/app-sidebar/AppSidebarConversationImportDialog.tsx";
const CONVERSATION_IMPORT_PROTOCOL_PATH =
  "lime-rs/crates/app-server-protocol/src/protocol/v0/conversation_import.rs";

function readFile(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function readFiles(...filePaths) {
  return filePaths.map(readFile).join("\n");
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function expectMatrixRow(matrix, category) {
  expect(matrix).toMatch(
    new RegExp(`^\\|\\s*${escapeRegExp(category)}\\s*\\|`, "m"),
  );
}

describe("codex import fidelity acceptance matrix guard", () => {
  it("documents the current fact source and excludes legacy implementation paths", () => {
    const matrix = readFile(MATRIX_PATH);

    expect(matrix).toContain("conversationImport/source/scan");
    expect(matrix).toContain("conversationImport/thread/preview");
    expect(matrix).toContain("conversationImport/thread/commit");
    expect(matrix).toContain("conversationImport/job/read");
    expect(matrix).toContain("RuntimeCore StoredSession + AgentEvent");
    expect(matrix).toContain("thread/read + thread/items/list");
    expect(matrix).toContain("Preview Artifact Contract");
    expect(matrix).toContain(
      "canonical Thread/Turn/Item read model / derived replay current 主链",
    );
    expect(matrix).toContain("Renderer 不直接扫描 `.codex`");
    expect(matrix).toContain(
      "旧 Tauri / `lime-rs/src/**` / 旧 `agent_runtime_*` 不作为新增能力落点",
    );
    expect(matrix).toContain("agentSession/read");
    expect(matrix).not.toContain("agentSession/export");
    expect(matrix).not.toContain("WebviewWindow");
  });

  it("keeps every high-fidelity source category tied to a concrete guard", () => {
    const matrix = readFile(MATRIX_PATH);
    const runtimeTests = readFiles(
      RUNTIME_EVENTS_TEST_PATH,
      SOURCE_SCAN_TEST_PATH,
      SOURCE_HEALTH_TEST_PATH,
      PATH_RESOLUTION_TEST_PATH,
      PERFORMANCE_TEST_PATH,
    );
    const smokeGuards = readFiles(
      CLICK_THROUGH_SMOKE_TEST_PATH,
      CLICK_THROUGH_SMOKE_PATH,
      CLICK_THROUGH_FIXTURE_HELPER_PATH,
      REAL_SAMPLE_SMOKE_TEST_PATH,
      REAL_SAMPLE_SMOKE_PATH,
      VISUAL_AUDIT_SMOKE_TEST_PATH,
      VISUAL_AUDIT_SMOKE_PATH,
      CONVERSATION_IMPORT_PROGRESS_TEST_PATH,
    );

    const requiredRows = [
      "source discovery",
      "read-only source health",
      "project path filtering",
      "rollout path repair",
      "message ordering",
      "attachments",
      "reasoning",
      "shell command",
      "file read preview",
      "Markdown / HTML / DOCX / XLSX / PPTX / PDF / image / binary fallback preview",
      "patch",
      "approval",
      "web search",
      "MCP / dynamic / view image / image generation",
      "plan",
      "context / review / subagent / collab",
      "incomplete lifecycle",
      "high-volume rollout",
      "background import lifecycle",
      "continue same session",
      "responsive chat / workbench",
      "evidence / replay",
      "session delete / retention boundary",
      "privacy / source leak boundary",
    ];

    for (const row of requiredRows) {
      expectMatrixRow(matrix, row);
    }

    const requiredRustEvidence = [
      "commit_preserves_codex_tool_command_and_patch_timeline",
      "commit_preserves_high_volume_codex_tool_events_in_canonical_projection",
      "starts_multi_turn_codex_history_in_background_and_reports_complete_progress",
      "scans_session_index_fallback_reports_read_only_health",
      "scans_missing_source_reports_read_only_health",
      "scans_codex_state_db_project_path_exact_prefix_and_contains",
      "commit_preserves_imported_commands_across_turns_without_projection_budget",
      "commit_preserves_imported_assistant_message_order_between_runtime_events",
      "commit_preserves_imported_update_plan_timeline_item",
      "commit_preserves_imported_completed_plan_item",
      "commit_projects_codex_runtime_specialized_items_into_existing_timeline_types",
      "commit_merges_duplicate_user_messages_when_response_item_precedes_event_msg",
      "commit_closes_incomplete_imported_lifecycles_as_failed_timeline_items",
      "mcp_tool_call_begin",
      "dynamic_tool_call_request",
      "view_image_tool_call",
      "image_generation_begin",
      "context_compacted",
      "entered_review_mode",
      "exited_review_mode",
      "subagent_activity",
      "collab_agent_spawn_begin",
    ];

    for (const token of requiredRustEvidence) {
      expect(runtimeTests).toContain(token);
    }

    const apiTests = readFile(CONVERSATION_IMPORT_API_TEST_PATH);
    const importDialog = readFile(CONVERSATION_IMPORT_DIALOG_PATH);
    expect(apiTests).toContain("sourceHomeExists");
    expect(apiTests).toContain("stateDbReadable");
    expect(apiTests).toContain("rolloutFileCount");
    expect(apiTests).toContain("waitForConversationImportJob");
    expect(apiTests).toContain('name: "AbortError"');
    expect(importDialog).toContain("readConversationImportJob");
    expect(importDialog).toContain("thread.importJobId");

    const requiredGuiEvidence = [
      "hasHistoricalReasoningVisible",
      "hasHistoricalCommandExecutionVisible",
      "hasCommandOutput",
      "hasPatchText",
      "hasSearchItem",
      "hasHistoricalApprovalText",
      "openedAllImportedPreviewArtifacts",
      "Markdown",
      "HTML",
      "DOCX",
      "sourceThreadId",
      "hidesSourceBrandText",
      "sourceMetadataUiVisible",
      "hidesRawSourceEventNames",
      "conversationImport/job/read",
      "waitForConversationImportJob",
      "app-sidebar-conversation-import-progress",
      "app-sidebar-conversation-import-close",
      "backgroundImportResume",
      "重新打开导入弹窗后未附着后台 job",
    ];

    for (const token of requiredGuiEvidence) {
      expect(smokeGuards).toContain(token);
    }
  });

  it("keeps preview artifacts as the unified opening path for imported files", () => {
    const matrix = readFile(MATRIX_PATH);
    const artifactRoadmap = readFile(ARTIFACT_ROADMAP_PATH);
    const previewArtifactTests = readFiles(
      PREVIEW_ARTIFACT_TEST_PATH,
      ARTIFACT_RENDERER_TEST_PATH,
      ARTIFACT_TOOLBAR_TEST_PATH,
      WORKSPACE_PREVIEW_ACTIONS_TEST_PATH,
    );
    const clickThroughSurface = readFiles(
      CLICK_THROUGH_SMOKE_PATH,
      CLICK_THROUGH_GUI_HELPER_PATH,
      CLICK_THROUGH_SMOKE_TEST_PATH,
    );

    expect(matrix).toContain("Preview Artifact Contract");
    expect(matrix).toContain(
      "`thread_read.tool_calls.arguments.path` + `inline-tool-open-file` + Preview Artifact Contract",
    );
    expect(matrix).toContain("URL / record / app shell source preview");
    expect(matrix).toContain("PreviewSourceSummaryRenderer");
    expect(artifactRoadmap).toContain(
      "打开链路 artifact 化，业务事实源不 artifact 化",
    );
    expect(artifactRoadmap).toContain("openArtifactInWorkbench");
    expect(artifactRoadmap).toContain("selectionKey=artifact:<id>");
    expect(artifactRoadmap).toContain("PreviewSourceSummaryRenderer");
    expect(artifactRoadmap).toContain("PDF、Excel、PPT");
    expect(artifactRoadmap).toContain("document-preview");
    expect(artifactRoadmap).toContain("renderMode=document_text");
    expect(artifactRoadmap).toContain("renderMode=system_open");
    expect(artifactRoadmap).toContain("open_with_default_app");
    expect(matrix).toContain(
      "DOCX / XLSX / PPTX 与可解析 PDF 文本流走 `document_text`",
    );
    expect(matrix).toContain("PDF OCR / 扫描件 / 复杂字体映射");
    expect(matrix).toContain("`entered_review_mode`、`exited_review_mode`");
    expect(matrix).toContain(
      "`context_compaction`、`reasoning`、`agent_message`",
    );
    expect(previewArtifactTests).toContain("research.pdf");
    expect(previewArtifactTests).toContain("budget.xlsx");
    expect(previewArtifactTests).toContain("deck.pptx");
    expect(previewArtifactTests).toContain("interview.wav");
    expect(previewArtifactTests).toContain("demo.mp4");
    expect(previewArtifactTests).toContain('source: "url"');
    expect(previewArtifactTests).toContain('source: "database_record"');
    expect(previewArtifactTests).toContain('source: "app"');
    expect(previewArtifactTests).toContain('contentKind: "app_shell"');
    expect(previewArtifactTests).toContain("preview-source-summary-renderer");
    expect(previewArtifactTests).toContain("openExternalUrlWithSystemBrowser");
    expect(previewArtifactTests).toContain('renderMode: "system_open"');
    expect(clickThroughSurface).toContain("inline-tool-open-file");
    expect(clickThroughSurface).toContain(
      "inspectImportedFilePreviewArtifacts",
    );
    expect(clickThroughSurface).toContain("openedAllImportedPreviewArtifacts");
    expect(clickThroughSurface).toContain("canvas-workbench-html-preview");
    expect(clickThroughSurface).toContain("word/document.xml");
    expect(clickThroughSurface).toContain("ZIP/OpenXML 噪音");
  });

  it("keeps continuation and evidence on the current App Server runtime chain", () => {
    const matrix = readFile(MATRIX_PATH);
    const tracker = readFile(TRACKER_PATH);
    const importCommit = readFile(IMPORT_COMMIT_PATH);
    const clickThroughSurface = readFiles(
      CLICK_THROUGH_SMOKE_TEST_PATH,
      CLICK_THROUGH_SMOKE_PATH,
    );

    expect(matrix).toContain("`turn/start` current 主链");
    expect(matrix).toContain(
      "`agentSession/read` / `thread/read` / `thread/items/list` 与 derived replay 使用 Lime canonical facts",
    );
    expect(tracker).toContain(
      "导入来源 runtime 的 `provider_name/model_name` 只表示来源上下文",
    );
    expect(tracker).toContain(
      "续聊 submit op 必须继续提交用户当前选择的 provider/model",
    );
    expect(importCommit).toContain("source provenance");
    expect(clickThroughSurface).toContain('"turn/start"');
    expect(clickThroughSurface).toContain("backendMetadataImported");
    expect(clickThroughSurface).toContain("hasContinueUserMessage");
    expect(clickThroughSurface).toContain("hasContinueAssistantMessage");
    expect(clickThroughSurface).toContain('not.toContain("agent_runtime_")');
  });

  it("keeps session delete separate from archive, reimport cleanup, and source retention", () => {
    const matrix = readFile(MATRIX_PATH);
    const sessionClient = readFile(SESSION_CLIENT_PATH);
    const sessionClientBoundary = readFile(SESSION_CLIENT_BOUNDARY_TEST_PATH);

    expect(matrix).toContain("session delete / retention boundary");
    expect(matrix).toContain(
      "`thread/delete` 清理 Lime memory / projection / event log / session-scoped sidecar",
    );
    expect(matrix).toContain("`thread/archive` / `thread/unarchive`");
    expect(matrix).toContain("不删除外部来源目录");
    expect(matrix).toContain(
      "导出后删除、删除前导出确认和保留期限策略还需要独立产品规则",
    );
    expect(sessionClient).toContain("deleteAgentRuntimeSession");
    expect(sessionClient).toContain(
      "appServerSessionClient.deleteAgentRuntimeSession(sessionId)",
    );
    expect(sessionClientBoundary).toContain(
      "archive projection must use thread/archive instead of session update",
    );
    expect(sessionClientBoundary).toContain(
      "unarchive projection must use thread/unarchive",
    );
    expect(sessionClientBoundary).toContain(
      "delete projection must resolve canonical id and use thread/delete",
    );
  });

  it("guards privacy and keeps the importer Codex-only", () => {
    const matrix = readFile(MATRIX_PATH);
    const protocol = readFile(CONVERSATION_IMPORT_PROTOCOL_PATH);
    const securityTests = readFile(SECURITY_TEST_PATH);
    const realSampleGuard = readFile(REAL_SAMPLE_SMOKE_PATH);
    const visualAuditGuard = readFile(VISUAL_AUDIT_SMOKE_PATH);
    const visualAuditGuardTest = readFile(VISUAL_AUDIT_SMOKE_TEST_PATH);

    expect(matrix).toContain("敏感文件路径");
    expect(matrix).not.toContain("non-Codex importer");
    expect(matrix).not.toContain("compat / deferred");
    expect(protocol).toContain("pub enum ConversationImportSourceClient");
    expect(protocol).toContain("Codex,");
    expect(protocol).not.toContain("ClaudeCode");
    expect(matrix).toContain(
      "未来 Codex 新事件类型默认先进入 unsupported / provenance-only",
    );
    expect(securityTests).toContain(
      "scan_rejects_sensitive_rollout_path_from_state_db",
    );
    expect(securityTests).toContain("preview_rejects_sensitive_source_path");
    expect(securityTests).toContain(
      "commit_rejects_source_path_outside_source_root",
    );
    expect(realSampleGuard).toContain("!audit.sourceMetadataUiVisible");
    expect(visualAuditGuard).toContain("collectVisibleTextLeaks");
    expect(visualAuditGuard).toContain("SOURCE_BRAND_PATTERN");
    expect(visualAuditGuard).toContain("GUI 可见文本仍泄漏来源品牌");
    expect(visualAuditGuardTest).toContain("collectVisibleTextLeaks");
  });
});
