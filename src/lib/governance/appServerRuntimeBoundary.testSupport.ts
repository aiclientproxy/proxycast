/* global process */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

export const REPO_ROOT = process.cwd();
export const APP_SERVER_SRC_DIR = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src",
);
export const LOCAL_DATA_SOURCE_SKILLS_DIR = join(
  REPO_ROOT,
  "lime-rs/crates/app-server/src/local_data_source/skills",
);
export const RUNTIME_BACKEND_REQUEST_CONTEXT_MAIN =
  "lime-rs/crates/app-server/src/runtime_backend/request_context.rs";
export const RUNTIME_BACKEND_REQUEST_CONTEXT_SPLIT_MODULES = [
  "lime-rs/crates/app-server/src/runtime_backend/request_context/session_config.rs",
  "lime-rs/crates/app-server/src/runtime_backend/request_context/turn_context.rs",
  "lime-rs/crates/app-server/src/runtime_backend/request_context/workspace_scope.rs",
];
export const RUNTIME_BACKEND_TESTS_MAIN =
  "lime-rs/crates/app-server/src/runtime_backend/tests.rs";
export const RUNTIME_BACKEND_TEST_SPLIT_MODULES = [
  "lime-rs/crates/app-server/src/runtime_backend/tests/coding_event_projection.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/image_tools.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/model_selection.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/session_prompt_context.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/session_skill_context.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/session_soul_context.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/tool_inventory.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/tool_policy_context.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/tool_surface.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/turn_flows.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tests/workspace_scope_context.rs",
];
export const IMAGE_COMMAND_MAIN =
  "lime-rs/crates/app-server/src/runtime_backend/image_command/mod.rs";
export const IMAGE_COMMAND_SPLIT_MODULES = [
  "lime-rs/crates/app-server/src/runtime_backend/image_command/events.rs",
  "lime-rs/crates/app-server/src/runtime_backend/image_command/intent.rs",
  "lime-rs/crates/app-server/src/runtime_backend/image_command/presentation.rs",
  "lime-rs/crates/app-server/src/runtime_backend/image_command/tests.rs",
];
export const RUNTIME_CORE_MAIN = "lime-rs/crates/app-server/src/runtime.rs";
export const RUNTIME_CORE_OWNER_MODULES = [
  "lime-rs/crates/app-server/src/runtime/error.rs",
  "lime-rs/crates/app-server/src/runtime/context_compaction.rs",
  "lime-rs/crates/app-server/src/runtime/execution_request.rs",
  "lime-rs/crates/app-server/src/runtime/model_providers.rs",
  "lime-rs/crates/app-server/src/runtime/read_model.rs",
  "lime-rs/crates/app-server/src/runtime/session_control.rs",
  "lime-rs/crates/app-server/src/runtime/session_lifecycle.rs",
  "lime-rs/crates/app-server/src/runtime/thread_item_projection.rs",
  "lime-rs/crates/app-server/src/runtime/tool_item_projection.rs",
  "lime-rs/crates/app-server/src/runtime/turn_execution.rs",
  "lime-rs/crates/app-server/src/runtime/value_fields.rs",
];
export const RUNTIME_READ_MODEL_MAIN =
  "lime-rs/crates/app-server/src/runtime/read_model.rs";
export const RUNTIME_READ_MODEL_OWNER_MODULES = [
  "lime-rs/crates/app-server/src/runtime/artifact_projection.rs",
  "lime-rs/crates/app-server/src/runtime/coding_activity_projection.rs",
  "lime-rs/crates/app-server/src/runtime/file_checkpoint_projection.rs",
  "lime-rs/crates/app-server/src/runtime/read_model/messages.rs",
  "lime-rs/crates/app-server/src/runtime/read_model/model_routing.rs",
  "lime-rs/crates/app-server/src/runtime/read_model/queued_turns.rs",
  "lime-rs/crates/app-server/src/runtime/read_model/runtime_items.rs",
  "lime-rs/crates/app-server/src/runtime/read_model/session_metadata.rs",
  "lime-rs/crates/app-server/src/runtime/read_model/tests.rs",
  "lime-rs/crates/app-server/src/runtime/read_model_turn_usage.rs",
  "lime-rs/crates/app-server/src/runtime/tool_item_projection.rs",
  "lime-rs/crates/app-server/src/runtime/workflow/read_model.rs",
];
export const RUNTIME_THREAD_ITEM_PROJECTION_MAIN =
  "lime-rs/crates/app-server/src/runtime/thread_item_projection.rs";
export const RUNTIME_THREAD_ITEM_PROJECTION_OWNER_MODULES = [
  "lime-rs/crates/app-server/src/runtime/thread_item_projection/agent_message.rs",
  "lime-rs/crates/app-server/src/runtime/thread_item_projection/plan.rs",
];
export const RUNTIME_BACKEND_MAIN =
  "lime-rs/crates/app-server/src/runtime_backend.rs";
export const RUNTIME_BACKEND_OWNER_MODULES = [
  "lime-rs/crates/app-server/src/runtime_backend/execution_backend.rs",
  "lime-rs/crates/app-server/src/runtime_backend/model_capability.rs",
  "lime-rs/crates/app-server/src/runtime_backend/model_routing.rs",
  "lime-rs/crates/app-server/src/runtime_backend/provider_config.rs",
  "lime-rs/crates/app-server/src/runtime_backend/request_context.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tool_inventory.rs",
];
export const AGENT_SESSION_EXECUTION_RUNTIME_MAIN =
  "lime-rs/crates/agent/src/session_execution_runtime.rs";
export const AGENT_SESSION_EXECUTION_RUNTIME_OWNER_MODULES = [
  "lime-rs/crates/agent/src/agent_tools/execution.rs",
  "lime-rs/crates/agent/src/agent_tools/tool_inventory_runtime_snapshot.rs",
  "lime-rs/crates/agent/src/provider_configuration.rs",
  "lime-rs/crates/agent/src/turn_context_configuration.rs",
  "lime-rs/crates/agent/src/turn_input_envelope.rs",
];
export const PROCESSOR_MAIN = "lime-rs/crates/app-server/src/processor/mod.rs";
export const PROCESSOR_DISPATCH =
  "lime-rs/crates/app-server/src/processor/dispatch.rs";
export const PROCESSOR_TESTS_MAIN =
  "lime-rs/crates/app-server/src/processor/tests.rs";
export const PROCESSOR_SPLIT_MODULES = [
  "lime-rs/crates/app-server/src/processor/dispatch.rs",
  "lime-rs/crates/app-server/src/processor/model.rs",
  "lime-rs/crates/app-server/src/processor/thread_resume_context.rs",
  "lime-rs/crates/app-server/src/processor/tests/artifact.rs",
  "lime-rs/crates/app-server/src/processor/tests/capability.rs",
  "lime-rs/crates/app-server/src/processor/tests/mcp.rs",
  "lime-rs/crates/app-server/src/processor/tests/project_git.rs",
  "lime-rs/crates/app-server/src/processor/tests/right_surface.rs",
  "lime-rs/crates/app-server/src/processor/tests/usage_stats.rs",
];
export const AGENT_PROVIDER_CONFIGURATION_BOUNDARY =
  "lime-rs/crates/agent/src/provider_configuration.rs";
export const AGENT_PROVIDER_RUNTIME_ADAPTER_BOUNDARY =
  "lime-rs/crates/agent/src/credential_bridge/runtime_provider_adapter.rs";
export const AGENT_SESSION_CONFIGURATION_BOUNDARY =
  "lime-rs/crates/agent/src/session_configuration.rs";
export const AGENT_TURN_CONTEXT_CONFIGURATION_BOUNDARY =
  "lime-rs/crates/agent/src/turn_context_configuration.rs";
export const RUNTIME_BOUNDARY_ROADMAP =
  "internal/roadmap/appserver/app-server-agent-runtime-boundary-governance.md";
export const EXTERNAL_BACKEND_SCAN_DIRS = [
  "electron",
  "packages/app-server-client",
  "scripts",
  "lime-rs/crates/app-server/src",
  "lime-rs/crates/app-server-daemon/src",
];

export const ALLOWED_AGENT_COUPLING_OWNER_FILES = new Set([
  "lime-rs/crates/app-server/src/agent_runtime_registry.rs",
  "lime-rs/crates/app-server/src/runtime_backend.rs",
  "lime-rs/crates/app-server/src/runtime_backend/action_response.rs",
  "lime-rs/crates/app-server/src/runtime_backend/image_tools.rs",
  "lime-rs/crates/app-server/src/runtime_backend/live_execution_process.rs",
  "lime-rs/crates/app-server/src/runtime_backend/mcp_bridges.rs",
  "lime-rs/crates/app-server/src/runtime_backend/memory_tools.rs",
  "lime-rs/crates/app-server/src/runtime_backend/native_tools.rs",
  "lime-rs/crates/app-server/src/runtime_backend/plugin_worker_generation.rs",
  "lime-rs/crates/app-server/src/runtime_backend/provider_config.rs",
  "lime-rs/crates/app-server/src/runtime_backend/tool_inventory.rs",
]);

export const KNOWN_OUT_OF_BOUND_AGENT_COUPLING_FILES = new Set<string>();
export const KNOWN_OUT_OF_BOUND_AGENT_EXECUTION_FILES = new Set<string>();

export const AGENT_COUPLING_SNIPPETS = ["use agent::"];

export const AGENT_EXECUTION_SNIPPETS = [
  "stream_current_provider_turn(",
  "run_agent_turn_with_policy(",
  "configure_model_route_provider_for_session_with_provider(",
];

export const AGENT_PROVIDER_CONFIGURATION_SNIPPETS = [
  ".configure_provider(",
  "configure_provider_from_pool(",
  "provider_config_from_pool(",
  "provider_config_with_route_protocol(",
  "RuntimeProviderProtocol",
  "RuntimeProviderProtocol",
  "runtime_provider_protocol_from_route",
  "runtime_provider_protocol_from_route",
  "route_protocol_from_agent_protocol",
  "route_protocol_from_runtime_protocol",
];

export const AGENT_SKILL_EXECUTION_SNIPPETS = [
  "execute_skill_prompt(",
  "execute_skill_workflow(",
  "SkillPromptExecution",
  "SkillWorkflowExecution",
];

export const EXTERNAL_BACKEND_LAUNCH_SNIPPETS = [
  'APP_SERVER_BACKEND_MODE: "external"',
  "APP_SERVER_BACKEND_MODE: 'external'",
  'backendMode: "external"',
  "backendMode: 'external'",
  '"--backend", "external"',
  "'--backend', 'external'",
  "--backend external",
];

export const ALLOWED_EXTERNAL_BACKEND_LAUNCH_FILES = new Set([
  "lime-rs/crates/app-server/src/main.rs",
  "lime-rs/crates/app-server-daemon/src/lib.rs",
  "packages/app-server-client/tests/client.test.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-smoke.mjs",
  "scripts/agent-runtime/claw-chat-current-fixture-smoke.test.mjs",
  "scripts/agent-runtime/claw-image-live-smoke.test.mjs",
  "scripts/agent-runtime/reopen-running-turn-cdp-gate.mjs",
  "scripts/agent-runtime/reopen-running-turn-cdp-gate.test.mjs",
  "scripts/agent-runtime/tool-execution-smoke.test.mjs",
  "scripts/app-server/external-backend-smoke.mjs",
  "scripts/app-server/packaged-external-backend-failure-smoke.mjs",
  "scripts/check-app-server-client-contract.mjs",
  "scripts/check-command-contracts.mjs",
  "scripts/electron/codex-import-click-through-fixture-smoke.mjs",
  "scripts/electron/codex-import-click-through-fixture-smoke.test.mjs",
  "scripts/electron/codex-import-continuation-fixture-smoke.mjs",
  "scripts/electron/codex-import-continuation-fixture-smoke.test.mjs",
  "scripts/electron/code-artifact-workbench-fixture-smoke.mjs",
  "scripts/electron/code-artifact-workbench-fixture-smoke.test.mjs",
  "scripts/electron/local-history-import-real-sample-visual-audit-smoke.test.mjs",
  "scripts/electron/session-history-fixture-smoke.test.mjs",
  "scripts/electron/strict-review-gate-b.mjs",
  "scripts/electron/strict-review-gate-b.test.mjs",
  "scripts/electron/thread-revert-gate-b.mjs",
  "scripts/electron/thread-revert-gate-b.test.mjs",
  "scripts/lib/electron-dev-sidecar.mjs",
  "scripts/lib/electron-dev-sidecar.test.mjs",
  "scripts/plugin/runtime-electron-fixture-smoke.mjs",
  "scripts/plugin/runtime-electron-fixture-smoke.test.mjs",
  "scripts/plugin/runtime-electron-sdk-fixture-smoke.mjs",
  "scripts/plugin/runtime-electron-sdk-fixture-smoke.test.mjs",
  "scripts/plugin/runtime-electron-task-fixture-smoke.mjs",
  "scripts/plugin/runtime-electron-task-fixture-smoke.test.mjs",
  "scripts/smoke/agent-session-messages-electron-fixture-smoke.mjs",
  "scripts/smoke/agent-session-messages-electron-fixture-smoke.test.mjs",
]);

export const KNOWN_OUT_OF_BOUND_AGENT_BASELINE: Array<{
  path: string;
  snippets: Record<string, number>;
}> = [];

export function collectRustFiles(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir)) {
    const fullPath = join(dir, entry);
    const stats = statSync(fullPath);
    if (stats.isDirectory()) {
      if (entry === "target") {
        continue;
      }
      files.push(...collectRustFiles(fullPath));
      continue;
    }
    if (fullPath.endsWith(".rs")) {
      files.push(fullPath);
    }
  }
  return files;
}

export function collectTextFiles(dir: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(dir)) {
    const fullPath = join(dir, entry);
    const stats = statSync(fullPath);
    if (stats.isDirectory()) {
      if (
        entry === "target" ||
        entry === "node_modules" ||
        entry === "dist" ||
        entry === "dist-electron"
      ) {
        continue;
      }
      files.push(...collectTextFiles(fullPath));
      continue;
    }
    if (/\.(?:cjs|js|mjs|rs|ts|tsx)$/u.test(fullPath)) {
      files.push(fullPath);
    }
  }
  return files;
}

export function repoRelative(path: string): string {
  return relative(REPO_ROOT, path).replace(/\\/g, "/");
}

export function productionSource(path: string): string {
  const relativePath = repoRelative(path);
  if (
    relativePath.includes("/tests/") ||
    relativePath.endsWith("/tests.rs") ||
    relativePath.endsWith("_tests.rs")
  ) {
    return "";
  }
  const source = readFileSync(path, "utf8");
  const testIndex = source.indexOf("#[cfg(test)]");
  return testIndex >= 0 ? source.slice(0, testIndex) : source;
}

export function countSnippet(source: string, snippet: string): number {
  return source.split(snippet).length - 1;
}
