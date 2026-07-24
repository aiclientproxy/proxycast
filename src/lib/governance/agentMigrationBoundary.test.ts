/* global process */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

const REPO_ROOT = process.cwd();

function repoPath(path: string): string {
  return join(REPO_ROOT, path);
}

function read(path: string): string {
  return readFileSync(repoPath(path), "utf8");
}

function filesUnder(path: string): string[] {
  const root = repoPath(path);
  if (!existsSync(root)) return [];
  return readdirSync(root, { recursive: true })
    .map(String)
    .map((entry) => join(root, entry))
    .filter((entry) => statSync(entry).isFile())
    .map((entry) => relative(REPO_ROOT, entry));
}

const CURRENT_RUNTIME_CRATES = [
  "agent-protocol",
  "agent-runtime",
  "model-provider",
  "thread-store",
  "tool-runtime",
];

const DELETED_AGENT_PATHS = [
  "lime-rs/crates/agent/src/agent_runtime_projection.rs",
  "lime-rs/crates/agent/src/agent_session_store.rs",
  "lime-rs/crates/agent/src/agent_session_store",
  "lime-rs/crates/agent/src/event_converter.rs",
  "lime-rs/crates/agent/src/identity_adapter.rs",
  "lime-rs/crates/agent/src/message_content_adapter.rs",
  "lime-rs/crates/agent/src/native_tools/runtime_tool_bridge.rs",
  "lime-rs/crates/agent/src/runtime_conversation_agent_adapter.rs",
  "lime-rs/crates/agent/src/runtime_snapshot_adapter.rs",
  "lime-rs/crates/agent/src/runtime_state/gateway_registration_tests.rs",
  "lime-rs/crates/agent/src/runtime_store_agent_adapter.rs",
  "lime-rs/crates/agent/src/runtime_timeline_adapter.rs",
  "lime-rs/crates/agent/src/session_config_adapter.rs",
  "lime-rs/crates/agent/src/execution_strategy_compat.rs",
  "lime-rs/crates/agent/src/session_execution_runtime_adapter.rs",
  "lime-rs/crates/agent/src/session_execution_runtime/runtime_payload.rs",
  "lime-rs/crates/agent/src/session_execution_runtime_query.rs",
  "lime-rs/crates/agent/src/session_runtime_conversation_query.rs",
  "lime-rs/crates/agent/src/session_state_snapshot.rs",
  "lime-rs/crates/agent/src/session_store.rs",
  "lime-rs/crates/agent/src/session_store_history_visibility.rs",
  "lime-rs/crates/agent/src/session_store_message_projection.rs",
  "lime-rs/crates/agent/src/session_store_provider_routing.rs",
  "lime-rs/crates/agent/src/session_store_runtime_detail.rs",
  "lime-rs/crates/agent/src/session_store_runtime_projection.rs",
  "lime-rs/crates/agent/src/session_store_subagent_context.rs",
  "lime-rs/crates/agent/src/session_store_subagent_projection.rs",
  "lime-rs/crates/agent/src/session_store_subagent_query.rs",
  "lime-rs/crates/agent/src/session_store_todo_agent_adapter.rs",
  "lime-rs/crates/agent/src/session_store_todo_projection.rs",
  "lime-rs/crates/agent/src/session_store_tests.rs",
  "lime-rs/crates/agent/src/session_store_types.rs",
  "lime-rs/crates/agent/src/subagent_control.rs",
  "lime-rs/crates/agent/src/subagent_profiles.rs",
  "lime-rs/crates/agent/src/subagent_runtime_adapter.rs",
  "lime-rs/crates/agent/src/runtime_support.rs",
  "lime-rs/crates/agent/src/tools/skill_tool_gate.rs",
  "lime-rs/crates/agent/src/turn_context_configuration/agent_adapter.rs",
];

const DELETED_THREAD_STORE_PATHS = [
  "lime-rs/crates/thread-store/src/conversation_transcript.rs",
  "lime-rs/crates/thread-store/src/history_search.rs",
  "lime-rs/crates/thread-store/src/in_memory_runtime_store.rs",
  "lime-rs/crates/thread-store/src/legacy_conversation.rs",
  "lime-rs/crates/thread-store/src/runtime_status_item.rs",
  "lime-rs/crates/thread-store/src/runtime_store.rs",
  "lime-rs/crates/thread-store/src/session_insights.rs",
  "lime-rs/crates/thread-store/src/sqlite_runtime_store.rs",
];

const DELETED_RUNTIME_QUEUE_PATHS = [
  "lime-rs/crates/agent-runtime/src/runtime_queue/sqlite.rs",
];

const DELETED_DEAD_SURFACE_PATHS = [
  "lime-rs/crates/thread-store/src/subagent_tree.rs",
  "lime-rs/crates/agent/src/team_runtime_governor.rs",
];

const RETIRED_AGENT_RUNTIME_TYPE_NAMES = [
  "AgentRuntimeFrontmatterHookMatcher",
  "AgentRuntimeFrontmatterHook",
  "AgentRuntimeFrontmatterHooks",
  "AgentRuntimeSpawnSubagentRequest",
  "AgentRuntimeSpawnSubagentResponse",
  "AgentRuntimeSendSubagentInputRequest",
  "AgentRuntimeSendSubagentInputResponse",
  "AgentRuntimeStatusSnapshot",
  "AgentRuntimeWaitSubagentsRequest",
  "AgentRuntimeWaitSubagentsResponse",
  "AgentRuntimeResumeSubagentRequest",
  "AgentRuntimeResumeSubagentResponse",
  "AgentRuntimeCloseSubagentRequest",
  "AgentRuntimeCloseSubagentResponse",
];

describe("Agent migration boundary", () => {
  it("agent-compat 必须保持物理删除", () => {
    expect(existsSync(repoPath("lime-rs/crates/agent-compat"))).toBe(false);
  });

  it("workspace 与 lime-agent 不得恢复 Agent 依赖", () => {
    const manifests = [
      "lime-rs/Cargo.toml",
      "lime-rs/crates/agent/Cargo.toml",
      ...CURRENT_RUNTIME_CRATES.map(
        (crateName) => `lime-rs/crates/${crateName}/Cargo.toml`,
      ),
    ];
    for (const manifest of manifests) {
      const source = read(manifest);
      expect(source, manifest).not.toMatch(/^agent\s*=|agent\.workspace/mu);
      expect(source, manifest).not.toContain("agent-compat");
    }
  });

  it("current runtime crates 不得导入 Agent", () => {
    const files = CURRENT_RUNTIME_CRATES.flatMap((crateName) =>
      filesUnder(`lime-rs/crates/${crateName}/src`).filter((path) =>
        path.endsWith(".rs"),
      ),
    );
    const leaks = files.filter((path) =>
      /\baster::|\buse\s+agent\b|extern\s+crate\s+agent/u.test(read(path)),
    );
    expect(leaks).toEqual([]);
  });

  it("lime-agent 生产源码不得导入 Agent", () => {
    const files = filesUnder("lime-rs/crates/agent/src").filter((path) =>
      path.endsWith(".rs"),
    );
    const leaks = files.filter((path) =>
      /\baster::|\buse\s+agent\b|extern\s+crate\s+agent/u.test(read(path)),
    );
    expect(leaks).toEqual([]);
  });

  it("已删除的 lime-agent Agent adapter 不得恢复", () => {
    expect(
      DELETED_AGENT_PATHS.filter((path) => existsSync(repoPath(path))),
    ).toEqual([]);
    expect(
      filesUnder("lime-rs/crates/agent/src/request_tool_policy").filter(
        (path) => /\/agent_[^/]+\.rs$/u.test(path),
      ),
    ).toEqual([]);
  });

  it("已删除的 dead roster/runtime surface 不得恢复", () => {
    expect(
      DELETED_DEAD_SURFACE_PATHS.filter((path) => existsSync(repoPath(path))),
    ).toEqual([]);

    const threadStoreSource = read("lime-rs/crates/thread-store/src/lib.rs");
    expect(threadStoreSource).not.toContain("pub mod subagent_tree");

    const agentSource = read("lime-rs/crates/agent/src/lib.rs");
    expect(agentSource).not.toContain("team_runtime_governor");

    const requestTypesSource = read("src/lib/api/agentRuntime/requestTypes.ts");
    for (const typeName of RETIRED_AGENT_RUNTIME_TYPE_NAMES) {
      expect(requestTypesSource, typeName).not.toContain(typeName);
    }
  });

  it("Runtime session 不得恢复 app-data fallback 或 hydration helper", () => {
    expect(
      existsSync(
        repoPath("lime-rs/crates/app-server/src/runtime/session_hydration.rs"),
      ),
    ).toBe(false);

    const runtimeFiles = filesUnder(
      "lime-rs/crates/app-server/src/runtime",
    ).filter((path) => path.endsWith(".rs"));
    const leaks = runtimeFiles.filter((path) =>
      /load_app_data_session|read_agent_session\(|\bmod session_hydration;/u.test(
        read(path),
      ),
    );
    expect(leaks).toEqual([]);
  });

  it("provider current owner 必须保留 OpenCode 风格中立 part 与集中 lowering", () => {
    const source = read("lime-rs/crates/model-provider/src/current_client.rs");
    expect(source).toContain("pub enum CurrentProviderContent");
    expect(source).toContain("Image {");
    expect(source).toContain("ToolCall(CurrentProviderToolCall)");
    expect(source).toContain("ToolResult(CurrentProviderToolResult)");
    expect(source).toContain("RuntimeProviderProtocol::Responses");
    expect(source).toContain("RuntimeProviderProtocol::AnthropicMessages");
    expect(source).toContain("RuntimeProviderProtocol::ChatCompletions");
  });

  it("provider turn 必须消费 current provider 与 typed tool executor", () => {
    const source = read("lime-rs/crates/agent-runtime/src/provider_turn.rs");
    expect(source).toContain("CurrentProvider");
    expect(source).toContain("RuntimeToolExecutorHandle");
    expect(source).not.toContain("agent::");
  });

  it("MCP current owner 必须由 tool-runtime registry 和每回合 snapshot 分工", () => {
    const registrySource = read(
      "lime-rs/crates/tool-runtime/src/mcp_connection/registry.rs",
    );
    const snapshotSource = read(
      "lime-rs/crates/tool-runtime/src/mcp_connection/step_snapshot.rs",
    );
    expect(registrySource).toContain("pub struct McpConnectionRegistry");
    expect(registrySource).toContain("pub async fn step_snapshot");
    expect(registrySource).not.toContain("agent::");
    expect(snapshotSource).toContain("pub struct McpStepSnapshot");
    expect(snapshotSource).toContain("pub async fn dispatch");
    expect(snapshotSource).not.toContain("agent::");
  });

  it("request_user_input pending state 必须归属 session/turn current state", () => {
    const source = read("lime-rs/crates/agent-runtime/src/action_required.rs");
    const turnSnapshotSource = read(
      "lime-rs/crates/agent/src/current_provider_turn/mcp_step_snapshot.rs",
    );
    const turnExecutorSource = read(
      "lime-rs/crates/agent/src/current_provider_turn/tool_executor.rs",
    );
    const bridgeSource = read(
      "lime-rs/crates/agent/src/request_user_input_bridge.rs",
    );
    expect(source).toContain("pub struct ActionRequiredState");
    expect(source).toContain("pending");
    expect(source).toContain("resolve_action");
    expect(source).not.toContain("static ACTION_REQUIRED");
    expect(turnSnapshotSource).toContain("request_user_input_tool_definition");
    expect(turnExecutorSource).toContain("execute_request_user_input");
    expect(bridgeSource).toContain("request_action_and_wait_with_notification");
    expect(bridgeSource).toContain("run_request_user_input");
    expect(bridgeSource).toContain("AgentEvent::ActionRequired");
  });

  it("ProjectionStore 必须是唯一 Thread/Turn/Item truth，queue payload 单独持久化", () => {
    const projectionStore = read(
      "lime-rs/crates/app-server/src/runtime/projection_store.rs",
    );
    const queueIntent = read(
      "lime-rs/crates/app-server/src/runtime/queued_turn_intent.rs",
    );
    for (const path of DELETED_THREAD_STORE_PATHS) {
      expect(existsSync(repoPath(path)), path).toBe(false);
    }
    for (const path of DELETED_RUNTIME_QUEUE_PATHS) {
      expect(existsSync(repoPath(path)), path).toBe(false);
    }
    expect(projectionStore).toContain("pub struct ProjectionStore");
    expect(projectionStore).toContain("pub fn apply_events");
    expect(projectionStore).toContain("pub fn read_session_projection");
    expect(queueIntent).toContain("queue.added");
    expect(queueIntent).toContain("snapshot_value");
    expect(queueIntent).toContain("runtime_options_from_events");
  });

  it("架构文档必须明确 Codex 与 OpenCode 的裁决边界", () => {
    const source = read(
      "internal/research/refactor/v1/opencode-reference-comparison.md",
    );
    expect(source).toContain("Codex 更适合指导 Agent loop");
    expect(source).toContain("Provider-specific lowering");
    expect(source).toContain("Model capability");
    expect(source).toContain("多模态 message part");
  });
});
