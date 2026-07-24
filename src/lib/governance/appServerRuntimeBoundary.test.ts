import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  AGENT_PROVIDER_CONFIGURATION_BOUNDARY,
  AGENT_PROVIDER_RUNTIME_ADAPTER_BOUNDARY,
  AGENT_SESSION_CONFIGURATION_BOUNDARY,
  AGENT_TURN_CONTEXT_CONFIGURATION_BOUNDARY,
  ALLOWED_AGENT_COUPLING_OWNER_FILES,
  APP_SERVER_SRC_DIR,
  AGENT_COUPLING_SNIPPETS,
  AGENT_EXECUTION_SNIPPETS,
  AGENT_PROVIDER_CONFIGURATION_SNIPPETS,
  AGENT_SKILL_EXECUTION_SNIPPETS,
  IMAGE_COMMAND_MAIN,
  IMAGE_COMMAND_SPLIT_MODULES,
  AGENT_SESSION_EXECUTION_RUNTIME_MAIN,
  AGENT_SESSION_EXECUTION_RUNTIME_OWNER_MODULES,
  KNOWN_OUT_OF_BOUND_AGENT_BASELINE,
  KNOWN_OUT_OF_BOUND_AGENT_COUPLING_FILES,
  KNOWN_OUT_OF_BOUND_AGENT_EXECUTION_FILES,
  LOCAL_DATA_SOURCE_SKILLS_DIR,
  PLUGIN_WORKER_TURN_MAIN,
  PLUGIN_WORKER_TURN_SPLIT_MODULES,
  PROCESSOR_DISPATCH,
  PROCESSOR_MAIN,
  PROCESSOR_SPLIT_MODULES,
  PROCESSOR_TESTS_MAIN,
  REPO_ROOT,
  RUNTIME_BACKEND_MAIN,
  RUNTIME_BACKEND_OWNER_MODULES,
  RUNTIME_BACKEND_REQUEST_CONTEXT_MAIN,
  RUNTIME_BACKEND_REQUEST_CONTEXT_SPLIT_MODULES,
  RUNTIME_BACKEND_TESTS_MAIN,
  RUNTIME_BACKEND_TEST_SPLIT_MODULES,
  RUNTIME_BOUNDARY_ROADMAP,
  RUNTIME_CORE_MAIN,
  RUNTIME_CORE_OWNER_MODULES,
  RUNTIME_READ_MODEL_MAIN,
  RUNTIME_READ_MODEL_OWNER_MODULES,
  RUNTIME_THREAD_ITEM_PROJECTION_MAIN,
  RUNTIME_THREAD_ITEM_PROJECTION_OWNER_MODULES,
  collectRustFiles,
  countSnippet,
  productionSource,
  repoRelative,
} from "./appServerRuntimeBoundary.testSupport";

describe("app-server runtime boundary", () => {
  it("App Server 生产代码不应新增未登记的 Agent 直接耦合", () => {
    const unregistered = collectRustFiles(APP_SERVER_SRC_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .filter(({ source }) =>
        AGENT_COUPLING_SNIPPETS.some((snippet) => source.includes(snippet)),
      )
      .filter(
        ({ path }) =>
          !ALLOWED_AGENT_COUPLING_OWNER_FILES.has(path) &&
          !KNOWN_OUT_OF_BOUND_AGENT_COUPLING_FILES.has(path),
      )
      .map(({ path }) => path);

    expect(
      unregistered,
      "新增 Agent 直接耦合前必须先迁到 lime-agent / runtime-core / 已登记 runtime_backend 子边界；已知越界文件只能减少，不能增加",
    ).toEqual([]);
  });

  it("已登记的 App Server 顶层 Agent 越界白名单只能减少，不能增长", () => {
    const increases = KNOWN_OUT_OF_BOUND_AGENT_BASELINE.flatMap(
      ({ path, snippets }) => {
        const fullPath = join(REPO_ROOT, path);
        const source = existsSync(fullPath) ? productionSource(fullPath) : "";

        return Object.entries(snippets)
          .map(([snippet, baseline]) => ({
            path,
            snippet,
            baseline,
            actual: countSnippet(source, snippet),
          }))
          .filter(({ actual, baseline }) => actual > baseline);
      },
    );

    expect(
      increases,
      "已知越界面是迁移负债，不是扩展许可；新增 Agent 状态、provider 配置或 Skill 执行调用前必须先迁到 runtime backend / lime-agent / runtime-core",
    ).toEqual([]);
  });

  it("App Server 顶层 Agent 越界白名单必须登记到治理路线图", () => {
    const roadmap = readFileSync(
      join(REPO_ROOT, RUNTIME_BOUNDARY_ROADMAP),
      "utf8",
    );
    const missing = [
      ...KNOWN_OUT_OF_BOUND_AGENT_COUPLING_FILES,
      ...KNOWN_OUT_OF_BOUND_AGENT_EXECUTION_FILES,
    ].filter((path) => !roadmap.includes(path));

    expect(
      missing,
      "新增 App Server 顶层 Agent 越界白名单时，必须同步登记原因、分类和退出条件",
    ).toEqual([]);
  });

  it("LocalAppDataSource skills 层不应直接触碰 Agent 状态或 reload API", () => {
    const forbiddenSnippets = [
      "AgentState",
      "lime_agent::reload_lime_skills",
      "reload_lime_skills(",
    ];
    const offenders = collectRustFiles(LOCAL_DATA_SOURCE_SKILLS_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .filter(({ source }) =>
        forbiddenSnippets.some((snippet) => source.includes(snippet)),
      )
      .map(({ path }) => path);

    expect(
      offenders,
      "LocalAppDataSource skills 层只能通知 App Server skill_registry 边界，不得直接触碰 Agent 状态或 lime_agent reload API",
    ).toEqual([]);
  });

  it("App Server 只能通过 agent runtime registry 初始化 Agent runtime", () => {
    const offenders = collectRustFiles(APP_SERVER_SRC_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .filter(({ source }) =>
        source.includes("lime_agent::initialize_agent_runtime("),
      )
      .map(({ path }) => path)
      .filter(
        (path) =>
          path !== "lime-rs/crates/app-server/src/agent_runtime_registry.rs",
      );

    expect(
      offenders,
      "Agent runtime 初始化只能停留在 App Server agent_runtime_registry 边界；数据源、processor 或 runtime backend 不得重新 direct import lime_agent::initialize_agent_runtime",
    ).toEqual([]);
  });

  it("App Server 不应新增 runtime_backend 之外的 Agent 回合执行链", () => {
    const unregistered = collectRustFiles(APP_SERVER_SRC_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .filter(({ source }) =>
        AGENT_EXECUTION_SNIPPETS.some((snippet) => source.includes(snippet)),
      )
      .filter(
        ({ path }) =>
          path !== "lime-rs/crates/app-server/src/runtime_backend.rs" &&
          !path.startsWith("lime-rs/crates/app-server/src/runtime_backend/") &&
          !KNOWN_OUT_OF_BOUND_AGENT_EXECUTION_FILES.has(path),
      )
      .map(({ path }) => path);

    expect(
      unregistered,
      "App Server 不应在 runtime_backend 之外继续复制 Agent provider 配置或 stream_reply 执行流；新增同类执行面必须先迁入 runtime backend / lime-agent / runtime-core",
    ).toEqual([]);
  });

  it("App Server provider adapter 不应直接配置 Agent provider", () => {
    const offenders = collectRustFiles(APP_SERVER_SRC_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .flatMap(({ path, source }) =>
        AGENT_PROVIDER_CONFIGURATION_SNIPPETS.filter((snippet) =>
          source.includes(snippet),
        ).map((snippet) => ({ path, snippet })),
      );
    const agentBoundary = readFileSync(
      join(REPO_ROOT, AGENT_PROVIDER_CONFIGURATION_BOUNDARY),
      "utf8",
    );
    const agentProviderRuntimeAdapter = readFileSync(
      join(REPO_ROOT, AGENT_PROVIDER_RUNTIME_ADAPTER_BOUNDARY),
      "utf8",
    );
    const agentProviderOwner = `${agentBoundary}\n${agentProviderRuntimeAdapter}`;
    const appServerProviderAdapter = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/provider_config.rs",
      ),
    );

    expect(agentProviderOwner).toContain("create_configured_reply_provider");
    expect(agentProviderOwner).toContain("install_provider_for_session");
    expect(agentProviderOwner).toContain("RuntimeProviderProtocol");
    expect(agentProviderOwner).toContain("SessionProviderConfig");
    expect(agentProviderOwner).toContain("ModelProviderProtocol");
    expect(agentProviderOwner).toContain("ProtocolKind");
    expect(agentProviderOwner).toContain(
      "route_protocol_from_session_provider_config",
    );
    expect(agentProviderOwner).toContain("ModelRouteProviderConfiguration");
    expect(agentProviderOwner).toContain(
      "configure_model_route_provider_for_session",
    );
    expect(agentProviderOwner).toContain("ProviderConfigurationRequest");
    expect(agentProviderOwner).toContain("RuntimeProviderProtocol::Responses");
    expect(agentProviderOwner).toContain("ModelProviderProtocol::Responses");
    expect(agentProviderOwner).toContain("ProtocolKind::OpenaiResponses");
    expect(agentProviderOwner).not.toContain(".configure_provider(");
    expect(agentProviderOwner).not.toContain("configure_provider_from_pool(");
    expect(appServerProviderAdapter).not.toContain(
      "configure_provider_for_session",
    );
    expect(appServerProviderAdapter).not.toContain(
      "ProviderConfigurationRequest",
    );
    expect(appServerProviderAdapter).not.toContain(
      "configure_provider_for_route",
    );
    expect(
      offenders,
      "runtime provider 配置与 provider protocol 映射属于 lime-agent provider_configuration 边界；App Server 只能传 route ProtocolKind 并做 façade 接线",
    ).toEqual([]);
  });

  it("App Server 不应直接复制 Agent Skill prompt/workflow 执行器", () => {
    const offenders = collectRustFiles(APP_SERVER_SRC_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .filter(({ source }) =>
        AGENT_SKILL_EXECUTION_SNIPPETS.some((snippet) =>
          source.includes(snippet),
        ),
      )
      .map(({ path }) => path);

    expect(
      offenders,
      "Knowledge / Skill 执行应下沉到 lime-agent 或统一 Skill runtime；App Server 只能做 JSON-RPC / RuntimeCore 投影和受控 backend adapter",
    ).toEqual([]);
  });

  it("Knowledge Builder App Server adapter 不应直接持有 Agent 状态", () => {
    const adapter = readFileSync(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/knowledge_builder_runtime.rs",
      ),
      "utf8",
    );

    expect(adapter).toContain("KnowledgeBuilderSkillRunner");
    expect(adapter).not.toContain("AgentState");
    expect(adapter).not.toContain("run_knowledge_builder_skill");
  });

  it("runtime_backend 主文件不应重新承接 action response 或事件 mapper 细节", () => {
    const runtimeBackend = readFileSync(
      join(REPO_ROOT, "lime-rs/crates/app-server/src/runtime_backend.rs"),
      "utf8",
    );

    expect(runtimeBackend).toContain("mod action_response;");
    expect(runtimeBackend).toContain("mod event_mapper;");
    expect(runtimeBackend).not.toContain("confirm_tool_action(");
    expect(runtimeBackend).not.toContain("submit_elicitation_response(");
    expect(runtimeBackend).not.toContain(
      "fn emit_runtime_agent_event_with_coding_mirror",
    );
    expect(runtimeBackend).not.toContain("fn emit_reasoning_finish");
    expect(runtimeBackend).not.toContain("fn emit_proposed_plan_parser_flush");
    expect(runtimeBackend).not.toContain(
      "WorkspacePatchHostToolPlan::from_events",
    );
    expect(runtimeBackend).not.toContain("execute_planned_tool_batch(");
    expect(runtimeBackend).not.toContain("ToolExecutionBatchInput");
  });

  it("runtime_backend request_context 主文件必须保持职责拆分", () => {
    const requestContext = readFileSync(
      join(REPO_ROOT, RUNTIME_BACKEND_REQUEST_CONTEXT_MAIN),
      "utf8",
    );
    const lineCount = requestContext.split(/\r?\n/u).length;
    const missingModules = RUNTIME_BACKEND_REQUEST_CONTEXT_SPLIT_MODULES.filter(
      (path) => !existsSync(join(REPO_ROOT, path)),
    );
    const returnedResponsibilities = [
      "fn session_config_from_request(",
      "fn turn_context_from_request(",
      "fn request_workspace_scope(",
      "fn request_system_prompt(",
      "fn w3c_trace_context_metadata_from_request(",
    ].filter((snippet) => requestContext.includes(snippet));

    expect(
      lineCount,
      "request_context.rs 超过 800 行前必须继续拆子模块",
    ).toBeLessThanOrEqual(800);
    expect(
      missingModules,
      "request_context.rs 的职责拆分模块不得被折回主文件",
    ).toEqual([]);
    expect(
      returnedResponsibilities,
      "SessionConfig / TurnContext / workspace scope 逻辑不得回流到 request_context.rs 主文件",
    ).toEqual([]);
  });

  it("plugin_worker_turn 主文件必须保持 worker turn 编排职责拆分", () => {
    const mainSource = readFileSync(
      join(REPO_ROOT, PLUGIN_WORKER_TURN_MAIN),
      "utf8",
    );
    const lineCount = mainSource.split(/\r?\n/u).length;
    const missingModules = PLUGIN_WORKER_TURN_SPLIT_MODULES.filter(
      (path) => !existsSync(join(REPO_ROOT, path)),
    );
    const returnedResponsibilities = [
      "fn resolve_plugin_activation_request(",
      "fn resolve_pane_action_request(",
      "fn validate_worker_cloud_release_signature(",
      "fn classify_worker_failure(",
      "fn worker_progress_events_for_sink(",
      "fn assistant_message_events_from_worker_events(",
      "fn json_string(",
    ].filter((snippet) => mainSource.includes(snippet));

    expect(
      lineCount,
      "plugin_worker_turn.rs 超过 800 行前必须继续拆子模块，不能把 worker request / failure / progress / launch gate 职责折回主文件",
    ).toBeLessThanOrEqual(800);
    expect(
      missingModules,
      "plugin_worker_turn.rs 的职责拆分模块不得被删除或折回主文件",
    ).toEqual([]);
    expect(
      returnedResponsibilities,
      "plugin worker request 解析、失败分类、progress 投影、launch gate 和 JSON helper 不得回流到 plugin_worker_turn.rs 主文件",
    ).toEqual([]);
  });

  it("runtime_backend tests 主文件必须保持测试职责拆分", () => {
    const mainSource = readFileSync(
      join(REPO_ROOT, RUNTIME_BACKEND_TESTS_MAIN),
      "utf8",
    );
    const lineCount = mainSource.split(/\r?\n/u).length;
    const missingModules = RUNTIME_BACKEND_TEST_SPLIT_MODULES.filter(
      (path) => !existsSync(join(REPO_ROOT, path)),
    );
    const returnedTestResponsibilities = [
      "fn explicit_runtime_preferences_win(",
      "fn session_config_appends_memory_context_to_system_prompt(",
      "fn runtime_agent_tool_events_are_mirrored_to_coding_facts(",
      "async fn runtime_backend_registers_memory_tools_in_agent_registry(",
      "fn request_working_dir_uses_host_turn_config_absolute_directory(",
    ].filter((snippet) => mainSource.includes(snippet));

    expect(
      lineCount,
      "runtime_backend/tests.rs 只能保留共享 fixture 和模块声明，具体 runtime backend 测试必须留在 tests/ 子模块",
    ).toBeLessThanOrEqual(800);
    expect(
      missingModules,
      "runtime_backend 测试职责拆分模块不得被删除或折回 tests.rs 主文件",
    ).toEqual([]);
    expect(
      returnedTestResponsibilities,
      "model selection、session prompt、tool policy、turn flow、coding projection 等测试不得回流到 runtime_backend/tests.rs 主文件",
    ).toEqual([]);
  });

  it("image_command 主文件必须保持 @配图 workflow 职责拆分", () => {
    const mainSource = readFileSync(
      join(REPO_ROOT, IMAGE_COMMAND_MAIN),
      "utf8",
    );
    const lineCount = mainSource.split(/\r?\n/u).length;
    const missingModules = IMAGE_COMMAND_SPLIT_MODULES.filter(
      (path) => !existsSync(join(REPO_ROOT, path)),
    );
    const returnedResponsibilities = [
      "struct ImageCommandIntent",
      "fn parse_image_command_intent(",
      "fn image_command_metadata(",
      "fn request_metadata_values(",
      "#[tokio::test]",
      "#[test]",
    ].filter((snippet) => mainSource.includes(snippet));

    expect(
      lineCount,
      "runtime_backend/image_command/mod.rs 只能保留 @配图 workflow 编排和事件投影，intent 解析、presentation 和测试必须留在子模块",
    ).toBeLessThanOrEqual(800);
    expect(
      missingModules,
      "@配图 image_command 职责拆分模块不得被删除或折回 mod.rs 主文件",
    ).toEqual([]);
    expect(
      returnedResponsibilities,
      "ImageCommandIntent、metadata 解析和内联测试不得回流到 image_command/mod.rs 主文件",
    ).toEqual([]);
  });

  it("processor 主文件必须保持 JSON-RPC facade 职责拆分", () => {
    const mainSource = readFileSync(join(REPO_ROOT, PROCESSOR_MAIN), "utf8");
    const dispatchSource = readFileSync(
      join(REPO_ROOT, PROCESSOR_DISPATCH),
      "utf8",
    );
    const testsSource = readFileSync(
      join(REPO_ROOT, PROCESSOR_TESTS_MAIN),
      "utf8",
    );
    const mainLineCount = mainSource.split(/\r?\n/u).length;
    const dispatchLineCount = dispatchSource.split(/\r?\n/u).length;
    const testsLineCount = testsSource.split(/\r?\n/u).length;
    const missingModules = PROCESSOR_SPLIT_MODULES.filter(
      (path) => !existsSync(join(REPO_ROOT, path)),
    );
    const returnedMainResponsibilities = [
      "async fn handle_request_inner(",
      "#[tokio::test]",
      "METHOD_PROJECT_GIT_STATUS =>",
      "METHOD_MCP_TOOL_CALL =>",
      "METHOD_EXECUTION_PROCESS_START =>",
    ].filter((snippet) => mainSource.includes(snippet));
    const returnedTestResponsibilities = [
      "async fn artifact_read_requires_initialized_and_returns_artifact_summaries(",
      "async fn mcp_runtime_methods_require_initialized_and_fail_closed_without_manager(",
      "async fn workspace_right_surface_methods_register_and_list_pending_requests(",
      "async fn execution_process_methods_start_drain_and_report_status(",
    ].filter((snippet) => testsSource.includes(snippet));

    expect(
      mainLineCount,
      "processor/mod.rs 只能保留 processor facade、初始化和共享 helper；分发表和领域测试不得折回主文件",
    ).toBeLessThanOrEqual(800);
    expect(
      dispatchLineCount,
      "processor/dispatch.rs 只能保留 JSON-RPC method 分发表；超过 800 行前必须继续拆 command group dispatch",
    ).toBeLessThanOrEqual(800);
    expect(
      testsLineCount,
      "processor/tests.rs 只能保留测试模块声明，具体 JSON-RPC 集成测试必须留在 tests/ 子模块",
    ).toBeLessThanOrEqual(120);
    expect(
      missingModules,
      "processor 的 dispatch 与测试职责拆分模块不得被删除或折回主文件",
    ).toEqual([]);
    expect(
      returnedMainResponsibilities,
      "JSON-RPC 分发表、MCP / git / execution process 分派和内联测试不得回流到 processor/mod.rs",
    ).toEqual([]);
    expect(
      returnedTestResponsibilities,
      "artifact / MCP / right surface / execution process 等 processor 集成测试不得回流到 processor/tests.rs",
    ).toEqual([]);
  });

  it("P1-4 core runtime owner 主文件不得继续承接 turn/model/tool/context domain 逻辑", () => {
    const runtimeCore = readFileSync(
      join(REPO_ROOT, RUNTIME_CORE_MAIN),
      "utf8",
    );
    const runtimeBackend = readFileSync(
      join(REPO_ROOT, RUNTIME_BACKEND_MAIN),
      "utf8",
    );
    const dispatchSource = readFileSync(
      join(REPO_ROOT, PROCESSOR_DISPATCH),
      "utf8",
    );
    const readModel = readFileSync(
      join(REPO_ROOT, RUNTIME_READ_MODEL_MAIN),
      "utf8",
    );
    const threadItemProjection = readFileSync(
      join(REPO_ROOT, RUNTIME_THREAD_ITEM_PROJECTION_MAIN),
      "utf8",
    );
    const agentSessionRuntime = readFileSync(
      join(REPO_ROOT, AGENT_SESSION_EXECUTION_RUNTIME_MAIN),
      "utf8",
    );
    const missingModules = [
      ...RUNTIME_CORE_OWNER_MODULES,
      ...RUNTIME_BACKEND_OWNER_MODULES,
      ...RUNTIME_READ_MODEL_OWNER_MODULES,
      ...RUNTIME_THREAD_ITEM_PROJECTION_OWNER_MODULES,
      ...AGENT_SESSION_EXECUTION_RUNTIME_OWNER_MODULES,
    ].filter((path) => !existsSync(join(REPO_ROOT, path)));

    const returnedRuntimeCoreResponsibilities = [
      "fn build_session_context_compaction(",
      "fn stored_session_to_overview(",
      "fn validate_runtime_resume_contract(",
      "fn runtime_session_read_detail_with_options(",
      "fn tool_items_from_events(",
    ].filter((snippet) => runtimeCore.includes(snippet));
    const returnedRuntimeBackendResponsibilities = [
      "fn resolve_runtime_model_selection(",
      "fn resolve_basic_model_capability(",
      "fn routing_decision_payload(",
      "fn read_agent_tool_inventory(",
      "fn build_agent_turn_context(",
      "fn build_agent_session_config(",
    ].filter((snippet) => runtimeBackend.includes(snippet));
    const returnedReadModelResponsibilities = [
      "fn stored_artifact_summaries_for_turn(",
      "fn stored_user_visible_artifact_summaries_for_turn(",
      "fn tool_items_from_events(",
      "fn tool_calls_from_events(",
      "fn workflow_read_model_from_events(",
      "fn coding_activity_from_events(",
      "fn file_artifact_items_from_events(",
      "fn permission_state_from_events(",
      "fn turns_with_usage(",
    ].filter((snippet) => readModel.includes(snippet));
    const returnedThreadProjectionResponsibilities = [
      "fn item_from_delta(",
      "fn upsert_from_item_event(",
      "fn plan_item(",
      "fn plan_metadata(",
      "fn merge_cumulative_text(",
    ].filter((snippet) => threadItemProjection.includes(snippet));
    const returnedAgentSessionRuntimeResponsibilities = [
      "fn project_agent_session_execution_runtime_session(",
      "fn project_agent_session_execution_runtime_snapshot(",
      "fn build_agent_turn_context(",
      "fn read_agent_tool_inventory(",
      "fn configure_model_route_provider_for_session(",
      "fn resolve_tool_execution_policy(",
      "pub(crate) fn extract_recent_harness_context_from_metadata(",
      "fn extract_recent_preferences_from_metadata(",
      "pub(crate) fn extract_recent_access_mode_from_metadata(",
      "fn extract_recent_team_selection_from_metadata(",
      "fn extract_task_profile_from_metadata(",
      "fn extract_routing_decision_from_metadata(",
      "fn extract_runtime_summary_from_metadata(",
      "fn apply_usage_to_cost_state(",
      "fn detect_runtime_limit_event(",
      "fn calculate_estimated_total_cost(",
    ].filter((snippet) => agentSessionRuntime.includes(snippet));
    const returnedAgentSessionRuntimeTestResponsibilities = [
      "fn keeps_recent_preferences_from_latest_turn_metadata(",
      "fn extracts_task_routing_and_limit_state_from_lime_runtime_metadata(",
      "fn apply_usage_to_cost_state_should_calculate_estimated_total_cost(",
    ].filter((snippet) => agentSessionRuntime.includes(snippet));

    expect(
      runtimeCore.split(/\r?\n/u).length,
      "runtime.rs 只能作为 RuntimeCore facade / state wiring；新增 session/turn/model/tool/context 逻辑必须进入 runtime/* domain 模块",
    ).toBeLessThanOrEqual(720);
    expect(
      runtimeBackend.split(/\r?\n/u).length,
      "runtime_backend.rs 只能保留 ExecutionBackend 编排；model/provider/tool/context 细节必须进入 runtime_backend/* domain 模块，且不得超过通用 facade 上限",
    ).toBeLessThanOrEqual(800);
    expect(
      dispatchSource.split(/\r?\n/u).length,
      "processor/dispatch.rs 只能保留 JSON-RPC method 分发表；新增 command group 必须先进 processor/* handler 模块",
    ).toBeLessThanOrEqual(800);
    expect(
      readModel.split(/\r?\n/u).length,
      "read_model.rs 已拆出 messages owner，新增 artifact/tool/workflow/coding/session runtime/message 投影必须先进 owner 模块",
    ).toBeLessThanOrEqual(840);
    expect(
      threadItemProjection.split(/\r?\n/u).length,
      "thread_item_projection.rs 已超过 1000 行，新增 Item 类型投影必须先进 thread_item_projection/* 子模块",
    ).toBeLessThanOrEqual(1445);
    expect(
      agentSessionRuntime.split(/\r?\n/u).length,
      "session_execution_runtime.rs 只保留 AgentEvent DTO 与 current session projection alias；新增 provider/tool/turn/context 投影必须进入 App Server 或 agent-runtime current owner",
    ).toBeLessThanOrEqual(660);
    expect(
      missingModules,
      "P1-4 owner 模块不得被删除或折回中心文件；新增 turn/model/tool/context 能力必须沿这些 owner 扩展",
    ).toEqual([]);
    expect(
      returnedRuntimeCoreResponsibilities,
      "RuntimeCore 主文件不得重新承接 session lifecycle、turn execution、read model 或 tool projection 细节",
    ).toEqual([]);
    expect(
      returnedRuntimeBackendResponsibilities,
      "runtime_backend 主文件不得重新承接 model routing、provider config、tool inventory 或 turn context 细节",
    ).toEqual([]);
    expect(
      returnedReadModelResponsibilities,
      "read_model.rs 不得回收 artifact/tool/workflow/coding/usage/permission 等已拆 owner 的函数定义",
    ).toEqual([]);
    expect(
      returnedThreadProjectionResponsibilities,
      "thread_item_projection.rs 不得回收 agent_message / plan 等已拆 Item 投影函数定义",
    ).toEqual([]);
    expect(
      returnedAgentSessionRuntimeResponsibilities,
      "session_execution_runtime.rs 不得回收 provider/tool/turn context/adapter/recent context/recent settings或已删除 aggregate runtime payload 的函数定义",
    ).toEqual([]);
    expect(
      returnedAgentSessionRuntimeTestResponsibilities,
      "session_execution_runtime.rs 不得回收 recent settings或已删除 aggregate runtime payload 的测试职责；新增测试必须进入 current owner",
    ).toEqual([]);
  });

  it("P1-5 UI execution runtime projection owner 不得折回 utils facade", () => {
    const utilsPath =
      "src/components/agent/chat/utils/sessionExecutionRuntime.ts";
    const projectionPath =
      "src/components/agent/chat/projection/sessionExecutionRuntimeProjection.ts";
    const utilsSource = readFileSync(join(REPO_ROOT, utilsPath), "utf8");
    const projectionSource = readFileSync(
      join(REPO_ROOT, projectionPath),
      "utf8",
    );
    const projectionHelpers = [
      "function mergeExecutionRuntime(",
      "export function applyTurnContextExecutionRuntime(",
      "export function applyModelChangeExecutionRuntime(",
    ];

    expect(existsSync(join(REPO_ROOT, projectionPath))).toBe(true);
    expect(utilsSource.split(/\r?\n/u).length).toBeLessThanOrEqual(390);
    expect(
      projectionHelpers.filter((snippet) => utilsSource.includes(snippet)),
    ).toEqual([]);
    expect(
      projectionHelpers.filter(
        (snippet) => !projectionSource.includes(snippet),
      ),
    ).toEqual([]);
  });

  it("App Server 主 turn 执行不应直接调用 Agent streaming loop", () => {
    const runtimeBackend = productionSource(
      join(REPO_ROOT, "lime-rs/crates/app-server/src/runtime_backend.rs"),
    );
    const runtimeBackendFull = readFileSync(
      join(REPO_ROOT, "lime-rs/crates/app-server/src/runtime_backend.rs"),
      "utf8",
    );
    const agentTurnExecution = readFileSync(
      join(REPO_ROOT, "lime-rs/crates/agent/src/turn_execution.rs"),
      "utf8",
    );
    const agentProviderConfiguration = readFileSync(
      join(REPO_ROOT, AGENT_PROVIDER_CONFIGURATION_BOUNDARY),
      "utf8",
    );
    const agentProviderRuntimeAdapter = readFileSync(
      join(REPO_ROOT, AGENT_PROVIDER_RUNTIME_ADAPTER_BOUNDARY),
      "utf8",
    );
    const agentProviderOwner = `${agentProviderConfiguration}\n${agentProviderRuntimeAdapter}`;

    expect(runtimeBackend).toContain("run_agent_turn_with_policy");
    expect(runtimeBackend).toContain("AgentTurnExecutionRequest");
    expect(runtimeBackend).toContain("AgentTurnProviderConfiguration");
    expect(runtimeBackendFull).toContain("provider_configuration_from_runtime");
    expect(runtimeBackend).not.toContain("ProviderConfigurationRequest");
    expect(runtimeBackend).not.toContain("stream_reply_with_policy(");
    expect(runtimeBackend).not.toContain("configure_provider_for_route(");
    expect(runtimeBackend).not.toContain("mark_current_healthy(");
    expect(runtimeBackend).not.toContain("create_cancel_token(");
    expect(runtimeBackend).not.toContain("remove_cancel_token(");
    expect(agentTurnExecution).toContain("stream_current_provider_turn(");
    expect(agentTurnExecution).not.toContain(
      "stream_runtime_reply_with_policy(",
    );
    expect(agentTurnExecution).not.toContain(
      "stream_runtime_reply_with_configured_provider(",
    );
    expect(agentTurnExecution).toContain(
      "configure_model_route_provider_for_session_with_provider_and_credential_ref(",
    );
    expect(agentTurnExecution).toContain("ModelRouteProviderConfiguration");
    expect(agentTurnExecution).toContain("AgentTurnProviderConfiguration");
    expect(agentTurnExecution).toContain("let configured_provider");
    expect(agentTurnExecution).toContain(
      "configured_provider.map(|configured| configured.into_config())",
    );
    expect(agentTurnExecution).not.toContain("mark_healthy(");
    expect(agentTurnExecution).toContain("create_cancel_token(");
    expect(agentTurnExecution).toContain("remove_cancel_token(");
    expect(agentProviderOwner).toContain("create_configured_reply_provider");
    expect(agentProviderOwner).toContain("install_provider_for_session");
    expect(agentProviderOwner).not.toContain("mark_healthy(");
  });

  it("App Server 不应直接构造 Agent SessionConfig", () => {
    const offenders = collectRustFiles(APP_SERVER_SRC_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .flatMap(({ path, source }) =>
        ["SessionConfigBuilder", "agent::agents::SessionConfig"]
          .filter((snippet) => source.includes(snippet))
          .map((snippet) => ({ path, snippet })),
      );
    const agentSessionConfiguration = readFileSync(
      join(REPO_ROOT, AGENT_SESSION_CONFIGURATION_BOUNDARY),
      "utf8",
    );
    const appServerAdapter = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/request_context/session_config.rs",
      ),
    );

    expect(agentSessionConfiguration).toContain(
      "agent_runtime::session_config",
    );
    expect(agentSessionConfiguration).toContain(
      "AgentSessionConfigurationRequest",
    );
    expect(agentSessionConfiguration).toContain("build_agent_session_config");
    expect(agentSessionConfiguration).not.toContain(
      "agent::agents::SessionConfig",
    );
    expect(
      existsSync(
        join(REPO_ROOT, "lime-rs/crates/agent/src/session_config_adapter.rs"),
      ),
      "已删除的 session_config_adapter 不得恢复；current façade 直接 re-export agent-runtime session_config",
    ).toBe(false);
    expect(appServerAdapter).toContain("build_agent_session_config");
    expect(appServerAdapter).toContain("AgentSessionConfigurationRequest");
    expect(
      offenders,
      "Agent SessionConfig 构造属于 lime-agent session_configuration 边界；App Server request_context 只能准备 prompt / turn_context 投影并调用 façade",
    ).toEqual([]);
  });

  it("App Server 不应直接引用 Agent TurnContext 类型", () => {
    const forbiddenSnippets = ["TurnContextOverride", "TurnOutputSchemaSource"];
    const offenders = collectRustFiles(APP_SERVER_SRC_DIR)
      .map((file) => ({
        path: repoRelative(file),
        source: productionSource(file),
      }))
      .flatMap(({ path, source }) =>
        forbiddenSnippets
          .filter((snippet) => source.includes(snippet))
          .map((snippet) => ({ path, snippet })),
      );
    const agentBoundary = readFileSync(
      join(REPO_ROOT, AGENT_TURN_CONTEXT_CONFIGURATION_BOUNDARY),
      "utf8",
    );
    const appServerAdapter = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/request_context/turn_context.rs",
      ),
    );
    const imagePresentationAdapter = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/image_command/presentation.rs",
      ),
    );

    expect(agentBoundary).toContain("TurnContextOverride");
    expect(agentBoundary).toContain("TurnOutputSchemaSource");
    expect(appServerAdapter).toContain("build_agent_turn_context");
    expect(appServerAdapter).toContain("AgentTurnContextConfigurationRequest");
    expect(imagePresentationAdapter).toContain("insert_agent_turn_metadata");
    expect(imagePresentationAdapter).toContain(
      "set_agent_turn_user_visible_input_text",
    );
    expect(
      offenders,
      "Agent TurnContextOverride / TurnOutputSchemaSource 属于 lime-agent turn_context_configuration 边界；App Server 只能准备投影数据并调用 façade",
    ).toEqual([]);
  });

  it("workspace patch host tool plan 和 evidence 语义应下沉到 lime-agent", () => {
    const appServerAdapter = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/workspace_patch_host_tools.rs",
      ),
    );
    const agentBoundary = readFileSync(
      join(
        REPO_ROOT,
        "lime-rs/crates/agent/src/agent_tools/workspace_patch_host.rs",
      ),
      "utf8",
    );

    expect(agentBoundary).toContain("WorkspacePatchHostToolPlan");
    expect(agentBoundary).toContain("hostToolRequests");
    expect(agentBoundary).toContain("searchRequests");
    expect(agentBoundary).toContain(
      "update_workspace_patch_with_host_tool_evidence",
    );
    expect(agentBoundary).toContain("execute_workspace_patch_host_tool_plan");
    expect(agentBoundary).toContain("execute_call(");
    expect(agentBoundary).toContain("RuntimeToolExecutorHandle");
    expect(agentBoundary).not.toContain("execute_planned_tool_batch");
    expect(agentBoundary).not.toContain("ToolExecutionBatchInput");
    expect(agentBoundary).not.toContain("agent.tool_registry().clone()");
    expect(
      existsSync(
        join(
          REPO_ROOT,
          "lime-rs/crates/agent/src/agent_tools/workspace_patch_runtime_adapter.rs",
        ),
      ),
      "已删除的 workspace_patch_runtime_adapter 不得恢复；RuntimeTool execute_call 留在 workspace_patch_host current owner",
    ).toBe(false);
    expect(appServerAdapter).toContain(
      "WorkspacePatchHostToolPlan::from_patch",
    );
    expect(appServerAdapter).toContain(
      "update_workspace_patch_with_host_tool_evidence",
    );
    expect(appServerAdapter).not.toContain("ToolExecutionOutcome");
    expect(appServerAdapter).not.toContain("hostToolRequests");
    expect(appServerAdapter).not.toContain("searchRequests");
    expect(appServerAdapter).not.toContain("hostToolEvidence");

    const executionAdapter = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/workspace_patch_host_execution.rs",
      ),
    );
    expect(executionAdapter).toContain(
      "execute_workspace_patch_host_tool_plan",
    );
    expect(executionAdapter).not.toContain("host_tool_plan.planned_tools()");
    expect(executionAdapter).not.toContain("execute_planned_tool_batch");
    expect(executionAdapter).not.toContain("tool_registry()");
    expect(executionAdapter).not.toContain("PlannedToolExecution {");
  });

  it("App Server 受控文本生成 adapter 不应重新复制 Agent streaming loop", () => {
    const pluginWorkerAdapter = {
      path: "lime-rs/crates/app-server/src/runtime_backend/plugin_worker_generation.rs",
      source: readFileSync(
        join(
          REPO_ROOT,
          "lime-rs/crates/app-server/src/runtime_backend/plugin_worker_generation.rs",
        ),
        "utf8",
      ),
    };
    const imagePresentationAdapter = {
      path: "lime-rs/crates/app-server/src/runtime_backend/image_command/presentation.rs",
      source: readFileSync(
        join(
          REPO_ROOT,
          "lime-rs/crates/app-server/src/runtime_backend/image_command/presentation.rs",
        ),
        "utf8",
      ),
    };
    const agentHostManagedGeneration = readFileSync(
      join(REPO_ROOT, "lime-rs/crates/agent/src/host_managed_generation.rs"),
      "utf8",
    );
    const agentDirectTextGeneration = readFileSync(
      join(REPO_ROOT, "lime-rs/crates/agent/src/direct_text_generation.rs"),
      "utf8",
    );
    const adapters = [pluginWorkerAdapter, imagePresentationAdapter].map(
      ({ path, source }) => ({
        path,
        source,
      }),
    );
    const forbiddenSnippets = [
      "stream_reply_with_policy",
      "resolve_request_tool_policy_with_mode",
      "SessionConfigBuilder",
      "RuntimeAgentEvent",
      "TextDeltaBatch",
      "configure_provider_for_route(",
    ];
    const pluginWorkerForbiddenSnippets = [
      "run_direct_text_generation",
      "DirectTextGenerationRequest",
      "fn generation_system_prompt",
      "fn generation_user_prompt",
      "MAX_GENERATION_REQUESTS",
      "MAX_GENERATED_CHARS",
    ].filter((snippet) => pluginWorkerAdapter.source.includes(snippet));

    const offenders = adapters.flatMap(({ path, source }) =>
      forbiddenSnippets
        .filter((snippet) => source.includes(snippet))
        .map((snippet) => ({ path, snippet })),
    );

    expect(agentHostManagedGeneration).toContain("run_direct_text_generation");
    expect(agentHostManagedGeneration).toContain("DirectTextGenerationRequest");
    expect(agentHostManagedGeneration).toContain("HostManagedGenerationPlan");
    expect(agentHostManagedGeneration).toContain("provider_configuration");
    expect(agentDirectTextGeneration).toContain(
      "ModelRouteProviderConfiguration",
    );
    expect(agentDirectTextGeneration).toContain(
      "configure_model_route_provider_for_session",
    );
    expect(pluginWorkerAdapter.source).toContain("run_host_managed_generation");
    expect(pluginWorkerAdapter.source).toContain("HostManagedGenerationPlan");
    expect(pluginWorkerAdapter.source).toContain(
      "provider_configuration_from_runtime",
    );
    expect(pluginWorkerForbiddenSnippets).toEqual([]);
    expect(imagePresentationAdapter.source).toContain(
      "run_direct_text_generation",
    );
    expect(imagePresentationAdapter.source).toContain(
      "DirectTextGenerationRequest",
    );
    expect(imagePresentationAdapter.source).toContain(
      "provider_configuration_from_runtime",
    );
    expect(
      offenders,
      "plugin worker 只能调用 lime-agent host_managed_generation，image presentation 只能调用 lime-agent direct_text_generation；App Server adapter 不得重新承接禁用工具的模型 streaming loop",
    ).toEqual([]);
  });

  it("App Server native tool adapter 不应重新实现 Agent Tool surface", () => {
    const nativeToolAdapter = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/native_tools.rs",
      ),
    );
    const adapters = [
      {
        path: "lime-rs/crates/app-server/src/runtime_backend/image_tools.rs",
        requiredSnippets: ["image_task_gateway", "ImageTaskGateway"],
      },
      {
        path: "lime-rs/crates/app-server/src/runtime_backend/memory_tools.rs",
        requiredSnippets: ["memory_store_gateway", "MemoryStoreGateway"],
      },
    ].map(({ path, requiredSnippets }) => ({
      path,
      requiredSnippets,
      source: productionSource(join(REPO_ROOT, path)),
    }));
    const forbiddenSnippets = [
      "impl Tool for",
      "ToolContext",
      "ToolError",
      "PermissionCheckResult",
      "ToolOptions",
      "fn input_schema",
    ];

    const offenders = adapters.flatMap(({ path, source }) =>
      forbiddenSnippets
        .filter((snippet) => source.includes(snippet))
        .map((snippet) => ({ path, snippet })),
    );
    const missingRequired = adapters.flatMap(
      ({ path, source, requiredSnippets }) =>
        requiredSnippets
          .filter((snippet) => !source.includes(snippet))
          .map((snippet) => ({ path, snippet })),
    );

    expect(
      missingRequired,
      "App Server native tool adapter 必须委托 lime-agent native_tools gateway，不能恢复本地 Tool 实现",
    ).toEqual([]);
    expect(
      offenders,
      "image/memory native tool 的 schema、权限检查和 ToolResult 拼装属于 lime-agent；App Server 只能注入 AppDataSource gateway",
    ).toEqual([]);
    expect(nativeToolAdapter).toContain("register_memory_store_tools");
    expect(nativeToolAdapter).toContain("register_image_task_tools");
    expect(nativeToolAdapter).not.toContain("create_memory_tools");
    expect(nativeToolAdapter).not.toContain("create_image_tools");
  });

  it("App Server tool inventory 不应直接读取 Agent tool registry", () => {
    const appServerAdapter = productionSource(
      join(
        REPO_ROOT,
        "lime-rs/crates/app-server/src/runtime_backend/tool_inventory.rs",
      ),
    );
    const agentBoundary = readFileSync(
      join(
        REPO_ROOT,
        "lime-rs/crates/agent/src/agent_tools/tool_inventory_runtime_snapshot.rs",
      ),
      "utf8",
    );
    const forbiddenSnippets = [
      "tool_registry()",
      "get_extension_configs()",
      ".list_tools(",
      "ExtensionConfig",
      "ExtensionToolInventorySeed",
      "build_mcp_extension_surface",
    ].filter((snippet) => appServerAdapter.includes(snippet));

    expect(agentBoundary).toContain("read_agent_tool_inventory");
    expect(agentBoundary).not.toContain(
      "pub struct AgentToolInventoryRuntimeSnapshot",
    );
    expect(agentBoundary).not.toContain(
      "pub async fn read_agent_tool_inventory_runtime_snapshot",
    );
    expect(appServerAdapter).toContain(
      "use lime_agent::agent_tools::{read_agent_tool_inventory, AgentToolInventoryReadInput};",
    );
    expect(appServerAdapter).toContain("read_agent_tool_inventory");
    expect(appServerAdapter).toContain("AgentToolInventoryReadInput");
    expect(appServerAdapter).not.toContain("build_tool_inventory");
    expect(appServerAdapter).not.toContain("AgentToolInventoryBuildInput");
    expect(
      forbiddenSnippets,
      "Agent tool registry / extension snapshot 语义属于 lime-agent agent_tools::inventory；App Server tool_inventory 只能合并 AppDataSource MCP snapshot 并投影 read-model",
    ).toEqual([]);
  });

  it("App Server execution process 不应直接注册 Agent shell tools", () => {
    const appServerExecutionProcess = productionSource(
      join(REPO_ROOT, "lime-rs/crates/app-server/src/execution_process.rs"),
    );
    const toolRuntimeShellPermission = readFileSync(
      join(REPO_ROOT, "lime-rs/crates/tool-runtime/src/shell_permission.rs"),
      "utf8",
    );
    const forbiddenSnippets = [
      "BashTool",
      "PowerShellTool",
      "ToolRegistry",
      "ToolContext",
      "check_tool_permissions",
      "agent::tools",
    ].filter((snippet) => appServerExecutionProcess.includes(snippet));

    expect(toolRuntimeShellPermission).toContain(
      "check_shell_command_permission",
    );
    expect(toolRuntimeShellPermission).toContain(
      "check_bash_command_permission",
    );
    expect(toolRuntimeShellPermission).toContain(
      "check_powershell_command_permission",
    );
    expect(appServerExecutionProcess).toContain(
      "check_shell_command_permission",
    );
    expect(appServerExecutionProcess).toContain("decide_tool_execution");
    expect(appServerExecutionProcess).not.toContain(
      "check_shell_tool_permissions",
    );
    expect(
      existsSync(
        join(
          REPO_ROOT,
          "lime-rs/crates/agent/src/agent_tools/tool_orchestrator.rs",
        ),
      ),
      "已删除的 Agent shell tool_orchestrator 不得恢复；权限和决策统一归 tool-runtime",
    ).toBe(false);
    expect(
      forbiddenSnippets,
      "shell permission 与 execution decision 属于 tool-runtime；App Server execution_process 只能做 process control / read-model 投影和委托预检",
    ).toEqual([]);
  });
});
