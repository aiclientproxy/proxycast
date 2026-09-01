import { describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  isElectronHostCommandAvailable: vi.fn(),
}));

vi.mock("@/lib/electron-host", () => ({
  isElectronHostCommandAvailable: mocks.isElectronHostCommandAvailable,
}));

import {
  areOptionalLegacyUxCommandsAvailable,
  isBridgeTruthCommand,
  isBridgeTruthEvent,
  isOptionalLegacyUxCommand,
  isOptionalLegacyUxCommandAvailable,
  resolveDevBridgeCommandTimeoutProfile,
  shouldDisallowMockFallbackCommand,
  shouldBypassDevBridgeCooldown,
  shouldRetryDevBridgeReadCommand,
} from "./commandPolicy";

describe("commandPolicy", () => {
  it("集中声明必须走真实桥接的前后端 truth 命令", () => {
    expect(isBridgeTruthCommand("app_server_handle_json_lines")).toBe(true);
    expect(isBridgeTruthCommand("agent_runtime_submit_turn")).toBe(false);
    expect(isBridgeTruthCommand("workspace_list")).toBe(true);
    for (const command of [
      "get_model_registry",
      "get_model_registry_provider_ids",
      "get_models_for_provider",
      "get_models_by_tier",
      "get_provider_alias_config",
      "get_all_alias_configs",
      "refresh_model_registry",
    ]) {
      expect(isBridgeTruthCommand(command)).toBe(false);
      expect(shouldDisallowMockFallbackCommand(command)).toBe(false);
    }
    expect(isBridgeTruthCommand("plugin_list_installed")).toBe(false);
    expect(isBridgeTruthCommand("plugin_launch_shell")).toBe(false);
    expect(isBridgeTruthCommand("knowledge_list_packs")).toBe(false);
    expect(isBridgeTruthCommand("get_automation_jobs")).toBe(false);
    expect(isBridgeTruthCommand("project_memory_get")).toBe(false);
    for (const command of [
      "save_layered_design_project_export",
      "read_layered_design_project_export",
      "recognize_layered_design_text",
      "analyze_layered_design_flat_image",
    ]) {
      expect(isBridgeTruthCommand(command)).toBe(false);
      expect(shouldDisallowMockFallbackCommand(command)).toBe(true);
    }
    expect(shouldDisallowMockFallbackCommand("plugin_select_directory")).toBe(
      false,
    );
    expect(isBridgeTruthCommand("get_local_skills_for_app")).toBe(false);
    expect(shouldDisallowMockFallbackCommand("get_local_skills_for_app")).toBe(
      false,
    );
    expect(
      isBridgeTruthCommand("take_pending_skill_package_open_requests"),
    ).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand(
        "take_pending_skill_package_open_requests",
      ),
    ).toBe(false);
    expect(
      isBridgeTruthCommand("get_skill_package_file_association_status"),
    ).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand(
        "get_skill_package_file_association_status",
      ),
    ).toBe(false);
    expect(
      isBridgeTruthCommand("set_skill_package_file_association_default"),
    ).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand(
        "set_skill_package_file_association_default",
      ),
    ).toBe(false);
    expect(isBridgeTruthCommand("agent_generate_title")).toBe(false);
    expect(isBridgeTruthCommand("get_hint_routes")).toBe(false);
    expect(isBridgeTruthCommand("agent_start_process")).toBe(false);
    expect(isBridgeTruthCommand("agent_stop_process")).toBe(false);
    expect(isBridgeTruthCommand("agent_get_process_status")).toBe(false);
    expect(isBridgeTruthCommand("agent_status")).toBe(false);
    expect(isBridgeTruthCommand("agent_configure_provider")).toBe(false);
    expect(isBridgeTruthCommand("agent_reset")).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_interrupt_turn")).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_respond_action")).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_get_thread_read")).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_export_evidence_pack")).toBe(
      false,
    );
    expect(
      isBridgeTruthCommand("agent_runtime_list_workspace_skill_bindings"),
    ).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand(
        "agent_runtime_list_workspace_skill_bindings",
      ),
    ).toBe(false);
    expect(isBridgeTruthCommand("workspace_ensure")).toBe(true);
    expect(shouldDisallowMockFallbackCommand("workspace_ensure")).toBe(true);
    for (const command of [
      "plugin_runtime_start_task",
      "plugin_runtime_cancel_task",
      "plugin_runtime_get_task",
      "plugin_runtime_submit_host_response",
    ]) {
      expect(isBridgeTruthCommand(command)).toBe(false);
      expect(shouldDisallowMockFallbackCommand(command)).toBe(false);
    }
    expect(isBridgeTruthCommand("create_image_generation_task_artifact")).toBe(
      false,
    );
    expect(shouldDisallowMockFallbackCommand("get_media_task_artifact")).toBe(
      false,
    );
    expect(isBridgeTruthCommand("agent_runtime_delete_session")).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand("agent_runtime_delete_session"),
    ).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_compact_session")).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_get_objective")).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand("agent_runtime_get_objective"),
    ).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand("agent_runtime_set_objective"),
    ).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand(
        "agent_runtime_update_objective_status",
      ),
    ).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand("agent_runtime_clear_objective"),
    ).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_list_file_checkpoints")).toBe(
      false,
    );
    expect(
      shouldDisallowMockFallbackCommand("agent_runtime_list_file_checkpoints"),
    ).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_export_handoff_bundle")).toBe(
      false,
    );
    expect(
      shouldDisallowMockFallbackCommand("agent_runtime_export_handoff_bundle"),
    ).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_spawn_subagent")).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_send_subagent_input")).toBe(
      false,
    );
    expect(isBridgeTruthCommand("agent_runtime_wait_subagents")).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_resume_subagent")).toBe(false);
    expect(isBridgeTruthCommand("agent_runtime_close_subagent")).toBe(false);
    expect(isBridgeTruthCommand("voice_models_download")).toBe(false);
    expect(shouldDisallowMockFallbackCommand("voice_models_download")).toBe(
      true,
    );
    expect(isBridgeTruthCommand("voice_models_delete")).toBe(false);
    expect(shouldDisallowMockFallbackCommand("voice_models_delete")).toBe(true);
    expect(isBridgeTruthCommand("voice_models_test_transcribe_file")).toBe(
      false,
    );
    expect(
      shouldDisallowMockFallbackCommand("voice_models_test_transcribe_file"),
    ).toBe(false);
    expect(isBridgeTruthCommand("voice_models_set_default")).toBe(false);
    expect(shouldDisallowMockFallbackCommand("voice_models_set_default")).toBe(
      false,
    );
    expect(isBridgeTruthCommand("open_system_settings_url")).toBe(false);
    expect(shouldDisallowMockFallbackCommand("open_system_settings_url")).toBe(
      true,
    );
    expect(isBridgeTruthCommand("macos_native_host_invoke")).toBe(false);
    expect(shouldDisallowMockFallbackCommand("macos_native_host_invoke")).toBe(
      true,
    );
    expect(isBridgeTruthCommand("open_file_preview_window")).toBe(false);
    expect(shouldDisallowMockFallbackCommand("open_file_preview_window")).toBe(
      true,
    );
    expect(isBridgeTruthCommand("open_resource_manager_window")).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand("open_resource_manager_window"),
    ).toBe(true);
    expect(isBridgeTruthCommand("windows_native_host_invoke")).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand("windows_native_host_invoke"),
    ).toBe(true);
  });

  it("P6 Session files 旧写读链已退役，不再作为 DevBridge policy surface", () => {
    for (const command of [
      "session_files_get_or_create",
      "session_files_update_meta",
      "session_files_save_file",
      "session_files_read_file",
      "session_files_resolve_file_path",
      "session_files_delete_file",
      "session_files_list_files",
    ]) {
      expect(isBridgeTruthCommand(command)).toBe(false);
      expect(shouldDisallowMockFallbackCommand(command)).toBe(false);
      expect(isOptionalLegacyUxCommand(command)).toBe(false);
    }
  });

  it("P9 process / Agent residual 已退役，不再作为 DevBridge policy surface", () => {
    for (const command of [
      "agent_start_process",
      "agent_stop_process",
      "agent_get_process_status",
      "agent_status",
      "agent_configure_provider",
      "agent_reset",
    ]) {
      expect(isBridgeTruthCommand(command)).toBe(false);
      expect(shouldDisallowMockFallbackCommand(command)).toBe(false);
    }
  });

  it("public subagent 旧 facade 已退役，不再作为 DevBridge policy surface", () => {
    for (const command of [
      "agent_runtime_spawn_subagent",
      "agent_runtime_send_subagent_input",
      "agent_runtime_wait_subagents",
      "agent_runtime_resume_subagent",
      "agent_runtime_close_subagent",
    ]) {
      expect(isBridgeTruthCommand(command)).toBe(false);
      expect(shouldDisallowMockFallbackCommand(command)).toBe(false);
    }
  });

  it("P10 Skill 管理旧 facade 已退役，不再作为 DevBridge policy surface", () => {
    for (const command of [
      "get_skills_for_app",
      "install_skill_for_app",
      "uninstall_skill_for_app",
      "get_skill_repos",
      "add_skill_repo",
      "remove_skill_repo",
      "get_installed_lime_skills",
      "refresh_skill_cache",
      "inspect_local_skill_for_app",
      "create_skill_scaffold_for_app",
      "import_local_skill_for_app",
      "inspect_remote_skill",
    ]) {
      expect(isBridgeTruthCommand(command)).toBe(false);
      expect(shouldDisallowMockFallbackCommand(command)).toBe(false);
    }
  });

  it("queue/session 旧 facade 已由 App Server current method 替换", () => {
    for (const command of [
      "agent_runtime_compact_session",
      "agent_runtime_resume_thread",
      "agent_runtime_promote_queued_turn",
      "agent_runtime_remove_queued_turn",
    ]) {
      expect(isBridgeTruthCommand(command)).toBe(false);
      expect(shouldDisallowMockFallbackCommand(command)).toBe(false);
    }
  });

  it("replay request 旧 facade 已退役，不再作为 DevBridge policy surface", () => {
    expect(isBridgeTruthCommand("agent_runtime_replay_request")).toBe(false);
    expect(
      shouldDisallowMockFallbackCommand("agent_runtime_replay_request"),
    ).toBe(false);
  });

  it("objective continue / audit 旧 facade 已退役，不再作为 DevBridge policy surface", () => {
    for (const command of [
      "agent_runtime_continue_objective",
      "agent_runtime_audit_objective",
    ]) {
      expect(isBridgeTruthCommand(command)).toBe(false);
      expect(shouldDisallowMockFallbackCommand(command)).toBe(false);
    }
  });

  it("集中声明 Claw 旧可选 UX 命令，只在 Electron host 明确支持时调用", () => {
    mocks.isElectronHostCommandAvailable.mockImplementation(
      (command: string) => command === "get_hint_routes",
    );

    expect(isOptionalLegacyUxCommand("get_hint_routes")).toBe(true);
    expect(isOptionalLegacyUxCommand("workspace_list")).toBe(false);
    expect(isOptionalLegacyUxCommandAvailable("get_hint_routes")).toBe(true);
    expect(
      isOptionalLegacyUxCommandAvailable("session_files_get_or_create"),
    ).toBe(false);
    expect(
      areOptionalLegacyUxCommandsAvailable([
        "session_files_get_or_create",
        "session_files_list_files",
      ]),
    ).toBe(false);
  });

  it("集中声明 DevBridge 超时、冷却绕过和读命令重试策略", () => {
    expect(resolveDevBridgeCommandTimeoutProfile("agent_init")).toBe("default");
    expect(
      resolveDevBridgeCommandTimeoutProfile("agent_runtime_get_session"),
    ).toBe("default");
    expect(
      resolveDevBridgeCommandTimeoutProfile("plugin_runtime_get_task"),
    ).toBe("default");
    expect(
      resolveDevBridgeCommandTimeoutProfile("plugin_start_ui_runtime"),
    ).toBe("default");
    expect(
      resolveDevBridgeCommandTimeoutProfile(
        "save_layered_design_project_export",
      ),
    ).toBe("layered-design-project");
    expect(
      resolveDevBridgeCommandTimeoutProfile(
        "read_layered_design_project_export",
      ),
    ).toBe("layered-design-project");
    expect(
      resolveDevBridgeCommandTimeoutProfile("plugin_select_directory"),
    ).toBe("default");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "read-codex-import-job",
              method: "conversationImport/job/read",
              params: { jobId: "import-job-1" },
            }),
          ],
        },
      }),
    ).toBe("app-server-import");
    expect(resolveDevBridgeCommandTimeoutProfile("execute_skill")).toBe(
      "default",
    );
    expect(resolveDevBridgeCommandTimeoutProfile("agent_generate_title")).toBe(
      "default",
    );
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 1,
              method: "thread/list",
              params: {},
            }),
          ],
        },
      }),
    ).toBe("agent-session-list");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 2,
              method: "turn/start",
              params: {},
            }),
          ],
        },
      }),
    ).toBe("app-server-turn-start");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "automation-run-now",
              method: "scheduledTask/run/start",
              params: { id: "job-1" },
            }),
          ],
        },
      }),
    ).toBe("app-server-long-running");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 6,
              method: "workspace/default/ensure",
              params: {},
            }),
          ],
        },
      }),
    ).toBe("startup-truth");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 7,
              method: "workspace/ensureReady",
              params: { workspaceId: "default" },
            }),
          ],
        },
      }),
    ).toBe("startup-truth");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 3,
              method: "thread/list",
              params: {},
            }),
            JSON.stringify({
              id: 4,
              method: "turn/start",
              params: {},
            }),
          ],
        },
      }),
    ).toBe("app-server-turn-start");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: 5,
              method: "thread/read",
              params: {},
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    for (const method of [
      "thread/items/list",
      "thread/turns/list",
      "futureDomain/futureMethod",
    ]) {
      expect(
        resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
          request: {
            lines: [JSON.stringify({ id: `read-${method}`, method, params: {} })],
          },
        }),
      ).toBe("app-server-read");
    }
    for (const method of [
      "pluginLocalPackage/inspect",
      "pluginLocalPackage/export",
      "pluginInstalled/save",
      "pluginUiRuntime/start",
    ]) {
      expect(
        resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
          request: {
            lines: [
              JSON.stringify({ id: `retired-${method}`, method, params: {} }),
            ],
          },
        }),
      ).toBe("app-server-read");
    }
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "scheduled-task-create",
              method: "scheduledTask/create",
              params: { task: { title: "task" } },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "workflow-cancel",
              method: "workflow/cancel",
              params: {
                sessionId: "session-1",
                workflowRunId: "run-1",
              },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "workflow-read",
              method: "workflow/read",
              params: { sessionId: "session-1" },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    for (const method of [
      "plugin/list",
      "plugin/search",
      "plugin/read",
      "plugin/install",
      "plugin/uninstall",
      "plugin/installed",
      "plugin/enabled/set",
    ]) {
      expect(
        resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
          request: {
            lines: [
              JSON.stringify({ id: `plugin-${method}`, method, params: {} }),
            ],
          },
        }),
      ).toBe("app-server-read");
    }
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "session-files",
              method: "sessionFile/list",
              params: { sessionId: "session-1" },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "project-git-status",
              method: "projectGit/status",
              params: { rootPath: "/workspace" },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "thread-settings-update",
              method: "thread/settings/update",
              params: {
                threadId: "thread-1",
                model: "gpt-5.4",
                modelProvider: "openai",
              },
            }),
            JSON.stringify({
              id: "session-start",
              method: "thread/start",
              params: { workspaceId: "workspace-1" },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "skill-management",
              method: "skillManagement/list",
              params: { app: "lime" },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "channel-status",
              method: "gatewayChannel/status",
              params: { channel: "wechat" },
            }),
            JSON.stringify({
              id: "wechat-accounts",
              method: "wechatChannel/accounts/list",
              params: {},
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "media-task",
              method: "mediaTaskArtifact/list",
              params: { projectRootPath: "/workspace" },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "right-surface-pending",
              method: "workspaceRightSurface/pending/list",
              params: { workspaceId: "workspace-1", limit: 50 },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "mcp-status",
              method: "mcpServerStatus/list",
              params: {},
            }),
            JSON.stringify({
              id: "mcp-start",
              method: "mcpServer/start",
              params: { name: "context7" },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "mcp-server-tool-call",
              method: "mcpServer/tool/call",
              params: {
                threadId: "thread-1",
                server: "context7",
                tool: "query-docs",
                arguments: { libraryId: "/openai/openai-agents-python" },
              },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "knowledge",
              method: "knowledgePack/compile",
              params: {},
            }),
            JSON.stringify({
              id: "models",
              method: "modelProvider/list",
              params: {},
            }),
          ],
        },
      }),
    ).toBe("knowledge-compile");
    for (const method of [
      "modelProvider/testConnection",
      "modelProvider/testChat",
      "modelProvider/fetchModels",
    ]) {
      expect(
        resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
          request: {
            lines: [
              JSON.stringify({
                id: `provider-network-${method}`,
                method,
                params: { providerId: "provider-1" },
              }),
            ],
          },
        }),
      ).toBe("app-server-provider-network");
    }
    for (const method of [
      "modelProvider/create",
      "modelProvider/update",
      "modelProviderKey/create",
      "modelProviderUiState/write",
    ]) {
      expect(
        resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
          request: {
            lines: [
              JSON.stringify({
                id: `provider-write-${method}`,
                method,
                params: {},
              }),
            ],
          },
        }),
      ).toBe("app-server-read");
    }
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "collaboration-mode-list",
              method: "collaborationMode/list",
              params: {},
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "file-write",
              method: "fs/writeFile",
              params: { path: "/tmp/demo.txt", dataBase64: "" },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "voice-model-test",
              method: "voiceModel/testTranscribeFile",
              params: {
                model_id: "sensevoice-small-int8-2024-07-17",
                file_path: "/tmp/interview.wav",
              },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_handle_json_lines", {
        request: {
          lines: [
            JSON.stringify({
              id: "voice-transcription-polish",
              method: "voiceTranscription/polishText",
              params: {
                text: "帮我整理整理这段话",
              },
            }),
          ],
        },
      }),
    ).toBe("app-server-read");
    expect(
      resolveDevBridgeCommandTimeoutProfile("app_server_drain_events", {
        request: { limit: 20 },
      }),
    ).toBe("app-server-read");
    expect(resolveDevBridgeCommandTimeoutProfile("unknown_command")).toBe(
      "default",
    );

    expect(shouldBypassDevBridgeCooldown("agent_runtime_get_session")).toBe(
      false,
    );
    expect(
      shouldBypassDevBridgeCooldown("agent_runtime_send_subagent_input"),
    ).toBe(false);
    expect(shouldRetryDevBridgeReadCommand("agent_runtime_get_session")).toBe(
      false,
    );
    expect(shouldRetryDevBridgeReadCommand("agent_runtime_submit_turn")).toBe(
      false,
    );
  });

  it("集中声明运行时真相事件前缀", () => {
    expect(isBridgeTruthEvent("agent_stream_session-1")).toBe(true);
    expect(isBridgeTruthEvent("agent_subagent_status:session-1")).toBe(false);
    expect(isBridgeTruthEvent("agent_subagent_stream:session-1")).toBe(false);
    expect(isBridgeTruthEvent("embedded-browser-view-state")).toBe(true);
    expect(isBridgeTruthEvent("embedded-browser-view-load-failed")).toBe(true);
    expect(isBridgeTruthEvent("retired-runtime-event")).toBe(false);
  });
});
