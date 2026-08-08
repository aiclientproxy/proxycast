import { describe, expect, it } from "vitest";

import {
  ELECTRON_APP_SERVER_COMMANDS,
  ELECTRON_APP_SERVER_TRUTH_BRIDGE_COMMANDS,
  IPC_DEEP_LINK_GET_CURRENT_CHANNEL,
  IPC_DEEP_LINK_GET_URLS_CHANNEL,
  IPC_DIALOG_OPEN_CHANNEL,
  IPC_DIALOG_SAVE_CHANNEL,
  IPC_EMIT_CHANNEL,
  IPC_GLOBAL_SHORTCUT_COMMAND_CHANNEL,
  IPC_INVOKE_CHANNEL,
  IPC_SHELL_OPEN_CHANNEL,
  IPC_WINDOW_COMMAND_CHANNEL,
  isElectronAppServerCommand,
  isElectronHostCommand,
} from "./ipcChannels";

describe("electron/ipcChannels", () => {
  it("Electron host command 白名单承接本地宿主与 App Server 桥接命令", () => {
    expect(isElectronHostCommand("app_server_handle_json_lines")).toBe(true);
    expect(isElectronHostCommand("agent_runtime_create_session")).toBe(false);
    expect(isElectronHostCommand("agent_runtime_submit_turn")).toBe(false);
    expect(isElectronHostCommand("agent_runtime_interrupt_turn")).toBe(false);
    expect(isElectronHostCommand("agent_runtime_update_session")).toBe(false);
    expect(isElectronHostCommand("agent_runtime_respond_action")).toBe(false);
    expect(isElectronHostCommand("agent_runtime_get_thread_read")).toBe(false);
    expect(isElectronHostCommand("agent_runtime_export_evidence_pack")).toBe(
      false,
    );
    expect(isElectronHostCommand("agent_runtime_get_tool_inventory")).toBe(
      false,
    );
    expect(isElectronHostCommand("agent_runtime_list_sessions")).toBe(false);
    expect(isElectronHostCommand("agent_runtime_get_session")).toBe(false);
    expect(isElectronHostCommand("agent_runtime_delete_session")).toBe(false);
    expect(
      isElectronHostCommand("agent_runtime_list_workspace_skill_bindings"),
    ).toBe(false);
    expect(isElectronHostCommand("fetch_provider_models_auto")).toBe(false);
    expect(isElectronHostCommand("get_provider_ui_state")).toBe(false);
    expect(isElectronHostCommand("get_local_skills_for_app")).toBe(false);
    expect(isElectronHostCommand("get_automation_jobs")).toBe(false);
    expect(isElectronHostCommand("get_automation_scheduler_config")).toBe(
      false,
    );
    expect(isElectronHostCommand("get_automation_status")).toBe(false);
    expect(isElectronHostCommand("get_automation_health")).toBe(false);
    expect(isElectronHostCommand("project_memory_get")).toBe(false);
    for (const command of [
      "get_model_registry",
      "get_model_preferences",
      "get_model_sync_state",
      "get_model_registry_provider_ids",
      "get_models_for_provider",
      "get_models_by_tier",
      "get_provider_alias_config",
      "get_all_alias_configs",
    ]) {
      expect(isElectronHostCommand(command)).toBe(false);
    }
    expect(isElectronHostCommand("get_api_key_providers")).toBe(false);
    expect(isElectronHostCommand("list_dir")).toBe(false);
    expect(isElectronHostCommand("read_file_preview_cmd")).toBe(false);
    expect(isElectronHostCommand("list_executable_skills")).toBe(false);
    expect(isElectronHostCommand("workspace_list")).toBe(true);
    expect(isElectronHostCommand("workspace_ensure")).toBe(true);
    expect(isElectronHostCommand("workspace_get_by_path")).toBe(true);
    expect(isElectronHostCommand("workspace_set_default")).toBe(false);
    expect(isElectronHostCommand("get_config")).toBe(true);
    expect(isElectronHostCommand("get_experimental_config")).toBe(true);
    expect(isElectronHostCommand("save_experimental_config")).toBe(true);
    expect(isElectronHostCommand("get_file_icon_data_url")).toBe(true);
    expect(isElectronHostCommand("get_file_manager_locations")).toBe(true);
    expect(isElectronHostCommand("get_home_dir")).toBe(true);
    expect(isElectronHostCommand("plugin_list_installed")).toBe(false);
    expect(isElectronHostCommand("plugin_select_directory")).toBe(false);
    expect(isElectronHostCommand("plugin_launch_shell")).toBe(false);
    expect(isElectronHostCommand("plugin_runtime_start_task")).toBe(false);
    expect(isElectronHostCommand("get_usage_stats")).toBe(false);
    expect(isElectronHostCommand("get_model_usage_ranking")).toBe(false);
    expect(isElectronHostCommand("get_daily_usage_trends")).toBe(false);
    expect(isElectronHostCommand("get_browser_connector_settings_cmd")).toBe(
      true,
    );
    expect(
      isElectronHostCommand("get_browser_connector_install_status_cmd"),
    ).toBe(true);
    expect(isElectronHostCommand("get_chrome_profile_sessions")).toBe(true);
    expect(isElectronHostCommand("get_chrome_bridge_endpoint_info")).toBe(true);
    expect(isElectronHostCommand("get_chrome_bridge_status")).toBe(true);
    expect(isElectronHostCommand("get_browser_backend_policy")).toBe(true);
    expect(isElectronHostCommand("get_browser_backends_status")).toBe(true);
    expect(isElectronHostCommand("get_voice_input_config")).toBe(false);
    expect(isElectronHostCommand("get_asr_credentials")).toBe(false);
    expect(isElectronHostCommand("list_audio_devices")).toBe(false);
    expect(isElectronHostCommand("get_voice_instructions")).toBe(false);
    expect(isElectronHostCommand("voice_models_delete")).toBe(true);
    expect(isElectronHostCommand("voice_models_download")).toBe(true);
    expect(isElectronHostCommand("voice_models_get_install_state")).toBe(true);
    expect(isElectronHostCommand("get_environment_preview")).toBe(true);
    expect(isElectronHostCommand("unified_memory_stats")).toBe(false);
    expect(isElectronHostCommand("get_mcp_servers")).toBe(false);
    expect(isElectronHostCommand("mcp_list_servers_with_status")).toBe(false);
    expect(isElectronHostCommand("mcp_list_tools")).toBe(false);
    expect(isElectronHostCommand("mcp_list_prompts")).toBe(false);
    expect(isElectronHostCommand("mcp_list_resources")).toBe(false);
    for (const command of [
      "knowledge_list_packs",
      "knowledge_get_pack",
      "knowledge_import_source",
      "knowledge_compile_pack",
      "knowledge_set_default_pack",
      "knowledge_update_pack_status",
      "knowledge_resolve_context",
      "knowledge_validate_context_run",
    ]) {
      expect(isElectronHostCommand(command)).toBe(false);
    }
    expect(isElectronHostCommand("site_get_adapter_catalog_status")).toBe(
      false,
    );
    expect(isElectronHostCommand("site_list_adapters")).toBe(false);
    expect(isElectronHostCommand("open_external_url")).toBe(true);
    expect(isElectronHostCommand("open_file_preview_window")).toBe(true);
    expect(isElectronHostCommand("open_resource_manager_window")).toBe(true);
    expect(isElectronHostCommand("show_desktop_notification")).toBe(true);
    expect(isElectronHostCommand("embedded_browser_view_mount")).toBe(true);
    expect(isElectronHostCommand("embedded_browser_view_set_bounds")).toBe(
      true,
    );
    expect(isElectronHostCommand("embedded_browser_view_navigate")).toBe(true);
    expect(isElectronHostCommand("embedded_browser_view_load_html")).toBe(true);
    expect(isElectronHostCommand("embedded_browser_view_reload")).toBe(true);
    expect(isElectronHostCommand("embedded_browser_view_stop")).toBe(true);
    expect(isElectronHostCommand("embedded_browser_view_find_in_page")).toBe(
      true,
    );
    expect(
      isElectronHostCommand("embedded_browser_view_stop_find_in_page"),
    ).toBe(true);
    expect(isElectronHostCommand("embedded_browser_view_set_zoom")).toBe(true);
    expect(isElectronHostCommand("embedded_browser_view_go_back")).toBe(true);
    expect(isElectronHostCommand("embedded_browser_view_go_forward")).toBe(
      true,
    );
    expect(isElectronHostCommand("embedded_browser_view_destroy")).toBe(true);
    expect(isElectronHostCommand("open_system_settings_url")).toBe(true);
    expect(isElectronHostCommand("open_with_default_app")).toBe(true);
    expect(isElectronHostCommand("open_project_path_with_tool")).toBe(true);
    expect(isElectronHostCommand("run_project_shell_command")).toBe(true);
    expect(isElectronHostCommand("project_shell_session_start")).toBe(true);
    expect(isElectronHostCommand("project_shell_session_write")).toBe(true);
    expect(isElectronHostCommand("project_shell_session_resize")).toBe(true);
    expect(isElectronHostCommand("project_shell_session_kill")).toBe(true);
    expect(isElectronHostCommand("reveal_in_finder")).toBe(true);
    expect(isElectronHostCommand("save_exported_document")).toBe(true);
    expect(isElectronHostCommand("save_layered_design_project_export")).toBe(
      true,
    );
    expect(isElectronHostCommand("read_layered_design_project_export")).toBe(
      true,
    );
    expect(isElectronHostCommand("recognize_layered_design_text")).toBe(true);
    expect(isElectronHostCommand("analyze_layered_design_flat_image")).toBe(
      true,
    );
    expect(isElectronHostCommand("start_oem_cloud_oauth_callback_bridge")).toBe(
      true,
    );
    expect(
      isElectronHostCommand("get_skill_package_file_association_status"),
    ).toBe(false);
    expect(
      isElectronHostCommand("set_skill_package_file_association_default"),
    ).toBe(false);
    expect(isElectronHostCommand("report_frontend_crash")).toBe(true);
    expect(isElectronHostCommand("sync_tray_model_shortcuts")).toBe(true);
    expect(
      isElectronHostCommand("take_pending_skill_package_open_requests"),
    ).toBe(true);
    expect(isElectronHostCommand("open_update_window")).toBe(true);
  });

  it("App Server 原始桥接命令只包含 JSONL sidecar 通道", () => {
    expect([...ELECTRON_APP_SERVER_COMMANDS].sort()).toEqual([
      "app_server_drain_events",
      "app_server_handle_json_lines",
    ]);
    expect(isElectronAppServerCommand("app_server_handle_json_lines")).toBe(
      true,
    );
    expect(isElectronAppServerCommand("agent_runtime_list_sessions")).toBe(
      false,
    );
    expect(isElectronAppServerCommand("sync_tray_model_shortcuts")).toBe(false);
    expect(
      isElectronAppServerCommand("take_pending_skill_package_open_requests"),
    ).toBe(false);
  });

  it("App Server truth bridge 命令由 Desktop Host 投影，不走原始 JSONL 通道", () => {
    expect([...ELECTRON_APP_SERVER_TRUTH_BRIDGE_COMMANDS].sort()).toEqual([
      "get_default_provider",
      "get_runtime_provider_selection",
      "workspace_ensure",
      "workspace_ensure_default_ready",
      "workspace_ensure_ready",
      "workspace_get",
      "workspace_get_by_path",
      "workspace_get_default",
      "workspace_get_projects_root",
      "workspace_list",
      "workspace_resolve_project_path",
    ]);
    expect(
      ELECTRON_APP_SERVER_TRUTH_BRIDGE_COMMANDS.every((command) =>
        isElectronHostCommand(command),
      ),
    ).toBe(true);
    expect(
      ELECTRON_APP_SERVER_TRUTH_BRIDGE_COMMANDS.some((command) =>
        isElectronAppServerCommand(command),
      ),
    ).toBe(false);
    expect(
      ELECTRON_APP_SERVER_TRUTH_BRIDGE_COMMANDS.includes(
        "start_oem_cloud_oauth_callback_bridge" as never,
      ),
    ).toBe(false);
    expect(
      ELECTRON_APP_SERVER_TRUTH_BRIDGE_COMMANDS.includes(
        "get_file_icon_data_url" as never,
      ),
    ).toBe(false);
    expect(
      ELECTRON_APP_SERVER_TRUTH_BRIDGE_COMMANDS.includes(
        "get_file_manager_locations" as never,
      ),
    ).toBe(false);
    expect(
      ELECTRON_APP_SERVER_TRUTH_BRIDGE_COMMANDS.includes(
        "get_home_dir" as never,
      ),
    ).toBe(false);
  });

  it("IPC channel 命名保持 app 前缀，避免裸 ipcRenderer 通道扩散", () => {
    const channels = [
      IPC_INVOKE_CHANNEL,
      IPC_EMIT_CHANNEL,
      IPC_DIALOG_OPEN_CHANNEL,
      IPC_DIALOG_SAVE_CHANNEL,
      IPC_SHELL_OPEN_CHANNEL,
      IPC_WINDOW_COMMAND_CHANNEL,
      IPC_GLOBAL_SHORTCUT_COMMAND_CHANNEL,
      IPC_DEEP_LINK_GET_URLS_CHANNEL,
      IPC_DEEP_LINK_GET_CURRENT_CHANNEL,
    ];

    expect(channels).toHaveLength(new Set(channels).size);
    expect(channels.every((channel) => channel.startsWith("app:"))).toBe(true);
  });
});
