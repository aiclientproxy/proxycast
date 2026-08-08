#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";

const repoRoot = process.cwd();

const sourceRoots = ["src"];
const productionRuntimeRoots = ["src", "electron", "packages"];
const sourceExtensions = new Set([".ts", ".tsx", ".js", ".jsx"]);
const ignoredDirectories = new Set([
  ".git",
  ".idea",
  ".vscode",
  "coverage",
  "dist",
  "docs",
  "node_modules",
  "target",
]);

const frontendCommandPatterns = [
  /\bsafeInvoke(?:<[^>]+>)?\s*\(\s*["'`]([^"'`]+)["'`]/g,
  /\binvoke(?:<[^>]+>)?\s*\(\s*["'`]([^"'`]+)["'`]/g,
  /\binvokeAgentRuntimeBridge(?:<[^>]+>)?\s*\(\s*["'`]([^"'`]+)["'`]/g,
];

const knownDeferredRegistrationReasons = new Map();
const retiredFileBrowserFacadeCommands = new Set([
  "list_dir",
  "read_file_preview_cmd",
  "create_file",
  "create_directory",
  "delete_file",
  "rename_file",
]);
const retiredFileSystemMethods = new Set([
  "fileSystem/listDirectory",
  "fileSystem/readFilePreview",
  "fileSystem/createFile",
  "fileSystem/createDirectory",
  "fileSystem/renameFile",
  "fileSystem/deleteFile",
]);
const retiredAutomationFacadeCommands = new Set([
  "get_automation_scheduler_config",
  "update_automation_scheduler_config",
  "get_automation_status",
  "get_automation_jobs",
  "get_automation_job",
  "create_automation_job",
  "update_automation_job",
  "delete_automation_job",
  "run_automation_job_now",
  "get_automation_health",
  "get_automation_run_history",
  "preview_automation_schedule",
  "validate_automation_schedule",
]);
const retiredApiKeyProviderFacadeCommands = new Set([
  "get_api_key_providers",
  "get_system_provider_catalog",
  "get_api_key_provider",
  "read_api_key_provider_config",
  "add_custom_api_key_provider",
  "create_api_key_provider",
  "update_api_key_provider",
  "delete_custom_api_key_provider",
  "delete_api_key_provider",
  "add_api_key",
  "create_api_key_provider_key",
  "delete_api_key",
  "delete_api_key_provider_key",
  "toggle_api_key",
  "update_api_key_alias",
  "update_api_key_provider_key",
  "get_next_api_key",
  "next_api_key_provider_key",
  "record_api_key_usage",
  "record_api_key_provider_key_usage",
  "record_api_key_error",
  "record_api_key_provider_key_error",
  "get_provider_ui_state",
  "read_api_key_provider_ui_state",
  "set_provider_ui_state",
  "write_api_key_provider_ui_state",
  "update_provider_sort_orders",
  "update_api_key_provider_sort_orders",
  "export_api_key_providers",
  "export_api_key_provider_config",
  "import_api_key_providers",
  "import_api_key_provider_config",
  "test_api_key_provider_connection",
  "test_api_key_provider_chat",
  "fetch_provider_models_auto",
  "fetch_provider_models_from_api",
]);
const retiredModelRegistryFacadeCommands = new Set([
  "get_model_registry",
  "get_model_registry_provider_ids",
  "refresh_model_registry",
  "search_models",
  "get_model_preferences",
  "toggle_model_favorite",
  "hide_model",
  "record_model_usage",
  "get_model_sync_state",
  "get_models_for_provider",
  "get_models_by_tier",
  "get_provider_alias_config",
  "get_all_alias_configs",
]);
const retiredPluginPackageFacadeCommands = new Set([
  "plugin_fetch_cloud_package",
  "plugin_inspect_local_package",
  "plugin_list_installed",
  "plugin_save_installed_state",
  "plugin_set_disabled",
  "plugin_uninstall",
  "plugin_uninstall_rehearsal",
]);
const retiredUsageStatsElectronFacadeCommands = new Set([
  "get_usage_stats",
  "get_model_usage_ranking",
  "get_daily_usage_trends",
]);
const retiredLogFacadeCommands = new Set([
  "get_logs",
  "get_persisted_logs_tail",
  "clear_logs",
  "clear_diagnostic_log_history",
  "get_log_storage_diagnostics",
  "export_support_bundle",
  "get_server_diagnostics",
  "get_windows_startup_diagnostics",
]);
const retiredLogTauriGenerateHandlerCommands = new Set(
  retiredLogFacadeCommands,
);
const retiredFrontendDiagnosticsTauriGenerateHandlerCommands = new Set([
  "report_frontend_crash",
  "report_frontend_debug_log",
]);
const retiredConfigTauriGenerateHandlerCommands = new Set([
  "get_config",
  "save_config",
  "get_environment_preview",
  "get_default_provider",
  "set_default_provider",
  "update_provider_env_vars",
]);
const retiredChannelReadOnlyFacadeCommands = new Set([
  "gateway_channel_status",
  "wechat_channel_list_accounts",
]);
const retiredChannelSideEffectFacadeCommands = new Set([
  "gateway_channel_start",
  "gateway_channel_stop",
  "telegram_channel_probe",
  "feishu_channel_probe",
  "discord_channel_probe",
  "wechat_channel_probe",
  "wechat_channel_login_start",
  "wechat_channel_login_wait",
  "wechat_channel_remove_account",
  "wechat_channel_set_runtime_model",
]);
const retiredGatewayTunnelFacadeCommands = new Set([
  "gateway_tunnel_probe",
  "gateway_tunnel_detect_cloudflared",
  "gateway_tunnel_install_cloudflared",
  "gateway_tunnel_create",
  "gateway_tunnel_start",
  "gateway_tunnel_stop",
  "gateway_tunnel_restart",
  "gateway_tunnel_status",
  "gateway_tunnel_sync_webhook_url",
]);
const retiredVoiceInputConfigElectronFacadeCommands = new Set([
  "get_voice_input_config",
  "save_voice_input_config",
]);
const retiredVoiceAudioDeviceFacadeCommands = new Set(["list_audio_devices"]);
const retiredVoiceAsrCredentialFacadeCommands = new Set([
  "get_asr_credentials",
  "add_asr_credential",
  "update_asr_credential",
  "delete_asr_credential",
  "set_default_asr_credential",
  "test_asr_credential",
]);
const retiredVoiceInstructionFacadeCommands = new Set([
  "get_voice_instructions",
  "save_voice_instruction",
  "delete_voice_instruction",
]);
const retiredVoiceRealtimeFacadeCommands = new Set([
  "transcribe_audio",
  "polish_voice_text",
  "output_voice_text",
  "start_recording",
  "stop_recording",
  "get_recording_snapshot",
  "get_recording_segment",
  "cancel_recording",
  "get_recording_status",
]);
const retiredPluginShellCommands = new Set([
  "plugin_launch_shell",
  "plugin_select_directory",
  "plugin_get_ui_runtime_status",
  "plugin_start_ui_runtime",
  "plugin_stop_ui_runtime",
]);
const retiredPluginRuntimeCommands = new Set([
  "plugin_runtime_start_task",
  "plugin_runtime_cancel_task",
  "plugin_runtime_get_task",
  "plugin_runtime_submit_host_response",
]);
const currentTrayDesktopHostShellCommands = new Set([
  "sync_tray_model_shortcuts",
]);
const currentHotkeyDesktopHostShellCommands = new Set([]);
const currentVoiceModelDesktopHostReadCommands = new Set([
  "voice_models_get_install_state",
  "voice_models_list_catalog",
]);
const currentVoiceModelDesktopHostSideEffectCommands = new Set([
  "voice_models_delete",
  "voice_models_download",
]);
const retiredVoiceModelDefaultFacadeCommands = new Set([
  "voice_models_set_default",
]);
const retiredVoiceModelTestTranscribeFacadeCommands = new Set([
  "voice_models_test_transcribe_file",
]);
const currentSkillDesktopHostShellCommands = new Set([
  "take_pending_skill_package_open_requests",
]);
const currentLayeredDesignDesktopHostShellCommands = new Set([
  "save_layered_design_project_export",
  "read_layered_design_project_export",
  "recognize_layered_design_text",
  "analyze_layered_design_flat_image",
]);
const retiredAgentRuntimeSessionRustCommands = new Set([
  "agent_runtime_create_session",
  "agent_runtime_list_sessions",
  "agent_runtime_get_session",
  "agent_runtime_update_session",
  "agent_runtime_delete_session",
]);
const retiredAgentRuntimeSubagentRustCommands = new Set([
  "agent_runtime_spawn_subagent",
  "agent_runtime_send_subagent_input",
  "agent_runtime_wait_subagents",
  "agent_runtime_resume_subagent",
  "agent_runtime_close_subagent",
]);
const retiredAgentRuntimeCoreCurrentBridgeRustCommands = new Set([
  "agent_runtime_submit_turn",
  "agent_runtime_interrupt_turn",
  "agent_runtime_respond_action",
  "agent_runtime_get_thread_read",
  "agent_runtime_export_evidence_pack",
  "agent_runtime_get_tool_inventory",
  "agent_runtime_list_workspace_skill_bindings",
]);
const retiredAgentRuntimeCheckpointQueueReplayRustCommands = new Set([
  "agent_runtime_list_file_checkpoints",
  "agent_runtime_get_file_checkpoint",
  "agent_runtime_diff_file_checkpoint",
  "agent_runtime_restore_file_checkpoint",
  "agent_runtime_replay_request",
  "agent_runtime_promote_queued_turn",
  "agent_runtime_remove_queued_turn",
]);
const retiredAgentRuntimeCheckpointCurrentCommands = new Set([
  "agent_runtime_list_file_checkpoints",
  "agent_runtime_get_file_checkpoint",
  "agent_runtime_diff_file_checkpoint",
  "agent_runtime_restore_file_checkpoint",
]);
const retiredAgentRuntimeCompactObjectiveRustCommands = new Set([
  "agent_runtime_compact_session",
  "agent_runtime_resume_thread",
  "agent_runtime_get_objective",
  "agent_runtime_set_objective",
  "agent_runtime_update_objective_status",
  "agent_runtime_clear_objective",
  "agent_runtime_continue_objective",
  "agent_runtime_audit_objective",
]);
const retiredAgentRuntimeObjectiveCrudGatewayCommands = new Set([
  "agent_runtime_get_objective",
  "agent_runtime_set_objective",
  "agent_runtime_update_objective_status",
  "agent_runtime_clear_objective",
]);
const retiredAgentRuntimeObjectiveContinuationGatewayCommands = new Set([
  "agent_runtime_continue_objective",
  "agent_runtime_audit_objective",
]);
const retiredAgentSessionObjectiveMethodPrefix = "agentSession/objective/";
const retiredAgentRuntimeProcessRustCommands = new Set([
  "agent_start_process",
  "agent_stop_process",
  "agent_get_process_status",
  "agent_init",
  "agent_status",
  "agent_configure_provider",
  "agent_reset",
]);
const retiredAgentRuntimeExportRustCommands = new Set([
  "agent_runtime_export_analysis_handoff",
  "agent_runtime_export_handoff_bundle",
  "agent_runtime_export_review_decision_template",
  "agent_runtime_save_review_decision",
  "agent_runtime_export_replay_case",
]);
const retiredSkillExecutionFacadeCommands = new Set(["execute_skill"]);
const retiredCapabilityDraftAuthoringCommands = new Set([
  "capability_draft_create",
  "capability_draft_list",
  "capability_draft_get",
  "capability_draft_verify",
  "capability_draft_register",
  "capability_draft_submit_approval_session_inputs",
  "capability_draft_execute_controlled_get",
]);
const retiredSkillManagementFacadeCommands = new Set([
  "get_local_skills_for_app",
  "get_skills",
  "get_skills_for_app",
  "install_skill",
  "install_skill_for_app",
  "uninstall_skill",
  "uninstall_skill_for_app",
  "get_skill_repos",
  "add_skill_repo",
  "remove_skill_repo",
  "refresh_skill_cache",
  "get_installed_lime_skills",
  "inspect_local_skill_for_app",
  "create_skill_scaffold_for_app",
  "import_local_skill_for_app",
  "inspect_remote_skill",
  "get_skill_package_file_association_status",
  "set_skill_package_file_association_default",
]);
const retiredSkillRevealFacadeCommands = new Set([
  "reveal_local_skill_for_app",
]);
const retiredSkillLocalManagementFacadeCommands = new Set([
  "inspect_local_skill_detail_for_app",
  "rename_local_skill_for_app",
  "replace_local_skill_package_for_app",
  "inspect_remote_skill",
]);
const retiredSkillPackageLocalFacadeCommands = new Set([
  "inspect_local_skill_package_for_app",
  "install_local_skill_package_for_app",
  "export_local_skill_package_for_app",
]);
const retiredSkillMarketplaceInstallFacadeCommands = new Set([
  "install_marketplace_skill_for_app",
  "install_skill_from_download_url_for_app",
]);
const retiredSiteAdapterFacadeCommands = new Set([
  "site_list_adapters",
  "site_recommend_adapters",
  "site_search_adapters",
  "site_get_adapter_info",
  "site_get_adapter_launch_readiness",
  "site_get_adapter_catalog_status",
  "site_apply_adapter_catalog_bootstrap",
  "site_clear_adapter_catalog_cache",
  "site_import_adapter_yaml_bundle",
  "site_run_adapter",
  "site_debug_run_adapter",
  "site_save_adapter_result",
]);
const retiredMediaTaskArtifactFacadeCommands = new Set([
  "create_image_generation_task_artifact",
  "create_audio_generation_task_artifact",
  "complete_audio_generation_task_artifact",
  "get_media_task_artifact",
  "list_media_task_artifacts",
  "cancel_media_task_artifact",
]);
const retiredVideoGenerationFacadeCommands = new Set([
  "create_video_generation_task",
  "get_video_generation_task",
  "list_video_generation_tasks",
  "cancel_video_generation_task",
]);
const retiredGalleryMaterialFacadeCommands = new Set([
  "create_gallery_material_metadata",
  "get_gallery_material_metadata",
  "get_gallery_material",
  "list_gallery_materials_by_image_category",
  "list_gallery_materials_by_layout_category",
  "list_gallery_materials_by_mood",
  "update_gallery_material_metadata",
  "delete_gallery_material_metadata",
]);
const retiredProjectMaterialFacadeCommands = new Set([
  "list_materials",
  "get_material",
  "get_material_count",
  "upload_material",
  "import_material_from_url",
  "update_material",
  "delete_material",
  "get_material_content",
]);
const retiredWorkspaceWriteFacadeCommands = new Set([
  "workspace_create",
  "workspace_update",
  "workspace_delete",
  "workspace_set_default",
]);
const retiredContentCrudFacadeCommands = new Set([
  "content_create",
  "content_get",
  "content_get_general_workbench_document_state",
  "content_list",
  "content_update",
  "content_delete",
  "content_reorder",
  "content_stats",
]);
const retiredExecutionRunFacadeCommands = new Set([
  "execution_run_list",
  "execution_run_get",
  "execution_run_get_general_workbench_state",
  "execution_run_list_general_workbench_history",
]);
const retiredFrontendSessionFileFacadeCommands = new Set([
  "session_files_create",
  "session_files_exists",
  "session_files_get_or_create",
  "session_files_delete",
  "session_files_list",
  "session_files_get_detail",
  "session_files_update_meta",
  "session_files_save_file",
  "session_files_read_file",
  "session_files_resolve_file_path",
  "session_files_delete_file",
  "session_files_list_files",
  "session_files_cleanup_expired",
  "session_files_cleanup_empty",
]);
const retiredUnifiedMemoryFacadeCommands = new Set([
  "unified_memory_list",
  "unified_memory_get",
  "unified_memory_create",
  "unified_memory_update",
  "unified_memory_delete",
  "unified_memory_search",
  "unified_memory_stats",
  "unified_memory_analyze",
  "unified_memory_semantic_search",
  "unified_memory_hybrid_search",
  "unified_memory_feedback",
]);
const retiredMemoryRuntimeTauriGenerateHandlerCommands = new Set([
  "memory_runtime_get_stats",
  "memory_runtime_get_overview",
  "memory_runtime_request_analysis",
  "memory_runtime_cleanup",
  "memory_runtime_get_working_memory",
  "memory_runtime_get_extraction_status",
  "memory_runtime_prefetch_for_turn",
  "memory_get_effective_sources",
  "memory_get_auto_index",
  "memory_toggle_auto",
  "memory_update_auto_note",
  "memory_cleanup_memdir",
  "memory_scaffold_memdir",
  "memory_scaffold_runtime_agents_template",
  "memory_ensure_workspace_local_agents_gitignore",
]);
const retiredMemoryCrudTauriGenerateHandlerCommands = new Set([
  "character_create",
  "character_get",
  "character_list",
  "character_update",
  "character_delete",
  "world_building_get",
  "world_building_update",
  "outline_node_create",
  "outline_node_get",
  "outline_node_list",
  "outline_node_update",
  "outline_node_delete",
  "project_memory_get",
]);
const retiredModelRegistryTauriGenerateHandlerCommands = new Set([
  "get_model_registry",
  "get_model_registry_provider_ids",
  "refresh_model_registry",
  "search_models",
  "get_model_preferences",
  "toggle_model_favorite",
  "hide_model",
  "record_model_usage",
  "get_model_sync_state",
  "get_models_for_provider",
  "get_models_by_tier",
  "get_provider_alias_config",
  "get_all_alias_configs",
]);
const retiredMemoryRuntimeCatalogSections = new Set(["memoryRuntimeCommands"]);
const retiredAgentRuntimeRustCommands = new Set([
  ...retiredAgentRuntimeSessionRustCommands,
  ...retiredAgentRuntimeSubagentRustCommands,
  ...retiredAgentRuntimeCoreCurrentBridgeRustCommands,
  ...retiredAgentRuntimeCheckpointQueueReplayRustCommands,
  ...retiredAgentRuntimeCompactObjectiveRustCommands,
  ...retiredAgentRuntimeProcessRustCommands,
  ...retiredAgentRuntimeExportRustCommands,
]);
const currentFileBrowserDesktopHostShellCommands = new Set([
  "get_home_dir",
  "get_file_manager_locations",
  "get_file_icon_data_url",
  "reveal_in_finder",
  "open_with_default_app",
  "save_exported_document",
]);
const retiredTauriGenerateHandlerCommands = new Set([
  ...retiredLogTauriGenerateHandlerCommands,
  ...retiredFrontendDiagnosticsTauriGenerateHandlerCommands,
  ...retiredConfigTauriGenerateHandlerCommands,
  ...retiredModelRegistryFacadeCommands,
  ...retiredGalleryMaterialFacadeCommands,
  ...retiredProjectMaterialFacadeCommands,
  ...retiredVideoGenerationFacadeCommands,
  ...retiredWorkspaceWriteFacadeCommands,
  ...retiredContentCrudFacadeCommands,
  ...retiredExecutionRunFacadeCommands,
  ...retiredUnifiedMemoryFacadeCommands,
  ...retiredMemoryRuntimeTauriGenerateHandlerCommands,
  ...retiredMemoryCrudTauriGenerateHandlerCommands,
  ...retiredModelRegistryTauriGenerateHandlerCommands,
  "unified_memory_feedback",
  "add_mcp_server",
  "add_model_to_provider",
  "add_prompt",
  "add_provider",
  "check_codex_cli_status",
  "create_a2ui_form",
  "auto_import_prompt",
  "clear_request_logs",
  "companion_get_pet_status",
  "companion_launch_pet",
  "companion_send_pet_command",
  "close_webview_panel",
  "create_webview_panel",
  "create_persona",
  "delete_mcp_server",
  "delete_prompt",
  "delete_a2ui_form",
  "delete_avatar",
  "delete_persona",
  "enable_prompt",
  "execute_ecommerce_review_reply",
  "expand_path",
  "export_bundle",
  "export_config",
  "export_config_yaml",
  "get_all_provider_models",
  "get_a2ui_form",
  "get_a2ui_forms_by_message",
  "get_a2ui_forms_by_session",
  "get_auto_launch_status",
  "get_config_dir_path",
  "get_config_paths",
  "get_config_status",
  "get_current_prompt_file_content",
  "get_daily_usage_trends",
  "get_default_persona",
  "get_external_tools",
  "get_experimental_config",
  "get_injection_config",
  "get_injection_rules",
  "get_model_usage_ranking",
  "get_models_config",
  "get_memory_feedback_stats",
  "get_materials_content",
  "get_provider_models",
  "get_persona",
  "get_prompts",
  "get_project_context",
  "get_mcp_servers",
  "get_relay_info",
  "get_request_log_detail",
  "get_request_logs",
  "get_stats_by_model",
  "get_stats_by_provider",
  "get_stats_summary",
  "get_sysinfo",
  "get_token_stats_by_day",
  "get_token_stats_by_model",
  "get_token_stats_by_provider",
  "get_token_summary",
  "get_tool_versions",
  "get_usage_stats",
  "get_available_voices",
  "get_webview_panels",
  "get_websocket_connections",
  "get_websocket_status",
  "handle_deep_link",
  "handle_open_deep_link",
  "import_bundle",
  "import_config",
  "import_document",
  "import_document_to_session",
  "import_mcp_from_app",
  "import_prompt_from_file",
  "get_telegram_remote_status",
  "list_relay_providers",
  "mcp_call_tool",
  "mcp_call_tool_with_caller",
  "mcp_get_prompt",
  "mcp_list_prompts",
  "mcp_list_resources",
  "mcp_list_servers_with_status",
  "mcp_list_tools",
  "mcp_list_tools_for_context",
  "mcp_read_resource",
  "mcp_search_tools",
  "mcp_start_server",
  "mcp_stop_server",
  "navigate_webview_panel",
  "open_auth_dir",
  "open_codex_cli_login",
  "open_codex_cli_logout",
  "open_config_folder",
  "open_external_url",
  "open_system_settings_url",
  "read_image_from_session",
  "refresh_relay_registry",
  "resize_webview_panel",
  "remove_model_from_provider",
  "remove_provider",
  "save_exported_document",
  "save_experimental_config",
  "save_models_config",
  "save_relay_api_key",
  "save_a2ui_form_data",
  "search_pixabay_images",
  "search_web_images",
  "send_connect_callback",
  "set_auto_launch",
  "set_injection_enabled",
  "set_websocket_enabled",
  "start_telegram_remote",
  "start_oem_cloud_oauth_callback_bridge",
  "stop_telegram_remote",
  "submit_a2ui_form",
  "sync_all_mcp_to_live",
  "sync_tray_model_shortcuts",
  "test_tts",
  "toggle_model_enabled",
  "toggle_mcp_server",
  "focus_webview_panel",
  "add_injection_rule",
  "remove_injection_rule",
  "update_prompt",
  "update_injection_rule",
  "update_mcp_server",
  "upload_image_to_session",
  "upsert_prompt",
  "validate_config_yaml",
  "validate_import",
]);
const retiredMcpDesktopFacadeCommands = new Set([
  "add_mcp_server",
  "delete_mcp_server",
  "get_mcp_servers",
  "import_mcp_from_app",
  "mcp_call_tool",
  "mcp_call_tool_with_caller",
  "mcp_get_prompt",
  "mcp_list_prompts",
  "mcp_list_resources",
  "mcp_list_servers_with_status",
  "mcp_list_tools",
  "mcp_list_tools_for_context",
  "mcp_read_resource",
  "mcp_search_tools",
  "mcp_start_server",
  "mcp_stop_server",
  "sync_all_mcp_to_live",
  "toggle_mcp_server",
  "update_mcp_server",
]);
const retiredTauriCommandModules = new Set([
  "a2ui_form_cmd",
  "config_cmd",
  "companion_cmd",
  "connect_cmd",
  "document_import_cmd",
  "ecommerce_review_reply_cmd",
  "experimental_cmd",
  "external_tools_cmd",
  "file_upload_cmd",
  "image_search_cmd",
  "image_upload_cmd",
  "injection_cmd",
  "knowledge_cmd",
  "layered_design_cmd",
  "memory_cmd",
  "memory_feedback_cmd",
  "models_cmd",
  "model_registry_cmd",
  "execution_run_cmd",
  "video_generation_cmd",
  "mcp_cmd",
  "persona_cmd",
  "prompt_cmd",
  "telemetry_cmd",
  "telegram_remote_cmd",
  "theme_context_cmd",
  "tray_cmd",
  "voice_model_cmd",
  "voice_test_cmd",
  "websocket_cmd",
  "windows_startup_cmd",
]);

function addDeferredCommands(commands, reason) {
  for (const command of commands) {
    knownDeferredRegistrationReasons.set(command, reason);
  }
}

const currentElectronHostRequiredCommands = new Set([
  "app_server_handle_json_lines",
  "app_server_drain_events",
  ...currentTrayDesktopHostShellCommands,
  ...currentHotkeyDesktopHostShellCommands,
  ...currentVoiceModelDesktopHostReadCommands,
  ...currentVoiceModelDesktopHostSideEffectCommands,
  ...currentSkillDesktopHostShellCommands,
  ...currentLayeredDesignDesktopHostShellCommands,
  ...currentFileBrowserDesktopHostShellCommands,
  "get_default_provider",
  "get_environment_preview",
  "get_experimental_config",
  "open_external_url",
  "open_system_settings_url",
  "save_experimental_config",
  "start_oem_cloud_oauth_callback_bridge",
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

const currentDevBridgeTruthRequiredCommands = new Set([
  "open_external_url",
  "start_oem_cloud_oauth_callback_bridge",
]);

const electronDiagnosticFacadeCommands = new Set([
  "get_browser_backend_policy",
  "get_browser_backends_status",
  "get_browser_connector_install_status_cmd",
  "get_browser_connector_settings_cmd",
  "get_chrome_bridge_endpoint_info",
  "get_chrome_bridge_status",
  "get_chrome_profile_sessions",
]);

function normalizePath(filePath) {
  return filePath.split(path.sep).join("/");
}

function isRuntimeSource(relativePath) {
  const normalizedPath = normalizePath(relativePath);
  const extension = path.extname(normalizedPath);
  if (!sourceExtensions.has(extension)) {
    return false;
  }
  if (normalizedPath.endsWith(".d.ts")) {
    return false;
  }
  if (
    normalizedPath.includes("/__tests__/") ||
    normalizedPath.includes("/__mocks__/") ||
    /\.test\.[^.]+$/.test(normalizedPath) ||
    /\.spec\.[^.]+$/.test(normalizedPath)
  ) {
    return false;
  }
  return true;
}

function walkDirectory(rootDirectory) {
  const results = [];
  if (!fs.existsSync(rootDirectory)) {
    return results;
  }

  const entries = fs.readdirSync(rootDirectory, { withFileTypes: true });
  for (const entry of entries) {
    if (ignoredDirectories.has(entry.name)) {
      continue;
    }

    const absolutePath = path.join(rootDirectory, entry.name);
    if (entry.isDirectory()) {
      results.push(...walkDirectory(absolutePath));
      continue;
    }

    const relativePath = normalizePath(path.relative(repoRoot, absolutePath));
    if (isRuntimeSource(relativePath)) {
      results.push(relativePath);
    }
  }

  return results;
}

function addUsage(map, command, relativePath) {
  if (!map.has(command)) {
    map.set(command, new Set());
  }
  map.get(command).add(relativePath);
}

function isFrameworkPluginCommand(command) {
  return command.startsWith("plugin:");
}

function extractCommandsFromSource(sourceCode) {
  const commands = new Set();
  for (const pattern of frontendCommandPatterns) {
    for (const match of sourceCode.matchAll(pattern)) {
      const command = match[1];
      if (isFrameworkPluginCommand(command)) {
        continue;
      }
      commands.add(command);
    }
  }
  return commands;
}

function collectFrontendCommandUsage() {
  const commandUsage = new Map();
  for (const root of sourceRoots) {
    const absoluteRoot = path.join(repoRoot, root);
    for (const relativePath of walkDirectory(absoluteRoot)) {
      const sourceCode = readSourceIfExists(relativePath);
      if (sourceCode === null) {
        continue;
      }
      for (const command of extractCommandsFromSource(sourceCode)) {
        addUsage(commandUsage, command, relativePath);
      }
    }
  }
  return commandUsage;
}

function extractBalancedBlock(sourceCode, startIndex, openChar, closeChar) {
  let depth = 0;
  let inSingleQuote = false;
  let inDoubleQuote = false;
  let inTemplateString = false;
  let inLineComment = false;
  let inBlockComment = false;
  let escaped = false;

  for (let index = startIndex; index < sourceCode.length; index += 1) {
    const currentChar = sourceCode[index];
    const nextChar = sourceCode[index + 1];

    if (inLineComment) {
      if (currentChar === "\n") {
        inLineComment = false;
      }
      continue;
    }

    if (inBlockComment) {
      if (currentChar === "*" && nextChar === "/") {
        inBlockComment = false;
        index += 1;
      }
      continue;
    }

    if (inSingleQuote) {
      if (!escaped && currentChar === "'") {
        inSingleQuote = false;
      }
      escaped = !escaped && currentChar === "\\";
      continue;
    }

    if (inDoubleQuote) {
      if (!escaped && currentChar === '"') {
        inDoubleQuote = false;
      }
      escaped = !escaped && currentChar === "\\";
      continue;
    }

    if (inTemplateString) {
      if (!escaped && currentChar === "`") {
        inTemplateString = false;
      }
      escaped = !escaped && currentChar === "\\";
      continue;
    }

    if (currentChar === "/" && nextChar === "/") {
      inLineComment = true;
      index += 1;
      continue;
    }

    if (currentChar === "/" && nextChar === "*") {
      inBlockComment = true;
      index += 1;
      continue;
    }

    if (currentChar === "'") {
      inSingleQuote = true;
      escaped = false;
      continue;
    }

    if (currentChar === '"') {
      inDoubleQuote = true;
      escaped = false;
      continue;
    }

    if (currentChar === "`") {
      inTemplateString = true;
      escaped = false;
      continue;
    }

    if (currentChar === openChar) {
      depth += 1;
      continue;
    }

    if (currentChar === closeChar) {
      depth -= 1;
      if (depth === 0) {
        return sourceCode.slice(startIndex + 1, index);
      }
    }
  }

  throw new Error(`无法提取 ${openChar}${closeChar} 平衡块`);
}

function collectElectronHostCommands() {
  const channelsPath = path.join(repoRoot, "electron/ipcChannels.ts");
  const sourceCode = fs.readFileSync(channelsPath, "utf8");
  const marker = "export const ELECTRON_HOST_COMMANDS = [";
  const markerIndex = sourceCode.indexOf(marker);
  if (markerIndex < 0) {
    throw new Error("未找到 Electron host command 白名单");
  }

  const bracketStart = markerIndex + marker.length - 1;
  const commandBody = extractBalancedBlock(sourceCode, bracketStart, "[", "]");
  const commands = new Set();

  for (const match of commandBody.matchAll(/["'`]([^"'`]+)["'`]/g)) {
    commands.add(match[1]);
  }

  return commands;
}

function collectMockPriorityCommands() {
  const filePath = path.join(
    repoRoot,
    "src/lib/dev-bridge/mockPriorityCommands.ts",
  );
  const sourceCode = fs.readFileSync(filePath, "utf8");
  const match = sourceCode.match(
    /const mockPriorityCommands = new Set<string>\(\[([\s\S]*?)\]\);/,
  );
  if (!match) {
    throw new Error("未找到 mockPriorityCommands 定义");
  }

  const commands = new Set();
  for (const stringMatch of match[1].matchAll(/["'`]([^"'`]+)["'`]/g)) {
    commands.add(stringMatch[1]);
  }
  return commands;
}

function collectBridgeTruthCommands() {
  const filePath = path.join(repoRoot, "src/lib/dev-bridge/commandPolicy.ts");
  const sourceCode = fs.readFileSync(filePath, "utf8");
  const match = sourceCode.match(
    /const bridgeTruthCommands = new Set<string>\(\[([\s\S]*?)\]\);/,
  );
  if (!match) {
    throw new Error("未找到 bridgeTruthCommands 定义");
  }

  const commands = new Set();
  for (const stringMatch of match[1].matchAll(/["'`]([^"'`]+)["'`]/g)) {
    commands.add(stringMatch[1]);
  }
  return commands;
}

function collectNoMockFallbackCompatCommands() {
  const filePath = path.join(repoRoot, "src/lib/dev-bridge/commandPolicy.ts");
  const sourceCode = fs.readFileSync(filePath, "utf8");
  const match = sourceCode.match(
    /const noMockFallbackCompatCommands = new Set<string>\(\[([\s\S]*?)\]\);/,
  );
  if (!match) {
    return new Set();
  }

  const commands = new Set();
  for (const stringMatch of match[1].matchAll(/["'`]([^"'`]+)["'`]/g)) {
    commands.add(stringMatch[1]);
  }
  return commands;
}

function readAgentCommandCatalog() {
  const catalogPath = path.join(
    repoRoot,
    "src/lib/governance/agentCommandCatalog.json",
  );
  return JSON.parse(fs.readFileSync(catalogPath, "utf8"));
}

function sortCommands(commands) {
  return [...commands].sort((left, right) => left.localeCompare(right));
}

function readSource(relativePath) {
  return fs.readFileSync(path.join(repoRoot, relativePath), "utf8");
}

function readSourceIfExists(relativePath) {
  const absolutePath = path.join(repoRoot, relativePath);
  if (!fs.existsSync(absolutePath)) {
    return null;
  }
  try {
    return fs.readFileSync(absolutePath, "utf8");
  } catch (error) {
    if (error?.code === "ENOENT") {
      return null;
    }
    throw error;
  }
}

function readExistingProductionSourceForGuard(relativePath) {
  const absolutePath = path.join(repoRoot, relativePath);
  if (!fs.existsSync(absolutePath)) {
    return "";
  }
  return readProductionSourceForGuard(relativePath);
}

function collectRustSourceFiles(relativeRoot) {
  const absoluteRoot = path.join(repoRoot, relativeRoot);
  return walkDirectory(absoluteRoot)
    .map((absolutePath) => normalizePath(path.relative(repoRoot, absolutePath)))
    .filter((relativePath) => relativePath.endsWith(".rs"))
    .sort();
}

function isAllowedTestMockFixtureSource(relativePath) {
  return (
    relativePath === "src/lib/dev-bridge/explicitMockFallback.ts" ||
    relativePath === "src/lib/desktop-host/core.ts" ||
    relativePath === "src/lib/desktop-host/event.ts" ||
    /^src\/lib\/desktop-host\/[A-Za-z0-9_-]+Mocks\.ts$/.test(relativePath)
  );
}

function printCommandGroup(title, commands, usageMap) {
  console.error(`\n## ${title}`);
  for (const command of sortCommands(commands)) {
    console.error(`- ${command}`);
    if (usageMap?.has(command)) {
      const files = sortCommands(usageMap.get(command));
      for (const file of files) {
        console.error(`  - ${file}`);
      }
    }
  }
}

function addForbiddenSubstringFailures(
  failures,
  relativePath,
  sourceCode,
  rules,
) {
  for (const rule of rules) {
    if (sourceCode.includes(rule.substring)) {
      failures.push({
        file: relativePath,
        message: rule.message,
        token: rule.substring,
      });
    }
  }
}

function addRequiredSubstringFailures(
  failures,
  relativePath,
  sourceCode,
  rules,
) {
  for (const rule of rules) {
    if (!sourceCode.includes(rule.substring)) {
      failures.push({
        file: relativePath,
        message: rule.message,
        token: rule.substring,
      });
    }
  }
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function hasStandaloneIdentifier(sourceCode, identifier) {
  const pattern = new RegExp(
    `(^|[^A-Za-z0-9_])${escapeRegExp(identifier)}([^A-Za-z0-9_]|$)`,
  );
  return pattern.test(sourceCode);
}

function hasTauriCommandRegistration(sourceCode, command) {
  const pattern = new RegExp(
    String.raw`\bcommands::[A-Za-z0-9_]+::${escapeRegExp(command)}\b`,
  );
  return pattern.test(sourceCode);
}

function hasFrontendCommandInvocation(sourceCode, command) {
  const escapedCommand = escapeRegExp(command);
  const pattern = new RegExp(
    String.raw`\b(?:safeInvoke|invoke)(?:<[^>]+>)?\s*\(\s*["'\`]${escapedCommand}["'\`]`,
  );
  return pattern.test(sourceCode);
}

function hasRustFunctionDefinition(sourceCode, functionName) {
  const pattern = new RegExp(
    String.raw`\b(?:pub(?:\s*\([^)]*\))?\s+)?(?:async\s+)?fn\s+${escapeRegExp(functionName)}\s*\(`,
  );
  return pattern.test(sourceCode);
}

function agentCommandCatalogRuntimeSurfaceHas(command) {
  const catalog = readAgentCommandCatalog();
  return (
    (catalog.runtimeGatewayCommands ?? []).includes(command) ||
    (catalog.capabilityDraftCommands ?? []).includes(command)
  );
}

function stripRustTestModules(sourceCode) {
  return sourceCode.replace(
    /(?:^|\n)\s*#\s*\[\s*cfg\s*\(\s*test\s*\)\s*\]\s*(?:pub\s+)?mod\s+\w+\s*(?:\{[\s\S]*$|;)/m,
    "\n",
  );
}

function readProductionSourceForGuard(relativePath) {
  const absolutePath = path.join(repoRoot, relativePath);
  if (!fs.existsSync(absolutePath)) {
    return "";
  }
  const sourceCode = readSource(relativePath);
  return relativePath.endsWith(".rs")
    ? stripRustTestModules(sourceCode)
    : sourceCode;
}

function extractNamedFunctionBody(sourceCode, marker) {
  const markerIndex = sourceCode.indexOf(marker);
  if (markerIndex < 0) {
    throw new Error(`未找到函数定义: ${marker}`);
  }
  const braceStart = sourceCode.indexOf("{", markerIndex);
  if (braceStart < 0) {
    throw new Error(`未找到函数体: ${marker}`);
  }
  return extractBalancedBlock(sourceCode, braceStart, "{", "}");
}

function collectProductionMockOnlyUsageFailures() {
  const failures = [];
  const forbiddenCallPatterns = [
    {
      pattern: /\binvokeMockOnly\s*\(/,
      token: "invokeMockOnly(",
      message: "生产源码不能调用测试 mock invoke 入口",
    },
    {
      pattern: /\bmockCommand\s*\(/,
      token: "mockCommand(",
      message: "生产源码不能注册 renderer mock command",
    },
    {
      pattern: /\bclearMocks\s*\(/,
      token: "clearMocks(",
      message: "生产源码不能清理测试 mock command",
    },
    {
      pattern: /\binvokeExplicitMock\s*\(/,
      token: "invokeExplicitMock(",
      message: "生产源码不能调用显式 renderer mock fallback",
    },
    {
      pattern: /\blistenExplicitMock\s*\(/,
      token: "listenExplicitMock(",
      message: "生产源码不能调用显式 renderer event mock fallback",
    },
  ];
  const forbiddenMockOnlyImports = [
    "invokeMockOnly",
    "mockCommand",
    "clearMocks",
  ];

  for (const root of productionRuntimeRoots) {
    const absoluteRoot = path.join(repoRoot, root);
    for (const relativePath of walkDirectory(absoluteRoot)) {
      if (isAllowedTestMockFixtureSource(relativePath)) {
        continue;
      }

      const sourceCode = readSourceIfExists(relativePath);
      if (sourceCode === null) {
        continue;
      }
      if (
        /from\s+["'`](?:@\/lib\/dev-bridge\/explicitMockFallback|\.{1,2}\/[^"'`]*explicitMockFallback|\.\/explicitMockFallback)["'`]/.test(
          sourceCode,
        )
      ) {
        failures.push({
          file: relativePath,
          message: "生产源码不能导入显式 renderer mock fallback",
          token: "explicitMockFallback",
        });
      }

      if (
        /from\s+["'`](?:\.\/__tests__\/testFixtures|\.\/testFixtures|\.{1,2}\/[^"'`]*\/testFixtures)["'`]/.test(
          sourceCode,
        )
      ) {
        failures.push({
          file: relativePath,
          message: "生产源码不能导入 Plugin mock SDK 测试夹具",
          token: "testFixtures",
        });
      }

      if (
        /from\s+["'`](?:\.\/MockCapabilityHost|\.\/mockCapabilityProfile|\.{1,2}\/[^"'`]*(?:MockCapabilityHost|mockCapabilityProfile))["'`]/.test(
          sourceCode,
        )
      ) {
        failures.push({
          file: relativePath,
          message: "生产源码不能导入 Plugin mock SDK host/profile",
          token: "MockCapabilityHost/mockCapabilityProfile",
        });
      }

      const desktopHostCoreImportPattern =
        /import\s+\{([^}]*)\}\s+from\s+["'`](?:@\/lib\/desktop-host\/core|\.\.?\/[^"'`]*desktop-host\/core|\.\/core)["'`]/g;
      for (const importMatch of sourceCode.matchAll(
        desktopHostCoreImportPattern,
      )) {
        const namedImports = importMatch[1]
          .split(",")
          .map((item) =>
            item
              .trim()
              .split(/\s+as\s+/)[0]
              ?.trim(),
          )
          .filter(Boolean);
        for (const importName of namedImports) {
          if (forbiddenMockOnlyImports.includes(importName)) {
            failures.push({
              file: relativePath,
              message: "生产源码不能导入 desktop-host 测试 mock 入口",
              token: importName,
            });
          }
        }
      }

      for (const rule of forbiddenCallPatterns) {
        if (rule.pattern.test(sourceCode)) {
          failures.push({
            file: relativePath,
            message: rule.message,
            token: rule.token,
          });
        }
      }
    }
  }

  for (const relativePath of collectRustSourceFiles(
    "lime-rs/src/dev_bridge/dispatcher",
  )) {
    const sourceCode = readProductionSourceForGuard(relativePath);
    for (const command of retiredMcpDesktopFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: relativePath,
          message:
            "已迁到 App Server MCP current API 的旧 MCP 命令不能回到 Rust DevBridge dispatcher",
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectProductionBridgeGuardFailures() {
  const failures = [];

  const safeInvokePath = "src/lib/dev-bridge/safeInvoke.ts";
  const safeInvokeSource = readSource(safeInvokePath);
  addForbiddenSubstringFailures(failures, safeInvokePath, safeInvokeSource, [
    {
      substring: "@/lib/desktop-host/api",
      message: "safeInvoke 不能再导入 legacy desktop-host api",
    },
    {
      substring: "./explicitMockFallback",
      message: "safeInvoke 不能再导入显式 renderer mock fallback",
    },
    {
      substring: "invokeExplicitMock",
      message: "safeInvoke 不能再调用显式 invoke mock",
    },
    {
      substring: "listenExplicitMock",
      message: "safeInvoke 不能再调用显式 event mock",
    },
    {
      substring: "getLegacyDesktopHostGlobal",
      message: "safeInvoke 不能再读取 legacy Tauri 全局对象",
    },
    {
      substring: "hasDesktopHostInvokeCapability",
      message: "safeInvoke 不能再探测 legacy Tauri invoke 能力",
    },
    {
      substring: "hasDesktopHostRuntimeMarkers",
      message: "safeInvoke 不能再依赖 legacy Tauri runtime marker",
    },
    {
      substring: "shouldPreferMockInBrowser",
      message: "safeInvoke 不能再按命令优先走 browser mock",
    },
    {
      substring: "shouldDisallowMockFallbackInBrowser",
      message: "safeInvoke 不应再保留 mock fallback 分流",
    },
    {
      substring: "fallback-invoke",
      message: "safeInvoke trace 不能再出现 fallback-invoke transport",
    },
    {
      substring: "legacy-ipc",
      message: "safeInvoke trace 不能再出现 legacy-ipc transport",
    },
  ]);

  const desktopHostPath = "src/lib/desktop-host/core.ts";
  const desktopHostSource = readSource(desktopHostPath);
  const invokeBody = extractNamedFunctionBody(
    desktopHostSource,
    "export async function invoke<T = any>",
  );
  addForbiddenSubstringFailures(failures, desktopHostPath, invokeBody, [
    {
      substring: "invokeDefaultMock",
      message: "生产 invoke 不能回退 invokeDefaultMock",
    },
    {
      substring: "loadDefaultMocks",
      message: "生产 invoke 不能加载 default mocks",
    },
    {
      substring: "mockCommands",
      message: "生产 invoke 不能读取测试 mockCommands",
    },
    {
      substring: "invokeMockOnly",
      message: "生产 invoke 不能委托测试夹具入口",
    },
  ]);
  if (!invokeBody.includes("getElectronHostBridge()")) {
    failures.push({
      file: desktopHostPath,
      message: "生产 invoke 必须优先检查 Electron Desktop Host IPC",
      token: "getElectronHostBridge()",
    });
  }
  if (!invokeBody.includes("invokeViaHttp<T>")) {
    failures.push({
      file: desktopHostPath,
      message: "生产 invoke 仅允许在 DevBridge 可用时走 HTTP bridge 诊断通道",
      token: "invokeViaHttp<T>",
    });
  }
  if (!invokeBody.includes("throw new Error(")) {
    failures.push({
      file: desktopHostPath,
      message: "生产 invoke 缺少无真实通道时的 fail-closed 错误",
      token: "throw new Error(",
    });
  }

  const appServerHostPath = "electron/appServerHost.ts";
  const appServerHostSource = readSource(appServerHostPath);
  if (/backendMode:\s*["'`]mock["'`]/.test(appServerHostSource)) {
    failures.push({
      file: appServerHostPath,
      message: "Electron App Server host 不能配置 mock backend",
      token: 'backendMode: "mock"',
    });
  }
  const resolveBackendModeBody = extractNamedFunctionBody(
    appServerHostSource,
    "function resolveBackendMode",
  );
  if (
    !resolveBackendModeBody.includes('normalized === "mock"') ||
    !resolveBackendModeBody.includes("throw new Error(")
  ) {
    failures.push({
      file: appServerHostPath,
      message: "APP_SERVER_BACKEND_MODE=mock 必须显式失败",
      token: 'normalized === "mock"',
    });
  }
  const runtimeBackendLaunchBody = extractNamedFunctionBody(
    appServerHostSource,
    "function resolveRuntimeBackendLaunchOptions",
  );
  for (const snippet of [
    "process.env.APP_SERVER_BACKEND_COMMAND?.trim()",
    "parseBackendArgs(process.env.APP_SERVER_BACKEND_ARGS)",
    "parsePositiveInteger(",
  ]) {
    if (!runtimeBackendLaunchBody.includes(snippet)) {
      failures.push({
        file: appServerHostPath,
        message: "Electron App Server host 必须保留 external backend env 投影",
        token: snippet,
      });
    }
  }
  const runtimeRequestTimeoutBody = extractNamedFunctionBody(
    appServerHostSource,
    "function resolveAppServerRequestTimeoutMs",
  );
  if (
    !runtimeRequestTimeoutBody.includes(
      "resolveDefaultAppServerRequestTimeoutMs(method)",
    )
  ) {
    failures.push({
      file: appServerHostPath,
      message:
        "Electron App Server host 的请求 timeout override 必须以默认 method timeout 为下限",
      token: "resolveDefaultAppServerRequestTimeoutMs(method)",
    });
  }
  const runtimeDefaultRequestTimeoutBody = extractNamedFunctionBody(
    appServerHostSource,
    "function resolveDefaultAppServerRequestTimeoutMs",
  );
  for (const snippet of [
    "method !== APP_SERVER_TURN_START_METHOD",
    "process.env.APP_SERVER_BACKEND_TIMEOUT_MS",
    "APP_SERVER_BACKEND_TIMEOUT_GRACE_MS",
    "DEFAULT_APP_SERVER_REQUEST_TIMEOUT_MS",
  ]) {
    if (!runtimeDefaultRequestTimeoutBody.includes(snippet)) {
      failures.push({
        file: appServerHostPath,
        message:
          "Electron App Server host 必须让长回合请求等待预算跟随 external backend timeout",
        token: snippet,
      });
    }
  }

  const devSidecarPath = "scripts/lib/electron-dev-sidecar.mjs";
  const devSidecarSource = readSource(devSidecarPath);
  for (const snippet of [
    "resolveDevAppServerBackendEnv",
    'defaultMode = "runtime"',
    "APP_SERVER_BACKEND_MODE: defaultMode",
    "APP_SERVER_BACKEND_MODE: requestedMode",
    'APP_SERVER_BACKEND_MODE: "external"',
    "APP_SERVER_BACKEND_COMMAND",
  ]) {
    if (!devSidecarSource.includes(snippet)) {
      failures.push({
        file: devSidecarPath,
        message:
          "Electron dev 必须默认接入 App Server runtime backend，并只保留显式 external override",
        token: snippet,
      });
    }
  }
  for (const token of [
    "appServerAgentBackendBinaryName",
    "localAppServerAgentBackendBinaryPath",
    "resolveDevAppServerAgentBackendBinary",
    "shouldUseDevAppServerExternalBackend",
    "app-server-agent-backend",
  ]) {
    if (devSidecarSource.includes(token)) {
      failures.push({
        file: devSidecarPath,
        message: "Electron dev 不能继续默认解析或构建旧 external agent backend",
        token,
      });
    }
  }
  if (/APP_SERVER_BACKEND_MODE:\s*["'`]mock["'`]/.test(devSidecarSource)) {
    failures.push({
      file: devSidecarPath,
      message: "Electron dev 不能注入 mock App Server backend",
      token: 'APP_SERVER_BACKEND_MODE: "mock"',
    });
  }

  const electronHostCommandsPath = "electron/hostCommands.ts";
  const electronHostCommandsSource = readSource(electronHostCommandsPath);
  const electronSystemUtilityHostPath = "electron/systemUtilityHost.ts";
  const electronSystemUtilityHostSource = readSource(
    electronSystemUtilityHostPath,
  );
  for (const { path: sourcePath, source } of [
    { path: electronHostCommandsPath, source: electronHostCommandsSource },
    {
      path: electronSystemUtilityHostPath,
      source: electronSystemUtilityHostSource,
    },
  ]) {
    if (source.includes('"current"')) {
      failures.push({
        file: sourcePath,
        message: "Electron 设置页诊断 facade 不能恢复 legacy Agent 浏览器后端",
        token: '"current"',
      });
    }
    if (source.includes("auto_fallback: true")) {
      failures.push({
        file: sourcePath,
        message: "Electron 设置页诊断 facade 不能默认启用浏览器后端自动回退",
        token: "auto_fallback: true",
      });
    }
  }
  for (const command of electronDiagnosticFacadeCommands) {
    if (!electronHostCommandsSource.includes(`"${command}"`)) {
      failures.push({
        file: electronHostCommandsPath,
        message: "设置页 Electron 诊断命令缺少 host facade 承接",
        token: command,
      });
    }
    const diagnosticMetaPattern = new RegExp(
      `diagnosticMeta\\([\\s\\S]*?["'\`]${escapeRegExp(command)}["'\`][\\s\\S]*?\\)`,
    );
    const emptyDiagnosticListPattern = new RegExp(
      `emptyDiagnosticList\\([\\s\\S]*?["'\`]${escapeRegExp(command)}["'\`][\\s\\S]*?\\)`,
    );
    const hasDiagnosticProjection =
      diagnosticMetaPattern.test(electronSystemUtilityHostSource) ||
      emptyDiagnosticListPattern.test(electronSystemUtilityHostSource) ||
      diagnosticMetaPattern.test(electronHostCommandsSource) ||
      emptyDiagnosticListPattern.test(electronHostCommandsSource);
    if (!hasDiagnosticProjection) {
      failures.push({
        file: electronSystemUtilityHostPath,
        message:
          "设置页 Electron 诊断命令必须显式标注 electron-host-diagnostic degraded",
        token: command,
      });
    }
  }

  const runElectronDevPath = "scripts/electron/run-dev.mjs";
  const runElectronDevSource = readSource(runElectronDevPath);
  for (const snippet of ["resolveDevAppServerBackendEnv"]) {
    if (!runElectronDevSource.includes(snippet)) {
      failures.push({
        file: runElectronDevPath,
        message: "Electron dev 启动必须注入 App Server runtime backend env",
        token: snippet,
      });
    }
  }
  for (const token of [
    "resolveDevAppServerAgentBackendBinary",
    "shouldUseDevAppServerExternalBackend",
    "backendCommand: appServerAgentBackendBin",
  ]) {
    if (runElectronDevSource.includes(token)) {
      failures.push({
        file: runElectronDevPath,
        message: "Electron dev 启动不能再自动接旧 external agent backend",
        token,
      });
    }
  }

  for (const smokePath of [
    "scripts/app-server/stdio-smoke.mjs",
    "scripts/app-server/sidecar-lifecycle-smoke.mjs",
  ]) {
    const smokeSource = readSource(smokePath);
    if (/backendMode:\s*["'`]mock["'`]/.test(smokeSource)) {
      failures.push({
        file: smokePath,
        message: "App Server smoke 不能用 mock backend 伪造 turn 成功",
        token: 'backendMode: "mock"',
      });
    }
  }

  failures.push(...collectProductionMockOnlyUsageFailures());

  const explicitMockFallbackPath = "src/lib/dev-bridge/explicitMockFallback.ts";
  const explicitMockFallbackSource = readSource(explicitMockFallbackPath);
  for (const snippet of [
    "assertExplicitMockFallbackTestEnvironment",
    'import.meta.env?.MODE === "test"',
    "invokeExplicitMock",
    "listenExplicitMock",
  ]) {
    if (!explicitMockFallbackSource.includes(snippet)) {
      failures.push({
        file: explicitMockFallbackPath,
        message: "显式 renderer mock fallback 必须只允许测试环境使用",
        token: snippet,
      });
    }
  }

  return failures;
}

function collectRetiredFileBrowserFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/desktop-host/fileSystemMocks.ts",
      message:
        "已迁到 App Server fs/* 的旧文件浏览 facade 命令不能继续保留 desktop-host mock fixture",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已迁到 App Server fs/* 的旧文件浏览 facade 命令不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/files.rs",
      message:
        "已迁到 App Server fs/* 的旧文件浏览 facade 命令不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/services/file_browser_service.rs",
      message:
        "已迁到 App Server fs/* 的旧文件浏览 facade 命令不能回到 Tauri command wrapper",
    },
    {
      path: "lime-rs/crates/services/src/file_browser_service.rs",
      message:
        "已迁到 App Server fs/* 的旧文件浏览 facade 命令不能回到 services compat wrapper",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readExistingProductionSourceForGuard(source.path);
    const commands = source.commands ?? retiredFileBrowserFacadeCommands;
    for (const command of commands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  for (const sourcePath of [
    "packages/app-server-client/src/protocol.ts",
    "packages/app-server-client/src/request-client.ts",
    "packages/app-server-client/src/request-client-methods.ts",
    "packages/app-server-client/src/connection-methods.ts",
    "src/lib/api/appServerConstants.ts",
    "src/lib/api/appServerTypes.ts",
    "src/lib/api/appServerClientMethods.ts",
    "src/lib/api/appServerClientMethodSpecs.ts",
    "src/lib/api/fileBrowser.ts",
    "src/lib/api/session-files.ts",
    "src/lib/dev-bridge/commandPolicy.ts",
    "src/lib/dev-bridge/mockPriorityCommands.ts",
    "src/lib/governance/agentCommandCatalog.json",
    "electron/hostCommands.ts",
    "electron/ipcChannels.ts",
  ]) {
    const sourceCode = readExistingProductionSourceForGuard(sourcePath);
    for (const method of retiredFileSystemMethods) {
      if (sourceCode.includes(method)) {
        failures.push({
          file: sourcePath,
          message:
            "旧 fileSystem/* method 不能回到 App Server client、Renderer、Electron 或 DevBridge current surface",
          token: method,
        });
      }
    }
  }

  return failures;
}

function collectRetiredAutomationFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "已迁到 App Server automation* 的旧自动化命令不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "已迁到 App Server automation* 的旧自动化命令不能继续作为 mock priority command",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已迁到 App Server automation* 的旧自动化命令不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message:
        "已迁到 App Server automation* 的旧自动化命令不能回到 legacy Tauri command module",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "已迁到 App Server automation* 的旧自动化命令不能回到 Rust DevBridge dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readExistingProductionSourceForGuard(source.path);
    for (const command of retiredAutomationFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredApiKeyProviderFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "已迁到 App Server modelProvider/* 的旧 Provider 命令不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "已迁到 App Server modelProvider/* 的旧 Provider 命令不能继续作为 mock priority command",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已迁到 App Server modelProvider/* 的旧 Provider 命令不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message:
        "已迁到 App Server modelProvider/* 的旧 Provider 命令不能回到 legacy Tauri command module",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "已迁到 App Server modelProvider/* 的旧 Provider 命令不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/providers.rs",
      message:
        "已迁到 App Server modelProvider/* 的旧 Provider 命令不能回到 Rust DevBridge provider dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredApiKeyProviderFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredMcpDesktopFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "已迁到 App Server MCP current API 的旧 MCP 命令不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "已迁到 App Server MCP current API 的旧 MCP 命令不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/core.ts",
      message:
        "已迁到 App Server MCP current API 的旧 MCP mock 模块不能继续被 desktop-host 默认 mock loader 加载",
      commands: ["mcpMocks"],
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已迁到 App Server MCP current API 的旧 MCP 命令不能回到 legacy Tauri generate_handler",
    },
    {
      paths: collectRustSourceFiles("lime-rs/src/dev_bridge/dispatcher"),
      message:
        "已迁到 App Server MCP current API 的旧 MCP 命令不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message:
        "mcp_cmd 已退为 cleanup-only residual，不能重新暴露为 legacy Tauri command module",
      commands: ["mcp_cmd"],
    },
  ];

  for (const retiredMockModulePath of [
    "src/lib/desktop-host/mcpMocks.ts",
    "src/lib/desktop-host/mcpMocks.d.ts",
  ]) {
    if (fs.existsSync(path.join(repoRoot, retiredMockModulePath))) {
      failures.push({
        file: retiredMockModulePath,
        message:
          "MCP desktop-host 默认 mock 模块已退役；旧 MCP 命令只能停留在 retired guard / contract forbidden snippet 中",
        token: retiredMockModulePath,
      });
    }
  }

  for (const source of restrictedSources) {
    const paths = source.paths ?? [source.path];
    const commands = source.commands ?? retiredMcpDesktopFacadeCommands;
    for (const path of paths) {
      const sourceCode = readProductionSourceForGuard(path);
      for (const command of commands) {
        if (hasStandaloneIdentifier(sourceCode, command)) {
          failures.push({
            file: path,
            message: source.message,
            token: command,
          });
        }
      }
    }
  }

  return failures;
}

function collectRetiredPluginPackageFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "已迁到 App Server plugin* 的旧 Plugin package/install 命令不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "已迁到 App Server plugin* 的旧 Plugin package/install 命令不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/pluginMocks.ts",
      message:
        "已迁到 App Server plugin* 的旧 Plugin package/install 命令不能继续保留 desktop-host mock fixture",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已迁到 App Server plugin* 的旧 Plugin package/install 命令不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/plugins.rs",
      message:
        "已迁到 App Server plugin* 的旧 Plugin package/install 命令不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message:
        "已迁到 App Server plugin* 的旧 Plugin package/install 命令不能重新暴露 legacy Tauri Plugin command module；commands/** 只允许清理旧逻辑",
      commands: ["plugin_cmd"],
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredPluginPackageFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredLogFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已迁到 App Server log/* 与 diagnostics/* 的旧诊断命令不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "已迁到 App Server log/* 与 diagnostics/* 的旧诊断命令不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/app_runtime.rs",
      message:
        "已迁到 App Server diagnostics/* 的旧诊断命令不能回到 Rust DevBridge app runtime dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/logs.rs",
      message:
        "已迁到 App Server log/* 与 diagnostics/logStorage/read 的旧日志命令不能回到 Rust DevBridge logs dispatcher",
    },
    {
      path: "scripts/social-workbench-e2e-smoke.mjs",
      message:
        "Social Workbench smoke 的诊断探活必须走 app_server_handle_json_lines -> diagnostics/server/read，不能继续把旧 get_* diagnostics facade 当正向证据",
      commands: [
        "get_log_storage_diagnostics",
        "export_support_bundle",
        "get_server_diagnostics",
        "get_windows_startup_diagnostics",
      ],
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    const commands = source.commands ?? retiredLogFacadeCommands;
    for (const command of commands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  const deletedWrapperSources = [
    "lime-rs/src/app/commands/logs.rs",
    "lime-rs/src/app/commands/server.rs",
    "lime-rs/src/commands/windows_startup_cmd.rs",
  ];
  for (const sourcePath of deletedWrapperSources) {
    if (fs.existsSync(path.join(repoRoot, sourcePath))) {
      failures.push({
        file: sourcePath,
        message:
          "P16 诊断旧 Tauri wrapper 文件已退场，不能在旧 command 目录恢复",
        token: sourcePath,
      });
    }
  }

  const appCommandsModPath = "lime-rs/src/app/commands/mod.rs";
  const appCommandsModSource = readProductionSourceForGuard(appCommandsModPath);
  addForbiddenSubstringFailures(
    failures,
    appCommandsModPath,
    appCommandsModSource,
    [
      {
        substring: "mod logs",
        message:
          "P16 日志 / support bundle 旧 app_commands 模块不能回到 app/commands 注册树",
      },
      {
        substring: "pub use logs",
        message: "P16 日志 / support bundle 旧 app_commands 模块不能重新导出",
      },
      {
        substring: "mod server",
        message:
          "P16 server diagnostics 旧 app_commands 模块不能回到 app/commands 注册树",
      },
      {
        substring: "pub use server",
        message: "P16 server diagnostics 旧 app_commands 模块不能重新导出",
      },
    ],
  );

  return failures;
}

function collectRetiredConfigFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "P16 config / provider residual 已迁出 legacy Tauri generate_handler，不能回到旧 runner 注册",
    },
    {
      path: "lime-rs/src/app/commands/config.rs",
      message:
        "P16 config / provider residual 旧 Tauri wrapper 已退场，不能回到 app_commands config 模块",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredConfigTauriGenerateHandlerCommands) {
      const hasRetiredConfigFacade =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : new RegExp(
              String.raw`\bpub\s+async\s+fn\s+${escapeRegExp(command)}\b`,
            ).test(sourceCode);
      if (hasRetiredConfigFacade) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  const deletedWrapperSources = [
    "lime-rs/src/dev_bridge/dispatcher/app_runtime.rs",
  ];
  for (const sourcePath of deletedWrapperSources) {
    if (fs.existsSync(path.join(repoRoot, sourcePath))) {
      failures.push({
        file: sourcePath,
        message:
          "P16 config / provider Rust DevBridge app runtime dispatcher 已退场，不能恢复旧 HTTP bridge 直连配置读写",
        token: sourcePath,
      });
    }
  }

  return failures;
}

function collectRetiredChannelSideEffectFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/api/channelsRuntime.ts",
      message:
        "Channels side-effect 前端网关必须通过 App Server gatewayChannel/* / *Channel/* current method，不能回到旧 safeInvoke facade",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "已迁到 App Server Channels current API 的旧 Channels side-effect 命令不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "已迁到 App Server Channels current API 的旧 Channels side-effect 命令不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "已迁到 App Server Channels current API 的旧 Channels side-effect 命令不能继续作为 runtime gateway command",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已迁到 App Server Channels current API 的旧 Channels side-effect 命令不能回到 legacy Tauri generate_handler",
    },
    {
      paths: collectRustSourceFiles("lime-rs/src/dev_bridge/dispatcher"),
      message:
        "已迁到 App Server Channels current API 的旧 Channels side-effect 命令不能回到 Rust DevBridge dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    const paths = source.paths ?? [source.path];
    for (const relativePath of paths) {
      if (!fs.existsSync(path.join(repoRoot, relativePath))) {
        continue;
      }
      const sourceCode = readProductionSourceForGuard(relativePath);
      for (const command of retiredChannelSideEffectFacadeCommands) {
        const hasRetiredChannelFacade =
          relativePath === "lime-rs/src/app/runner.rs"
            ? hasTauriCommandRegistration(sourceCode, command)
            : relativePath === "src/lib/governance/agentCommandCatalog.json"
              ? agentCommandCatalogRuntimeSurfaceHas(command)
              : hasStandaloneIdentifier(sourceCode, command);
        if (hasRetiredChannelFacade) {
          failures.push({
            file: relativePath,
            message: source.message,
            token: command,
          });
        }
      }
    }
  }

  const deletedDispatcherPath =
    "lime-rs/src/dev_bridge/dispatcher/runtime_queries.rs";
  if (fs.existsSync(path.join(repoRoot, deletedDispatcherPath))) {
    failures.push({
      file: deletedDispatcherPath,
      message:
        "Channels runtime model 旧 Rust DevBridge dispatcher 已退场，不能恢复旧 wechat_channel_set_runtime_model 分发",
      token: deletedDispatcherPath,
    });
  }

  const commandsModPath = "lime-rs/src/commands/mod.rs";
  const commandsModSource = readProductionSourceForGuard(commandsModPath);
  for (const moduleName of ["gateway_channel_cmd", "wechat_channel_cmd"]) {
    if (hasStandaloneIdentifier(commandsModSource, moduleName)) {
      failures.push({
        file: commandsModPath,
        message:
          "Channels/WeChat 旧 Tauri wrapper 已迁到 App Server current，不能回到 commands 模块树",
        token: moduleName,
      });
    }
  }

  for (const retiredWrapperPath of [
    "lime-rs/src/commands/gateway_channel_cmd.rs",
    "lime-rs/src/commands/wechat_channel_cmd.rs",
  ]) {
    if (fs.existsSync(path.join(repoRoot, retiredWrapperPath))) {
      failures.push({
        file: retiredWrapperPath,
        message:
          "Channels/WeChat 旧 Tauri wrapper 文件已删除，不能恢复旧 wrapper 文件本体",
        token: retiredWrapperPath,
      });
    }
  }

  return failures;
}

function collectRetiredGatewayTunnelFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/api/channelsRuntime.ts",
      message:
        "Gateway Tunnel 前端网关必须通过 App Server gatewayTunnel/* current method，不能回到旧 safeInvoke facade",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "已迁到 App Server gatewayTunnel/* 的旧 tunnel 命令不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "已迁到 App Server gatewayTunnel/* 的旧 tunnel 命令不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "已迁到 App Server gatewayTunnel/* 的旧 tunnel 命令不能继续作为 runtime gateway command",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已迁到 App Server gatewayTunnel/* 的旧 tunnel 命令不能回到 legacy Tauri generate_handler",
    },
    {
      paths: collectRustSourceFiles("lime-rs/src/dev_bridge/dispatcher"),
      message:
        "已迁到 App Server gatewayTunnel/* 的旧 tunnel 命令不能回到 Rust DevBridge dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    const paths = source.paths ?? [source.path];
    for (const relativePath of paths) {
      if (!fs.existsSync(path.join(repoRoot, relativePath))) {
        continue;
      }
      const sourceCode = readProductionSourceForGuard(relativePath);
      for (const command of retiredGatewayTunnelFacadeCommands) {
        const hasRetiredTunnelFacade =
          relativePath === "lime-rs/src/app/runner.rs"
            ? hasTauriCommandRegistration(sourceCode, command)
            : relativePath === "src/lib/governance/agentCommandCatalog.json"
              ? agentCommandCatalogRuntimeSurfaceHas(command)
              : hasStandaloneIdentifier(sourceCode, command);
        if (hasRetiredTunnelFacade) {
          failures.push({
            file: relativePath,
            message: source.message,
            token: command,
          });
        }
      }
    }
  }

  const commandsModPath = "lime-rs/src/commands/mod.rs";
  const commandsModSource = readProductionSourceForGuard(commandsModPath);
  if (hasStandaloneIdentifier(commandsModSource, "gateway_tunnel_cmd")) {
    failures.push({
      file: commandsModPath,
      message:
        "Gateway Tunnel 旧 Tauri wrapper 已迁到 App Server current，不能回到 commands 模块树",
      token: "gateway_tunnel_cmd",
    });
  }

  const runnerPath = "lime-rs/src/app/runner.rs";
  const runnerSource = readProductionSourceForGuard(runnerPath);
  for (const token of ["GatewayTunnelState", "lime_gateway::tunnel::"]) {
    if (runnerSource.includes(token)) {
      failures.push({
        file: runnerPath,
        message:
          "Gateway Tunnel 生命周期已迁到 App Server LocalAppDataSource，legacy Tauri runner 不能重新持有 tunnel 状态或守护器",
        token,
      });
    }
  }

  return failures;
}

function collectRetiredPluginShellSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "plugin_launch_shell 与 pluginShell/prepare 已退役，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "plugin_launch_shell 与 pluginShell/prepare 已退役，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/pluginMocks.ts",
      message:
        "plugin_launch_shell 与 pluginShell/prepare 已退役，不能继续保留 desktop-host mock fixture",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "plugin_launch_shell 与 pluginShell/prepare 已退役，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/plugins.rs",
      message:
        "plugin_launch_shell 与 pluginShell/prepare 已退役，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message:
        "Plugin shell / picker 已退役，不能重新暴露 legacy Tauri Plugin command module",
      commands: ["plugin_cmd"],
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredPluginShellCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredPluginRuntimeSourceFailures() {
  const failures = [];
  for (const relativePath of [
    "src/features/plugin/featureFlag.ts",
    "src/features/plugin/index.ts",
    "src/features/plugin/sdk",
  ]) {
    if (fs.existsSync(path.join(repoRoot, relativePath))) {
      failures.push({
        file: relativePath,
        message:
          "已删除的 Plugin feature flag / public SDK 不得恢复；标准包只通过 App Server current owner 运行",
        token: relativePath,
      });
    }
  }

  const restrictedSources = [
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "plugin_runtime_* 已由 Plugin v3 App Server 主链取代，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message:
        "plugin_runtime_cmd 已退为 cleanup-only residual，不能重新暴露为 legacy Tauri command module",
      commands: ["plugin_runtime_cmd"],
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readProductionSourceForGuard(source.path);
    const commands = source.commands ?? retiredPluginRuntimeCommands;
    for (const command of commands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectCurrentTrayDesktopHostShellSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "sync_tray_model_shortcuts 是 Electron Desktop Host 托盘壳能力，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "sync_tray_model_shortcuts 是 Electron Desktop Host 托盘壳能力，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/modelMocks.ts",
      message:
        "sync_tray_model_shortcuts 是 Electron Desktop Host 托盘壳能力，不能保留 desktop-host 默认 mock",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "sync_tray_model_shortcuts 是 Electron Desktop Host 托盘壳能力，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "sync_tray_model_shortcuts 是 Electron Desktop Host 托盘壳能力，不能回到 Rust DevBridge dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of currentTrayDesktopHostShellCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectCurrentHotkeyDesktopHostShellSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "快捷键状态 / 校验是 Electron Desktop Host 快捷键壳能力，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "快捷键状态 / 校验是 Electron Desktop Host 快捷键壳能力，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/configSystemMocks.ts",
      message:
        "快捷键状态 / 校验是 Electron Desktop Host 快捷键壳能力，不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/desktop-host/voiceMocks.ts",
      message:
        "快捷键状态 / 校验是 Electron Desktop Host 快捷键壳能力，不能保留 desktop-host 默认 mock",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "快捷键状态 / 校验是 Electron Desktop Host 快捷键壳能力，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "快捷键状态 / 校验是 Electron Desktop Host 快捷键壳能力，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/voice.rs",
      message:
        "快捷键状态 / 校验是 Electron Desktop Host 快捷键壳能力，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/voice/commands.rs",
      message:
        "快捷键状态 / 校验是 Electron Desktop Host 快捷键壳能力，不能回到旧 Tauri voice command",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of currentHotkeyDesktopHostShellCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  for (const retiredPath of ["lime-rs/src/commands/hotkey_cmd.rs"]) {
    if (fs.existsSync(path.join(repoRoot, retiredPath))) {
      failures.push({
        file: retiredPath,
        message:
          "快捷键校验旧 Tauri wrapper 已迁到 Electron Desktop Host current，不能恢复旧文件本体",
        token: retiredPath,
      });
    }
  }

  const commandsModPath = "lime-rs/src/commands/mod.rs";
  const commandsModSource = readProductionSourceForGuard(commandsModPath);
  if (hasStandaloneIdentifier(commandsModSource, "hotkey_cmd")) {
    failures.push({
      file: commandsModPath,
      message:
        "快捷键校验旧 Tauri wrapper 已迁到 Electron Desktop Host current，不能回到 commands 模块树",
      token: "hotkey_cmd",
    });
  }

  return failures;
}

function collectCurrentVoiceModelDesktopHostSourceFailures() {
  const failures = [];
  const bridgeTruthCommands = collectBridgeTruthCommands();
  const noMockFallbackCompatCommands = collectNoMockFallbackCompatCommands();
  const currentVoiceModelDesktopHostCommands = new Set([
    ...currentVoiceModelDesktopHostReadCommands,
    ...currentVoiceModelDesktopHostSideEffectCommands,
  ]);
  const voiceModelDesktopHostCommandReason = (command) =>
    currentVoiceModelDesktopHostSideEffectCommands.has(command)
      ? "语音模型删除是 Electron Desktop Host current 删除壳能力"
      : "语音模型目录 / 安装状态是 Electron Desktop Host 读链壳能力";
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      messageForCommand: (command) =>
        `${voiceModelDesktopHostCommandReason(command)}，不能继续作为 DevBridge truth command`,
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      messageForCommand: (command) =>
        `${voiceModelDesktopHostCommandReason(command)}，不能继续作为 mock priority command`,
    },
    {
      path: "src/lib/desktop-host/voiceMocks.ts",
      messageForCommand: (command) =>
        `${voiceModelDesktopHostCommandReason(command)}，不能保留 desktop-host 默认 mock`,
    },
    {
      path: "lime-rs/src/app/runner.rs",
      messageForCommand: (command) =>
        `${voiceModelDesktopHostCommandReason(command)}，不能回到 legacy Tauri generate_handler`,
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      messageForCommand: (command) =>
        `${voiceModelDesktopHostCommandReason(command)}，不能回到 Rust DevBridge dispatcher`,
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/voice.rs",
      messageForCommand: (command) =>
        `${voiceModelDesktopHostCommandReason(command)}，不能回到 Rust DevBridge dispatcher`,
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of currentVoiceModelDesktopHostCommands) {
      const hasVoiceModelDesktopHostLeak =
        source.path === "src/lib/dev-bridge/commandPolicy.ts"
          ? bridgeTruthCommands.has(command) ||
            noMockFallbackCompatCommands.has(command)
          : hasStandaloneIdentifier(sourceCode, command);
      if (hasVoiceModelDesktopHostLeak) {
        failures.push({
          file: source.path,
          message: source.messageForCommand(command),
          token: command,
        });
      }
    }
  }

  const voiceModelCommandPath = "lime-rs/src/commands/voice_model_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, voiceModelCommandPath))) {
    const sourceCode = readProductionSourceForGuard(voiceModelCommandPath);
    for (const command of currentVoiceModelDesktopHostCommands) {
      if (hasRustFunctionDefinition(sourceCode, command)) {
        failures.push({
          file: voiceModelCommandPath,
          message: `${voiceModelDesktopHostCommandReason(command)}，旧 Rust wrapper 不能继续保留同名正向函数`,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredVoiceModelDefaultFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/api/voiceModels.ts",
      message:
        "语音模型设默认已迁到 App Server voiceModel/default/set current，前端 API 不能继续 safeInvoke 旧 voice_models_set_default 命令",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "语音模型设默认已迁到 App Server voiceModel/default/set current，不能回到 Electron Host 白名单",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "语音模型设默认已迁到 App Server voiceModel/default/set current，不能回到 Electron Host facade",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "语音模型设默认已迁到 App Server voiceModel/default/set current，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "语音模型设默认已迁到 App Server voiceModel/default/set current，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/voiceMocks.ts",
      message:
        "语音模型设默认已迁到 App Server voiceModel/default/set current，不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "语音模型设默认已迁到 App Server voiceModel/default/set current，不能回到 runtime/capability catalog",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "语音模型设默认已迁到 App Server voiceModel/default/set current，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "语音模型设默认已迁到 App Server voiceModel/default/set current，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/voice.rs",
      message:
        "语音模型设默认已迁到 App Server voiceModel/default/set current，不能回到 Rust DevBridge voice dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredVoiceModelDefaultFacadeCommands) {
      const hasRetiredVoiceModelDefaultLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/governance/agentCommandCatalog.json"
            ? agentCommandCatalogRuntimeSurfaceHas(command)
            : source.path === "src/lib/api/voiceModels.ts"
              ? hasFrontendCommandInvocation(sourceCode, command)
              : hasStandaloneIdentifier(sourceCode, command);
      if (hasRetiredVoiceModelDefaultLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  const voiceModelCommandPath = "lime-rs/src/commands/voice_model_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, voiceModelCommandPath))) {
    const sourceCode = readProductionSourceForGuard(voiceModelCommandPath);
    for (const command of retiredVoiceModelDefaultFacadeCommands) {
      if (hasRustFunctionDefinition(sourceCode, command)) {
        failures.push({
          file: voiceModelCommandPath,
          message:
            "语音模型设默认已迁到 App Server voiceModel/default/set current，旧 Rust wrapper 不能继续保留同名正向函数",
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredVoiceModelTestTranscribeFacadeSourceFailures() {
  const failures = [];
  const bridgeTruthCommands = collectBridgeTruthCommands();
  const noMockFallbackCompatCommands = collectNoMockFallbackCompatCommands();
  const restrictedSources = [
    {
      path: "src/lib/api/voiceModels.ts",
      message:
        "语音模型测试转写已迁到 App Server voiceModel/testTranscribeFile current，前端 API 不能继续 safeInvoke 旧 voice_models_test_transcribe_file 命令",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "语音模型测试转写已迁到 App Server voiceModel/testTranscribeFile current，不能回到 Electron Host 白名单",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "语音模型测试转写已迁到 App Server voiceModel/testTranscribeFile current，不能回到 Electron Host facade",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "语音模型测试转写已迁到 App Server voiceModel/testTranscribeFile current，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "语音模型测试转写已迁到 App Server voiceModel/testTranscribeFile current，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/voiceMocks.ts",
      message:
        "语音模型测试转写已迁到 App Server voiceModel/testTranscribeFile current，不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "语音模型测试转写已迁到 App Server voiceModel/testTranscribeFile current，不能回到 runtime/capability catalog",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "语音模型测试转写已迁到 App Server voiceModel/testTranscribeFile current，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "语音模型测试转写已迁到 App Server voiceModel/testTranscribeFile current，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/voice.rs",
      message:
        "语音模型测试转写已迁到 App Server voiceModel/testTranscribeFile current，不能回到 Rust DevBridge voice dispatcher",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message:
        "语音模型测试转写旧 Tauri wrapper 已删除，不能回到 commands/mod.rs",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredVoiceModelTestTranscribeFacadeCommands) {
      const hasRetiredVoiceModelTestTranscribeLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/governance/agentCommandCatalog.json"
            ? agentCommandCatalogRuntimeSurfaceHas(command)
            : source.path === "src/lib/api/voiceModels.ts"
              ? hasFrontendCommandInvocation(sourceCode, command)
              : source.path === "src/lib/dev-bridge/commandPolicy.ts"
                ? bridgeTruthCommands.has(command) ||
                  noMockFallbackCompatCommands.has(command)
                : hasStandaloneIdentifier(sourceCode, command);
      if (hasRetiredVoiceModelTestTranscribeLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  const retiredWrapperPath = "lime-rs/src/commands/voice_model_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, retiredWrapperPath))) {
    failures.push({
      file: retiredWrapperPath,
      message: "语音模型测试转写旧 Tauri wrapper 已删除，不能恢复旧文件本体",
      token: retiredWrapperPath,
    });
  }

  return failures;
}

function collectRetiredVoiceInputConfigFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "已收敛到 app config current 网关的旧 Voice Input config 命令不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "已收敛到 app config current 网关的旧 Voice Input config 命令不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/voiceMocks.ts",
      message:
        "已收敛到 app config current 网关的旧 Voice Input config 命令不能保留 desktop-host 默认 mock",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已收敛到 app config current 网关的旧 Voice Input config 命令不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "已收敛到 app config current 网关的旧 Voice Input config 命令不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/voice.rs",
      message:
        "已收敛到 app config current 网关的旧 Voice Input config 命令不能回到 Rust DevBridge voice dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredVoiceInputConfigElectronFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  const voiceCommandPath = "lime-rs/src/voice/commands.rs";
  if (fs.existsSync(path.join(repoRoot, voiceCommandPath))) {
    const sourceCode = readProductionSourceForGuard(voiceCommandPath);
    for (const command of retiredVoiceInputConfigElectronFacadeCommands) {
      if (hasRustFunctionDefinition(sourceCode, command)) {
        failures.push({
          file: voiceCommandPath,
          message:
            "Voice Input config 已收敛到前端 app config current 网关，旧 Tauri voice command wrapper 不能继续保留同名正向函数",
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredVoiceAudioDeviceFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/api/asrProvider.ts",
      message:
        "麦克风设备列表已收敛到 renderer media devices current，前端 API 不能继续 safeInvoke 旧 list_audio_devices 命令",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "麦克风设备列表已收敛到 renderer media devices current，不能回到 Electron Host 白名单",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "麦克风设备列表已收敛到 renderer media devices current，不能回到 Electron Host diagnostic facade",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "麦克风设备列表已收敛到 renderer media devices current，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "麦克风设备列表已收敛到 renderer media devices current，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/voiceMocks.ts",
      message:
        "麦克风设备列表已收敛到 renderer media devices current，不能保留 desktop-host 默认 mock",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "麦克风设备列表已收敛到 renderer media devices current，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "麦克风设备列表已收敛到 renderer media devices current，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/voice.rs",
      message:
        "麦克风设备列表已收敛到 renderer media devices current，不能回到 Rust DevBridge voice dispatcher",
    },
    {
      path: "lime-rs/src/voice/commands.rs",
      message:
        "麦克风设备列表已收敛到 renderer media devices current，旧 Tauri voice command 不能继续保留同名正向函数",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredVoiceAudioDeviceFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredVoiceAsrCredentialFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/api/asrProvider.ts",
      message:
        "ASR 凭证 CRUD 已迁到 App Server voiceAsrCredential/* current，前端 API 不能继续 safeInvoke 旧 ASR 凭证命令",
    },
    {
      path: "src/lib/api/voiceModels.ts",
      message:
        "默认本地语音模型 readiness 必须经 asrProvider App Server current 间接读取 ASR 凭证，不能直接调用旧 ASR 凭证命令",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "ASR 凭证 CRUD 已迁到 App Server voiceAsrCredential/* current，不能回到 Electron Host 白名单",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "ASR 凭证 CRUD 已迁到 App Server voiceAsrCredential/* current，不能回到 Electron Host diagnostic facade",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "ASR 凭证 CRUD 已迁到 App Server voiceAsrCredential/* current，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "ASR 凭证 CRUD 已迁到 App Server voiceAsrCredential/* current，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/voiceMocks.ts",
      message:
        "ASR 凭证 CRUD 已迁到 App Server voiceAsrCredential/* current，不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "ASR 凭证 CRUD 已迁到 App Server voiceAsrCredential/* current，不能回到 runtime/capability catalog",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "ASR 凭证 CRUD 已迁到 App Server voiceAsrCredential/* current，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "ASR 凭证 CRUD 已迁到 App Server voiceAsrCredential/* current，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/voice.rs",
      message:
        "ASR 凭证 CRUD 已迁到 App Server voiceAsrCredential/* current，不能回到 Rust DevBridge voice dispatcher",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message: "ASR 凭证旧 Tauri wrapper 已删除，不能回到 commands/mod.rs",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredVoiceAsrCredentialFacadeCommands) {
      const hasRetiredAsrCredentialLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/governance/agentCommandCatalog.json"
            ? agentCommandCatalogRuntimeSurfaceHas(command)
            : hasStandaloneIdentifier(sourceCode, command);
      if (hasRetiredAsrCredentialLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  const retiredWrapperPath = "lime-rs/src/commands/asr_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, retiredWrapperPath))) {
    failures.push({
      file: retiredWrapperPath,
      message:
        "ASR 凭证旧 Tauri wrapper 文件已物理删除，不能恢复 asr_cmd.rs 文件本体",
      token: retiredWrapperPath,
    });
  }

  return failures;
}

function collectRetiredVoiceInstructionFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/api/asrProvider.ts",
      message:
        "Voice instructions 已迁到 App Server voiceInstruction/* current，前端 API 不能继续 safeInvoke 旧 voice instruction 命令",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "Voice instructions 已迁到 App Server voiceInstruction/* current，不能回到 Electron Host 白名单",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "Voice instructions 已迁到 App Server voiceInstruction/* current，不能回到 Electron Host diagnostic facade",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Voice instructions 已迁到 App Server voiceInstruction/* current，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Voice instructions 已迁到 App Server voiceInstruction/* current，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/voiceMocks.ts",
      message:
        "Voice instructions 已迁到 App Server voiceInstruction/* current，不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Voice instructions 已迁到 App Server voiceInstruction/* current，不能回到 runtime/capability catalog",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Voice instructions 已迁到 App Server voiceInstruction/* current，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "Voice instructions 已迁到 App Server voiceInstruction/* current，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/voice.rs",
      message:
        "Voice instructions 已迁到 App Server voiceInstruction/* current，不能回到 Rust DevBridge voice dispatcher",
    },
    {
      path: "lime-rs/src/voice/commands.rs",
      message:
        "Voice instructions 旧 Tauri wrapper 已删除，不能回到 voice/commands.rs 正向函数",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredVoiceInstructionFacadeCommands) {
      const hasRetiredVoiceInstructionLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/governance/agentCommandCatalog.json"
            ? agentCommandCatalogRuntimeSurfaceHas(command)
            : hasStandaloneIdentifier(sourceCode, command);
      if (hasRetiredVoiceInstructionLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredVoiceRealtimeFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/api/asrProvider.ts",
      message:
        "语音转写、润色、输出与录音控制尚未接入 App Server / Electron current，前端 API 不能继续 safeInvoke 旧实时语音命令",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "旧实时语音命令不能回到 Electron Host 白名单；后续必须先定义 current voice runtime 通道",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "旧实时语音命令不能回到 Electron Host diagnostic facade；后续必须先定义 current voice runtime 通道",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "旧实时语音命令不能继续作为 DevBridge truth command 或 no-mock fallback",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message: "旧实时语音命令不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/voiceMocks.ts",
      message: "旧实时语音命令不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message: "旧实时语音命令不能回到 runtime/capability catalog",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message: "旧实时语音命令不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message: "旧实时语音命令不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/voice.rs",
      message: "旧实时语音命令不能回到 Rust DevBridge voice dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredVoiceRealtimeFacadeCommands) {
      const hasRetiredVoiceRealtimeLeak =
        source.path === "src/lib/api/asrProvider.ts"
          ? hasFrontendCommandInvocation(sourceCode, command)
          : source.path === "lime-rs/src/app/runner.rs"
            ? hasTauriCommandRegistration(sourceCode, command)
            : source.path === "src/lib/governance/agentCommandCatalog.json"
              ? agentCommandCatalogRuntimeSurfaceHas(command)
              : hasStandaloneIdentifier(sourceCode, command);
      if (hasRetiredVoiceRealtimeLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  const retiredWrapperPath = "lime-rs/src/voice/commands.rs";
  if (fs.existsSync(path.join(repoRoot, retiredWrapperPath))) {
    failures.push({
      file: retiredWrapperPath,
      message:
        "旧实时语音 Tauri wrapper 文件已物理删除，不能恢复 voice/commands.rs 文件本体",
      token: retiredWrapperPath,
    });
  }

  return failures;
}

function collectCurrentSkillDesktopHostShellSourceFailures() {
  const failures = [];
  const retiredSkillCommandPath = "lime-rs/src/commands/skill_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, retiredSkillCommandPath))) {
    failures.push({
      file: retiredSkillCommandPath,
      message:
        "Skill 旧 Tauri wrapper 文件已物理删除，不能恢复 skill_cmd.rs 文件本体",
      token: retiredSkillCommandPath,
    });
  }
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Skill list/open-request/file-association 是 Electron Desktop Host 壳能力，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Skill list/open-request/file-association 是 Electron Desktop Host 壳能力，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/skillManagementMocks.ts",
      message:
        "Skill list/open-request/file-association 是 Electron Desktop Host 壳能力，不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Skill list/open-request 是 Electron Desktop Host 壳能力，不能继续作为 runtime gateway command",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Skill list/open-request 是 Electron Desktop Host 壳能力，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/skills.rs",
      message:
        "Skill list/open-request 是 Electron Desktop Host 壳能力，不能回到 Rust DevBridge skills dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of currentSkillDesktopHostShellCommands) {
      const hasCurrentSkillShellLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/governance/agentCommandCatalog.json"
            ? agentCommandCatalogRuntimeSurfaceHas(command)
            : hasStandaloneIdentifier(sourceCode, command);
      if (hasCurrentSkillShellLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectCurrentLayeredDesignDesktopHostShellSourceFailures() {
  const failures = [];
  const bridgeTruthCommands = collectBridgeTruthCommands();
  const retiredLayeredDesignCommandPath =
    "lime-rs/src/commands/layered_design_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, retiredLayeredDesignCommandPath))) {
    failures.push({
      file: retiredLayeredDesignCommandPath,
      message:
        "Layered Design 已迁到 Electron Desktop Host current 壳能力，不能恢复旧 Rust layered_design_cmd.rs 文件本体",
      token: retiredLayeredDesignCommandPath,
    });
  }
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Layered Design project/extraction 是 Electron Desktop Host 壳能力，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Layered Design project/extraction 是 Electron Desktop Host 壳能力，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Layered Design project/extraction 是 Electron Desktop Host 壳能力，不能继续作为 runtime gateway command",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Layered Design project/extraction 是 Electron Desktop Host 壳能力，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/files.rs",
      message:
        "Layered Design project/extraction 是 Electron Desktop Host 壳能力，不能回到 Rust DevBridge files dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of currentLayeredDesignDesktopHostShellCommands) {
      const hasCurrentLayeredDesignShellLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/governance/agentCommandCatalog.json"
            ? agentCommandCatalogRuntimeSurfaceHas(command)
            : source.path === "src/lib/dev-bridge/commandPolicy.ts"
              ? bridgeTruthCommands.has(command)
              : hasStandaloneIdentifier(sourceCode, command);
      if (hasCurrentLayeredDesignShellLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredAgentRuntimeRustSourceFailures() {
  const failures = [];
  const retiredCommandsRoot = "lime-rs/src/commands";
  if (fs.existsSync(path.join(repoRoot, retiredCommandsRoot))) {
    failures.push({
      file: retiredCommandsRoot,
      message:
        "旧 Tauri command tree 已删除；Agent runtime 只能经 App Server JSON-RPC 进入 current 主链",
      token: retiredCommandsRoot,
    });
  }
  const restrictedSources = [
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Agent Runtime session CRUD / public subagent facade / core current bridge / checkpoint queue replay / compact objective 旧入口已撤下，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/agent_sessions.rs",
      message:
        "Agent Runtime session CRUD / public subagent facade / core current bridge / checkpoint queue replay / compact objective / process Agent 旧入口已撤下，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/providers.rs",
      message:
        "Agent Runtime process 旧 DevBridge provider dispatcher 已撤下，不能恢复旧 Rust DevBridge dispatcher",
      commands: retiredAgentRuntimeProcessRustCommands,
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    const commands = source.commands ?? retiredAgentRuntimeRustCommands;
    for (const command of commands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredSkillExecutionFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "已下线的 execute_skill 独立执行命令不能继续作为 DevBridge truth 或特殊策略命令",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "已下线的 execute_skill 独立执行命令不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/skillForgeMocks.ts",
      message:
        "已下线的 execute_skill 独立执行命令不能继续保留 desktop-host mock fixture",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已下线的 execute_skill 独立执行命令不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message:
        "skill_exec_cmd 已删除，不能重新暴露为 legacy Tauri command module",
      tokens: ["skill_exec_cmd"],
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "已下线的 execute_skill 独立执行命令不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/skills.rs",
      message:
        "已下线的 execute_skill 独立执行命令不能回到 Rust DevBridge skills dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    const absolutePath = path.join(repoRoot, source.path);
    if (!fs.existsSync(absolutePath)) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    const tokens = source.tokens ?? retiredSkillExecutionFacadeCommands;
    for (const token of tokens) {
      if (hasStandaloneIdentifier(sourceCode, token)) {
        failures.push({
          file: source.path,
          message: source.message,
          token,
        });
      }
    }
  }

  const retiredWrapperPath = "lime-rs/src/commands/skill_exec_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, retiredWrapperPath))) {
    failures.push({
      file: retiredWrapperPath,
      message: "execute_skill 独立 Tauri wrapper 已下线，文件不能恢复",
      token: retiredWrapperPath,
    });
  }

  return failures;
}

function collectRetiredSkillRevealFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "本地 Skill reveal 已迁到 App Server skills/list localDirectoryPath projection + Electron reveal_in_finder，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "本地 Skill reveal 已迁到 App Server skills/list localDirectoryPath projection + Electron reveal_in_finder，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "本地 Skill reveal 已迁到 App Server skills/list + Electron reveal_in_finder，不能继续作为 runtime gateway command",
    },
    {
      path: "src/lib/api/skills.ts",
      message:
        "skillsApi.revealLocalSkill 必须通过 current App Server catalog 读取 localDirectoryPath 后调用 reveal_in_finder，不能回到旧 Desktop facade",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredSkillRevealFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredSkillManagementFacadeSourceFailures() {
  const failures = [];
  const retiredSkillCommandPath = "lime-rs/src/commands/skill_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, retiredSkillCommandPath))) {
    failures.push({
      file: retiredSkillCommandPath,
      message:
        "Skill 管理旧 Tauri wrapper 文件已物理删除，不能恢复 skill_cmd.rs 文件本体",
      token: retiredSkillCommandPath,
    });
  }
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Skill 管理旧 facade 已迁到 App Server skillManagement/* / skillRepository/* / skillLocal/* / skillRemote/inspect，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Skill 管理旧 facade 已迁到 App Server current，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/skillManagementMocks.ts",
      message:
        "Skill 管理旧 facade 已迁到 App Server current，不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Skill 管理旧 facade 已迁到 App Server current，不能继续作为 runtime gateway command",
    },
    {
      path: "src/lib/api/skills.ts",
      message:
        "skillsApi Skill 管理必须通过 App Server skillManagement/* / skillRepository/* / skillLocal/* / skillRemote/inspect，不能回到旧 Tauri facade",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Skill 管理旧 facade 已迁到 App Server current，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/skills.rs",
      message:
        "Skill 管理旧 facade 已迁到 App Server current，不能回到 Rust DevBridge skills dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readExistingProductionSourceForGuard(source.path);
    for (const command of retiredSkillManagementFacadeCommands) {
      const hasRetiredSkillManagementLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/governance/agentCommandCatalog.json"
            ? agentCommandCatalogRuntimeSurfaceHas(command)
            : hasStandaloneIdentifier(sourceCode, command);
      if (hasRetiredSkillManagementLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredSkillPackageLocalFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "本地 Skill package inspect/install/export 已迁到 App Server skillPackage/local/* + skillPackage/export，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "本地 Skill package inspect/install/export 已迁到 App Server skillPackage/local/* + skillPackage/export，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "本地 Skill package inspect/install/export 已迁到 App Server skillPackage/local/* + skillPackage/export，不能继续作为 runtime gateway command",
    },
    {
      path: "src/lib/api/skills.ts",
      message:
        "skillsApi 本地 Skill package inspect/install/export 必须通过 App Server skillPackage/local/* + skillPackage/export，不能回到旧 Tauri facade",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredSkillPackageLocalFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredSkillLocalManagementFacadeSourceFailures() {
  const failures = [];
  const retiredSkillCommandPath = "lime-rs/src/commands/skill_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, retiredSkillCommandPath))) {
    failures.push({
      file: retiredSkillCommandPath,
      message:
        "本地 Skill 旧 Tauri wrapper 文件已物理删除，不能恢复 skill_cmd.rs 文件本体",
      token: retiredSkillCommandPath,
    });
  }
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "本地 Skill detail/rename/replace 与远程 inspect 已迁到 App Server skillLocal/* / skillPackage/local/replace / skillRemote/inspect，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "本地 Skill detail/rename/replace 与远程 inspect 已迁到 App Server skillLocal/* / skillPackage/local/replace / skillRemote/inspect，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "本地 Skill detail/rename/replace 与远程 inspect 已迁到 App Server skillLocal/* / skillPackage/local/replace / skillRemote/inspect，不能继续作为 runtime gateway command",
    },
    {
      path: "src/lib/api/skills.ts",
      message:
        "skillsApi 本地 Skill detail/rename/replace 与远程 inspect 必须通过 App Server skillLocal/* / skillPackage/local/replace / skillRemote/inspect，不能回到旧 Tauri facade",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "本地 Skill detail/rename/replace 与远程 inspect 已迁到 App Server current，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/skills.rs",
      message:
        "本地 Skill detail/rename/replace 与远程 inspect 已迁到 App Server current，不能回到 Rust DevBridge skills dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readExistingProductionSourceForGuard(source.path);
    for (const command of retiredSkillLocalManagementFacadeCommands) {
      const hasRetiredSkillLocalManagementLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/governance/agentCommandCatalog.json"
            ? agentCommandCatalogRuntimeSurfaceHas(command)
            : hasStandaloneIdentifier(sourceCode, command);
      if (hasRetiredSkillLocalManagementLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredSkillMarketplaceInstallFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "官方 Skill marketplace/download 安装已迁到 App Server skillMarketplace/install + skillPackage/download/install，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "官方 Skill marketplace/download 安装已迁到 App Server skillMarketplace/install + skillPackage/download/install，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "官方 Skill marketplace/download 安装已迁到 App Server skillMarketplace/install + skillPackage/download/install，不能继续作为 runtime gateway command",
    },
    {
      path: "src/lib/api/skills.ts",
      message:
        "skillsApi 官方 Skill marketplace/download 安装必须通过 App Server skillMarketplace/install + skillPackage/download/install，不能回到旧 Tauri facade",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredSkillMarketplaceInstallFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredSiteAdapterFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Site Adapter 旧命令已退为 fail-closed guard，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Site Adapter 旧命令已退为 fail-closed guard，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Site Adapter 旧命令已退为 fail-closed guard，不能继续作为 runtime gateway command",
    },
    {
      path: "src/lib/api/agentRuntime/siteClient.ts",
      message:
        "Site Adapter 前端 client 必须保持 fail-closed，不能重新调用旧 bridge facade",
      commands: ["bridgeInvoke("],
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "Site Adapter 旧命令不能回到 Electron Host 白名单；后续必须先定义 App Server current method",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "Site Adapter 旧命令不能回到 Electron Host diagnostic facade；后续必须先定义 App Server current method",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Site Adapter 旧命令不能回到 legacy Tauri generate_handler；后续必须先定义 App Server current method",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "Site Adapter 旧命令不能回到 Rust DevBridge dispatcher；后续必须先定义 App Server current method",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readProductionSourceForGuard(source.path);
    const commands = source.commands ?? retiredSiteAdapterFacadeCommands;
    for (const command of commands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredMediaTaskArtifactFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Media task artifact 旧命令已迁到 App Server mediaTaskArtifact/*，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Media task artifact 旧命令已迁到 App Server mediaTaskArtifact/*，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Media task artifact 旧命令已迁到 App Server mediaTaskArtifact/*，不能继续作为 runtime gateway command",
    },
    {
      path: "src/lib/api/mediaTasks.ts",
      message:
        "mediaTasks API 必须通过 App Server mediaTaskArtifact/*，不能回到旧 Tauri facade",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Media task artifact 旧命令不能回到 legacy Tauri generate_handler；current 入口是 App Server mediaTaskArtifact/*",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "Media task artifact 旧命令不能回到 Rust DevBridge dispatcher；current 入口是 App Server mediaTaskArtifact/*",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredMediaTaskArtifactFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  for (const retiredPath of [
    "lime-rs/src/commands/media_task_cmd.rs",
    "lime-rs/src/dev_bridge/dispatcher/media_tasks.rs",
  ]) {
    if (fs.existsSync(path.join(repoRoot, retiredPath))) {
      failures.push({
        file: retiredPath,
        message:
          "Media task artifact 旧 Tauri wrapper / creation tool runtime 已迁到 App Server current，不能恢复旧文件本体",
        token: retiredPath,
      });
    }
  }

  const commandsModPath = "lime-rs/src/commands/mod.rs";
  const commandsModSource = readProductionSourceForGuard(commandsModPath);
  if (hasStandaloneIdentifier(commandsModSource, "media_task_cmd")) {
    failures.push({
      file: commandsModPath,
      message:
        "Media task artifact 旧 Tauri wrapper 已迁到 App Server current，不能回到 commands 模块树",
      token: "media_task_cmd",
    });
  }

  return failures;
}

function collectRetiredVideoGenerationPromptSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "lime-rs/resources/default-skills/video_generate/SKILL.md",
      message:
        "默认 video_generate skill 必须走 CLI/App Server mediaTaskArtifact/video/* current 主链，不能允许或提示回退旧 video generation facade",
    },
  ];
  const retiredPromptTokens = new Set([
    "lime_create_video_generation_task",
    ...retiredVideoGenerationFacadeCommands,
  ]);

  for (const source of restrictedSources) {
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const token of retiredPromptTokens) {
      if (sourceCode.includes(token)) {
        failures.push({
          file: source.path,
          message: source.message,
          token,
        });
      }
    }
  }

  const themeVideoModulePath = "lime-rs/src/theme/video/mod.rs";
  const themeVideoModuleSource =
    readProductionSourceForGuard(themeVideoModulePath);
  if (hasStandaloneIdentifier(themeVideoModuleSource, "command")) {
    failures.push({
      file: themeVideoModulePath,
      message:
        "theme::video::command 是旧视频任务内部 owner residual，不能再从 theme/video 模块树暴露",
      token: "command",
    });
  }

  return failures;
}

function collectCurrentFileBrowserDesktopHostShellSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/desktop-host/fileSystemMocks.ts",
      message:
        "已迁到 Electron Desktop Host 的文件浏览壳命令不能继续保留 desktop-host mock fixture",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "已迁到 Electron Desktop Host 的文件浏览壳命令不能继续作为 DevBridge truth command",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "已迁到 Electron Desktop Host 的文件浏览壳命令不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/files.rs",
      message:
        "已迁到 Electron Desktop Host 的文件浏览壳命令不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/services/file_browser_service.rs",
      message:
        "已迁到 Electron Desktop Host 的文件浏览壳命令不能回到 Tauri command wrapper",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of currentFileBrowserDesktopHostShellCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredGalleryMaterialFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Gallery material 旧 Tauri facade 已迁到 App Server galleryMaterial*，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Gallery material 旧 Tauri facade 已迁到 App Server galleryMaterial*，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Gallery material 旧 Tauri facade 已迁到 App Server galleryMaterial*，不能继续作为 runtime gateway command",
    },
    {
      path: "src/lib/api/galleryMaterials.ts",
      message:
        "galleryMaterials API 必须通过 App Server galleryMaterial*，不能回到旧 Tauri facade",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "Gallery material 旧 Tauri facade 不能回到 Electron Host 白名单；current 入口是 App Server galleryMaterial*",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "Gallery material 旧 Tauri facade 不能回到 Electron Host diagnostic facade；current 入口是 App Server galleryMaterial*",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Gallery material 旧 Tauri facade 不能回到 legacy Tauri generate_handler；current 入口是 App Server galleryMaterial*",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "Gallery material 旧 Tauri facade 不能回到 Rust DevBridge dispatcher；current 入口是 App Server galleryMaterial*",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/project_resources.rs",
      message:
        "Gallery material 旧 Tauri facade 不能回到 Rust DevBridge project resource dispatcher；current 入口是 App Server galleryMaterial*",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readExistingProductionSourceForGuard(source.path);
    for (const command of retiredGalleryMaterialFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  for (const retiredPath of ["lime-rs/src/commands/gallery_material_cmd.rs"]) {
    if (fs.existsSync(path.join(repoRoot, retiredPath))) {
      failures.push({
        file: retiredPath,
        message:
          "Gallery material 旧 Tauri wrapper 已迁到 App Server current，不能恢复旧文件本体",
        token: retiredPath,
      });
    }
  }

  const commandsModPath = "lime-rs/src/commands/mod.rs";
  const commandsModSource = readProductionSourceForGuard(commandsModPath);
  if (hasStandaloneIdentifier(commandsModSource, "gallery_material_cmd")) {
    failures.push({
      file: commandsModPath,
      message:
        "Gallery material 旧 Tauri wrapper 已迁到 App Server current，不能恢复 commands/mod.rs 模块声明",
      token: "gallery_material_cmd",
    });
  }

  return failures;
}

function collectRetiredProjectMaterialFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Project material 旧 Tauri facade 已迁到 App Server projectMaterial/*，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Project material 旧 Tauri facade 已迁到 App Server projectMaterial/*，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Project material 旧 Tauri facade 已迁到 App Server projectMaterial/*，不能继续作为 runtime gateway command",
    },
    {
      path: "src/lib/api/materials.ts",
      message:
        "materials API 必须通过 App Server projectMaterial/*，不能回到旧 Tauri facade",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "Project material 旧 Tauri facade 不能回到 Electron Host 白名单；current 入口是 App Server projectMaterial/*",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "Project material 旧 Tauri facade 不能回到 Electron Host diagnostic facade；current 入口是 App Server projectMaterial/*",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Project material 旧 Tauri facade 不能回到 legacy Tauri generate_handler；current 入口是 App Server projectMaterial/*",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "Project material 旧 Tauri facade 不能回到 Rust DevBridge dispatcher；current 入口是 App Server projectMaterial/*",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/project_resources.rs",
      message:
        "Project material 旧 Tauri facade 不能回到 Rust DevBridge project resource dispatcher；current 入口是 App Server projectMaterial/*",
    },
  ];

  for (const source of restrictedSources) {
    const sourceCode = readExistingProductionSourceForGuard(source.path);
    for (const command of retiredProjectMaterialFacadeCommands) {
      if (hasStandaloneIdentifier(sourceCode, command)) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  for (const retiredPath of ["lime-rs/src/commands/material_cmd.rs"]) {
    if (fs.existsSync(path.join(repoRoot, retiredPath))) {
      failures.push({
        file: retiredPath,
        message:
          "Project material 旧 Tauri wrapper 已迁到 App Server current，不能恢复旧文件本体",
        token: retiredPath,
      });
    }
  }

  const commandsModPath = "lime-rs/src/commands/mod.rs";
  const commandsModSource = readProductionSourceForGuard(commandsModPath);
  if (hasStandaloneIdentifier(commandsModSource, "material_cmd")) {
    failures.push({
      file: commandsModPath,
      message:
        "Project material 旧 Tauri wrapper 已迁到 App Server current，不能恢复 commands/mod.rs 模块声明",
      token: "material_cmd",
    });
  }

  return failures;
}

function collectRetiredWorkspaceContentFacadeSourceFailures() {
  const failures = [];
  const retiredCommands = [
    ...retiredWorkspaceWriteFacadeCommands,
    ...retiredContentCrudFacadeCommands,
  ];
  const restrictedSources = [
    {
      path: "src/lib/api/project.ts",
      message:
        "Workspace 写链 / Content CRUD 已在前端 fail closed，不能重新 safeInvoke 旧 Tauri facade",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "Workspace 写链 / Content CRUD 未接 App Server current，不能回到 Electron Host 白名单",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "Workspace 写链 / Content CRUD 未接 App Server current，不能回到 Electron Host adapter / diagnostic facade",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Workspace 写链 / Content CRUD 已 retired，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Workspace 写链 / Content CRUD 已 retired，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/workspaceMocks.ts",
      message:
        "Workspace 写链 / Content CRUD 已 retired，不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Workspace 写链 / Content CRUD 已 retired，不能回到 runtime/capability catalog",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Workspace 写链 / Content CRUD 已从前端退场，不能回到 legacy Tauri generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "Workspace 写链 / Content CRUD 已从前端退场，不能回到 Rust DevBridge dispatcher",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredCommands) {
      const hasLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/api/project.ts"
            ? hasFrontendCommandInvocation(sourceCode, command)
            : source.path === "src/lib/governance/agentCommandCatalog.json"
              ? agentCommandCatalogRuntimeSurfaceHas(command)
              : hasStandaloneIdentifier(sourceCode, command);
      if (hasLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  return failures;
}

function collectRetiredExecutionRunFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/api/executionRun.ts",
      message:
        "Execution run 读链已在前端 fail closed，不能重新 safeInvoke 旧 Tauri facade",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "Execution run 读链未接 App Server current，不能回到 Electron Host 白名单",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "Execution run 读链未接 App Server current，不能回到 Electron Host diagnostic facade",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Execution run 读链已 retired，不能继续作为 DevBridge truth command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Execution run 读链已 retired，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/core.ts",
      message: "Execution run 读链已 retired，不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Execution run 读链已 retired，不能回到 runtime/capability catalog",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Execution run 旧 Tauri command surface 已 retired，不能回到 legacy generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "Execution run 旧 Tauri command surface 已 retired，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/runtime_queries.rs",
      message:
        "Execution run 旧 Tauri command surface 已 retired，不能回到 Rust DevBridge runtime query dispatcher",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message:
        "Execution run 旧 Tauri wrapper 已 retired，不能回到 commands 模块树",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredExecutionRunFacadeCommands) {
      const hasLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/api/executionRun.ts"
            ? hasFrontendCommandInvocation(sourceCode, command)
            : source.path === "src/lib/governance/agentCommandCatalog.json"
              ? agentCommandCatalogRuntimeSurfaceHas(command)
              : hasStandaloneIdentifier(sourceCode, command);
      if (hasLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  const retiredWrapperPath = "lime-rs/src/commands/execution_run_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, retiredWrapperPath))) {
    failures.push({
      file: retiredWrapperPath,
      message: "Execution run 旧 Tauri wrapper 已 retired，不能恢复旧文件本体",
      token: retiredWrapperPath,
    });
  }

  return failures;
}

function collectRetiredFrontendSessionFileFacadeSourceFailures() {
  const failures = [];
  const restrictedSources = [
    {
      path: "src/lib/api/session-files.ts",
      message:
        "Session files 工作台写读链已前端 fail closed，不能重新 safeInvoke 旧 Tauri facade",
    },
    {
      path: "src/lib/dev-bridge/commandPolicy.ts",
      message:
        "Session files 旧写读链已 retired，不能继续作为 DevBridge truth / optional legacy command",
    },
    {
      path: "src/lib/dev-bridge/mockPriorityCommands.ts",
      message:
        "Session files 旧写读链已 retired，不能继续作为 mock priority command",
    },
    {
      path: "src/lib/desktop-host/sessionFileMocks.ts",
      message:
        "Session files 旧写读链已 retired，不能保留 desktop-host 默认 mock",
    },
    {
      path: "src/lib/governance/agentCommandCatalog.json",
      message:
        "Session files 旧写读链已 retired，不能回到 runtime/capability catalog",
    },
    {
      path: "electron/ipcChannels.ts",
      message:
        "Session files 旧写读链未接 App Server current，不能回到 Electron Host 白名单",
    },
    {
      path: "electron/hostCommands.ts",
      message:
        "Session files 旧写读链未接 App Server current，不能回到 Electron Host diagnostic facade",
    },
    {
      path: "lime-rs/src/app/runner.rs",
      message:
        "Session files 旧 Tauri command surface 已 retired，不能回到 legacy generate_handler",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher.rs",
      message:
        "Session files 旧 Tauri command surface 已 retired，不能回到 Rust DevBridge dispatcher",
    },
    {
      path: "lime-rs/src/dev_bridge/dispatcher/files.rs",
      message:
        "Session files 旧 Tauri command surface 已 retired，不能回到 Rust DevBridge files dispatcher",
    },
    {
      path: "lime-rs/src/commands/mod.rs",
      message:
        "Session files 旧 Tauri wrapper 已 retired，不能回到 commands 模块树",
    },
  ];

  for (const source of restrictedSources) {
    if (!fs.existsSync(path.join(repoRoot, source.path))) {
      continue;
    }
    const sourceCode = readProductionSourceForGuard(source.path);
    for (const command of retiredFrontendSessionFileFacadeCommands) {
      const hasLeak =
        source.path === "lime-rs/src/app/runner.rs"
          ? hasTauriCommandRegistration(sourceCode, command)
          : source.path === "src/lib/api/session-files.ts"
            ? hasFrontendCommandInvocation(sourceCode, command)
            : source.path === "src/lib/governance/agentCommandCatalog.json"
              ? agentCommandCatalogRuntimeSurfaceHas(command)
              : hasStandaloneIdentifier(sourceCode, command);
      if (hasLeak) {
        failures.push({
          file: source.path,
          message: source.message,
          token: command,
        });
      }
    }
  }

  const retiredWrapperPath = "lime-rs/src/commands/session_files_cmd.rs";
  if (fs.existsSync(path.join(repoRoot, retiredWrapperPath))) {
    failures.push({
      file: retiredWrapperPath,
      message: "Session files 旧 Tauri wrapper 已 retired，不能恢复旧文件本体",
      token: retiredWrapperPath,
    });
  }

  return failures;
}

function collectRetiredTauriGenerateHandlerFailures() {
  const failures = [];
  const source = {
    path: "lime-rs/src/app/runner.rs",
    message:
      "已撤注册的 legacy Tauri command 不能回到 generate_handler；业务能力必须走 App Server current，桌面壳能力必须走 Electron Desktop Host current",
  };
  const sourceCode = readProductionSourceForGuard(source.path);
  for (const command of retiredTauriGenerateHandlerCommands) {
    if (hasTauriCommandRegistration(sourceCode, command)) {
      failures.push({
        file: source.path,
        message: source.message,
        token: command,
      });
    }
  }

  return failures;
}

function collectRetiredTauriCommandModuleFailures() {
  const failures = [];
  const source = {
    path: "lime-rs/src/commands/mod.rs",
    message:
      "已撤注册的 legacy Tauri command module 不能回到 commands/mod.rs；旧 wrapper 文件只能等待确认后物理删除或登记 blocker",
  };
  const sourceCode = readProductionSourceForGuard(source.path);
  for (const moduleName of retiredTauriCommandModules) {
    if (hasStandaloneIdentifier(sourceCode, moduleName)) {
      failures.push({
        file: source.path,
        message: source.message,
        token: moduleName,
      });
    }
    const modulePath = `lime-rs/src/commands/${moduleName}.rs`;
    if (fs.existsSync(path.join(repoRoot, modulePath))) {
      failures.push({
        file: modulePath,
        message:
          "已撤注册的 legacy Tauri command module 文件不能恢复；旧能力必须迁到 App Server current 或 Electron Desktop Host current",
        token: modulePath,
      });
    }
  }

  return failures;
}

function collectRetiredAgentSessionObjectiveBridgeSourceFailures() {
  const failures = [];
  for (const root of productionRuntimeRoots) {
    for (const relativePath of walkDirectory(path.join(repoRoot, root))) {
      const sourceCode = readSourceIfExists(relativePath);
      if (sourceCode?.includes(retiredAgentSessionObjectiveMethodPrefix)) {
        failures.push({
          file: relativePath,
          message:
            "Managed Objective 已整体删除，agentSession/objective/* 不得恢复到 Renderer、Electron Host 或 App Server client",
          token: retiredAgentSessionObjectiveMethodPrefix,
        });
      }
    }
  }
  return failures;
}

function printGuardFailures(title, failures) {
  console.error(`\n## ${title}`);
  for (const failure of failures) {
    console.error(`- ${failure.file}: ${failure.message}`);
    console.error(`  - ${failure.token}`);
  }
}

function main() {
  const frontendUsage = collectFrontendCommandUsage();
  const frontendCommands = new Set(frontendUsage.keys());
  const registeredCommands = collectElectronHostCommands();
  const mockPriorityCommands = collectMockPriorityCommands();
  const bridgeTruthCommands = collectBridgeTruthCommands();
  const noMockFallbackCompatCommands = collectNoMockFallbackCompatCommands();
  const agentCommandCatalog = readAgentCommandCatalog();
  const retiredMemoryRuntimeCatalogSectionLeaks = Object.keys(
    agentCommandCatalog,
  ).filter((sectionName) =>
    retiredMemoryRuntimeCatalogSections.has(sectionName),
  );
  const productionBridgeGuardFailures = collectProductionBridgeGuardFailures();
  const retiredFileBrowserFacadeSourceFailures =
    collectRetiredFileBrowserFacadeSourceFailures();
  const retiredAutomationFacadeSourceFailures =
    collectRetiredAutomationFacadeSourceFailures();
  const retiredApiKeyProviderFacadeSourceFailures =
    collectRetiredApiKeyProviderFacadeSourceFailures();
  const retiredMcpDesktopFacadeSourceFailures =
    collectRetiredMcpDesktopFacadeSourceFailures();
  const retiredPluginPackageFacadeSourceFailures =
    collectRetiredPluginPackageFacadeSourceFailures();
  const retiredLogFacadeSourceFailures =
    collectRetiredLogFacadeSourceFailures();
  const retiredConfigFacadeSourceFailures =
    collectRetiredConfigFacadeSourceFailures();
  const retiredChannelSideEffectFacadeSourceFailures =
    collectRetiredChannelSideEffectFacadeSourceFailures();
  const retiredGatewayTunnelFacadeSourceFailures =
    collectRetiredGatewayTunnelFacadeSourceFailures();
  const retiredPluginShellSourceFailures =
    collectRetiredPluginShellSourceFailures();
  const retiredPluginRuntimeSourceFailures =
    collectRetiredPluginRuntimeSourceFailures();
  const currentTrayDesktopHostShellSourceFailures =
    collectCurrentTrayDesktopHostShellSourceFailures();
  const currentHotkeyDesktopHostShellSourceFailures =
    collectCurrentHotkeyDesktopHostShellSourceFailures();
  const currentVoiceModelDesktopHostSourceFailures =
    collectCurrentVoiceModelDesktopHostSourceFailures();
  const retiredVoiceModelDefaultFacadeSourceFailures =
    collectRetiredVoiceModelDefaultFacadeSourceFailures();
  const retiredVoiceModelTestTranscribeFacadeSourceFailures =
    collectRetiredVoiceModelTestTranscribeFacadeSourceFailures();
  const retiredVoiceInputConfigFacadeSourceFailures =
    collectRetiredVoiceInputConfigFacadeSourceFailures();
  const retiredVoiceAudioDeviceFacadeSourceFailures =
    collectRetiredVoiceAudioDeviceFacadeSourceFailures();
  const retiredVoiceAsrCredentialFacadeSourceFailures =
    collectRetiredVoiceAsrCredentialFacadeSourceFailures();
  const retiredVoiceInstructionFacadeSourceFailures =
    collectRetiredVoiceInstructionFacadeSourceFailures();
  const retiredVoiceRealtimeFacadeSourceFailures =
    collectRetiredVoiceRealtimeFacadeSourceFailures();
  const retiredWorkspaceContentFacadeSourceFailures =
    collectRetiredWorkspaceContentFacadeSourceFailures();
  const retiredExecutionRunFacadeSourceFailures =
    collectRetiredExecutionRunFacadeSourceFailures();
  const retiredFrontendSessionFileFacadeSourceFailures =
    collectRetiredFrontendSessionFileFacadeSourceFailures();
  const currentSkillDesktopHostShellSourceFailures =
    collectCurrentSkillDesktopHostShellSourceFailures();
  const currentLayeredDesignDesktopHostShellSourceFailures =
    collectCurrentLayeredDesignDesktopHostShellSourceFailures();
  const retiredAgentRuntimeRustSourceFailures =
    collectRetiredAgentRuntimeRustSourceFailures();
  const retiredAgentSessionObjectiveBridgeSourceFailures =
    collectRetiredAgentSessionObjectiveBridgeSourceFailures();
  const retiredSkillExecutionFacadeSourceFailures =
    collectRetiredSkillExecutionFacadeSourceFailures();
  const retiredSkillRevealFacadeSourceFailures =
    collectRetiredSkillRevealFacadeSourceFailures();
  const retiredSkillManagementFacadeSourceFailures =
    collectRetiredSkillManagementFacadeSourceFailures();
  const retiredSkillLocalManagementFacadeSourceFailures =
    collectRetiredSkillLocalManagementFacadeSourceFailures();
  const retiredSkillPackageLocalFacadeSourceFailures =
    collectRetiredSkillPackageLocalFacadeSourceFailures();
  const retiredSkillMarketplaceInstallFacadeSourceFailures =
    collectRetiredSkillMarketplaceInstallFacadeSourceFailures();
  const retiredSiteAdapterFacadeSourceFailures =
    collectRetiredSiteAdapterFacadeSourceFailures();
  const retiredMediaTaskArtifactFacadeSourceFailures =
    collectRetiredMediaTaskArtifactFacadeSourceFailures();
  const retiredVideoGenerationPromptSourceFailures =
    collectRetiredVideoGenerationPromptSourceFailures();
  const retiredGalleryMaterialFacadeSourceFailures =
    collectRetiredGalleryMaterialFacadeSourceFailures();
  const retiredProjectMaterialFacadeSourceFailures =
    collectRetiredProjectMaterialFacadeSourceFailures();
  const currentFileBrowserDesktopHostShellSourceFailures =
    collectCurrentFileBrowserDesktopHostShellSourceFailures();
  const retiredTauriGenerateHandlerFailures =
    collectRetiredTauriGenerateHandlerFailures();
  const retiredTauriCommandModuleFailures =
    collectRetiredTauriCommandModuleFailures();

  const deprecatedCommands = new Set(
    Object.keys(agentCommandCatalog.deprecatedCommandReplacements ?? {}),
  );
  const runtimeGatewayCommands = new Set(
    agentCommandCatalog.runtimeGatewayCommands ?? [],
  );
  const capabilityDraftCommands = new Set(
    agentCommandCatalog.capabilityDraftCommands ?? [],
  );

  const deferredCommands = new Set(knownDeferredRegistrationReasons.keys());
  const currentDeferredConflicts = new Set(
    [...currentElectronHostRequiredCommands].filter((command) =>
      deferredCommands.has(command),
    ),
  );

  const missingCurrentRegistrations = new Set(
    [...currentElectronHostRequiredCommands].filter(
      (command) => !registeredCommands.has(command),
    ),
  );
  const missingDevBridgeTruthCommands = new Set(
    [...currentDevBridgeTruthRequiredCommands].filter(
      (command) => !bridgeTruthCommands.has(command),
    ),
  );
  const retiredPluginRuntimeDevBridgeTruthLeaks = new Set(
    [...retiredPluginRuntimeCommands].filter((command) =>
      bridgeTruthCommands.has(command),
    ),
  );
  const retiredPluginRuntimeNoMockCompatLeaks = new Set(
    [...retiredPluginRuntimeCommands].filter((command) =>
      noMockFallbackCompatCommands.has(command),
    ),
  );
  const retiredPluginRuntimeMockPriorityLeaks = new Set(
    [...retiredPluginRuntimeCommands].filter((command) =>
      mockPriorityCommands.has(command),
    ),
  );
  const deprecatedCommandsStillUsed = new Set(
    [...frontendCommands].filter((command) => deprecatedCommands.has(command)),
  );
  const retiredCapabilityDraftAuthoringFrontendLeaks = new Set(
    [...retiredCapabilityDraftAuthoringCommands].filter((command) =>
      frontendCommands.has(command),
    ),
  );
  const retiredFileBrowserFacadeLeaks = new Set(
    [...retiredFileBrowserFacadeCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredAutomationFacadeLeaks = new Set(
    [...retiredAutomationFacadeCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredApiKeyProviderFacadeLeaks = new Set(
    [...retiredApiKeyProviderFacadeCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredMcpDesktopFacadeLeaks = new Set(
    [...retiredMcpDesktopFacadeCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredPluginPackageFacadeLeaks = new Set(
    [...retiredPluginPackageFacadeCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredUsageStatsElectronFacadeLeaks = new Set(
    [...retiredUsageStatsElectronFacadeCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredLogFacadeLeaks = new Set(
    [...retiredLogFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredChannelReadOnlyFacadeLeaks = new Set(
    [...retiredChannelReadOnlyFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredChannelSideEffectFacadeLeaks = new Set(
    [...retiredChannelSideEffectFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredGatewayTunnelFacadeLeaks = new Set(
    [...retiredGatewayTunnelFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredVoiceInputConfigElectronFacadeLeaks = new Set(
    [...retiredVoiceInputConfigElectronFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredVoiceAudioDeviceFacadeLeaks = new Set(
    [...retiredVoiceAudioDeviceFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredVoiceAsrCredentialFacadeLeaks = new Set(
    [...retiredVoiceAsrCredentialFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredVoiceInstructionFacadeLeaks = new Set(
    [...retiredVoiceInstructionFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredVoiceRealtimeFacadeLeaks = new Set(
    [...retiredVoiceRealtimeFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredVoiceModelDefaultFacadeLeaks = new Set(
    [...retiredVoiceModelDefaultFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredVoiceModelTestTranscribeFacadeLeaks = new Set(
    [...retiredVoiceModelTestTranscribeFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredAgentRuntimeExportSurfaceLeaks = new Set(
    [...retiredAgentRuntimeExportRustCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredAgentRuntimeCheckpointCurrentSurfaceLeaks = new Set(
    [...retiredAgentRuntimeCheckpointCurrentCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredAgentRuntimeSubagentSurfaceLeaks = new Set(
    [...retiredAgentRuntimeSubagentRustCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredAgentRuntimeProcessSurfaceLeaks = new Set(
    [...retiredAgentRuntimeProcessRustCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredAgentRuntimeDeleteSessionSurfaceLeaks = new Set(
    ["agent_runtime_delete_session"].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredAgentRuntimeObjectiveCrudGatewayLeaks = new Set(
    [...retiredAgentRuntimeObjectiveCrudGatewayCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredAgentRuntimeObjectiveContinuationGatewayLeaks = new Set(
    [...retiredAgentRuntimeObjectiveContinuationGatewayCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredSkillExecutionFacadeLeaks = new Set(
    [...retiredSkillExecutionFacadeCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredCapabilityDraftAuthoringSurfaceLeaks = new Set(
    [...retiredCapabilityDraftAuthoringCommands].filter(
      (command) =>
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const retiredSkillRevealFacadeLeaks = new Set(
    [...retiredSkillRevealFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredSkillManagementFacadeLeaks = new Set(
    [...retiredSkillManagementFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredSkillPackageLocalFacadeLeaks = new Set(
    [...retiredSkillPackageLocalFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredSkillLocalManagementFacadeLeaks = new Set(
    [...retiredSkillLocalManagementFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredSkillMarketplaceInstallFacadeLeaks = new Set(
    [...retiredSkillMarketplaceInstallFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredSiteAdapterFacadeLeaks = new Set(
    [...retiredSiteAdapterFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredMediaTaskArtifactFacadeLeaks = new Set(
    [...retiredMediaTaskArtifactFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredGalleryMaterialFacadeLeaks = new Set(
    [...retiredGalleryMaterialFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredFrontendSessionFileFacadeLeaks = new Set(
    [...retiredFrontendSessionFileFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredExecutionRunFacadeLeaks = new Set(
    [...retiredExecutionRunFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredUnifiedMemoryFacadeLeaks = new Set(
    [...retiredUnifiedMemoryFacadeCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const retiredMemoryRuntimeFacadeLeaks = new Set(
    [...retiredMemoryRuntimeTauriGenerateHandlerCommands].filter(
      (command) =>
        frontendCommands.has(command) ||
        registeredCommands.has(command) ||
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const currentFileBrowserDesktopHostShellBridgeLeaks = new Set(
    [...currentFileBrowserDesktopHostShellCommands].filter(
      (command) =>
        bridgeTruthCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command),
    ),
  );
  const currentTrayDesktopHostShellBridgeLeaks = new Set(
    [...currentTrayDesktopHostShellCommands].filter(
      (command) =>
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const currentHotkeyDesktopHostShellBridgeLeaks = new Set(
    [...currentHotkeyDesktopHostShellCommands].filter(
      (command) =>
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const currentVoiceModelDesktopHostCommandBridgeLeaks = new Set(
    [
      ...currentVoiceModelDesktopHostReadCommands,
      ...currentVoiceModelDesktopHostSideEffectCommands,
    ].filter(
      (command) =>
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const currentSkillDesktopHostShellBridgeLeaks = new Set(
    [...currentSkillDesktopHostShellCommands].filter(
      (command) =>
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const currentLayeredDesignDesktopHostShellBridgeLeaks = new Set(
    [...currentLayeredDesignDesktopHostShellCommands].filter(
      (command) =>
        bridgeTruthCommands.has(command) ||
        mockPriorityCommands.has(command) ||
        runtimeGatewayCommands.has(command) ||
        capabilityDraftCommands.has(command) ||
        deferredCommands.has(command),
    ),
  );
  const runtimeGatewayMissingRegistrations = new Set(
    [...runtimeGatewayCommands].filter(
      (command) =>
        !registeredCommands.has(command) && !deferredCommands.has(command),
    ),
  );
  const unclassifiedRuntimeGatewayCommands = new Set(
    [...runtimeGatewayCommands].filter(
      (command) =>
        !currentElectronHostRequiredCommands.has(command) &&
        !deferredCommands.has(command),
    ),
  );
  const capabilityDraftMissingRegistrations = new Set(
    [...capabilityDraftCommands].filter(
      (command) =>
        !registeredCommands.has(command) && !deferredCommands.has(command),
    ),
  );

  console.log("[command-contracts] frontend commands:", frontendCommands.size);
  console.log(
    "[command-contracts] Electron host commands:",
    registeredCommands.size,
  );
  console.log(
    "[command-contracts] mock priority commands:",
    mockPriorityCommands.size,
  );
  console.log(
    "[command-contracts] DevBridge truth commands:",
    bridgeTruthCommands.size,
  );

  if (knownDeferredRegistrationReasons.size > 0) {
    console.log("\n[command-contracts] 已登记的延期命令：");
    for (const command of sortCommands(
      knownDeferredRegistrationReasons.keys(),
    )) {
      console.log(`- ${command}`);
      console.log(`  ${knownDeferredRegistrationReasons.get(command)}`);
    }
  }

  let hasError = false;

  if (currentDeferredConflicts.size > 0) {
    hasError = true;
    printCommandGroup(
      "命令不能同时标记为 current 与 compat/deferred",
      currentDeferredConflicts,
    );
  }

  if (missingCurrentRegistrations.size > 0) {
    hasError = true;
    printCommandGroup(
      "current Electron/App Server 命令缺少 Electron host 承接",
      missingCurrentRegistrations,
      frontendUsage,
    );
  }

  if (missingDevBridgeTruthCommands.size > 0) {
    hasError = true;
    printCommandGroup(
      "current App Server 数据面命令缺少 DevBridge truth 分类",
      missingDevBridgeTruthCommands,
      frontendUsage,
    );
  }

  if (retiredPluginRuntimeDevBridgeTruthLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已退役 Plugin runtime task 命令不能回到 DevBridge truth surface",
      retiredPluginRuntimeDevBridgeTruthLeaks,
    );
  }

  if (retiredPluginRuntimeNoMockCompatLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已退役 Plugin runtime task 命令不能留在 no-mock compat surface",
      retiredPluginRuntimeNoMockCompatLeaks,
    );
  }

  if (retiredPluginRuntimeMockPriorityLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已退役 Plugin runtime task 命令不能回到 mock priority surface",
      retiredPluginRuntimeMockPriorityLeaks,
    );
  }

  if (productionBridgeGuardFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "生产桥接路径不能回退 legacy Tauri / renderer mock / mock backend",
      productionBridgeGuardFailures,
    );
  }

  if (deprecatedCommandsStillUsed.size > 0) {
    hasError = true;
    printCommandGroup(
      "前端仍在调用的废弃命令",
      deprecatedCommandsStillUsed,
      frontendUsage,
    );
  }

  if (retiredCapabilityDraftAuthoringFrontendLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已退役的 Capability Draft authoring 命令不能回到前端生产 safeInvoke / invoke",
      retiredCapabilityDraftAuthoringFrontendLeaks,
      frontendUsage,
    );
  }

  if (retiredFileBrowserFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server fs/* 的旧文件浏览命令不能回到 Electron Host 或 DevBridge truth surface",
      retiredFileBrowserFacadeLeaks,
    );
  }

  if (retiredFileBrowserFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server fs/* 的旧文件浏览命令不能回到旧客户端源码",
      retiredFileBrowserFacadeSourceFailures,
    );
  }

  if (retiredAutomationFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server automation* 的旧自动化命令不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredAutomationFacadeLeaks,
    );
  }

  if (retiredAutomationFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server automation* 的旧自动化命令不能回到旧客户端源码",
      retiredAutomationFacadeSourceFailures,
    );
  }

  if (retiredApiKeyProviderFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server modelProvider/* 的旧 Provider 命令不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredApiKeyProviderFacadeLeaks,
    );
  }

  if (retiredMcpDesktopFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server MCP current API 的旧 MCP Desktop facade 不能回到 Electron Host / DevBridge / mock / runtime catalog",
      retiredMcpDesktopFacadeLeaks,
    );
  }

  if (retiredAgentRuntimeObjectiveCrudGatewayLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "Managed Objective 已整体删除，旧 objective CRUD 命令不能回到 Electron Host / DevBridge / mock / runtime catalog",
      retiredAgentRuntimeObjectiveCrudGatewayLeaks,
    );
  }

  if (retiredAgentRuntimeObjectiveContinuationGatewayLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "Managed Objective 已整体删除，旧 objective continue/audit 命令不能回到 Electron Host / DevBridge / mock / runtime catalog",
      retiredAgentRuntimeObjectiveContinuationGatewayLeaks,
    );
  }

  if (retiredAgentSessionObjectiveBridgeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "agentSession/objective/* 已 dead / deleted / forbidden-to-restore",
      retiredAgentSessionObjectiveBridgeSourceFailures,
    );
  }

  if (retiredMcpDesktopFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server MCP current API 的旧 MCP 命令不能回到旧客户端源码",
      retiredMcpDesktopFacadeSourceFailures,
    );
  }

  if (retiredApiKeyProviderFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server modelProvider/* 的旧 Provider 命令不能回到旧客户端源码",
      retiredApiKeyProviderFacadeSourceFailures,
    );
  }

  if (retiredPluginPackageFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server plugin* 的旧 Plugin lifecycle 命令不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredPluginPackageFacadeLeaks,
    );
  }

  if (retiredPluginPackageFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server plugin* 的旧 Plugin package/install 命令不能回到旧客户端源码",
      retiredPluginPackageFacadeSourceFailures,
    );
  }

  if (retiredUsageStatsElectronFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server usageStats/* 的旧 Usage Stats Electron façade 不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredUsageStatsElectronFacadeLeaks,
    );
  }

  if (retiredLogFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server log/* 与 diagnostics/* 的旧诊断 facade 不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredLogFacadeLeaks,
    );
  }

  if (retiredLogFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server log/* 与 diagnostics/* 的旧诊断命令不能回到旧客户端源码",
      retiredLogFacadeSourceFailures,
    );
  }

  if (retiredChannelReadOnlyFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server gatewayChannel/status + wechatChannel/accounts/list 的旧 Channels/WeChat 只读 facade 不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredChannelReadOnlyFacadeLeaks,
    );
  }

  if (retiredChannelSideEffectFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server Channels side-effect current API 的旧 Channels/WeChat facade 不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredChannelSideEffectFacadeLeaks,
      frontendUsage,
    );
  }

  if (retiredChannelSideEffectFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server Channels side-effect current API 的旧 Channels/WeChat 命令不能回到旧客户端源码",
      retiredChannelSideEffectFacadeSourceFailures,
    );
  }

  if (retiredGatewayTunnelFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server gatewayTunnel/* 的旧 tunnel facade 不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredGatewayTunnelFacadeLeaks,
      frontendUsage,
    );
  }

  if (retiredGatewayTunnelFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server gatewayTunnel/* 的旧 tunnel 命令不能回到旧客户端源码",
      retiredGatewayTunnelFacadeSourceFailures,
    );
  }

  if (retiredConfigFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 Electron Desktop Host current 的旧 config facade 不能回到 Rust 旧源码",
      retiredConfigFacadeSourceFailures,
    );
  }

  if (retiredVoiceInputConfigElectronFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已收敛到 app config current 网关的旧 Voice Input config Electron façade 不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredVoiceInputConfigElectronFacadeLeaks,
    );
  }

  if (retiredVoiceAudioDeviceFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已收敛到 renderer media devices current 的旧音频设备命令不能回到前端 safeInvoke、Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredVoiceAudioDeviceFacadeLeaks,
    );
  }

  if (retiredVoiceAsrCredentialFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server voiceAsrCredential/* 的旧 ASR 凭证命令不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredVoiceAsrCredentialFacadeLeaks,
      frontendUsage,
    );
  }

  if (retiredVoiceInstructionFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server voiceInstruction/* 的旧 Voice instruction 命令不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredVoiceInstructionFacadeLeaks,
      frontendUsage,
    );
  }

  if (retiredVoiceRealtimeFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "旧实时语音转写 / 录音命令不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单；后续必须先定义 current voice runtime 通道",
      retiredVoiceRealtimeFacadeLeaks,
      frontendUsage,
    );
  }

  if (retiredVoiceModelDefaultFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server voiceModel/default/set 的旧语音模型设默认命令不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredVoiceModelDefaultFacadeLeaks,
      frontendUsage,
    );
  }

  if (retiredVoiceModelTestTranscribeFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "语音模型测试转写旧 facade 已退役，不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredVoiceModelTestTranscribeFacadeLeaks,
      frontendUsage,
    );
  }

  if (retiredAgentRuntimeExportSurfaceLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server agentSession export current 的旧 Agent Runtime export 命令不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredAgentRuntimeExportSurfaceLeaks,
    );
  }

  if (retiredAgentRuntimeCheckpointCurrentSurfaceLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server agentSession/fileCheckpoint current 的旧 Agent Runtime checkpoint 命令不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredAgentRuntimeCheckpointCurrentSurfaceLeaks,
    );
  }

  if (retiredAgentRuntimeSubagentSurfaceLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已退役的 public subagent facade 不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredAgentRuntimeSubagentSurfaceLeaks,
    );
  }

  if (retiredAgentRuntimeProcessSurfaceLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已退役的 Agent Runtime process side-effect facade 不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredAgentRuntimeProcessSurfaceLeaks,
    );
  }

  if (retiredAgentRuntimeDeleteSessionSurfaceLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已收敛到 App Server thread/delete 的旧 delete session 命令不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredAgentRuntimeDeleteSessionSurfaceLeaks,
    );
  }

  if (retiredSkillExecutionFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已下线的 execute_skill 独立执行命令不能回到 Electron Host、DevBridge truth、mock priority 或 runtime surface",
      retiredSkillExecutionFacadeLeaks,
    );
  }

  if (retiredCapabilityDraftAuthoringSurfaceLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已退役的 Capability Draft authoring 命令不能回到 Electron Host、DevBridge truth、mock priority 或 runtime/capability catalog",
      retiredCapabilityDraftAuthoringSurfaceLeaks,
    );
  }

  if (retiredSkillRevealFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server skills/list + Electron reveal_in_finder 的旧 Skill reveal facade 不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredSkillRevealFacadeLeaks,
    );
  }

  if (retiredSkillManagementFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server skillManagement/* / skillRepository/* / skillLocal/* / skillRemote/inspect 的旧 Skill 管理 facade 不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredSkillManagementFacadeLeaks,
    );
  }

  if (retiredSkillPackageLocalFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server skillPackage/local/* + skillPackage/export 的旧本地 Skill package facade 不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredSkillPackageLocalFacadeLeaks,
    );
  }

  if (retiredSkillLocalManagementFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server skillLocal/* + skillPackage/local/replace 的旧本地 Skill 管理 facade 不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredSkillLocalManagementFacadeLeaks,
    );
  }

  if (retiredSkillMarketplaceInstallFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 App Server skillMarketplace/install + skillPackage/download/install 的旧官方 Skill 安装 facade 不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredSkillMarketplaceInstallFacadeLeaks,
    );
  }

  if (retiredSiteAdapterFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "Site Adapter 旧 facade 已退役，不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredSiteAdapterFacadeLeaks,
    );
  }

  if (retiredMediaTaskArtifactFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "Media task artifact 旧 facade 已迁到 App Server current，不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredMediaTaskArtifactFacadeLeaks,
    );
  }

  if (retiredGalleryMaterialFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "Gallery material 旧 facade 已迁到 App Server current，不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredGalleryMaterialFacadeLeaks,
    );
  }

  if (retiredWorkspaceContentFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "Workspace 写链 / Content CRUD retired facade 不能回到旧客户端源码",
      retiredWorkspaceContentFacadeSourceFailures,
    );
  }

  if (retiredExecutionRunFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "Execution run 旧读链已前端 fail closed，不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredExecutionRunFacadeLeaks,
      frontendUsage,
    );
  }

  if (retiredExecutionRunFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "Execution run 旧读链 retired facade 不能回到受限事实源",
      retiredExecutionRunFacadeSourceFailures,
    );
  }

  if (retiredFrontendSessionFileFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "Session files 旧写读链已前端 fail closed，不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredFrontendSessionFileFacadeLeaks,
      frontendUsage,
    );
  }

  if (retiredFrontendSessionFileFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "Session files 旧写读链 retired facade 不能回到受限事实源",
      retiredFrontendSessionFileFacadeSourceFailures,
    );
  }

  if (retiredUnifiedMemoryFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "旧 unified_memory_* 已被 memoryStore/* current 取代，不能回到前端调用、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredUnifiedMemoryFacadeLeaks,
      frontendUsage,
    );
  }
  if (retiredMemoryRuntimeCatalogSectionLeaks.length > 0) {
    hasError = true;
    console.error(
      `\n[command-contracts] retired Memory runtime catalog sections must not come back: ${retiredMemoryRuntimeCatalogSectionLeaks.join(
        ", ",
      )}`,
    );
  }
  if (retiredMemoryRuntimeFacadeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已退役的 Memory runtime Tauri 命令不能回到前端调用、Electron Host、DevBridge truth、mock priority、runtime catalog 或 deferred 白名单",
      retiredMemoryRuntimeFacadeLeaks,
      frontendUsage,
    );
  }

  if (retiredSkillExecutionFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已下线的 execute_skill 独立执行命令不能回到旧客户端源码",
      retiredSkillExecutionFacadeSourceFailures,
    );
  }

  if (retiredSkillRevealFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server skills/list + Electron reveal_in_finder 的旧 Skill reveal facade 不能回到旧客户端源码",
      retiredSkillRevealFacadeSourceFailures,
    );
  }

  if (retiredSkillManagementFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server skillManagement/* / skillRepository/* / skillLocal/* / skillRemote/inspect 的旧 Skill 管理 facade 不能回到旧客户端源码",
      retiredSkillManagementFacadeSourceFailures,
    );
  }

  if (retiredSkillPackageLocalFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server skillPackage/local/* + skillPackage/export 的旧本地 Skill package facade 不能回到旧客户端源码",
      retiredSkillPackageLocalFacadeSourceFailures,
    );
  }

  if (retiredSkillLocalManagementFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server skillLocal/* + skillPackage/local/replace 的旧本地 Skill 管理 facade 不能回到旧客户端源码",
      retiredSkillLocalManagementFacadeSourceFailures,
    );
  }

  if (retiredSkillMarketplaceInstallFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server skillMarketplace/install + skillPackage/download/install 的旧官方 Skill 安装 facade 不能回到旧客户端源码",
      retiredSkillMarketplaceInstallFacadeSourceFailures,
    );
  }

  if (retiredSiteAdapterFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "Site Adapter 旧 facade 已退役，不能回到受限事实源",
      retiredSiteAdapterFacadeSourceFailures,
    );
  }

  if (retiredGalleryMaterialFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "Gallery material 旧 facade 已迁到 App Server galleryMaterial/*，不能回到受限事实源",
      retiredGalleryMaterialFacadeSourceFailures,
    );
  }

  if (retiredProjectMaterialFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "Project material 旧 facade 已迁到 App Server projectMaterial/*，不能回到受限事实源",
      retiredProjectMaterialFacadeSourceFailures,
    );
  }

  if (retiredMediaTaskArtifactFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "Media task artifact 旧 facade 已迁到 App Server mediaTaskArtifact/*，不能回到受限事实源",
      retiredMediaTaskArtifactFacadeSourceFailures,
    );
  }

  if (retiredVideoGenerationPromptSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "Video generation prompt 不能回退旧 video generation facade",
      retiredVideoGenerationPromptSourceFailures,
    );
  }

  if (retiredPluginShellSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已退役 Plugin shell 命令不能回到旧客户端源码",
      retiredPluginShellSourceFailures,
    );
  }

  if (retiredPluginRuntimeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已退役 Plugin runtime task 命令不能回到旧客户端源码",
      retiredPluginRuntimeSourceFailures,
    );
  }

  if (currentTrayDesktopHostShellBridgeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 Electron Desktop Host 的托盘壳命令不能回到 DevBridge truth、mock priority、runtime gateway 或 deferred surface",
      currentTrayDesktopHostShellBridgeLeaks,
    );
  }

  if (currentTrayDesktopHostShellSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 Electron Desktop Host 的托盘壳命令不能回到旧客户端源码",
      currentTrayDesktopHostShellSourceFailures,
    );
  }

  if (currentHotkeyDesktopHostShellBridgeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 Electron Desktop Host 的快捷键壳命令不能回到 DevBridge truth、mock priority、runtime gateway 或 deferred surface",
      currentHotkeyDesktopHostShellBridgeLeaks,
    );
  }

  if (currentHotkeyDesktopHostShellSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 Electron Desktop Host 的快捷键壳命令不能回到旧客户端源码",
      currentHotkeyDesktopHostShellSourceFailures,
    );
  }

  if (currentVoiceModelDesktopHostCommandBridgeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 Electron Desktop Host 的语音模型命令不能回到 DevBridge truth、mock priority、runtime gateway 或 deferred surface",
      currentVoiceModelDesktopHostCommandBridgeLeaks,
    );
  }

  if (currentVoiceModelDesktopHostSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 Electron Desktop Host 的语音模型命令不能回到旧客户端源码",
      currentVoiceModelDesktopHostSourceFailures,
    );
  }

  if (retiredVoiceModelDefaultFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server voiceModel/default/set 的旧语音模型设默认命令不能回到旧客户端源码",
      retiredVoiceModelDefaultFacadeSourceFailures,
    );
  }

  if (retiredVoiceModelTestTranscribeFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "语音模型测试转写旧 facade 已退役，不能回到旧客户端源码",
      retiredVoiceModelTestTranscribeFacadeSourceFailures,
    );
  }

  if (retiredVoiceInputConfigFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已收敛到 app config current 网关的旧 Voice Input config 命令不能回到旧客户端源码",
      retiredVoiceInputConfigFacadeSourceFailures,
    );
  }

  if (retiredVoiceAudioDeviceFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已收敛到 renderer media devices current 的旧音频设备命令不能回到旧客户端源码",
      retiredVoiceAudioDeviceFacadeSourceFailures,
    );
  }

  if (retiredVoiceAsrCredentialFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server voiceAsrCredential/* 的旧 ASR 凭证命令不能回到旧客户端源码",
      retiredVoiceAsrCredentialFacadeSourceFailures,
    );
  }

  if (retiredVoiceInstructionFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 App Server voiceInstruction/* 的旧 Voice instruction 命令不能回到旧客户端源码",
      retiredVoiceInstructionFacadeSourceFailures,
    );
  }

  if (retiredVoiceRealtimeFacadeSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "旧实时语音转写 / 录音命令不能回到旧客户端源码；后续必须先定义 current voice runtime 通道",
      retiredVoiceRealtimeFacadeSourceFailures,
    );
  }

  if (currentSkillDesktopHostShellBridgeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 Electron Desktop Host 的 Skill 壳命令不能回到 DevBridge truth、mock priority、runtime gateway 或 deferred surface",
      currentSkillDesktopHostShellBridgeLeaks,
    );
  }

  if (currentSkillDesktopHostShellSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 Electron Desktop Host 的 Skill 壳命令不能回到旧客户端源码",
      currentSkillDesktopHostShellSourceFailures,
    );
  }

  if (currentLayeredDesignDesktopHostShellBridgeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 Electron Desktop Host 的 Layered Design 壳命令不能回到 DevBridge truth、mock priority、runtime gateway 或 deferred surface",
      currentLayeredDesignDesktopHostShellBridgeLeaks,
    );
  }

  if (currentLayeredDesignDesktopHostShellSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 Electron Desktop Host 的 Layered Design 壳命令不能回到旧客户端源码",
      currentLayeredDesignDesktopHostShellSourceFailures,
    );
  }

  if (retiredAgentRuntimeRustSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "Agent Runtime residual 旧 Rust Tauri 入口不能回流",
      retiredAgentRuntimeRustSourceFailures,
    );
  }

  if (currentFileBrowserDesktopHostShellBridgeLeaks.size > 0) {
    hasError = true;
    printCommandGroup(
      "已迁到 Electron Desktop Host 的文件浏览壳命令不能回到 DevBridge truth 或 runtime gateway surface",
      currentFileBrowserDesktopHostShellBridgeLeaks,
    );
  }

  if (currentFileBrowserDesktopHostShellSourceFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已迁到 Electron Desktop Host 的文件浏览壳命令不能回到旧客户端源码",
      currentFileBrowserDesktopHostShellSourceFailures,
    );
  }

  if (retiredTauriGenerateHandlerFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已撤注册的 legacy Tauri command 不能回到 runner generate_handler",
      retiredTauriGenerateHandlerFailures,
    );
  }

  if (retiredTauriCommandModuleFailures.length > 0) {
    hasError = true;
    printGuardFailures(
      "已撤注册的 legacy Tauri command module 不能回到 commands/mod.rs",
      retiredTauriCommandModuleFailures,
    );
  }

  if (runtimeGatewayMissingRegistrations.size > 0) {
    hasError = true;
    printCommandGroup(
      "runtime gateway 命令缺少 Electron host 承接",
      runtimeGatewayMissingRegistrations,
    );
  }

  if (unclassifiedRuntimeGatewayCommands.size > 0) {
    hasError = true;
    printCommandGroup(
      "runtime gateway 命令未分类为 current 或 compat/deferred",
      unclassifiedRuntimeGatewayCommands,
    );
  }

  if (capabilityDraftMissingRegistrations.size > 0) {
    hasError = true;
    printCommandGroup(
      "capability draft 命令缺少 Electron host 承接",
      capabilityDraftMissingRegistrations,
    );
  }

  if (hasError) {
    process.exitCode = 1;
    return;
  }

  console.log("\n[command-contracts] 所有命令契约检查通过。");
}

main();
