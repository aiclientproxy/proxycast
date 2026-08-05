//! JSON-RPC method dispatch for the App Server processor.

mod response;
mod v2_ingress;
use super::thread_resume_context::request_id_for_thread_resume;
use super::{ConnectionRequestId, JsonRpcError, RequestProcessor};
use crate::AppServerError;
use app_server_protocol::error_codes;
use app_server_protocol::JsonRpcErrorResponse;
use app_server_protocol::JsonRpcMessage;
use app_server_protocol::JsonRpcRequest;
use app_server_protocol::*;
use futures::future::ready;
use futures::FutureExt;

impl RequestProcessor {
    pub(super) async fn handle_request_inner(
        &self,
        request: JsonRpcRequest,
        connection_request_id: Option<ConnectionRequestId>,
        event_callback: Option<&mut (dyn FnMut(JsonRpcMessage) + Send)>,
    ) -> Result<Vec<JsonRpcMessage>, AppServerError> {
        let v2_request = match v2_ingress::decode(&request) {
            Ok(request) => request,
            Err(error) => {
                return Ok(vec![JsonRpcMessage::Error(JsonRpcErrorResponse {
                    id: request.id,
                    error,
                })]);
            }
        };
        let request_id = request.id.clone();
        let (id, method, params) = if let Some(v2_request) = v2_request {
            match v2_ingress::into_parts(v2_request) {
                Ok(parts) => parts,
                Err(error) => {
                    return Ok(vec![JsonRpcMessage::Error(JsonRpcErrorResponse {
                        id: request_id,
                        error,
                    })]);
                }
            }
        } else {
            let request = match AppServerClientRequest::try_from(request) {
                Ok(request) => request,
                Err(error) => {
                    return Ok(vec![JsonRpcMessage::Error(JsonRpcErrorResponse {
                        id: request_id,
                        error: JsonRpcError::new(error_codes::METHOD_NOT_FOUND, error),
                    })]);
                }
            };
            let (id, method, params) = request.into_jsonrpc_parts();
            (id, method.as_str().to_string(), params)
        };
        let method = method.as_str();
        let is_transport_request = connection_request_id.is_some();
        let thread_resume_request_id =
            request_id_for_thread_resume(method, connection_request_id.as_ref());
        if self.is_request_canceled(&id) {
            self.clear_request_cancel_state(&id);
            return Ok(vec![JsonRpcMessage::Error(JsonRpcErrorResponse {
                id,
                error: JsonRpcError::new(error_codes::REQUEST_CANCELLED, "request canceled"),
            })]);
        }
        let result = match method {
            METHOD_INITIALIZE if is_transport_request => {
                ready(self.handle_transport_initialize(params)).boxed()
            }
            METHOD_INITIALIZE => ready(self.handle_initialize(params)).boxed(),
            METHOD_CAPABILITY_LIST => ready(self.handle_capability_list(params)).boxed(),
            METHOD_ARTIFACT_READ => ready(self.handle_artifact_read(params)).boxed(),
            app_server_protocol::protocol::v2::METHOD_ARTIFACT_WRITE => {
                self.handle_artifact_write_v2_impl(params).boxed()
            }
            METHOD_FILE_SYSTEM_LIST_DIRECTORY => {
                self.handle_file_system_list_directory_impl(params).boxed()
            }
            METHOD_FILE_SYSTEM_READ_FILE_PREVIEW => self
                .handle_file_system_read_file_preview_impl(params)
                .boxed(),
            METHOD_FILE_SYSTEM_CREATE_FILE => {
                self.handle_file_system_create_file_impl(params).boxed()
            }
            METHOD_FILE_SYSTEM_CREATE_DIRECTORY => self
                .handle_file_system_create_directory_impl(params)
                .boxed(),
            METHOD_FILE_SYSTEM_RENAME_FILE => {
                self.handle_file_system_rename_file_impl(params).boxed()
            }
            METHOD_FILE_SYSTEM_DELETE_FILE => {
                self.handle_file_system_delete_file_impl(params).boxed()
            }
            METHOD_PROJECT_GIT_STATUS => self.handle_project_git_status_impl(params).boxed(),
            METHOD_PROJECT_GIT_DIFF => self.handle_project_git_diff_impl(params).boxed(),
            METHOD_PROJECT_GIT_COMMITS_LIST => {
                self.handle_project_git_commits_list_impl(params).boxed()
            }
            METHOD_PROJECT_GIT_BRANCH_CHECKOUT => {
                self.handle_project_git_branch_checkout_impl(params).boxed()
            }
            METHOD_PROJECT_GIT_BRANCH_CREATE => {
                self.handle_project_git_branch_create_impl(params).boxed()
            }
            METHOD_PROJECT_GIT_WORKTREE_CREATE => {
                self.handle_project_git_worktree_create_impl(params).boxed()
            }
            METHOD_PROJECT_SHELL_SESSION_START => {
                self.handle_project_shell_session_start_impl(params).boxed()
            }
            METHOD_PROJECT_SHELL_SESSION_WRITE => {
                self.handle_project_shell_session_write_impl(params).boxed()
            }
            METHOD_PROJECT_SHELL_SESSION_RESIZE => self
                .handle_project_shell_session_resize_impl(params)
                .boxed(),
            METHOD_PROJECT_SHELL_SESSION_KILL => {
                self.handle_project_shell_session_kill_impl(params).boxed()
            }
            METHOD_PROJECT_SHELL_SESSION_DRAIN_EVENTS => self
                .handle_project_shell_session_drain_events_impl(params)
                .boxed(),
            METHOD_EXECUTION_PROCESS_START => {
                self.handle_execution_process_start_impl(params).boxed()
            }
            METHOD_EXECUTION_PROCESS_WRITE_STDIN => self
                .handle_execution_process_write_stdin_impl(params)
                .boxed(),
            METHOD_EXECUTION_PROCESS_INTERRUPT => {
                self.handle_execution_process_interrupt_impl(params).boxed()
            }
            METHOD_EXECUTION_PROCESS_TERMINATE => {
                self.handle_execution_process_terminate_impl(params).boxed()
            }
            METHOD_EXECUTION_PROCESS_STATUS => {
                self.handle_execution_process_status_impl(params).boxed()
            }
            METHOD_EXECUTION_PROCESS_DRAIN_OUTPUT => self
                .handle_execution_process_drain_output_impl(params)
                .boxed(),
            METHOD_EVIDENCE_EXPORT => self.handle_evidence_export(params).boxed(),
            METHOD_AGENT_SESSION_HANDOFF_BUNDLE_EXPORT => {
                self.handle_handoff_bundle_export(params).boxed()
            }
            METHOD_AGENT_SESSION_REPLAY_CASE_EXPORT => {
                self.handle_replay_case_export(params).boxed()
            }
            METHOD_AGENT_SESSION_ANALYSIS_HANDOFF_EXPORT => {
                self.handle_analysis_handoff_export(params).boxed()
            }
            METHOD_AGENT_SESSION_REVIEW_DECISION_TEMPLATE_EXPORT => {
                self.handle_review_decision_template_export(params).boxed()
            }
            METHOD_AGENT_SESSION_REVIEW_DECISION_SAVE => {
                self.handle_review_decision_save(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_COMPACT_START => {
                self.handle_thread_compact_start_v2(params).boxed()
            }
            METHOD_THREAD_RESUME => self
                .handle_thread_resume_v2(params, thread_resume_request_id)
                .boxed(),
            METHOD_AGENT_SESSION_FILE_CHECKPOINT_LIST => {
                self.handle_file_checkpoint_list_impl(params).boxed()
            }
            METHOD_AGENT_SESSION_FILE_CHECKPOINT_GET => {
                self.handle_file_checkpoint_get_impl(params).boxed()
            }
            METHOD_AGENT_SESSION_FILE_CHECKPOINT_DIFF => {
                self.handle_file_checkpoint_diff_impl(params).boxed()
            }
            METHOD_AGENT_SESSION_FILE_CHECKPOINT_RESTORE => {
                self.handle_file_checkpoint_restore_impl(params).boxed()
            }
            METHOD_SESSION_FILE_GET_OR_CREATE => {
                self.handle_session_file_get_or_create_impl(params).boxed()
            }
            METHOD_SESSION_FILE_UPDATE_META => {
                self.handle_session_file_update_meta_impl(params).boxed()
            }
            METHOD_SESSION_FILE_SAVE => self.handle_session_file_save_impl(params).boxed(),
            METHOD_SESSION_FILE_READ => self.handle_session_file_read_impl(params).boxed(),
            METHOD_SESSION_FILE_RESOLVE_PATH => {
                self.handle_session_file_resolve_path_impl(params).boxed()
            }
            METHOD_SESSION_FILE_DELETE => self.handle_session_file_delete_impl(params).boxed(),
            METHOD_SESSION_FILE_LIST => self.handle_session_file_list_impl(params).boxed(),
            METHOD_THREAD_START => self.handle_thread_start_v2(params).boxed(),
            app_server_protocol::protocol::v2::METHOD_THREAD_FORK => {
                self.handle_thread_fork_v2(params).boxed()
            }
            METHOD_THREAD_READ => self.handle_thread_read_impl(params).boxed(),
            METHOD_THREAD_LIST => self.handle_thread_list_impl(params).boxed(),
            app_server_protocol::protocol::v2::METHOD_THREAD_LOADED_LIST => {
                self.handle_thread_loaded_list_v2(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_UNSUBSCRIBE => self
                .handle_thread_unsubscribe_v2(params, connection_request_id)
                .boxed(),
            app_server_protocol::protocol::v2::METHOD_THREAD_INCREMENT_ELICITATION => {
                self.handle_thread_increment_elicitation_v2(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_DECREMENT_ELICITATION => {
                self.handle_thread_decrement_elicitation_v2(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_ARCHIVE => {
                self.handle_thread_archive_v2(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_DELETE => {
                self.handle_thread_delete_v2(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_UNARCHIVE => {
                self.handle_thread_unarchive_v2(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_NAME_SET => {
                self.handle_thread_name_set_v2(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_METADATA_UPDATE => {
                self.handle_thread_metadata_update_v2(params).boxed()
            }
            METHOD_THREAD_TURNS_LIST => self.handle_thread_turns_list_impl(params).boxed(),
            METHOD_THREAD_ITEMS_LIST => self.handle_thread_items_list_impl(params).boxed(),
            app_server_protocol::protocol::v2::METHOD_THREAD_INJECT_ITEMS => {
                self.handle_thread_inject_items_v2(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_SEARCH => {
                self.handle_thread_search_v2(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_SEARCH_OCCURRENCES => {
                self.handle_thread_search_occurrences_v2(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_SETTINGS_UPDATE => {
                self.handle_thread_settings_update_impl(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_MEMORY_MODE_SET => {
                self.handle_thread_memory_mode_set_impl(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_SHELL_COMMAND => {
                self.handle_thread_shell_command_impl(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_APPROVE_GUARDIAN_DENIED_ACTION => self
                .handle_thread_approve_guardian_denied_action_v2(params)
                .boxed(),
            app_server_protocol::protocol::v2::METHOD_THREAD_BACKGROUND_TERMINALS_CLEAN => self
                .handle_thread_background_terminals_clean_v2(params)
                .boxed(),
            app_server_protocol::protocol::v2::METHOD_THREAD_BACKGROUND_TERMINALS_LIST => self
                .handle_thread_background_terminals_list_v2(params)
                .boxed(),
            app_server_protocol::protocol::v2::METHOD_THREAD_BACKGROUND_TERMINALS_TERMINATE => self
                .handle_thread_background_terminals_terminate_v2(params)
                .boxed(),
            app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_SET => {
                self.handle_thread_goal_set(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_GET => {
                self.handle_thread_goal_get(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_THREAD_GOAL_CLEAR => {
                self.handle_thread_goal_clear(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_MEDIA_READ => self
                .handle_media_read_v2_impl(&id, params, event_callback)
                .boxed(),
            METHOD_WORKFLOW_READ => self.handle_workflow_read_impl(params).boxed(),
            METHOD_WORKFLOW_CANCEL => self.handle_workflow_cancel_impl(params).boxed(),
            METHOD_WORKFLOW_RETRY => self.handle_workflow_retry_impl(params).boxed(),
            METHOD_WORKFLOW_RESPOND => self.handle_workflow_respond_impl(params).boxed(),
            METHOD_WORKSPACE_LIST => self.handle_workspace_list_impl().boxed(),
            METHOD_WORKSPACE_READ => self.handle_workspace_read_impl(params).boxed(),
            METHOD_WORKSPACE_UPDATE => self.handle_workspace_update_impl(params).boxed(),
            METHOD_WORKSPACE_DELETE => self.handle_workspace_delete_impl(params).boxed(),
            METHOD_WORKSPACE_ENSURE => self.handle_workspace_ensure_impl(params).boxed(),
            METHOD_WORKSPACE_BY_PATH_READ => {
                self.handle_workspace_by_path_read_impl(params).boxed()
            }
            METHOD_WORKSPACE_DEFAULT_READ => self.handle_workspace_default_read_impl().boxed(),
            METHOD_WORKSPACE_DEFAULT_ENSURE => self.handle_workspace_default_ensure_impl().boxed(),
            METHOD_WORKSPACE_PROJECTS_ROOT_READ => {
                self.handle_workspace_projects_root_read_impl().boxed()
            }
            METHOD_WORKSPACE_PROJECT_PATH_RESOLVE => self
                .handle_workspace_project_path_resolve_impl(params)
                .boxed(),
            METHOD_WORKSPACE_ENSURE_READY => {
                self.handle_workspace_ensure_ready_impl(params).boxed()
            }
            METHOD_SKILL_LIST => self.handle_skill_list_impl().boxed(),
            METHOD_SKILL_READ => self.handle_skill_read_impl(params).boxed(),
            METHOD_SKILL_MANAGEMENT_LIST => self.handle_skill_management_list_impl(params).boxed(),
            METHOD_SKILL_MANAGEMENT_INSTALL => {
                self.handle_skill_management_install_impl(params).boxed()
            }
            METHOD_SKILL_MANAGEMENT_UNINSTALL => {
                self.handle_skill_management_uninstall_impl(params).boxed()
            }
            METHOD_SKILL_REPOSITORY_LIST => self.handle_skill_repository_list_impl().boxed(),
            METHOD_SKILL_REPOSITORY_SAVE => self.handle_skill_repository_save_impl(params).boxed(),
            METHOD_SKILL_REPOSITORY_DELETE => {
                self.handle_skill_repository_delete_impl(params).boxed()
            }
            METHOD_SKILL_CACHE_REFRESH => self.handle_skill_cache_refresh_impl().boxed(),
            METHOD_SKILL_INSTALLED_DIRECTORIES_LIST => {
                self.handle_skill_installed_directories_list_impl().boxed()
            }
            METHOD_SKILL_LOCAL_INSPECT => self.handle_skill_local_inspect_impl(params).boxed(),
            METHOD_SKILL_LOCAL_DETAIL_INSPECT => {
                self.handle_skill_local_detail_inspect_impl(params).boxed()
            }
            METHOD_SKILL_LOCAL_SCAFFOLD_CREATE => {
                self.handle_skill_local_scaffold_create_impl(params).boxed()
            }
            METHOD_SKILL_LOCAL_IMPORT => self.handle_skill_local_import_impl(params).boxed(),
            METHOD_SKILL_LOCAL_RENAME => self.handle_skill_local_rename_impl(params).boxed(),
            METHOD_SKILL_REMOTE_INSPECT => self.handle_skill_remote_inspect_impl(params).boxed(),
            METHOD_SKILL_PACKAGE_LOCAL_INSPECT => {
                self.handle_skill_package_local_inspect_impl(params).boxed()
            }
            METHOD_SKILL_PACKAGE_LOCAL_INSTALL => {
                self.handle_skill_package_local_install_impl(params).boxed()
            }
            METHOD_SKILL_PACKAGE_LOCAL_REPLACE => {
                self.handle_skill_package_local_replace_impl(params).boxed()
            }
            METHOD_SKILL_PACKAGE_EXPORT => self.handle_skill_package_export_impl(params).boxed(),
            METHOD_SKILL_MARKETPLACE_INSTALL => {
                self.handle_skill_marketplace_install_impl(params).boxed()
            }
            METHOD_SKILL_PACKAGE_DOWNLOAD_INSTALL => {
                self.handle_skill_download_install_impl(params).boxed()
            }
            METHOD_GATEWAY_CHANNEL_START => self.handle_gateway_channel_start_impl(params).boxed(),
            METHOD_GATEWAY_CHANNEL_STOP => self.handle_gateway_channel_stop_impl(params).boxed(),
            METHOD_GATEWAY_CHANNEL_STATUS => {
                self.handle_gateway_channel_status_impl(params).boxed()
            }
            METHOD_GATEWAY_TUNNEL_PROBE => self.handle_gateway_tunnel_probe_impl().boxed(),
            METHOD_GATEWAY_TUNNEL_CLOUDFLARED_DETECT => {
                self.handle_gateway_tunnel_cloudflared_detect_impl().boxed()
            }
            METHOD_GATEWAY_TUNNEL_CLOUDFLARED_INSTALL => self
                .handle_gateway_tunnel_cloudflared_install_impl(params)
                .boxed(),
            METHOD_GATEWAY_TUNNEL_CREATE => self.handle_gateway_tunnel_create_impl(params).boxed(),
            METHOD_GATEWAY_TUNNEL_START => self.handle_gateway_tunnel_start_impl().boxed(),
            METHOD_GATEWAY_TUNNEL_STOP => self.handle_gateway_tunnel_stop_impl().boxed(),
            METHOD_GATEWAY_TUNNEL_RESTART => self.handle_gateway_tunnel_restart_impl().boxed(),
            METHOD_GATEWAY_TUNNEL_STATUS => self.handle_gateway_tunnel_status_impl().boxed(),
            METHOD_GATEWAY_TUNNEL_SYNC_WEBHOOK_URL => self
                .handle_gateway_tunnel_sync_webhook_url_impl(params)
                .boxed(),
            METHOD_TELEGRAM_CHANNEL_PROBE => self.handle_telegram_channel_probe(params).boxed(),
            METHOD_FEISHU_CHANNEL_PROBE => self.handle_feishu_channel_probe(params).boxed(),
            METHOD_DISCORD_CHANNEL_PROBE => self.handle_discord_channel_probe(params).boxed(),
            METHOD_WECHAT_CHANNEL_PROBE => self.handle_wechat_channel_probe_impl(params).boxed(),
            METHOD_WECHAT_CHANNEL_LOGIN_START => {
                self.handle_wechat_channel_login_start_impl(params).boxed()
            }
            METHOD_WECHAT_CHANNEL_LOGIN_WAIT => {
                self.handle_wechat_channel_login_wait_impl(params).boxed()
            }
            METHOD_WECHAT_CHANNEL_ACCOUNT_LIST => {
                self.handle_wechat_channel_account_list_impl().boxed()
            }
            METHOD_WECHAT_CHANNEL_ACCOUNT_REMOVE => self
                .handle_wechat_channel_account_remove_impl(params)
                .boxed(),
            METHOD_WECHAT_CHANNEL_RUNTIME_MODEL_SET => self
                .handle_wechat_channel_runtime_model_set_impl(params)
                .boxed(),
            METHOD_MEDIA_TASK_ARTIFACT_IMAGE_CREATE => self
                .handle_media_task_artifact_image_create_impl(params)
                .boxed(),
            METHOD_MEDIA_TASK_ARTIFACT_AUDIO_CREATE => self
                .handle_media_task_artifact_audio_create_impl(params)
                .boxed(),
            METHOD_MEDIA_TASK_ARTIFACT_VIDEO_CREATE => self
                .handle_media_task_artifact_video_create_impl(params)
                .boxed(),
            METHOD_MEDIA_TASK_ARTIFACT_AUDIO_COMPLETE => self
                .handle_media_task_artifact_audio_complete_impl(params)
                .boxed(),
            METHOD_MEDIA_TASK_ARTIFACT_IMAGE_COMPLETE => self
                .handle_media_task_artifact_image_complete_impl(params)
                .boxed(),
            METHOD_MEDIA_TASK_ARTIFACT_GET => {
                self.handle_media_task_artifact_get_impl(params).boxed()
            }
            METHOD_MEDIA_TASK_ARTIFACT_LIST => {
                self.handle_media_task_artifact_list_impl(params).boxed()
            }
            METHOD_MEDIA_TASK_ARTIFACT_CANCEL => {
                self.handle_media_task_artifact_cancel_impl(params).boxed()
            }
            METHOD_GALLERY_MATERIAL_GET => self.handle_gallery_material_get_impl(params).boxed(),
            METHOD_GALLERY_MATERIAL_METADATA_CREATE => self
                .handle_gallery_material_metadata_create_impl(params)
                .boxed(),
            METHOD_GALLERY_MATERIAL_METADATA_GET => self
                .handle_gallery_material_metadata_get_impl(params)
                .boxed(),
            METHOD_GALLERY_MATERIAL_METADATA_UPDATE => self
                .handle_gallery_material_metadata_update_impl(params)
                .boxed(),
            METHOD_GALLERY_MATERIAL_METADATA_DELETE => self
                .handle_gallery_material_metadata_delete_impl(params)
                .boxed(),
            METHOD_GALLERY_MATERIAL_LIST_BY_IMAGE_CATEGORY => self
                .handle_gallery_material_list_by_image_category_impl(params)
                .boxed(),
            METHOD_GALLERY_MATERIAL_LIST_BY_LAYOUT_CATEGORY => self
                .handle_gallery_material_list_by_layout_category_impl(params)
                .boxed(),
            METHOD_GALLERY_MATERIAL_LIST_BY_MOOD => self
                .handle_gallery_material_list_by_mood_impl(params)
                .boxed(),
            METHOD_PROJECT_MATERIAL_LIST => self.handle_project_material_list_impl(params).boxed(),
            METHOD_PROJECT_MATERIAL_GET => self.handle_project_material_get_impl(params).boxed(),
            METHOD_PROJECT_MATERIAL_COUNT => {
                self.handle_project_material_count_impl(params).boxed()
            }
            METHOD_PROJECT_MATERIAL_UPLOAD => {
                self.handle_project_material_upload_impl(params).boxed()
            }
            METHOD_PROJECT_MATERIAL_IMPORT_FROM_URL => self
                .handle_project_material_import_from_url_impl(params)
                .boxed(),
            METHOD_PROJECT_MATERIAL_UPDATE => {
                self.handle_project_material_update_impl(params).boxed()
            }
            METHOD_PROJECT_MATERIAL_DELETE => {
                self.handle_project_material_delete_impl(params).boxed()
            }
            METHOD_PROJECT_MATERIAL_CONTENT => {
                self.handle_project_material_content_impl(params).boxed()
            }
            METHOD_VOICE_ASR_CREDENTIAL_LIST => {
                self.handle_voice_asr_credential_list_impl().boxed()
            }
            METHOD_VOICE_ASR_CREDENTIAL_CREATE => {
                self.handle_voice_asr_credential_create_impl(params).boxed()
            }
            METHOD_VOICE_ASR_CREDENTIAL_UPDATE => {
                self.handle_voice_asr_credential_update_impl(params).boxed()
            }
            METHOD_VOICE_ASR_CREDENTIAL_DELETE => {
                self.handle_voice_asr_credential_delete_impl(params).boxed()
            }
            METHOD_VOICE_ASR_CREDENTIAL_DEFAULT_SET => self
                .handle_voice_asr_credential_default_set_impl(params)
                .boxed(),
            METHOD_VOICE_ASR_CREDENTIAL_TEST => {
                self.handle_voice_asr_credential_test_impl(params).boxed()
            }
            METHOD_VOICE_INSTRUCTION_LIST => self.handle_voice_instruction_list_impl().boxed(),
            METHOD_VOICE_INSTRUCTION_SAVE => {
                self.handle_voice_instruction_save_impl(params).boxed()
            }
            METHOD_VOICE_INSTRUCTION_DELETE => {
                self.handle_voice_instruction_delete_impl(params).boxed()
            }
            METHOD_VOICE_MODEL_DEFAULT_SET => {
                self.handle_voice_model_default_set_impl(params).boxed()
            }
            METHOD_VOICE_MODEL_TEST_TRANSCRIBE_FILE => self
                .handle_voice_model_test_transcribe_file_impl(params)
                .boxed(),
            METHOD_VOICE_TRANSCRIPTION_TRANSCRIBE_AUDIO => self
                .handle_voice_transcription_transcribe_audio_impl(params)
                .boxed(),
            METHOD_VOICE_TRANSCRIPTION_POLISH_TEXT => self
                .handle_voice_transcription_polish_text_impl(params)
                .boxed(),
            METHOD_WORKSPACE_SKILL_BINDINGS_LIST => self
                .handle_workspace_skill_bindings_list_impl(params)
                .boxed(),
            METHOD_WORKSPACE_REGISTERED_SKILLS_LIST => self
                .handle_workspace_registered_skills_list_impl(params)
                .boxed(),
            METHOD_WORKSPACE_RIGHT_SURFACE_REQUEST => self
                .handle_workspace_right_surface_request_impl(params)
                .boxed(),
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_LIST => self
                .handle_workspace_right_surface_pending_list_impl(params)
                .boxed(),
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_CONSUME => self
                .handle_workspace_right_surface_pending_consume_impl(params)
                .boxed(),
            METHOD_WORKSPACE_RIGHT_SURFACE_PENDING_DISMISS => self
                .handle_workspace_right_surface_pending_dismiss_impl(params)
                .boxed(),
            METHOD_BROWSER_SESSION_TARGET_LIST => {
                self.handle_browser_session_target_list_impl(params).boxed()
            }
            METHOD_BROWSER_SESSION_OPEN => self.handle_browser_session_open_impl(params).boxed(),
            METHOD_BROWSER_SESSION_READ => self.handle_browser_session_read_impl(params).boxed(),
            METHOD_BROWSER_SESSION_CLOSE => self.handle_browser_session_close_impl(params).boxed(),
            METHOD_BROWSER_SESSION_EVENT_LIST => {
                self.handle_browser_session_event_list_impl(params).boxed()
            }
            METHOD_BROWSER_SESSION_ACTION_EXECUTE => self
                .handle_browser_session_action_execute_impl(params)
                .boxed(),
            app_server_protocol::protocol::v2::METHOD_PLUGIN_LIST => {
                self.handle_plugin_catalog_list_v2_impl(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_PLUGIN_READ => {
                self.handle_plugin_catalog_read_v2_impl(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_PLUGIN_INSTALL => {
                self.handle_plugin_catalog_install_v2_impl(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_PLUGIN_UNINSTALL => {
                self.handle_plugin_catalog_uninstall_v2_impl(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_PLUGIN_INSTALLED => {
                self.handle_plugin_catalog_installed_v2_impl(params).boxed()
            }
            app_server_protocol::protocol::v2::METHOD_PLUGIN_ENABLED_SET => self
                .handle_plugin_catalog_enabled_set_v2_impl(params)
                .boxed(),
            METHOD_PLUGIN_LOCAL_PACKAGE_INSPECT => self
                .handle_plugin_local_package_inspect_impl(params)
                .boxed(),
            METHOD_PLUGIN_LOCAL_PACKAGE_EXPORT => {
                self.handle_plugin_local_package_export_impl(params).boxed()
            }
            METHOD_PLUGIN_PACKAGE_FETCH_CLOUD => {
                self.handle_plugin_package_fetch_cloud_impl(params).boxed()
            }
            METHOD_PLUGIN_INSTALLED_SAVE => self.handle_plugin_installed_save_impl(params).boxed(),
            METHOD_PLUGIN_INSTALLED_LIST => self.handle_plugin_installed_list_impl().boxed(),
            METHOD_PLUGIN_INSTALLED_DISABLED_SET => self
                .handle_plugin_installed_disabled_set_impl(params)
                .boxed(),
            METHOD_PLUGIN_INSTALLED_UNINSTALL_REHEARSAL => self
                .handle_plugin_installed_uninstall_rehearsal_impl(params)
                .boxed(),
            METHOD_PLUGIN_INSTALLED_UNINSTALL => {
                self.handle_plugin_installed_uninstall_impl(params).boxed()
            }
            METHOD_PLUGIN_HOST_LIFECYCLE_LIST => {
                self.handle_plugin_host_lifecycle_list_impl().boxed()
            }
            METHOD_PLUGIN_SHELL_PREPARE => self.handle_plugin_shell_prepare_impl(params).boxed(),
            METHOD_PLUGIN_UI_RUNTIME_START => {
                self.handle_plugin_ui_runtime_start_impl(params).boxed()
            }
            METHOD_PLUGIN_UI_RUNTIME_STATUS => {
                self.handle_plugin_ui_runtime_status_impl(params).boxed()
            }
            METHOD_PLUGIN_UI_RUNTIME_STOP => {
                self.handle_plugin_ui_runtime_stop_impl(params).boxed()
            }
            METHOD_SOUL_STYLE_PACK_INSTALL => {
                self.handle_soul_style_pack_install_impl(params).boxed()
            }
            METHOD_SOUL_STYLE_PACK_LIST => self.handle_soul_style_pack_list_impl(params).boxed(),
            METHOD_SOUL_STYLE_PACK_STATUS_SET => {
                self.handle_soul_style_pack_status_set_impl(params).boxed()
            }
            METHOD_SOUL_STYLE_PACK_UNINSTALL => {
                self.handle_soul_style_pack_uninstall_impl(params).boxed()
            }
            METHOD_KNOWLEDGE_PACK_LIST => self.handle_knowledge_pack_list_impl(params).boxed(),
            METHOD_KNOWLEDGE_PACK_READ => self.handle_knowledge_pack_read_impl(params).boxed(),
            METHOD_KNOWLEDGE_SOURCE_IMPORT => {
                self.handle_knowledge_source_import_impl(params).boxed()
            }
            METHOD_KNOWLEDGE_PACK_COMPILE => {
                self.handle_knowledge_pack_compile_impl(params).boxed()
            }
            METHOD_KNOWLEDGE_PACK_DEFAULT_SET => {
                self.handle_knowledge_pack_default_set_impl(params).boxed()
            }
            METHOD_KNOWLEDGE_PACK_STATUS_UPDATE => self
                .handle_knowledge_pack_status_update_impl(params)
                .boxed(),
            METHOD_KNOWLEDGE_CONTEXT_RESOLVE => {
                self.handle_knowledge_context_resolve_impl(params).boxed()
            }
            METHOD_KNOWLEDGE_CONTEXT_RUN_VALIDATE => self
                .handle_knowledge_context_run_validate_impl(params)
                .boxed(),
            METHOD_AUTOMATION_SCHEDULER_CONFIG_READ => {
                self.handle_automation_scheduler_config_read_impl().boxed()
            }
            METHOD_AUTOMATION_SCHEDULER_CONFIG_UPDATE => self
                .handle_automation_scheduler_config_update_impl(params)
                .boxed(),
            METHOD_AUTOMATION_SCHEDULER_STATUS => {
                self.handle_automation_scheduler_status_impl().boxed()
            }
            METHOD_AUTOMATION_JOB_LIST => self.handle_automation_job_list_impl().boxed(),
            METHOD_AUTOMATION_JOB_READ => self.handle_automation_job_read_impl(params).boxed(),
            METHOD_AUTOMATION_JOB_CREATE => self.handle_automation_job_create_impl(params).boxed(),
            METHOD_AUTOMATION_JOB_UPDATE => self.handle_automation_job_update_impl(params).boxed(),
            METHOD_AUTOMATION_JOB_DELETE => self.handle_automation_job_delete_impl(params).boxed(),
            METHOD_AUTOMATION_JOB_RUN_NOW => {
                self.handle_automation_job_run_now_impl(params).boxed()
            }
            METHOD_AUTOMATION_JOB_HEALTH => self.handle_automation_job_health_impl(params).boxed(),
            METHOD_AUTOMATION_JOB_RUN_HISTORY => {
                self.handle_automation_job_run_history_impl(params).boxed()
            }
            METHOD_AUTOMATION_SCHEDULE_PREVIEW => {
                self.handle_automation_schedule_preview_impl(params).boxed()
            }
            METHOD_AUTOMATION_SCHEDULE_VALIDATE => self
                .handle_automation_schedule_validate_impl(params)
                .boxed(),
            METHOD_MCP_SERVER_LIST => self.handle_mcp_server_list_impl().boxed(),
            METHOD_MCP_SERVER_STATUS_LIST => self.handle_mcp_server_status_list_impl().boxed(),
            METHOD_MCP_SERVER_CREATE => self.handle_mcp_server_create_impl(params).boxed(),
            METHOD_MCP_SERVER_UPDATE => self.handle_mcp_server_update_impl(params).boxed(),
            METHOD_MCP_SERVER_DELETE => self.handle_mcp_server_delete_impl(params).boxed(),
            METHOD_MCP_SERVER_ENABLED_SET => {
                self.handle_mcp_server_enabled_set_impl(params).boxed()
            }
            METHOD_MCP_SERVER_IMPORT_FROM_APP => {
                self.handle_mcp_server_import_from_app_impl(params).boxed()
            }
            METHOD_MCP_SERVER_SYNC_ALL_TO_LIVE => {
                self.handle_mcp_server_sync_all_to_live_impl().boxed()
            }
            METHOD_MCP_SERVER_OAUTH_LOGIN => {
                self.handle_mcp_server_oauth_login_impl(params).boxed()
            }
            METHOD_MCP_SERVER_START => self.handle_mcp_server_start_impl(params).boxed(),
            METHOD_MCP_SERVER_STOP => self.handle_mcp_server_stop_impl(params).boxed(),
            METHOD_MCP_TOOL_LIST => self.handle_mcp_tool_list_impl().boxed(),
            METHOD_MCP_TOOL_LIST_FOR_CONTEXT => {
                self.handle_mcp_tool_list_for_context_impl(params).boxed()
            }
            METHOD_MCP_TOOL_SEARCH => self.handle_mcp_tool_search_impl(params).boxed(),
            METHOD_MCP_TOOL_CALL => self.handle_mcp_tool_call_impl(params).boxed(),
            METHOD_MCP_TOOL_CALL_WITH_CALLER => {
                self.handle_mcp_tool_call_with_caller_impl(params).boxed()
            }
            METHOD_MCP_PROMPT_LIST => self.handle_mcp_prompt_list_impl().boxed(),
            METHOD_MCP_PROMPT_GET => self.handle_mcp_prompt_get_impl(params).boxed(),
            METHOD_MCP_RESOURCE_LIST => self.handle_mcp_resource_list_impl().boxed(),
            METHOD_MCP_RESOURCE_READ => self.handle_mcp_resource_read_impl(params).boxed(),
            METHOD_MCP_RESOURCE_SUBSCRIBE => {
                self.handle_mcp_resource_subscribe_impl(params).boxed()
            }
            METHOD_MCP_RESOURCE_UNSUBSCRIBE => {
                self.handle_mcp_resource_unsubscribe_impl(params).boxed()
            }
            METHOD_PROJECT_MEMORY_READ => self.handle_project_memory_read_impl(params).boxed(),
            METHOD_MEMORY_STORE_LIST => self.handle_memory_store_list_impl(params).boxed(),
            METHOD_MEMORY_STORE_READ => self.handle_memory_store_read_impl(params).boxed(),
            METHOD_MEMORY_STORE_SEARCH => self.handle_memory_store_search_impl(params).boxed(),
            METHOD_MEMORY_STORE_ADD_NOTE => self.handle_memory_store_add_note_impl(params).boxed(),
            METHOD_MEMORY_STORE_CONSOLIDATE => {
                self.handle_memory_store_consolidate_impl(params).boxed()
            }
            METHOD_MEMORY_STORE_REVIEW_LIST => {
                self.handle_memory_store_review_list_impl(params).boxed()
            }
            METHOD_MEMORY_STORE_REVIEW_RESOLVE => {
                self.handle_memory_store_review_resolve_impl(params).boxed()
            }
            METHOD_MEMORY_STORE_HEALTH => self.handle_memory_store_health_impl(params).boxed(),
            METHOD_MEMORY_STORE_RESET => self.handle_memory_store_reset_impl(params).boxed(),
            METHOD_MEMORY_STORE_INDEX_REBUILD => {
                self.handle_memory_store_index_rebuild_impl(params).boxed()
            }
            METHOD_LOG_LIST => self.handle_log_list_impl().boxed(),
            METHOD_LOG_PERSISTED_TAIL => self.handle_log_persisted_tail_impl(params).boxed(),
            METHOD_LOG_CLEAR => self.handle_log_clear_impl().boxed(),
            METHOD_LOG_DIAGNOSTIC_HISTORY_CLEAR => {
                self.handle_log_diagnostic_history_clear_impl().boxed()
            }
            METHOD_DIAGNOSTICS_LOG_STORAGE_READ => {
                self.handle_diagnostics_log_storage_read_impl().boxed()
            }
            METHOD_DIAGNOSTICS_SUPPORT_BUNDLE_EXPORT => self
                .handle_diagnostics_support_bundle_export_impl(params)
                .boxed(),
            METHOD_DIAGNOSTICS_SERVER_READ => self.handle_diagnostics_server_read_impl().boxed(),
            METHOD_DIAGNOSTICS_WINDOWS_STARTUP_READ => {
                self.handle_diagnostics_windows_startup_read_impl().boxed()
            }
            METHOD_DIAGNOSTICS_TRACE_LIST => {
                self.handle_diagnostics_trace_list_impl(params).boxed()
            }
            METHOD_DIAGNOSTICS_TRACE_READ => {
                self.handle_diagnostics_trace_read_impl(params).boxed()
            }
            METHOD_DIAGNOSTICS_TRACE_EXPORT => {
                self.handle_diagnostics_trace_export_impl(params).boxed()
            }
            METHOD_USAGE_STATS_READ => self.handle_usage_stats_read(params).boxed(),
            METHOD_USAGE_STATS_MODEL_RANKING_LIST => {
                self.handle_usage_stats_model_ranking_list(params).boxed()
            }
            METHOD_USAGE_STATS_DAILY_TRENDS_LIST => {
                self.handle_usage_stats_daily_trends_list(params).boxed()
            }
            METHOD_MODEL_LIST => self.handle_model_list_impl(params).boxed(),
            METHOD_MODEL_PREFERENCES_LIST => self.handle_model_preferences_list_impl().boxed(),
            METHOD_MODEL_SYNC_STATE_READ => self.handle_model_sync_state_read_impl().boxed(),
            METHOD_MODEL_PROVIDER_LIST => self.handle_model_provider_list_impl().boxed(),
            METHOD_MODEL_PROVIDER_CATALOG_LIST => {
                self.handle_model_provider_catalog_list_impl().boxed()
            }
            METHOD_MODEL_PROVIDER_READ => self.handle_model_provider_read_impl(params).boxed(),
            METHOD_MODEL_PROVIDER_CREATE => self.handle_model_provider_create_impl(params).boxed(),
            METHOD_MODEL_PROVIDER_UPDATE => self.handle_model_provider_update_impl(params).boxed(),
            METHOD_MODEL_PROVIDER_DELETE => self.handle_model_provider_delete_impl(params).boxed(),
            METHOD_MODEL_PROVIDER_SORT_ORDERS_UPDATE => self
                .handle_model_provider_sort_orders_update_impl(params)
                .boxed(),
            METHOD_MODEL_PROVIDER_CONFIG_EXPORT => self
                .handle_model_provider_config_export_impl(params)
                .boxed(),
            METHOD_MODEL_PROVIDER_CONFIG_IMPORT => self
                .handle_model_provider_config_import_impl(params)
                .boxed(),
            METHOD_MODEL_PROVIDER_TEST_CONNECTION => self
                .handle_model_provider_test_connection_impl(params)
                .boxed(),
            METHOD_MODEL_PROVIDER_TEST_CHAT => {
                self.handle_model_provider_test_chat_impl(params).boxed()
            }
            METHOD_MODEL_PROVIDER_FETCH_MODELS => {
                self.handle_model_provider_fetch_models_impl(params).boxed()
            }
            METHOD_MODEL_PROVIDER_KEY_CREATE => {
                self.handle_model_provider_key_create_impl(params).boxed()
            }
            METHOD_MODEL_PROVIDER_KEY_UPDATE => {
                self.handle_model_provider_key_update_impl(params).boxed()
            }
            METHOD_MODEL_PROVIDER_KEY_DELETE => {
                self.handle_model_provider_key_delete_impl(params).boxed()
            }
            METHOD_MODEL_PROVIDER_UI_STATE_READ => self
                .handle_model_provider_ui_state_read_impl(params)
                .boxed(),
            METHOD_MODEL_PROVIDER_UI_STATE_WRITE => self
                .handle_model_provider_ui_state_write_impl(params)
                .boxed(),
            METHOD_MODEL_PROVIDER_ALIAS_READ => {
                self.handle_model_provider_alias_read_impl(params).boxed()
            }
            METHOD_MODEL_PROVIDER_ALIAS_LIST => {
                self.handle_model_provider_alias_list_impl().boxed()
            }
            METHOD_CONNECT_DEEP_LINK_RESOLVE => {
                self.handle_connect_deep_link_resolve_impl(params).boxed()
            }
            METHOD_CONNECT_OPEN_DEEP_LINK_RESOLVE => self
                .handle_connect_open_deep_link_resolve_impl(params)
                .boxed(),
            METHOD_CONNECT_RELAY_API_KEY_SAVE => {
                self.handle_connect_relay_api_key_save_impl(params).boxed()
            }
            METHOD_CONNECT_CALLBACK_SEND => self.handle_connect_callback_send_impl(params).boxed(),
            METHOD_CONVERSATION_IMPORT_SOURCE_SCAN => self
                .handle_conversation_import_source_scan_impl(params)
                .boxed(),
            METHOD_CONVERSATION_IMPORT_THREAD_PREVIEW => self
                .handle_conversation_import_thread_preview_impl(params)
                .boxed(),
            METHOD_CONVERSATION_IMPORT_THREAD_COMMIT => self
                .handle_conversation_import_thread_commit_impl(params)
                .boxed(),
            METHOD_CONVERSATION_IMPORT_JOB_READ => self
                .handle_conversation_import_job_read_impl(params)
                .boxed(),
            METHOD_TURN_START => self
                .handle_turn_start_v2_impl(params, event_callback)
                .boxed(),
            app_server_protocol::protocol::v2::METHOD_TURN_STEER => {
                self.handle_turn_steer_impl(params, event_callback).boxed()
            }
            METHOD_TURN_INTERRUPT => self.handle_turn_interrupt_v2_impl(params).boxed(),
            METHOD_AGENT_SESSION_ACTION_RESPOND => self.handle_action_respond(params).boxed(),
            METHOD_AGENT_SESSION_TOOL_INVENTORY_READ => {
                self.handle_tool_inventory_read_impl(params).boxed()
            }
            _ => ready(Err(JsonRpcError::new(
                error_codes::METHOD_NOT_FOUND,
                format!("method not found: {method}"),
            )))
            .boxed(),
        }
        .await;

        self.clear_request_cancel_state(&id);
        response::into_messages(id, result)
    }
}
