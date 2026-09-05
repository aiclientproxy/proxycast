use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use app_server_client::{
    ClientSession, RemoteTransportConfig, RequestHandle, SessionEvent, StdioTransportConfig,
};
use app_server_protocol::protocol::v2::{
    CurrentTimeReadResponse, METHOD_PERMISSION_PROFILE_LIST, METHOD_PROMPT_HISTORY_APPEND,
    METHOD_PROMPT_HISTORY_READ, METHOD_THREAD_ARCHIVE, METHOD_THREAD_QUEUE_ADD,
    METHOD_THREAD_QUEUE_DELETE, METHOD_THREAD_QUEUE_LIST, METHOD_THREAD_READ, METHOD_THREAD_RESUME,
    METHOD_THREAD_SETTINGS_UPDATE, METHOD_THREAD_START, METHOD_TURN_INTERRUPT, METHOD_TURN_START,
    METHOD_TURN_STEER, ModelListParams, ModelListResponse, PermissionProfileListParams,
    PermissionProfileListResponse, PromptHistoryAppendParams, PromptHistoryAppendResponse,
    PromptHistoryReadParams, PromptHistoryReadResponse, QueuedSubmission, ServerRequest,
    ThreadForkParams, ThreadForkResponse, ThreadListParams, ThreadListResponse,
    ThreadQueueAddParams, ThreadQueueAddResponse, ThreadQueueDeleteParams,
    ThreadQueueDeleteResponse, ThreadQueueListParams, ThreadQueueListResponse, ThreadReadParams,
    ThreadReadResponse, ThreadResumeParams, ThreadResumeResponse, ThreadSettingsUpdateParams,
    ThreadSettingsUpdateResponse, ThreadStartParams, ThreadStartResponse, ThreadUnarchiveParams,
    ThreadUnarchiveResponse, TurnInterruptParams, TurnInterruptResponse, TurnStartParams,
    TurnStartResponse, TurnSteerParams, TurnSteerResponse, UserInput,
};
use app_server_protocol::{
    ClientCapabilities, ClientInfo, InitializeParams, JsonRpcError, JsonRpcRequest, RequestId,
};
use serde_json::Value;

use crate::bottom_pane::AppServerResponse;

fn permission_profile_id(value: &Value) -> Option<String> {
    value
        .as_object()
        .and_then(|profile| profile.get("id"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|id| !id.trim().is_empty())
}

pub(crate) struct AppServerSession {
    session: ClientSession,
    request_handle: RequestHandle,
    thread_id: Option<String>,
    session_id: Option<String>,
    active_permission_profile: Option<String>,
}

impl AppServerSession {
    pub(crate) async fn connect(config: StdioTransportConfig) -> Result<Self> {
        let app_server_bin = config.app_server_bin.clone();
        let session = ClientSession::start_stdio(config, initialize_params())
            .await
            .with_context(|| {
                format!(
                    "failed to initialize App Server at {}",
                    app_server_bin.display()
                )
            })?;
        Ok(Self {
            request_handle: session.request_handle(),
            session,
            thread_id: None,
            session_id: None,
            active_permission_profile: None,
        })
    }

    pub(crate) async fn connect_remote(config: RemoteTransportConfig) -> Result<Self> {
        let websocket_url = config.websocket_url.clone();
        let session = ClientSession::start_remote(config, initialize_params())
            .await
            .with_context(|| {
                format!("failed to initialize remote App Server at {websocket_url}")
            })?;
        Ok(Self {
            request_handle: session.request_handle(),
            session,
            thread_id: None,
            session_id: None,
            active_permission_profile: None,
        })
    }

    pub(crate) async fn start_thread(
        &mut self,
        cwd: PathBuf,
        model: Option<String>,
        model_provider: Option<String>,
    ) -> Result<String> {
        let cwd = cwd.to_string_lossy().into_owned();
        let params = ThreadStartParams {
            cwd: Some(cwd.clone()),
            runtime_workspace_roots: Some(vec![cwd]),
            model,
            model_provider,
            experimental_raw_events: false,
            ..ThreadStartParams::default()
        };
        let response: ThreadStartResponse = self
            .request_handle
            .request(METHOD_THREAD_START, params)
            .await
            .context("failed to start App Server thread")?;
        let thread_id = response.thread.id.clone();
        self.session_id = Some(response.thread.session_id.clone());
        self.thread_id = Some(thread_id.clone());
        self.active_permission_profile = response
            .active_permission_profile
            .as_ref()
            .and_then(permission_profile_id);
        Ok(thread_id)
    }

    pub(crate) async fn resume_thread(
        &mut self,
        thread_id: String,
    ) -> Result<ThreadResumeResponse> {
        let response: ThreadResumeResponse = self
            .request_handle
            .request(
                METHOD_THREAD_RESUME,
                ThreadResumeParams {
                    thread_id,
                    ..ThreadResumeParams::default()
                },
            )
            .await
            .context("failed to resume App Server thread")?;
        self.thread_id = Some(response.thread.id.clone());
        self.session_id = Some(response.thread.session_id.clone());
        self.active_permission_profile = response
            .active_permission_profile
            .as_ref()
            .and_then(permission_profile_id);
        Ok(response)
    }

    pub(crate) fn active_permission_profile(&self) -> Option<&str> {
        self.active_permission_profile.as_deref()
    }

    pub(crate) async fn list_permission_profiles(
        &self,
        cwd: Option<String>,
    ) -> Result<PermissionProfileListResponse> {
        let mut cursor = None;
        let mut data = Vec::new();
        let mut seen_cursors = HashSet::new();
        for _ in 0..16 {
            let page: PermissionProfileListResponse = self
                .request_handle
                .request(
                    METHOD_PERMISSION_PROFILE_LIST,
                    PermissionProfileListParams {
                        cursor,
                        limit: Some(64),
                        cwd: cwd.clone(),
                    },
                )
                .await
                .context("failed to list App Server permission profiles")?;
            data.extend(page.data);
            let Some(next_cursor) = page.next_cursor else {
                return Ok(PermissionProfileListResponse {
                    data,
                    next_cursor: None,
                });
            };
            if !seen_cursors.insert(next_cursor.clone()) {
                bail!("permission profile list pagination repeated cursor {next_cursor}");
            }
            cursor = Some(next_cursor);
        }
        bail!("permission profile list pagination exceeded 16 pages")
    }

    #[allow(dead_code)]
    pub(crate) async fn list_thread_page(
        &self,
        params: ThreadListParams,
    ) -> Result<ThreadListResponse> {
        self.request_handle
            .request(
                app_server_protocol::protocol::v2::METHOD_THREAD_LIST,
                params,
            )
            .await
            .context("failed to list App Server thread page")
    }

    /// Return a cloneable request boundary for background TUI loaders.
    ///
    /// Requests are serialized by the App Server client worker, so handing a
    /// clone to a short-lived loader does not create a second transport or a
    /// parallel session. The owning session still controls lifecycle and
    /// shutdown.
    pub(crate) fn request_handle(&self) -> app_server_client::RequestHandle {
        self.request_handle.clone()
    }

    #[allow(dead_code)]
    pub(crate) async fn read_thread(
        &self,
        thread_id: impl Into<String>,
        include_turns: bool,
    ) -> Result<ThreadReadResponse> {
        self.request_handle
            .request(
                METHOD_THREAD_READ,
                ThreadReadParams {
                    thread_id: thread_id.into(),
                    include_turns,
                },
            )
            .await
            .context("failed to read App Server thread")
    }

    #[allow(dead_code)]
    pub(crate) async fn archive_thread(&self, thread_id: impl Into<String>) -> Result<()> {
        self.request_handle
            .request(
                METHOD_THREAD_ARCHIVE,
                app_server_protocol::protocol::v2::ThreadArchiveParams {
                    thread_id: thread_id.into(),
                },
            )
            .await
            .map(|_: app_server_protocol::protocol::v2::ThreadArchiveResponse| ())
            .context("failed to archive App Server thread")
    }

    #[allow(dead_code)]
    pub(crate) async fn unarchive_thread(
        &self,
        thread_id: impl Into<String>,
    ) -> Result<ThreadUnarchiveResponse> {
        self.request_handle
            .request(
                app_server_protocol::protocol::v2::METHOD_THREAD_UNARCHIVE,
                ThreadUnarchiveParams {
                    thread_id: thread_id.into(),
                },
            )
            .await
            .context("failed to restore archived App Server thread")
    }

    pub(crate) async fn fork_thread(
        &self,
        thread_id: impl Into<String>,
        cwd: Option<PathBuf>,
        model: Option<String>,
        model_provider: Option<String>,
    ) -> Result<ThreadForkResponse> {
        let cwd = cwd.map(|path| path.to_string_lossy().into_owned());
        self.request_handle
            .request(
                app_server_protocol::protocol::v2::METHOD_THREAD_FORK,
                ThreadForkParams {
                    thread_id: thread_id.into(),
                    cwd: cwd.clone(),
                    runtime_workspace_roots: cwd.map(|cwd| vec![cwd]),
                    model,
                    model_provider,
                    ..ThreadForkParams::default()
                },
            )
            .await
            .context("failed to fork App Server thread")
    }

    pub(crate) async fn list_models(&self, limit: u32) -> Result<ModelListResponse> {
        let mut cursor = None;
        let mut seen_cursors = HashSet::new();
        let mut data = Vec::new();
        for _ in 0..16 {
            let page: ModelListResponse = self
                .request_handle
                .request(
                    app_server_protocol::protocol::v2::METHOD_MODEL_LIST,
                    ModelListParams {
                        cursor,
                        limit: Some(limit),
                        include_hidden: Some(false),
                    },
                )
                .await
                .context("failed to list App Server models")?;
            data.extend(page.data);
            let Some(next_cursor) = page.next_cursor else {
                return Ok(ModelListResponse {
                    data,
                    next_cursor: None,
                });
            };
            if !seen_cursors.insert(next_cursor.clone()) {
                bail!("model list pagination repeated cursor {next_cursor}");
            }
            cursor = Some(next_cursor);
        }
        bail!("model list pagination exceeded 16 pages")
    }

    pub(crate) async fn list_queued_submissions(
        &self,
        limit: u32,
    ) -> Result<Vec<QueuedSubmission>> {
        let thread_id = self.thread_id()?.to_string();
        let mut cursor = None;
        let mut seen_cursors = HashSet::new();
        let mut data = Vec::new();
        for _ in 0..16 {
            let page: ThreadQueueListResponse = self
                .request_handle
                .request(
                    METHOD_THREAD_QUEUE_LIST,
                    ThreadQueueListParams {
                        thread_id: thread_id.clone(),
                        cursor,
                        limit: Some(limit),
                    },
                )
                .await
                .context("failed to list queued submissions")?;
            data.extend(page.data);
            let Some(next_cursor) = page.next_cursor else {
                return Ok(data);
            };
            if !seen_cursors.insert(next_cursor.clone()) {
                bail!("thread queue pagination repeated cursor {next_cursor}");
            }
            cursor = Some(next_cursor);
        }
        bail!("thread queue pagination exceeded 16 pages")
    }

    pub(crate) fn session_id(&self) -> Result<&str> {
        self.session_id
            .as_deref()
            .ok_or_else(|| anyhow!("App Server session has not been started"))
    }

    pub(crate) async fn update_settings(
        &self,
        model: Option<String>,
        model_provider: Option<String>,
        effort: Option<String>,
        permissions: Option<String>,
    ) -> Result<()> {
        if model.is_none() && model_provider.is_none() && effort.is_none() && permissions.is_none()
        {
            return Ok(());
        }
        let thread_id = self.thread_id()?.to_string();
        let _: ThreadSettingsUpdateResponse = self
            .request_handle
            .request(
                METHOD_THREAD_SETTINGS_UPDATE,
                ThreadSettingsUpdateParams {
                    thread_id,
                    model,
                    model_provider,
                    effort,
                    permissions,
                    ..ThreadSettingsUpdateParams::default()
                },
            )
            .await
            .context("failed to update App Server thread settings")?;
        Ok(())
    }

    pub(crate) async fn read_prompt_history(
        &self,
        limit: u32,
    ) -> Result<PromptHistoryReadResponse> {
        self.request_handle
            .request(
                METHOD_PROMPT_HISTORY_READ,
                PromptHistoryReadParams {
                    limit: Some(limit),
                    ..PromptHistoryReadParams::default()
                },
            )
            .await
            .context("failed to read prompt history")
    }

    pub(crate) async fn append_prompt_history(
        &self,
        text: String,
    ) -> Result<PromptHistoryAppendResponse> {
        let session_id = self.session_id()?.to_string();
        self.request_handle
            .request(
                METHOD_PROMPT_HISTORY_APPEND,
                PromptHistoryAppendParams { session_id, text },
            )
            .await
            .context("failed to append prompt history")
    }

    pub(crate) fn thread_id(&self) -> Result<&str> {
        self.thread_id
            .as_deref()
            .ok_or_else(|| anyhow!("App Server thread has not been started"))
    }

    pub(crate) async fn start_turn(&self, prompt: String) -> Result<String> {
        self.start_turn_input(vec![UserInput::Text {
            text: prompt,
            text_elements: Vec::new(),
        }])
        .await
    }

    pub(crate) async fn start_turn_input(&self, input: Vec<UserInput>) -> Result<String> {
        let thread_id = self.thread_id()?.to_string();
        let response: TurnStartResponse = self
            .request_handle
            .request(
                METHOD_TURN_START,
                TurnStartParams {
                    thread_id,
                    input,
                    ..TurnStartParams::default()
                },
            )
            .await
            .context("failed to start turn")?;
        Ok(response.turn.id)
    }

    pub(crate) async fn interrupt(&self, turn_id: &str) -> Result<()> {
        let thread_id = self.thread_id()?.to_string();
        let _: TurnInterruptResponse = self
            .request_handle
            .request(
                METHOD_TURN_INTERRUPT,
                TurnInterruptParams {
                    thread_id,
                    turn_id: turn_id.to_string(),
                },
            )
            .await
            .context("failed to interrupt turn")?;
        Ok(())
    }

    pub(crate) async fn steer_turn_input(
        &self,
        turn_id: &str,
        input: Vec<UserInput>,
    ) -> Result<String> {
        let thread_id = self.thread_id()?.to_string();
        let response: TurnSteerResponse = self
            .request_handle
            .request(
                METHOD_TURN_STEER,
                TurnSteerParams {
                    thread_id,
                    input,
                    expected_turn_id: turn_id.to_string(),
                    ..TurnSteerParams::default()
                },
            )
            .await
            .context("failed to steer turn")?;
        Ok(response.turn_id)
    }

    pub(crate) async fn queue_input(&self, input: Vec<UserInput>) -> Result<QueuedSubmission> {
        let thread_id = self.thread_id()?.to_string();
        let response: ThreadQueueAddResponse = self
            .request_handle
            .request(
                METHOD_THREAD_QUEUE_ADD,
                ThreadQueueAddParams {
                    thread_id,
                    input,
                    client_user_message_id: client_message_id(),
                },
            )
            .await
            .context("failed to queue prompt")?;
        Ok(response.queued_submission)
    }

    pub(crate) async fn delete_queued_submission(
        &self,
        queued_submission_id: String,
    ) -> Result<bool> {
        let thread_id = self.thread_id()?.to_string();
        let response: ThreadQueueDeleteResponse = self
            .request_handle
            .request(
                METHOD_THREAD_QUEUE_DELETE,
                ThreadQueueDeleteParams {
                    thread_id,
                    queued_submission_id,
                },
            )
            .await
            .context("failed to delete queued submission")?;
        Ok(response.deleted)
    }

    pub(crate) async fn next_event(&mut self) -> Option<SessionEvent> {
        self.session.next_event().await
    }

    pub(crate) async fn respond(&self, response: AppServerResponse) -> Result<()> {
        match response {
            AppServerResponse::Command { id, response } => {
                self.request_handle.respond(id, response).await
            }
            AppServerResponse::FileChange { id, response } => {
                self.request_handle.respond(id, response).await
            }
            AppServerResponse::Permissions { id, response } => {
                self.request_handle.respond(id, response).await
            }
            AppServerResponse::UserInput { id, response } => {
                self.request_handle.respond(id, response).await
            }
        }
        .context("failed to respond to App Server request")
    }

    pub(crate) async fn respond_current_time(&self, id: RequestId) -> Result<()> {
        let seconds = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| anyhow!("system clock is before Unix epoch: {error}"))?
            .as_secs();
        let current_time_at = i64::try_from(seconds)
            .map_err(|_| anyhow!("system clock is outside the supported range"))?;
        self.request_handle
            .respond(id, CurrentTimeReadResponse { current_time_at })
            .await
            .context("failed to respond to currentTime/read")
    }

    pub(crate) async fn reject_server_request(&self, request: Box<ServerRequest>) -> Result<()> {
        let method = request.method();
        self.request_handle
            .reject(
                request.id().clone(),
                JsonRpcError::new(
                    app_server_protocol::error_codes::METHOD_NOT_FOUND,
                    format!("TUI client does not support server request {method} yet"),
                ),
            )
            .await
            .context("failed to reject unsupported server request")
    }

    pub(crate) async fn reject_raw_server_request(&self, request: JsonRpcRequest) -> Result<()> {
        let method = request.method.clone();
        self.request_handle
            .reject(
                request.id,
                JsonRpcError::new(
                    app_server_protocol::error_codes::METHOD_NOT_FOUND,
                    format!("TUI client does not support server request {method}"),
                ),
            )
            .await
            .context("failed to reject unknown server request")
    }

    pub(crate) async fn shutdown(self) -> Result<()> {
        self.session
            .shutdown()
            .await
            .context("failed to stop App Server")
    }
}

fn client_message_id() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("lime-tui-{nanos}")
}

fn initialize_params() -> InitializeParams {
    InitializeParams {
        client_info: ClientInfo {
            name: "lime-tui".to_string(),
            title: Some("Lime TUI".to_string()),
            version: Some(env!("CARGO_PKG_VERSION").to_string()),
        },
        capabilities: ClientCapabilities {
            event_methods: Vec::new(),
            experimental_api: true,
            opt_out_notification_methods: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tui_client_uses_stable_identity_and_v2_capabilities() {
        let params = initialize_params();

        assert_eq!(params.client_info.name, "lime-tui");
        assert_eq!(params.client_info.title.as_deref(), Some("Lime TUI"));
        assert!(params.capabilities.experimental_api);
        assert!(params.capabilities.event_methods.is_empty());
    }

    #[test]
    fn active_permission_profile_requires_a_non_empty_id() {
        assert_eq!(
            permission_profile_id(&serde_json::json!({"id": ":workspace"})),
            Some(":workspace".to_string())
        );
        assert_eq!(
            permission_profile_id(&serde_json::json!({"id": "  "})),
            None
        );
        assert_eq!(permission_profile_id(&serde_json::json!({})), None);
        assert_eq!(permission_profile_id(&serde_json::json!(null)), None);
    }
}
