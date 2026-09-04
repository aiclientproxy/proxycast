use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use app_server_client::{ClientSession, RequestHandle, SessionEvent, StdioTransportConfig};
use app_server_protocol::protocol::v2::{
    CurrentTimeReadResponse, ModelListParams, ModelListResponse, PromptHistoryAppendParams,
    PromptHistoryAppendResponse, PromptHistoryReadParams, PromptHistoryReadResponse,
    QueuedSubmission, ServerRequest, SortDirection, ThreadListParams, ThreadListResponse,
    ThreadQueueAddParams, ThreadQueueAddResponse, ThreadResumeParams, ThreadResumeResponse,
    ThreadSettingsUpdateParams, ThreadSettingsUpdateResponse, ThreadSortKey, ThreadStartParams,
    ThreadStartResponse, TurnInterruptParams, TurnInterruptResponse, TurnStartParams,
    TurnStartResponse, TurnSteerParams, TurnSteerResponse, UserInput, METHOD_PROMPT_HISTORY_APPEND,
    METHOD_PROMPT_HISTORY_READ, METHOD_THREAD_QUEUE_ADD, METHOD_THREAD_RESUME,
    METHOD_THREAD_SETTINGS_UPDATE, METHOD_THREAD_START, METHOD_TURN_INTERRUPT, METHOD_TURN_START,
    METHOD_TURN_STEER,
};
use app_server_protocol::{
    ClientCapabilities, ClientInfo, InitializeParams, JsonRpcError, JsonRpcRequest, RequestId,
};

use crate::bottom_pane::AppServerResponse;

pub(crate) struct AppServerSession {
    session: ClientSession,
    request_handle: RequestHandle,
    thread_id: Option<String>,
    session_id: Option<String>,
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
        Ok(response)
    }

    pub(crate) async fn list_threads(&self, limit: u32) -> Result<ThreadListResponse> {
        let mut cursor = None;
        let mut seen_cursors = HashSet::new();
        let mut data = Vec::new();
        for _ in 0..16 {
            let page: ThreadListResponse = self
                .request_handle
                .request(
                    app_server_protocol::protocol::v2::METHOD_THREAD_LIST,
                    ThreadListParams {
                        cursor,
                        limit: Some(limit),
                        sort_key: Some(ThreadSortKey::UpdatedAt),
                        sort_direction: Some(SortDirection::Desc),
                        ..ThreadListParams::default()
                    },
                )
                .await
                .context("failed to list App Server threads")?;
            data.extend(page.data);
            let Some(next_cursor) = page.next_cursor else {
                return Ok(ThreadListResponse {
                    data,
                    next_cursor: None,
                    backwards_cursor: None,
                });
            };
            if !seen_cursors.insert(next_cursor.clone()) {
                bail!("thread list pagination repeated cursor {next_cursor}");
            }
            cursor = Some(next_cursor);
        }
        bail!("thread list pagination exceeded 16 pages")
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
        let thread_id = self.thread_id()?.to_string();
        let response: TurnStartResponse = self
            .request_handle
            .request(
                METHOD_TURN_START,
                TurnStartParams {
                    thread_id,
                    input: vec![UserInput::Text {
                        text: prompt,
                        text_elements: Vec::new(),
                    }],
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

    pub(crate) async fn steer_turn(&self, turn_id: &str, prompt: String) -> Result<String> {
        let thread_id = self.thread_id()?.to_string();
        let response: TurnSteerResponse = self
            .request_handle
            .request(
                METHOD_TURN_STEER,
                TurnSteerParams {
                    thread_id,
                    input: vec![UserInput::Text {
                        text: prompt,
                        text_elements: Vec::new(),
                    }],
                    expected_turn_id: turn_id.to_string(),
                    ..TurnSteerParams::default()
                },
            )
            .await
            .context("failed to steer turn")?;
        Ok(response.turn_id)
    }

    pub(crate) async fn queue_prompt(&self, prompt: String) -> Result<QueuedSubmission> {
        let thread_id = self.thread_id()?.to_string();
        let response: ThreadQueueAddResponse = self
            .request_handle
            .request(
                METHOD_THREAD_QUEUE_ADD,
                ThreadQueueAddParams {
                    thread_id,
                    input: vec![UserInput::Text {
                        text: prompt,
                        text_elements: Vec::new(),
                    }],
                    client_user_message_id: client_message_id(),
                },
            )
            .await
            .context("failed to queue prompt")?;
        Ok(response.queued_submission)
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
}
