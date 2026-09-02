//! ChromeBridge WebSocket 会话桥接
//!
//! 提供双通道能力：
//! - observer 通道：Chrome 扩展上报页面信息、心跳、命令执行结果
//! - control 通道：外部控制端下发命令并接收回传
//! - API 通道：App Server 请求触发命令并等待结果

use chrono::{DateTime, Utc};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, oneshot, Mutex};
use uuid::Uuid;

const DEFAULT_PROFILE_KEY: &str = "default";
const DEFAULT_COMMAND_TIMEOUT_MS: u64 = 30_000;
const MIN_COMMAND_TIMEOUT_MS: u64 = 1_000;
const MAX_COMMAND_TIMEOUT_MS: u64 = 120_000;
const DEFAULT_MESSAGE_PREVIEW_LIMIT: usize = 2_000;

const COMMAND_WHITELIST: &[&str] = &[
    "open_url",
    "click",
    "type",
    "scroll",
    "scroll_page",
    "get_page_info",
    "refresh_page",
    "go_back",
    "go_forward",
    "switch_tab",
    "list_tabs",
    "run_adapter",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeBridgePageInfo {
    pub title: Option<String>,
    pub url: Option<String>,
    pub markdown: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeBridgeObserverSnapshot {
    pub client_id: String,
    pub profile_key: String,
    pub connected_at: String,
    pub user_agent: Option<String>,
    pub last_heartbeat_at: Option<String>,
    pub last_page_info: Option<ChromeBridgePageInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeBridgeControlSnapshot {
    pub client_id: String,
    pub connected_at: String,
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeBridgePendingCommandSnapshot {
    pub request_id: String,
    pub source_type: String,
    pub command: String,
    pub observer_client_id: String,
    pub wait_for_page_info: bool,
    pub command_completed: bool,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeBridgeStatusSnapshot {
    pub observer_count: usize,
    pub control_count: usize,
    pub pending_command_count: usize,
    pub observers: Vec<ChromeBridgeObserverSnapshot>,
    pub controls: Vec<ChromeBridgeControlSnapshot>,
    pub pending_commands: Vec<ChromeBridgePendingCommandSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeBridgeDisconnectResult {
    pub disconnected_observer_count: usize,
    pub disconnected_control_count: usize,
    pub status: ChromeBridgeStatusSnapshot,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeBridgeCommandRequest {
    #[serde(default)]
    pub profile_key: Option<String>,
    pub command: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub payload: Option<Value>,
    #[serde(default)]
    pub wait_for_page_info: bool,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChromeBridgeCommandResult {
    pub success: bool,
    pub request_id: String,
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_info: Option<ChromeBridgePageInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ObserverCommandResultPayload {
    #[serde(alias = "requestId")]
    pub request_id: String,
    pub status: String,
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
    #[serde(default)]
    pub data: Option<Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ControlCommandPayload {
    #[serde(alias = "requestId")]
    pub request_id: String,
    pub command: String,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub payload: Option<Value>,
    #[serde(default)]
    pub wait_for_page_info: bool,
}

pub struct ChromeBridgeHub {
    inner: Mutex<ChromeBridgeState>,
}

struct ChromeBridgeState {
    observers: HashMap<String, ObserverConnection>,
    controls: HashMap<String, ControlConnection>,
    pending_commands: HashMap<String, PendingCommand>,
}

#[derive(Clone)]
struct ObserverConnection {
    sender: mpsc::UnboundedSender<String>,
    profile_key: String,
    connected_at: DateTime<Utc>,
    user_agent: Option<String>,
    last_heartbeat_at: Option<DateTime<Utc>>,
    last_page_info: Option<ChromeBridgePageInfo>,
}

#[derive(Clone)]
struct ControlConnection {
    sender: mpsc::UnboundedSender<String>,
    connected_at: DateTime<Utc>,
    user_agent: Option<String>,
}

enum PendingSource {
    Api(oneshot::Sender<ChromeBridgeCommandResult>),
    Control { control_client_id: String },
}

struct PendingCommand {
    request_id: String,
    source: PendingSource,
    command: String,
    observer_client_id: String,
    wait_for_page_info: bool,
    command_completed: bool,
    execution_message: Option<String>,
    created_at: DateTime<Utc>,
    expires_at: Instant,
}

impl PendingCommand {
    fn source_type(&self) -> &'static str {
        match self.source {
            PendingSource::Api(_) => "api",
            PendingSource::Control { .. } => "control",
        }
    }
}

impl ChromeBridgeState {
    fn new() -> Self {
        Self {
            observers: HashMap::new(),
            controls: HashMap::new(),
            pending_commands: HashMap::new(),
        }
    }
}

impl ChromeBridgeHub {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(ChromeBridgeState::new()),
        }
    }

    pub async fn register_observer(
        &self,
        client_id: String,
        profile_key: Option<String>,
        user_agent: Option<String>,
        sender: mpsc::UnboundedSender<String>,
    ) {
        let mut inner = self.inner.lock().await;
        inner.observers.insert(
            client_id,
            ObserverConnection {
                sender,
                profile_key: normalize_profile_key(profile_key),
                connected_at: Utc::now(),
                user_agent,
                last_heartbeat_at: None,
                last_page_info: None,
            },
        );
    }

    pub async fn unregister_observer(&self, client_id: &str) {
        let pending = {
            let mut inner = self.inner.lock().await;
            inner.observers.remove(client_id);
            take_pending_by_observer(&mut inner.pending_commands, client_id)
        };
        self.resolve_pending_with_disconnect(pending).await;
    }

    pub async fn register_control(
        &self,
        client_id: String,
        user_agent: Option<String>,
        sender: mpsc::UnboundedSender<String>,
    ) {
        let mut inner = self.inner.lock().await;
        inner.controls.insert(
            client_id,
            ControlConnection {
                sender,
                connected_at: Utc::now(),
                user_agent,
            },
        );
    }

    pub async fn unregister_control(&self, client_id: &str) {
        let mut inner = self.inner.lock().await;
        inner.controls.remove(client_id);
        let pending_ids: Vec<String> = inner
            .pending_commands
            .iter()
            .filter_map(|(request_id, pending)| match &pending.source {
                PendingSource::Control { control_client_id } if control_client_id == client_id => {
                    Some(request_id.clone())
                }
                _ => None,
            })
            .collect();

        for request_id in pending_ids {
            inner.pending_commands.remove(&request_id);
        }
    }

    pub async fn disconnect_connections(
        &self,
        profile_key: Option<&str>,
    ) -> ChromeBridgeDisconnectResult {
        let normalized_profile =
            profile_key.map(|value| normalize_profile_key(Some(value.to_string())));
        let (observer_senders, control_senders, pending) = {
            let mut inner = self.inner.lock().await;

            let observer_ids: Vec<String> = inner
                .observers
                .iter()
                .filter_map(|(client_id, conn)| {
                    let matches_profile = normalized_profile
                        .as_ref()
                        .map(|profile| conn.profile_key == *profile)
                        .unwrap_or(true);
                    if matches_profile {
                        Some(client_id.clone())
                    } else {
                        None
                    }
                })
                .collect();

            let observer_senders = observer_ids
                .iter()
                .filter_map(|client_id| {
                    inner
                        .observers
                        .get(client_id)
                        .map(|conn| conn.sender.clone())
                })
                .collect::<Vec<_>>();

            let mut pending = Vec::new();
            for client_id in &observer_ids {
                inner.observers.remove(client_id);
                pending.extend(take_pending_by_observer(
                    &mut inner.pending_commands,
                    client_id,
                ));
            }

            let control_ids: Vec<String> = if observer_ids.is_empty() {
                Vec::new()
            } else {
                inner.controls.keys().cloned().collect()
            };

            let control_senders = control_ids
                .iter()
                .filter_map(|client_id| {
                    inner
                        .controls
                        .get(client_id)
                        .map(|conn| conn.sender.clone())
                })
                .collect::<Vec<_>>();

            for client_id in &control_ids {
                inner.controls.remove(client_id);
                let pending_ids: Vec<String> = inner
                    .pending_commands
                    .iter()
                    .filter_map(|(request_id, pending)| match &pending.source {
                        PendingSource::Control { control_client_id }
                            if control_client_id == client_id =>
                        {
                            Some(request_id.clone())
                        }
                        _ => None,
                    })
                    .collect();
                for request_id in pending_ids {
                    inner.pending_commands.remove(&request_id);
                }
            }

            (observer_senders, control_senders, pending)
        };

        for sender in &observer_senders {
            let _ = sender.send(
                json!({
                    "type": "force_disconnect",
                    "message": "Lime 已主动断开当前扩展连接。",
                })
                .to_string(),
            );
        }

        for sender in &control_senders {
            let _ = sender.send(
                json!({
                    "type": "force_disconnect",
                    "message": "Lime 已主动断开当前控制连接。",
                })
                .to_string(),
            );
        }

        self.resolve_pending_with_disconnect(pending).await;
        let status = self.get_status_snapshot().await;

        ChromeBridgeDisconnectResult {
            disconnected_observer_count: observer_senders.len(),
            disconnected_control_count: control_senders.len(),
            status,
        }
    }

    pub async fn execute_api_command(
        &self,
        request: ChromeBridgeCommandRequest,
    ) -> Result<ChromeBridgeCommandResult, String> {
        self.sweep_expired_pending().await;
        validate_command(&request.command, &request.url, &request.payload)?;

        let timeout = normalize_timeout_ms(request.timeout_ms);
        let source_client_id = format!("lime-api-{}", Uuid::new_v4());
        let request_id = format!("cb-api-{}", Uuid::new_v4());

        let (observer_id, observer_sender) = {
            let inner = self.inner.lock().await;
            select_observer(&inner.observers, request.profile_key.as_deref())
                .ok_or_else(|| "没有可用的 Chrome observer 连接，请先连接扩展。".to_string())?
        };

        let command = request.command.trim().to_string();
        let payload = build_command_payload(
            &request_id,
            &source_client_id,
            &command,
            request.target.clone(),
            request.text.clone(),
            request.url.clone(),
            request.payload.clone(),
            request.wait_for_page_info,
        );

        let (tx, rx) = oneshot::channel();
        {
            let mut inner = self.inner.lock().await;
            inner.pending_commands.insert(
                request_id.clone(),
                PendingCommand {
                    request_id: request_id.clone(),
                    source: PendingSource::Api(tx),
                    command: command.clone(),
                    observer_client_id: observer_id.clone(),
                    wait_for_page_info: request.wait_for_page_info,
                    command_completed: false,
                    execution_message: None,
                    created_at: Utc::now(),
                    expires_at: Instant::now() + Duration::from_millis(timeout),
                },
            );
        }

        if observer_sender.send(payload.to_string()).is_err() {
            let pending = {
                let mut inner = self.inner.lock().await;
                inner.pending_commands.remove(&request_id)
            };
            if let Some(pending_cmd) = pending {
                self.dispatch_pending_result(
                    pending_cmd,
                    ChromeBridgeCommandResult {
                        success: false,
                        request_id,
                        command,
                        message: None,
                        error: Some("observer 通道发送失败，连接可能已断开。".to_string()),
                        page_info: None,
                        data: None,
                    },
                    None,
                )
                .await;
            }
            return Err("observer 通道发送失败，连接可能已断开。".to_string());
        }

        let wait_result = tokio::time::timeout(Duration::from_millis(timeout), rx).await;
        match wait_result {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Err("命令结果通道关闭，命令未完成。".to_string()),
            Err(_) => {
                let pending = {
                    let mut inner = self.inner.lock().await;
                    inner.pending_commands.remove(&request_id)
                };
                if let Some(pending_cmd) = pending {
                    self.dispatch_pending_result(
                        pending_cmd,
                        ChromeBridgeCommandResult {
                            success: false,
                            request_id,
                            command,
                            message: None,
                            error: Some("等待 Chrome 执行结果超时。".to_string()),
                            page_info: None,
                            data: None,
                        },
                        None,
                    )
                    .await;
                }
                Err("等待 Chrome 执行结果超时。".to_string())
            }
        }
    }

    pub async fn handle_control_command(
        &self,
        control_client_id: &str,
        payload: ControlCommandPayload,
    ) {
        self.sweep_expired_pending().await;

        if let Err(error) = validate_command(&payload.command, &payload.url, &payload.payload) {
            self.send_control_error(control_client_id, &payload.request_id, &error)
                .await;
            return;
        }

        let (observer_id, observer_sender) = {
            let inner = self.inner.lock().await;
            select_observer(&inner.observers, None)
                .map(|(id, sender)| (id, sender))
                .unwrap_or_else(|| (String::new(), mpsc::unbounded_channel::<String>().0))
        };

        if observer_id.is_empty() {
            self.send_control_error(
                control_client_id,
                &payload.request_id,
                "没有可用的 Chrome observer 连接。",
            )
            .await;
            return;
        }

        let message = build_command_payload(
            &payload.request_id,
            control_client_id,
            &payload.command,
            payload.target,
            payload.text,
            payload.url,
            payload.payload,
            payload.wait_for_page_info,
        )
        .to_string();

        {
            let mut inner = self.inner.lock().await;
            inner.pending_commands.insert(
                payload.request_id.clone(),
                PendingCommand {
                    request_id: payload.request_id.clone(),
                    source: PendingSource::Control {
                        control_client_id: control_client_id.to_string(),
                    },
                    command: payload.command,
                    observer_client_id: observer_id,
                    wait_for_page_info: payload.wait_for_page_info,
                    command_completed: false,
                    execution_message: None,
                    created_at: Utc::now(),
                    expires_at: Instant::now() + Duration::from_millis(DEFAULT_COMMAND_TIMEOUT_MS),
                },
            );
        }

        if observer_sender.send(message).is_err() {
            let pending = {
                let mut inner = self.inner.lock().await;
                inner.pending_commands.remove(&payload.request_id)
            };
            if let Some(pending_cmd) = pending {
                self.dispatch_pending_result(
                    pending_cmd,
                    ChromeBridgeCommandResult {
                        success: false,
                        request_id: payload.request_id,
                        command: "unknown".to_string(),
                        message: None,
                        error: Some("observer 通道发送失败，连接可能已断开。".to_string()),
                        page_info: None,
                        data: None,
                    },
                    None,
                )
                .await;
            }
        }
    }

    pub async fn handle_observer_heartbeat(&self, observer_client_id: &str) {
        let sender = {
            let mut inner = self.inner.lock().await;
            if let Some(observer) = inner.observers.get_mut(observer_client_id) {
                observer.last_heartbeat_at = Some(Utc::now());
                Some(observer.sender.clone())
            } else {
                None
            }
        };

        if let Some(observer_sender) = sender {
            let _ = observer_sender.send(
                json!({
                    "type": "heartbeat_ack",
                    "timestamp": Utc::now().timestamp_millis(),
                })
                .to_string(),
            );
        }
    }

    pub async fn handle_observer_command_result(
        &self,
        observer_client_id: &str,
        payload: ObserverCommandResultPayload,
    ) {
        let status = payload.status.trim().to_ascii_lowercase();
        let mut control_success_notice: Option<(String, String, String)> = None;

        let pending = {
            let mut inner = self.inner.lock().await;
            let Some(existing) = inner.pending_commands.get(&payload.request_id) else {
                return;
            };

            if existing.observer_client_id != observer_client_id {
                return;
            }

            if status != "success" {
                inner.pending_commands.remove(&payload.request_id)
            } else {
                let wait_for_page_info = inner
                    .pending_commands
                    .get(&payload.request_id)
                    .map(|pending| pending.wait_for_page_info)
                    .unwrap_or(false);

                if wait_for_page_info {
                    if let Some(pending) = inner.pending_commands.get_mut(&payload.request_id) {
                        pending.command_completed = true;
                        pending.execution_message = payload.message.clone();

                        // control source 在成功后先回 command_result，再等待 pageInfoUpdate。
                        if let PendingSource::Control { control_client_id } = &pending.source {
                            let success_message = payload
                                .message
                                .clone()
                                .unwrap_or_else(|| "命令执行成功".to_string());
                            control_success_notice = Some((
                                control_client_id.clone(),
                                pending.request_id.clone(),
                                success_message,
                            ));
                        }
                    }
                    None
                } else {
                    inner.pending_commands.remove(&payload.request_id)
                }
            }
        };

        if let Some((control_id, request_id, success_message)) = control_success_notice {
            self.send_message_to_control(
                &control_id,
                json!({
                    "type": "command_result",
                    "data": {
                        "requestId": request_id,
                        "status": "success",
                        "message": success_message,
                    }
                }),
            )
            .await;
        }

        if let Some(pending_cmd) = pending {
            let result = if status == "success" {
                ChromeBridgeCommandResult {
                    success: true,
                    request_id: payload.request_id,
                    command: pending_cmd.command.clone(),
                    message: payload.message.or(Some("命令执行成功".to_string())),
                    error: None,
                    page_info: None,
                    data: payload.data,
                }
            } else {
                ChromeBridgeCommandResult {
                    success: false,
                    request_id: payload.request_id,
                    command: pending_cmd.command.clone(),
                    message: None,
                    error: payload.error.or(Some("命令执行失败".to_string())),
                    page_info: None,
                    data: payload.data,
                }
            };

            self.dispatch_pending_result(pending_cmd, result, None)
                .await;
        }
    }

    pub async fn handle_observer_page_info_update(
        &self,
        observer_client_id: &str,
        markdown: String,
    ) {
        let page_info = parse_page_info(markdown);

        let pending_to_resolve = {
            let mut inner = self.inner.lock().await;
            if let Some(observer) = inner.observers.get_mut(observer_client_id) {
                observer.last_page_info = Some(page_info.clone());
            }

            let request_ids: Vec<String> = inner
                .pending_commands
                .iter()
                .filter_map(|(request_id, pending)| {
                    if pending.observer_client_id == observer_client_id
                        && pending.wait_for_page_info
                        && pending.command_completed
                    {
                        Some(request_id.clone())
                    } else {
                        None
                    }
                })
                .collect();

            let mut removed = Vec::new();
            for request_id in request_ids {
                if let Some(pending) = inner.pending_commands.remove(&request_id) {
                    removed.push(pending);
                }
            }
            removed
        };

        for pending_cmd in pending_to_resolve {
            let result = ChromeBridgeCommandResult {
                success: true,
                request_id: pending_cmd.request_id.clone(),
                command: pending_cmd.command.clone(),
                message: pending_cmd.execution_message.clone(),
                error: None,
                page_info: Some(page_info.clone()),
                data: None,
            };
            self.dispatch_pending_result(pending_cmd, result, Some(true))
                .await;
        }
    }

    pub async fn get_status_snapshot(&self) -> ChromeBridgeStatusSnapshot {
        self.sweep_expired_pending().await;
        let inner = self.inner.lock().await;

        let observers = inner
            .observers
            .iter()
            .map(|(client_id, conn)| ChromeBridgeObserverSnapshot {
                client_id: client_id.clone(),
                profile_key: conn.profile_key.clone(),
                connected_at: conn.connected_at.to_rfc3339(),
                user_agent: conn.user_agent.clone(),
                last_heartbeat_at: conn.last_heartbeat_at.map(|v| v.to_rfc3339()),
                last_page_info: conn.last_page_info.clone(),
            })
            .collect::<Vec<_>>();

        let controls = inner
            .controls
            .iter()
            .map(|(client_id, conn)| ChromeBridgeControlSnapshot {
                client_id: client_id.clone(),
                connected_at: conn.connected_at.to_rfc3339(),
                user_agent: conn.user_agent.clone(),
            })
            .collect::<Vec<_>>();

        let pending_commands = inner
            .pending_commands
            .values()
            .map(|pending| ChromeBridgePendingCommandSnapshot {
                request_id: pending.request_id.clone(),
                source_type: pending.source_type().to_string(),
                command: pending.command.clone(),
                observer_client_id: pending.observer_client_id.clone(),
                wait_for_page_info: pending.wait_for_page_info,
                command_completed: pending.command_completed,
                created_at: pending.created_at.to_rfc3339(),
            })
            .collect::<Vec<_>>();

        ChromeBridgeStatusSnapshot {
            observer_count: observers.len(),
            control_count: controls.len(),
            pending_command_count: pending_commands.len(),
            observers,
            controls,
            pending_commands,
        }
    }

    pub async fn send_message_to_control(&self, control_client_id: &str, message: Value) {
        let sender = {
            let inner = self.inner.lock().await;
            inner
                .controls
                .get(control_client_id)
                .map(|conn| conn.sender.clone())
        };
        if let Some(tx) = sender {
            let _ = tx.send(message.to_string());
        }
    }

    async fn send_control_error(&self, control_client_id: &str, request_id: &str, error: &str) {
        self.send_message_to_control(
            control_client_id,
            json!({
                "type": "command_result",
                "data": {
                    "requestId": request_id,
                    "status": "error",
                    "error": truncate_message(error),
                }
            }),
        )
        .await;
    }

    async fn sweep_expired_pending(&self) {
        let expired = {
            let mut inner = self.inner.lock().await;
            let now = Instant::now();
            let expired_ids: Vec<String> = inner
                .pending_commands
                .iter()
                .filter_map(|(request_id, pending)| {
                    if pending.expires_at <= now {
                        Some(request_id.clone())
                    } else {
                        None
                    }
                })
                .collect();

            let mut expired = Vec::new();
            for request_id in expired_ids {
                if let Some(pending) = inner.pending_commands.remove(&request_id) {
                    expired.push(pending);
                }
            }
            expired
        };

        for pending in expired {
            let result = ChromeBridgeCommandResult {
                success: false,
                request_id: pending.request_id.clone(),
                command: pending.command.clone(),
                message: None,
                error: Some("命令执行超时。".to_string()),
                page_info: None,
                data: None,
            };
            self.dispatch_pending_result(pending, result, None).await;
        }
    }

    async fn resolve_pending_with_disconnect(&self, pendings: Vec<PendingCommand>) {
        for pending in pendings {
            let result = ChromeBridgeCommandResult {
                success: false,
                request_id: pending.request_id.clone(),
                command: pending.command.clone(),
                message: None,
                error: Some("observer 已断开连接。".to_string()),
                page_info: None,
                data: None,
            };
            self.dispatch_pending_result(pending, result, None).await;
        }
    }

    async fn dispatch_pending_result(
        &self,
        pending: PendingCommand,
        mut result: ChromeBridgeCommandResult,
        from_page_info_update: Option<bool>,
    ) {
        result.message = result.message.map(|v| truncate_message(&v));
        result.error = result.error.map(|v| truncate_message(&v));

        match pending.source {
            PendingSource::Api(tx) => {
                let _ = tx.send(result);
            }
            PendingSource::Control { control_client_id } => {
                let is_page_info_event = from_page_info_update.unwrap_or(false);
                if is_page_info_event {
                    let page_payload = result.page_info.clone().map(|page| {
                        json!({
                            "requestId": result.request_id,
                            "markdown": page.markdown,
                            "title": page.title,
                            "url": page.url,
                            "updatedAt": page.updated_at,
                        })
                    });
                    if let Some(page_data) = page_payload {
                        self.send_message_to_control(
                            &control_client_id,
                            json!({
                                "type": "page_info_update",
                                "data": page_data,
                            }),
                        )
                        .await;
                    }
                    return;
                }

                if result.success {
                    self.send_message_to_control(
                        &control_client_id,
                        json!({
                            "type": "command_result",
                            "data": {
                                "requestId": result.request_id,
                                "status": "success",
                                "message": result.message,
                                "data": result.data,
                            }
                        }),
                    )
                    .await;
                } else {
                    self.send_message_to_control(
                        &control_client_id,
                        json!({
                            "type": "command_result",
                            "data": {
                                "requestId": result.request_id,
                                "status": "error",
                                "error": result.error,
                            }
                        }),
                    )
                    .await;
                }
            }
        }
    }
}

impl Default for ChromeBridgeHub {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_command(
    command: &str,
    url: &Option<String>,
    payload: &Option<Value>,
) -> Result<(), String> {
    let normalized = command.trim().to_ascii_lowercase();
    if !COMMAND_WHITELIST.contains(&normalized.as_str()) {
        return Err(format!(
            "不允许的命令: {}，仅允许: {}",
            command,
            COMMAND_WHITELIST.join(", ")
        ));
    }

    if normalized == "open_url" {
        let Some(url_value) = url else {
            return Err("open_url 命令需要提供 url。".to_string());
        };
        if url_value.trim().is_empty() {
            return Err("open_url 命令的 url 不能为空。".to_string());
        }
    }

    if normalized == "run_adapter" {
        let adapter_name = payload
            .as_ref()
            .and_then(|value| value.get("adapter_name"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        if adapter_name.is_none() {
            return Err("run_adapter 命令需要提供 payload.adapter_name。".to_string());
        }
    }

    Ok(())
}

fn take_pending_by_observer(
    pending_map: &mut HashMap<String, PendingCommand>,
    observer_client_id: &str,
) -> Vec<PendingCommand> {
    let request_ids: Vec<String> = pending_map
        .iter()
        .filter_map(|(request_id, pending)| {
            if pending.observer_client_id == observer_client_id {
                Some(request_id.clone())
            } else {
                None
            }
        })
        .collect();

    let mut removed = Vec::new();
    for request_id in request_ids {
        if let Some(pending) = pending_map.remove(&request_id) {
            removed.push(pending);
        }
    }
    removed
}

fn normalize_timeout_ms(input: Option<u64>) -> u64 {
    input
        .unwrap_or(DEFAULT_COMMAND_TIMEOUT_MS)
        .clamp(MIN_COMMAND_TIMEOUT_MS, MAX_COMMAND_TIMEOUT_MS)
}

fn normalize_profile_key(input: Option<String>) -> String {
    let raw = input.unwrap_or_else(|| DEFAULT_PROFILE_KEY.to_string());
    let normalized: String = raw
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect();

    if normalized.trim_matches('_').is_empty() {
        DEFAULT_PROFILE_KEY.to_string()
    } else {
        normalized
    }
}

fn select_observer(
    observers: &HashMap<String, ObserverConnection>,
    profile_key: Option<&str>,
) -> Option<(String, mpsc::UnboundedSender<String>)> {
    if observers.is_empty() {
        return None;
    }

    if let Some(profile_key_value) = profile_key {
        let normalized_profile = normalize_profile_key(Some(profile_key_value.to_string()));
        if let Some((client_id, conn)) = observers
            .iter()
            .find(|(_, conn)| conn.profile_key == normalized_profile)
        {
            return Some((client_id.clone(), conn.sender.clone()));
        }
    }

    observers
        .iter()
        .next()
        .map(|(client_id, conn)| (client_id.clone(), conn.sender.clone()))
}

fn build_command_payload(
    request_id: &str,
    source_client_id: &str,
    command: &str,
    target: Option<String>,
    text: Option<String>,
    url: Option<String>,
    payload: Option<Value>,
    wait_for_page_info: bool,
) -> Value {
    json!({
        "type": "command",
        "data": {
            "requestId": request_id,
            "sourceClientId": source_client_id,
            "command": command,
            "target": target,
            "text": text,
            "url": url,
            "payload": payload,
            "wait_for_page_info": wait_for_page_info,
        }
    })
}

fn parse_page_info(markdown: String) -> ChromeBridgePageInfo {
    let mut title = None;
    let mut url = None;

    for line in markdown.lines().take(6) {
        let trimmed = line.trim();
        if title.is_none() && trimmed.starts_with('#') {
            let extracted = trimmed.trim_start_matches('#').trim();
            if !extracted.is_empty() {
                title = Some(extracted.to_string());
            }
        }
        if url.is_none() {
            let lower = trimmed.to_ascii_lowercase();
            if lower.starts_with("url:") {
                let extracted = trimmed[4..].trim();
                if !extracted.is_empty() {
                    url = Some(extracted.to_string());
                }
            }
        }
        if title.is_some() && url.is_some() {
            break;
        }
    }

    ChromeBridgePageInfo {
        title,
        url,
        markdown,
        updated_at: Utc::now().to_rfc3339(),
    }
}

fn truncate_message(input: &str) -> String {
    let mut chars = input.chars();
    let truncated: String = chars.by_ref().take(DEFAULT_MESSAGE_PREVIEW_LIMIT).collect();
    if chars.next().is_some() {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

static CHROME_BRIDGE_HUB: Lazy<Arc<ChromeBridgeHub>> =
    Lazy::new(|| Arc::new(ChromeBridgeHub::new()));

pub fn chrome_bridge_hub() -> Arc<ChromeBridgeHub> {
    CHROME_BRIDGE_HUB.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn should_normalize_profile_key() {
        assert_eq!(
            normalize_profile_key(Some("search/google".to_string())),
            "search_google"
        );
        assert_eq!(normalize_profile_key(Some("___".to_string())), "default");
        assert_eq!(normalize_profile_key(None), "default");
    }

    #[test]
    fn should_validate_command_whitelist() {
        assert!(
            validate_command("open_url", &Some("https://example.com".to_string()), &None,).is_ok()
        );
        assert!(validate_command("click", &None, &None).is_ok());
        assert!(validate_command("list_tabs", &None, &None).is_ok());
        assert!(validate_command(
            "run_adapter",
            &None,
            &Some(json!({
                "adapter_name": "github/search",
                "args": {"query": "mcp"},
            })),
        )
        .is_ok());
        assert!(validate_command("run_adapter", &None, &None).is_err());
        assert!(validate_command("eval_js", &None, &None).is_err());
    }

    #[test]
    fn should_parse_page_info() {
        let markdown = "# Title\nURL: https://example.com\ncontent".to_string();
        let parsed = parse_page_info(markdown.clone());
        assert_eq!(parsed.title.as_deref(), Some("Title"));
        assert_eq!(parsed.url.as_deref(), Some("https://example.com"));
        assert_eq!(parsed.markdown, markdown);
    }

    #[tokio::test]
    async fn api_command_should_timeout_without_result() {
        let hub = ChromeBridgeHub::new();
        let (observer_tx, _observer_rx) = mpsc::unbounded_channel::<String>();
        hub.register_observer(
            "observer-a".to_string(),
            Some("search_google".to_string()),
            None,
            observer_tx,
        )
        .await;

        let result = hub
            .execute_api_command(ChromeBridgeCommandRequest {
                profile_key: Some("search_google".to_string()),
                command: "click".to_string(),
                target: Some("#btn".to_string()),
                text: None,
                url: None,
                payload: None,
                wait_for_page_info: false,
                timeout_ms: Some(10),
            })
            .await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn should_cleanup_pending_when_observer_disconnects() {
        let hub = Arc::new(ChromeBridgeHub::new());

        let (observer_tx, _observer_rx) = mpsc::unbounded_channel::<String>();
        hub.register_observer(
            "observer-a".to_string(),
            Some("search_google".to_string()),
            None,
            observer_tx,
        )
        .await;

        let (result_tx, result_rx) = oneshot::channel();
        {
            let mut inner = hub.inner.lock().await;
            inner.pending_commands.insert(
                "req-1".to_string(),
                PendingCommand {
                    request_id: "req-1".to_string(),
                    source: PendingSource::Api(result_tx),
                    command: "click".to_string(),
                    observer_client_id: "observer-a".to_string(),
                    wait_for_page_info: false,
                    command_completed: false,
                    execution_message: None,
                    created_at: Utc::now(),
                    expires_at: Instant::now() + Duration::from_secs(30),
                },
            );
        }

        hub.unregister_observer("observer-a").await;
        let result = result_rx.await.expect("must receive disconnect result");
        assert!(!result.success);
        assert!(result.error.unwrap_or_default().contains("observer"));
    }

    #[tokio::test]
    async fn should_force_disconnect_connections_and_notify_clients() {
        let hub = Arc::new(ChromeBridgeHub::new());

        let (observer_tx, mut observer_rx) = mpsc::unbounded_channel::<String>();
        hub.register_observer(
            "observer-a".to_string(),
            Some("search_google".to_string()),
            None,
            observer_tx,
        )
        .await;

        let (control_tx, mut control_rx) = mpsc::unbounded_channel::<String>();
        hub.register_control("control-a".to_string(), None, control_tx)
            .await;

        let (result_tx, result_rx) = oneshot::channel();
        {
            let mut inner = hub.inner.lock().await;
            inner.pending_commands.insert(
                "req-disconnect".to_string(),
                PendingCommand {
                    request_id: "req-disconnect".to_string(),
                    source: PendingSource::Api(result_tx),
                    command: "click".to_string(),
                    observer_client_id: "observer-a".to_string(),
                    wait_for_page_info: false,
                    command_completed: false,
                    execution_message: None,
                    created_at: Utc::now(),
                    expires_at: Instant::now() + Duration::from_secs(30),
                },
            );
        }

        let result = hub.disconnect_connections(Some("search_google")).await;
        assert_eq!(result.disconnected_observer_count, 1);
        assert_eq!(result.disconnected_control_count, 1);
        assert_eq!(result.status.observer_count, 0);
        assert_eq!(result.status.control_count, 0);
        assert_eq!(result.status.pending_command_count, 0);

        let observer_message = observer_rx
            .recv()
            .await
            .expect("observer should receive force_disconnect");
        let control_message = control_rx
            .recv()
            .await
            .expect("control should receive force_disconnect");

        let observer_payload: Value =
            serde_json::from_str(&observer_message).expect("observer message should be valid json");
        let control_payload: Value =
            serde_json::from_str(&control_message).expect("control message should be valid json");

        assert_eq!(
            observer_payload.get("type").and_then(Value::as_str),
            Some("force_disconnect")
        );
        assert_eq!(
            control_payload.get("type").and_then(Value::as_str),
            Some("force_disconnect")
        );

        let pending_result = result_rx
            .await
            .expect("pending api command should resolve after disconnect");
        assert!(!pending_result.success);
        assert!(pending_result
            .error
            .unwrap_or_default()
            .contains("observer"));
    }

    #[tokio::test]
    async fn wait_for_page_info_should_resolve_after_update() {
        let hub = Arc::new(ChromeBridgeHub::new());
        let (observer_tx, _observer_rx) = mpsc::unbounded_channel::<String>();
        hub.register_observer(
            "observer-a".to_string(),
            Some("default".to_string()),
            None,
            observer_tx,
        )
        .await;

        let (result_tx, result_rx) = oneshot::channel();
        {
            let mut inner = hub.inner.lock().await;
            inner.pending_commands.insert(
                "req-2".to_string(),
                PendingCommand {
                    request_id: "req-2".to_string(),
                    source: PendingSource::Api(result_tx),
                    command: "open_url".to_string(),
                    observer_client_id: "observer-a".to_string(),
                    wait_for_page_info: true,
                    command_completed: true,
                    execution_message: Some("ok".to_string()),
                    created_at: Utc::now(),
                    expires_at: Instant::now() + Duration::from_secs(30),
                },
            );
        }

        hub.handle_observer_page_info_update(
            "observer-a",
            "# T\nURL: https://example.com".to_string(),
        )
        .await;

        let result = result_rx.await.expect("must receive page info result");
        assert!(result.success);
        assert!(result.page_info.is_some());
    }

    #[tokio::test]
    async fn observer_command_result_should_preserve_data_payload() {
        let hub = Arc::new(ChromeBridgeHub::new());
        let (observer_tx, _observer_rx) = mpsc::unbounded_channel::<String>();
        hub.register_observer(
            "observer-a".to_string(),
            Some("default".to_string()),
            None,
            observer_tx,
        )
        .await;

        let (result_tx, result_rx) = oneshot::channel();
        {
            let mut inner = hub.inner.lock().await;
            inner.pending_commands.insert(
                "req-tabs".to_string(),
                PendingCommand {
                    request_id: "req-tabs".to_string(),
                    source: PendingSource::Api(result_tx),
                    command: "list_tabs".to_string(),
                    observer_client_id: "observer-a".to_string(),
                    wait_for_page_info: false,
                    command_completed: false,
                    execution_message: None,
                    created_at: Utc::now(),
                    expires_at: Instant::now() + Duration::from_secs(30),
                },
            );
        }

        hub.handle_observer_command_result(
            "observer-a",
            ObserverCommandResultPayload {
                request_id: "req-tabs".to_string(),
                status: "success".to_string(),
                message: Some("ok".to_string()),
                error: None,
                data: Some(json!({
                    "tabs": [
                        {
                            "id": 101,
                            "index": 0,
                            "title": "首页",
                            "url": "https://weibo.com/home",
                            "active": true,
                        }
                    ],
                })),
            },
        )
        .await;

        let result = result_rx.await.expect("must receive tabs result");
        assert!(result.success);
        assert_eq!(result.command, "list_tabs");
        assert_eq!(
            result.data,
            Some(json!({
                "tabs": [
                    {
                        "id": 101,
                        "index": 0,
                        "title": "首页",
                        "url": "https://weibo.com/home",
                        "active": true,
                    }
                ],
            })),
        );
    }

    #[test]
    fn build_command_payload_should_include_payload() {
        let payload = build_command_payload(
            "req-1",
            "lime-api-1",
            "run_adapter",
            Some("202".to_string()),
            None,
            None,
            Some(json!({
                "adapter_name": "github/search",
                "args": {"query": "mcp"},
            })),
            false,
        );

        assert_eq!(payload["data"]["command"], json!("run_adapter"));
        assert_eq!(payload["data"]["target"], json!("202"));
        assert_eq!(
            payload["data"]["payload"]["adapter_name"],
            json!("github/search")
        );
        assert_eq!(payload["data"]["payload"]["args"]["query"], json!("mcp"));
    }
}
