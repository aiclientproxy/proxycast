use base64::Engine;
use futures::{SinkExt, StreamExt};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tokio::time::{timeout, Duration};
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(2);

type EnvironmentSocket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteEnvironmentInfo {
    pub(crate) shell: RemoteShellInfo,
    pub(crate) cwd: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InitializeResponse {
    session_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteEnvironmentStatus {
    pub(crate) status: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct RemoteShellInfo {
    pub(crate) name: String,
    pub(crate) path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteFsReadFileResponse {
    pub(crate) data_base64: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteFsMetadataResponse {
    pub(crate) is_directory: bool,
    pub(crate) is_file: bool,
    pub(crate) is_symlink: bool,
    pub(crate) size: u64,
    pub(crate) created_at_ms: i64,
    pub(crate) modified_at_ms: i64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteFsCanonicalizeResponse {
    pub(crate) path: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteFsReadDirectoryResponse {
    pub(crate) entries: Vec<RemoteFsReadDirectoryEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteFsReadDirectoryEntry {
    pub(crate) file_name: String,
    pub(crate) is_directory: bool,
    pub(crate) is_file: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteFsRemoveResponse {}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteFsWalkResponse {
    pub(crate) entries: Vec<RemoteFsWalkEntry>,
    #[serde(default)]
    pub(crate) errors: Vec<Value>,
    #[serde(default)]
    pub(crate) truncated: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct RemoteFsWalkEntry {
    pub(crate) path: String,
    pub(crate) kind: String,
}

enum ClientCommand {
    Request {
        method: String,
        params: Value,
        response: oneshot::Sender<Result<Value, String>>,
    },
    Notify {
        method: String,
        params: Value,
    },
}

#[derive(Clone)]
pub(crate) struct RemoteExecClient {
    commands: mpsc::UnboundedSender<ClientCommand>,
}

impl std::fmt::Debug for RemoteExecClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RemoteExecClient")
            .finish_non_exhaustive()
    }
}

impl RemoteExecClient {
    pub(crate) async fn connect(
        exec_server_url: &str,
        connect_timeout: Duration,
    ) -> Result<(Arc<Self>, RemoteEnvironmentInfo), String> {
        let socket = timeout(connect_timeout, connect_async(exec_server_url))
            .await
            .map_err(|_| "timed out connecting to exec-server websocket".to_string())?
            .map_err(|error| format!("websocket connection failed: {error}"))?
            .0;
        let (commands, receiver) = mpsc::unbounded_channel();
        let client = Arc::new(Self { commands });
        tokio::spawn(run_client(socket, receiver));

        let initialize: InitializeResponse = client
            .request("initialize", json!({ "clientName": "lime-environment" }))
            .await?;
        if initialize.session_id.trim().is_empty() {
            return Err("exec-server initialize returned an empty sessionId".to_string());
        }
        client.notify("initialized", json!({}))?;
        let info: RemoteEnvironmentInfo = client.request("environment/info", json!({})).await?;
        let status: RemoteEnvironmentStatus =
            client.request("environment/status", json!({})).await?;
        if status.status != "ready" {
            return Err(format!(
                "exec-server reported environment status `{}`",
                status.status
            ));
        }
        Ok((client, info))
    }

    pub(crate) async fn request<T: DeserializeOwned>(
        &self,
        method: &str,
        params: Value,
    ) -> Result<T, String> {
        let (response_sender, response_receiver) = oneshot::channel();
        self.commands
            .send(ClientCommand::Request {
                method: method.to_string(),
                params,
                response: response_sender,
            })
            .map_err(|_| "exec-server websocket client is closed".to_string())?;
        let value = timeout(DEFAULT_REQUEST_TIMEOUT, response_receiver)
            .await
            .map_err(|_| format!("exec-server request '{method}' timed out"))?
            .map_err(|_| "exec-server websocket client stopped".to_string())??;
        if let Some(error) = value.get("__lime_rpc_error").and_then(Value::as_str) {
            return Err(format!("exec-server rejected '{method}': {error}"));
        }
        serde_json::from_value(value)
            .map_err(|error| format!("invalid exec-server '{method}' result: {error}"))
    }

    pub(crate) fn notify(&self, method: &str, params: Value) -> Result<(), String> {
        self.commands
            .send(ClientCommand::Notify {
                method: method.to_string(),
                params,
            })
            .map_err(|_| "exec-server websocket client is closed".to_string())
    }

    pub(crate) async fn fs_read_file(
        &self,
        path: &app_server_protocol::protocol::v2::PathUri,
        sandbox: Option<Value>,
    ) -> Result<Vec<u8>, String> {
        let response: RemoteFsReadFileResponse = self
            .request(
                "fs/readFile",
                json!({
                    "path": path,
                    "followSymlinks": true,
                    "sandbox": sandbox,
                }),
            )
            .await?;
        base64::engine::general_purpose::STANDARD
            .decode(response.data_base64)
            .map_err(|error| format!("exec-server fs/readFile returned invalid base64: {error}"))
    }

    pub(crate) async fn fs_write_file(
        &self,
        path: &app_server_protocol::protocol::v2::PathUri,
        data: &[u8],
        sandbox: Option<Value>,
    ) -> Result<(), String> {
        let _: Value = self
            .request(
                "fs/writeFile",
                json!({
                    "path": path,
                    "dataBase64": base64::engine::general_purpose::STANDARD.encode(data),
                    "followSymlinks": true,
                    "sandbox": sandbox,
                }),
            )
            .await?;
        Ok(())
    }

    pub(crate) async fn fs_get_metadata(
        &self,
        path: &app_server_protocol::protocol::v2::PathUri,
        sandbox: Option<Value>,
    ) -> Result<RemoteFsMetadataResponse, String> {
        self.request(
            "fs/getMetadata",
            json!({
                "path": path,
                "followSymlinks": true,
                "sandbox": sandbox,
            }),
        )
        .await
    }

    pub(crate) async fn fs_canonicalize(
        &self,
        path: &app_server_protocol::protocol::v2::PathUri,
        sandbox: Option<Value>,
    ) -> Result<RemoteFsCanonicalizeResponse, String> {
        self.request(
            "fs/canonicalize",
            json!({ "path": path, "sandbox": sandbox }),
        )
        .await
    }

    pub(crate) async fn fs_read_directory(
        &self,
        path: &app_server_protocol::protocol::v2::PathUri,
        sandbox: Option<Value>,
    ) -> Result<RemoteFsReadDirectoryResponse, String> {
        self.request(
            "fs/readDirectory",
            json!({ "path": path, "sandbox": sandbox }),
        )
        .await
    }

    pub(crate) async fn fs_remove(
        &self,
        path: &app_server_protocol::protocol::v2::PathUri,
        sandbox: Option<Value>,
    ) -> Result<(), String> {
        let _: RemoteFsRemoveResponse = self
            .request(
                "fs/remove",
                json!({
                    "path": path,
                    "recursive": true,
                    "force": false,
                    "followSymlinks": true,
                    "sandbox": sandbox,
                }),
            )
            .await?;
        Ok(())
    }

    pub(crate) async fn fs_create_directory(
        &self,
        path: &app_server_protocol::protocol::v2::PathUri,
        sandbox: Option<Value>,
    ) -> Result<(), String> {
        let _: Value = self
            .request(
                "fs/createDirectory",
                json!({ "path": path, "recursive": true, "sandbox": sandbox }),
            )
            .await?;
        Ok(())
    }

    pub(crate) async fn fs_walk(
        &self,
        path: &app_server_protocol::protocol::v2::PathUri,
        options: Value,
        sandbox: Option<Value>,
    ) -> Result<RemoteFsWalkResponse, String> {
        self.request(
            "fs/walk",
            json!({ "path": path, "options": options, "sandbox": sandbox }),
        )
        .await
    }
}

async fn run_client(
    mut socket: EnvironmentSocket,
    mut commands: mpsc::UnboundedReceiver<ClientCommand>,
) {
    let mut next_id = 1_u64;
    let mut pending = HashMap::<u64, oneshot::Sender<Result<Value, String>>>::new();
    loop {
        tokio::select! {
            command = commands.recv() => {
                let Some(command) = command else { break };
                match command {
                    ClientCommand::Notify { method, params } => {
                        if socket.send(Message::Text(json!({
                            "jsonrpc": "2.0",
                            "method": method,
                            "params": params,
                        }).to_string())).await.is_err() {
                            break;
                        }
                    }
                    ClientCommand::Request { method, params, response } => {
                        let id = next_id;
                        next_id = next_id.saturating_add(1);
                        if socket.send(Message::Text(json!({
                            "jsonrpc": "2.0",
                            "id": id,
                            "method": method,
                            "params": params,
                        }).to_string())).await.is_err() {
                            let _ = response.send(Err("exec-server websocket write failed".to_string()));
                            break;
                        }
                        pending.insert(id, response);
                    }
                }
            }
            message = socket.next() => {
                let Some(message) = message else { break };
                let message = match message {
                    Ok(message) => message,
                    Err(error) => {
                        for (_, response) in pending.drain() {
                            let _ = response.send(Err(format!("exec-server websocket read failed: {error}")));
                        }
                        break;
                    }
                };
                let Message::Text(text) = message else { continue };
                let value: Value = match serde_json::from_str(&text) {
                    Ok(value) => value,
                    Err(error) => {
                        for (_, response) in pending.drain() {
                            let _ = response.send(Err(format!("invalid exec-server JSON: {error}")));
                        }
                        break;
                    }
                };
                let Some(id) = value.get("id").and_then(Value::as_u64) else { continue };
                let Some(response) = pending.remove(&id) else { continue };
                if let Some(error) = value.get("error") {
                    let _ = response.send(Ok(json!({
                        "__lime_rpc_error": error.to_string(),
                    })));
                } else {
                    let _ = response.send(Ok(value.get("result").cloned().unwrap_or(Value::Null)));
                }
            }
        }
    }
    for (_, response) in pending {
        let _ = response.send(Err("exec-server websocket client stopped".to_string()));
    }
}
