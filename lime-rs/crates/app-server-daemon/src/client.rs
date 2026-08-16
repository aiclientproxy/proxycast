use app_server_protocol::ClientCapabilities;
use app_server_protocol::ClientInfo;
use app_server_protocol::InitializeParams;
use app_server_protocol::InitializeResponse;
use app_server_protocol::JsonRpcMessage;
use app_server_protocol::JsonRpcNotification;
use app_server_protocol::JsonRpcRequest;
use app_server_protocol::JsonRpcResponse;
use app_server_protocol::RequestId;

pub const DAEMON_CLIENT_NAME: &str = "app_server_daemon";
pub const INITIALIZE_REQUEST_ID: RequestId = RequestId::Integer(1);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProbeInfo {
    pub app_server_version: String,
    pub protocol_version: String,
}

pub fn initialize_probe_request(client_version: impl Into<String>) -> JsonRpcMessage {
    JsonRpcMessage::Request(JsonRpcRequest::new(
        INITIALIZE_REQUEST_ID,
        app_server_protocol::METHOD_INITIALIZE,
        Some(
            serde_json::to_value(InitializeParams {
                client_info: ClientInfo {
                    name: DAEMON_CLIENT_NAME.to_string(),
                    title: Some("App Server Daemon".to_string()),
                    version: Some(client_version.into()),
                },
                capabilities: ClientCapabilities::default(),
            })
            .expect("initialize probe params"),
        ),
    ))
}

pub fn initialized_notification() -> JsonRpcMessage {
    JsonRpcMessage::Notification(JsonRpcNotification::new(
        app_server_protocol::METHOD_INITIALIZED,
        None,
    ))
}

pub fn probe_info_from_initialize_response(
    message: &JsonRpcMessage,
) -> Result<Option<ProbeInfo>, String> {
    let JsonRpcMessage::Response(JsonRpcResponse { id, result }) = message else {
        return Ok(None);
    };
    if id != &INITIALIZE_REQUEST_ID {
        return Ok(None);
    }
    let response: InitializeResponse = serde_json::from_value(result.clone())
        .map_err(|error| format!("failed to parse app-server initialize response: {error}"))?;
    Ok(Some(ProbeInfo {
        app_server_version: response.server_info.version,
        protocol_version: response.server_info.protocol_version,
    }))
}

pub fn parse_version_from_user_agent(user_agent: &str) -> Result<String, String> {
    let (_originator, rest) = user_agent
        .split_once('/')
        .ok_or_else(|| "app-server user-agent omitted version separator".to_string())?;
    let version = rest
        .split_whitespace()
        .next()
        .filter(|version| !version.is_empty())
        .ok_or_else(|| "app-server user-agent omitted version".to_string())?;
    Ok(version.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::PlatformInfo;
    use app_server_protocol::ServerCapabilities;
    use app_server_protocol::ServerInfo;
    use serde_json::json;

    #[test]
    fn initialize_probe_request_uses_lime_current_protocol() {
        let request = initialize_probe_request("1.2.3");
        let value = serde_json::to_value(request).expect("json");

        assert_eq!(value["id"], 1);
        assert_eq!(value["method"], app_server_protocol::METHOD_INITIALIZE);
        assert_eq!(value["params"]["clientInfo"]["name"], DAEMON_CLIENT_NAME);
        assert_eq!(value["params"]["clientInfo"]["version"], "1.2.3");
    }

    #[test]
    fn probe_info_reads_initialize_response() {
        let response = JsonRpcMessage::Response(
            JsonRpcResponse::new(
                INITIALIZE_REQUEST_ID,
                InitializeResponse {
                    server_info: ServerInfo {
                        name: "app-server".to_string(),
                        version: "1.2.3".to_string(),
                        protocol_version: "appserver.v0".to_string(),
                    },
                    platform: PlatformInfo {
                        family: "unix".to_string(),
                        os: "macos".to_string(),
                    },
                    capabilities: ServerCapabilities {
                        agent_session: true,
                        capability_discovery: true,
                        artifact: true,
                        workspace: true,
                    },
                },
            )
            .expect("response"),
        );

        assert_eq!(
            probe_info_from_initialize_response(&response).expect("probe"),
            Some(ProbeInfo {
                app_server_version: "1.2.3".to_string(),
                protocol_version: "appserver.v0".to_string(),
            })
        );
        assert_eq!(
            probe_info_from_initialize_response(&JsonRpcMessage::Notification(
                JsonRpcNotification::new("initialized", None)
            ))
            .expect("ignore"),
            None
        );
    }

    #[test]
    fn parses_codex_style_user_agent_version_for_compat_probe() {
        assert_eq!(
            parse_version_from_user_agent("app_server_daemon/1.2.3 (Darwin) app-server/1.2.3")
                .expect("version"),
            "1.2.3"
        );
        assert!(parse_version_from_user_agent("app_server_daemon").is_err());
        assert!(parse_version_from_user_agent("app_server_daemon/   ").is_err());
    }

    #[test]
    fn rejects_wrong_initialize_response_id() {
        let response = JsonRpcMessage::Response(
            JsonRpcResponse::new(RequestId::Integer(9), json!({ "serverInfo": {} }))
                .expect("response"),
        );

        assert_eq!(
            probe_info_from_initialize_response(&response).expect("ignore"),
            None
        );
    }
}
