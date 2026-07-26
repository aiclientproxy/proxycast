use crate::events::{McpOAuthCompletedPayload, McpServerErrorPayload};
use crate::oauth_store::PersistentCredentialStore;
use crate::streamable_http::build_default_headers;
use crate::types::{McpError, McpServerConfig, McpServerTransport};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use axum::routing::get;
use axum::Router;
use lime_core::DynEmitter;
use rmcp::transport::auth::{AuthorizationManager, CredentialStore, OAuthState, StoredCredentials};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{oneshot, Mutex};

const DEFAULT_OAUTH_TIMEOUT_SECS: u64 = 300;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct McpOAuthLoginParams {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scopes: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub struct McpOAuthLoginResponse {
    pub authorization_url: String,
    pub state: String,
}

/// OAuth 凭据根只能由组合根注入。未注入时所有凭据操作 fail closed，
/// 不回退平台默认目录——旧的平台根旁路已删除。
#[derive(Debug, Clone, Default)]
pub struct McpOAuthRegistry {
    store_root: Option<PathBuf>,
}

impl McpOAuthRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_in(root_dir: impl Into<PathBuf>) -> Self {
        Self {
            store_root: Some(root_dir.into()),
        }
    }

    pub async fn start_login(
        &self,
        server_name: &str,
        config: &McpServerConfig,
        scopes: Option<Vec<String>>,
        timeout_secs: Option<u64>,
        emitter: Option<DynEmitter>,
    ) -> Result<McpOAuthLoginResponse, McpError> {
        let (url, bearer_token_env_var, http_headers, env_http_headers) = match &config.transport {
            McpServerTransport::StreamableHttp {
                url,
                bearer_token_env_var,
                http_headers,
                env_http_headers,
            } => (url, bearer_token_env_var, http_headers, env_http_headers),
            McpServerTransport::Stdio { .. } => {
                return Err(McpError::ConfigError(
                    "MCP OAuth login only supports streamable_http transport".to_string(),
                ));
            }
        };

        if bearer_token_env_var.is_some() {
            return Err(McpError::ConfigError(
                "MCP OAuth login cannot be used with bearer_token_env_var".to_string(),
            ));
        }

        if config
            .oauth
            .as_ref()
            .and_then(|oauth| oauth.client_id.as_deref())
            .is_some_and(|client_id| !client_id.trim().is_empty())
            || config
                .oauth_resource
                .as_deref()
                .is_some_and(|resource| !resource.trim().is_empty())
        {
            return Err(McpError::ConfigError(
                "MCP OAuth login with explicit client_id or oauth_resource is not yet supported by the runtime connector"
                    .to_string(),
            ));
        }

        ensure_loopback_no_proxy_env(url.as_str());
        let client = oauth_http_client(
            Some(url.as_str()),
            None,
            http_headers.as_ref(),
            env_http_headers.as_ref(),
        )?;
        let discovered_scopes = if scopes.is_none() && config.scopes.is_none() {
            discover_supported_scopes(url.as_str(), client.clone()).await?
        } else {
            None
        };
        let store = self.store_for(server_name, url.as_str())?;
        let mut auth_state = OAuthState::new(url.as_str(), Some(client))
            .await
            .map_err(oauth_error)?;
        let (callback_tx, callback_rx) = oneshot::channel::<OAuthCallbackResult>();
        let redirect_uri = spawn_callback_server(callback_tx, server_name).await?;
        let resolved_scopes = resolve_login_scopes(scopes, config, discovered_scopes);
        let scope_refs = resolved_scopes
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        auth_state
            .start_authorization(&scope_refs, redirect_uri.as_str(), Some("Lime MCP Client"))
            .await
            .map_err(oauth_error)?;
        let authorization_url = auth_state
            .get_authorization_url()
            .await
            .map_err(oauth_error)?;

        let server_name = server_name.to_string();
        let wait_secs = timeout_secs
            .unwrap_or(DEFAULT_OAUTH_TIMEOUT_SECS)
            .clamp(30, 900);
        let server_name_for_error = server_name.clone();
        let emitter_for_error = emitter.clone();
        tokio::spawn(async move {
            if let Err(error) = complete_oauth_login(
                auth_state,
                store,
                callback_rx,
                wait_secs,
                server_name.clone(),
                emitter,
            )
            .await
            {
                tracing::warn!(
                    server_name = %server_name,
                    error = %error,
                    "MCP OAuth login did not complete"
                );
                emit_oauth_error(
                    emitter_for_error,
                    &server_name_for_error,
                    &error.to_string(),
                );
            }
        });

        Ok(McpOAuthLoginResponse {
            authorization_url,
            state: "pending".to_string(),
        })
    }

    pub async fn authorized_manager_for(
        &self,
        server_name: &str,
        config: &McpServerConfig,
    ) -> Result<Option<AuthorizationManager>, McpError> {
        if !config.has_oauth_settings() {
            return Ok(None);
        }

        let url = match &config.transport {
            McpServerTransport::StreamableHttp { url, .. } => url,
            McpServerTransport::Stdio { .. } => return Ok(None),
        };

        let McpServerTransport::StreamableHttp {
            http_headers,
            env_http_headers,
            ..
        } = &config.transport
        else {
            return Ok(None);
        };

        let store = self.store_for(server_name, url.as_str())?;
        let client = oauth_http_client(
            Some(url.as_str()),
            None,
            http_headers.as_ref(),
            env_http_headers.as_ref(),
        )?;
        let mut manager = AuthorizationManager::new(url.as_str())
            .await
            .map_err(oauth_error)?;
        manager.with_client(client).map_err(oauth_error)?;
        manager.set_credential_store(store);
        if !manager.initialize_from_store().await.map_err(oauth_error)? {
            return Err(McpError::ConfigError(
                "MCP OAuth credentials are missing; run mcpServer/oauth/login first".to_string(),
            ));
        }
        Ok(Some(manager))
    }

    pub async fn has_credentials(
        &self,
        server_name: &str,
        config: &McpServerConfig,
    ) -> Result<bool, McpError> {
        if !config.has_oauth_settings() || config.has_unsupported_oauth_runtime_settings() {
            return Ok(false);
        }
        let url = match &config.transport {
            McpServerTransport::StreamableHttp { url, .. } => url,
            McpServerTransport::Stdio { .. } => return Ok(false),
        };
        self.store_for(server_name, url.as_str())?
            .has_credentials()
            .await
            .map_err(oauth_error)
    }

    fn store_for(
        &self,
        server_name: &str,
        server_url: &str,
    ) -> Result<PersistentCredentialStore, McpError> {
        let root_dir = self.store_root.as_ref().ok_or_else(|| {
            McpError::ConfigError(
                "MCP OAuth credential root is not injected; refusing to guess a platform directory"
                    .to_string(),
            )
        })?;
        Ok(PersistentCredentialStore::new_in(
            root_dir,
            server_name,
            server_url,
        ))
    }
}

#[derive(Clone)]
struct OAuthCallbackState {
    server_name: String,
    callback_tx: Arc<Mutex<Option<oneshot::Sender<OAuthCallbackResult>>>>,
}

#[derive(Debug, Deserialize)]
struct OAuthCallbackParams {
    code: Option<String>,
    state: Option<String>,
    error: Option<String>,
    error_description: Option<String>,
}

#[derive(Debug)]
enum OAuthCallbackResult {
    Success {
        code: String,
        state: String,
    },
    ProviderError {
        error: Option<String>,
        error_description: Option<String>,
    },
}

impl OAuthCallbackResult {
    fn from_params(params: OAuthCallbackParams) -> Self {
        match (params.code, params.state) {
            (Some(code), Some(state)) => Self::Success { code, state },
            _ => Self::ProviderError {
                error: params.error,
                error_description: params.error_description,
            },
        }
    }

    fn error_message(error: Option<&str>, error_description: Option<&str>) -> String {
        match (error, error_description) {
            (Some(error), Some(error_description)) => {
                format!("OAuth provider returned `{error}`: {error_description}")
            }
            (Some(error), None) => format!("OAuth provider returned `{error}`"),
            (None, Some(error_description)) => format!("OAuth error: {error_description}"),
            (None, None) => "OAuth callback did not include authorization code".to_string(),
        }
    }
}

async fn spawn_callback_server(
    callback_tx: oneshot::Sender<OAuthCallbackResult>,
    server_name: &str,
) -> Result<String, McpError> {
    let state = OAuthCallbackState {
        server_name: server_name.to_string(),
        callback_tx: Arc::new(Mutex::new(Some(callback_tx))),
    };
    let app = Router::new()
        .route("/oauth/callback", get(handle_oauth_callback))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .map_err(|error| {
            McpError::ConfigError(format!("MCP OAuth callback bind failed: {error}"))
        })?;
    let local_addr = listener.local_addr().map_err(|error| {
        McpError::ConfigError(format!("MCP OAuth callback local address failed: {error}"))
    })?;
    tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, app).await {
            tracing::warn!(error = %error, "MCP OAuth callback server stopped with error");
        }
    });
    Ok(format!(
        "http://127.0.0.1:{}/oauth/callback",
        local_addr.port()
    ))
}

async fn handle_oauth_callback(
    Query(params): Query<OAuthCallbackParams>,
    State(state): State<OAuthCallbackState>,
) -> impl IntoResponse {
    let result = OAuthCallbackResult::from_params(params);
    let success = matches!(result, OAuthCallbackResult::Success { .. });
    if let Some(tx) = state.callback_tx.lock().await.take() {
        let _ = tx.send(result);
    }
    if success {
        (
            StatusCode::OK,
            Html(format!(
                "<!doctype html><meta charset=\"utf-8\"><title>MCP OAuth</title><p>MCP OAuth authorization for {} has completed. You can close this window.</p>",
                escape_html(&state.server_name)
            )),
        )
            .into_response()
    } else {
        (
            StatusCode::BAD_REQUEST,
            Html(format!(
                "<!doctype html><meta charset=\"utf-8\"><title>MCP OAuth</title><p>MCP OAuth authorization for {} failed. You can close this window.</p>",
                escape_html(&state.server_name)
            )),
        )
            .into_response()
    }
}

async fn complete_oauth_login(
    mut auth_state: OAuthState,
    store: PersistentCredentialStore,
    callback_rx: oneshot::Receiver<OAuthCallbackResult>,
    timeout_secs: u64,
    server_name: String,
    emitter: Option<DynEmitter>,
) -> Result<(), McpError> {
    let login_timeout = Duration::from_secs(timeout_secs);
    let started_at = Instant::now();
    let params = tokio::time::timeout(login_timeout, callback_rx)
        .await
        .map_err(|_| McpError::Timeout)?
        .map_err(|error| {
            McpError::ProtocolError(format!("MCP OAuth callback canceled: {error}"))
        })?;
    let (code, state) = match params {
        OAuthCallbackResult::Success { code, state } => (code, state),
        OAuthCallbackResult::ProviderError {
            error,
            error_description,
        } => {
            return Err(McpError::ConfigError(OAuthCallbackResult::error_message(
                error.as_deref(),
                error_description.as_deref(),
            )));
        }
    };
    let remaining = login_timeout
        .checked_sub(started_at.elapsed())
        .ok_or(McpError::Timeout)?;
    tokio::time::timeout(remaining, async {
        tracing::debug!(server_name = %server_name, "MCP OAuth callback received; exchanging token");
        auth_state
            .handle_callback(&code, &state)
            .await
            .map_err(oauth_error)?;
        let (client_id, token_response) =
            auth_state.get_credentials().await.map_err(oauth_error)?;
        store
            .save(StoredCredentials {
                client_id,
                token_response,
            })
            .await
            .map_err(oauth_error)?;
        let mut manager = auth_state.into_authorization_manager().ok_or_else(|| {
            McpError::ProtocolError(
                "MCP OAuth flow completed without authorization manager".to_string(),
            )
        })?;
        manager.set_credential_store(store);
        Ok::<(), McpError>(())
    })
    .await
    .map_err(|_| McpError::Timeout)??;
    emit_oauth_completed(emitter, &server_name);
    Ok(())
}

fn emit_oauth_completed(emitter: Option<DynEmitter>, server_name: &str) {
    let Some(emitter) = emitter else {
        return;
    };
    let payload = McpOAuthCompletedPayload {
        server_name: server_name.to_string(),
    };
    let Ok(value) = serde_json::to_value(&payload) else {
        return;
    };
    if let Err(error) = emitter.emit_event("mcp:oauth_completed", &value) {
        tracing::warn!(
            server_name = %server_name,
            error = %error,
            "MCP OAuth completion event failed"
        );
    }
}

fn emit_oauth_error(emitter: Option<DynEmitter>, server_name: &str, error: &str) {
    let Some(emitter) = emitter else {
        return;
    };
    let payload = McpServerErrorPayload {
        server_name: server_name.to_string(),
        error: error.to_string(),
    };
    let Ok(value) = serde_json::to_value(&payload) else {
        return;
    };
    if let Err(error) = emitter.emit_event("mcp:server_error", &value) {
        tracing::warn!(
            server_name = %server_name,
            error = %error,
            "MCP OAuth error event failed"
        );
    }
}

async fn discover_supported_scopes(
    url: &str,
    client: reqwest::Client,
) -> Result<Option<Vec<String>>, McpError> {
    let mut manager = AuthorizationManager::new(url).await.map_err(oauth_error)?;
    manager.with_client(client).map_err(oauth_error)?;
    match manager.discover_metadata().await {
        Ok(metadata) => Ok(normalize_scopes(metadata.scopes_supported)),
        Err(error) => {
            tracing::debug!(
                url = %url,
                error = %error,
                "MCP OAuth scope discovery failed"
            );
            Ok(None)
        }
    }
}

fn resolve_login_scopes(
    requested_scopes: Option<Vec<String>>,
    config: &McpServerConfig,
    discovered_scopes: Option<Vec<String>>,
) -> Vec<String> {
    requested_scopes
        .or_else(|| config.scopes.clone())
        .or(discovered_scopes)
        .unwrap_or_default()
        .into_iter()
        .map(|scope| scope.trim().to_string())
        .filter(|scope| !scope.is_empty())
        .collect()
}

fn normalize_scopes(scopes: Option<Vec<String>>) -> Option<Vec<String>> {
    let mut normalized = Vec::new();
    for scope in scopes? {
        let scope = scope.trim();
        if scope.is_empty() {
            continue;
        }
        let scope = scope.to_string();
        if !normalized.contains(&scope) {
            normalized.push(scope);
        }
    }
    (!normalized.is_empty()).then_some(normalized)
}

pub(crate) fn oauth_http_client(
    base_url: Option<&str>,
    bearer_token_env_var: Option<&str>,
    http_headers: Option<&std::collections::HashMap<String, String>>,
    env_http_headers: Option<&std::collections::HashMap<String, String>>,
) -> Result<reqwest::Client, McpError> {
    let headers = build_default_headers(
        bearer_token_env_var,
        http_headers,
        env_http_headers,
        |name| std::env::var(name),
    )?;
    let mut builder = reqwest::Client::builder();
    if base_url.is_some_and(is_loopback_http_url) {
        builder = builder.no_proxy();
    }
    if !headers.is_empty() {
        builder = builder.default_headers(headers);
    }
    builder
        .build()
        .map_err(|error| McpError::ConfigError(format!("MCP OAuth HTTP client 构造失败: {error}")))
}

pub(crate) fn is_loopback_http_url(value: &str) -> bool {
    reqwest::Url::parse(value)
        .ok()
        .and_then(|url| {
            let is_http = matches!(url.scheme(), "http" | "https");
            let is_loopback = matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "::1"));
            (is_http && is_loopback).then_some(())
        })
        .is_some()
}

fn ensure_loopback_no_proxy_env(base_url: &str) {
    if !is_loopback_http_url(base_url) {
        return;
    }

    for key in ["NO_PROXY", "no_proxy"] {
        let current = std::env::var(key).unwrap_or_default();
        if let Some(next) = merge_loopback_no_proxy_hosts(&current) {
            std::env::set_var(key, next);
        }
    }
}

fn merge_loopback_no_proxy_hosts(current: &str) -> Option<String> {
    let mut parts = current
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let original_len = parts.len();
    for host in ["127.0.0.1", "localhost", "::1"] {
        if !parts.iter().any(|part| part == host) {
            parts.push(host.to_string());
        }
    }
    (parts.len() != original_len).then(|| parts.join(","))
}

pub(crate) fn oauth_error(error: impl std::fmt::Display) -> McpError {
    McpError::ConfigError(format!("MCP OAuth error: {error}"))
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
#[path = "oauth_tests.rs"]
mod oauth_tests;
