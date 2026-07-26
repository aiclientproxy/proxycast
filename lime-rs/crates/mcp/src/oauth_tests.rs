use super::*;
use crate::oauth_store::PersistentCredentialStore;
use crate::types::{McpServerOAuthConfig, McpServerTransport};
use axum::extract::Form;
use axum::http::{header::LOCATION, HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use lime_core::EventEmit;
use serde_json::Value;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Default)]
struct RecordingEmitter {
    events: Arc<Mutex<Vec<(String, Value)>>>,
}

impl EventEmit for RecordingEmitter {
    fn emit_event(&self, event: &str, payload: &Value) -> Result<(), String> {
        self.events
            .lock()
            .expect("events lock")
            .push((event.to_string(), payload.clone()));
        Ok(())
    }
}

fn oauth_config_with_explicit_client_id() -> McpServerConfig {
    McpServerConfig {
        transport: McpServerTransport::StreamableHttp {
            url: "https://example.com/mcp".to_string(),
            bearer_token_env_var: None,
            http_headers: None,
            env_http_headers: None,
        },
        environment_id: crate::types::DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string(),
        enabled: true,
        startup_timeout: 30,
        tool_timeout: None,
        enabled_tools: None,
        disabled_tools: Vec::new(),
        required: false,
        supports_parallel_tool_calls: false,
        scopes: Some(vec![" search.read ".to_string(), "".to_string()]),
        oauth: Some(McpServerOAuthConfig {
            client_id: Some("client".to_string()),
        }),
        oauth_resource: None,
    }
}

fn dynamic_oauth_config(url: String) -> McpServerConfig {
    McpServerConfig {
        transport: McpServerTransport::StreamableHttp {
            url,
            bearer_token_env_var: None,
            http_headers: None,
            env_http_headers: None,
        },
        environment_id: crate::types::DEFAULT_MCP_SERVER_ENVIRONMENT_ID.to_string(),
        enabled: true,
        startup_timeout: 30,
        tool_timeout: None,
        enabled_tools: None,
        disabled_tools: Vec::new(),
        required: false,
        supports_parallel_tool_calls: false,
        scopes: Some(vec!["search.read".to_string()]),
        oauth: None,
        oauth_resource: None,
    }
}

#[derive(Clone)]
struct TestOAuthProviderState {
    base_url: String,
    required_header: Option<(String, String)>,
    authorize_queries: Arc<Mutex<Vec<HashMap<String, String>>>>,
}

struct TestOAuthProvider {
    base_url: String,
    authorize_queries: Arc<Mutex<Vec<HashMap<String, String>>>>,
}

async fn spawn_test_oauth_provider() -> TestOAuthProvider {
    spawn_test_oauth_provider_with_header(None).await
}

async fn spawn_test_oauth_provider_with_header(
    required_header: Option<(String, String)>,
) -> TestOAuthProvider {
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
        .await
        .expect("bind OAuth provider");
    let local_addr = listener.local_addr().expect("OAuth provider local addr");
    let base_url = format!("http://127.0.0.1:{}", local_addr.port());
    let authorize_queries = Arc::new(Mutex::new(Vec::new()));
    let state = TestOAuthProviderState {
        base_url: base_url.clone(),
        required_header,
        authorize_queries: authorize_queries.clone(),
    };
    let app = Router::new()
        .route(
            "/.well-known/oauth-authorization-server",
            get(test_oauth_metadata),
        )
        .route(
            "/.well-known/oauth-authorization-server/mcp",
            get(test_oauth_metadata),
        )
        .route(
            "/mcp/.well-known/oauth-authorization-server",
            get(test_oauth_metadata),
        )
        .route("/register", post(test_oauth_register))
        .route("/authorize", get(test_oauth_authorize))
        .route("/token", post(test_oauth_token))
        .with_state(state);
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("serve OAuth provider");
    });
    TestOAuthProvider {
        base_url,
        authorize_queries,
    }
}

fn require_test_header(state: &TestOAuthProviderState, headers: &HeaderMap) {
    let Some((name, expected)) = &state.required_header else {
        return;
    };
    let actual = headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .expect("required OAuth provider header");
    assert_eq!(actual, expected);
}

async fn wait_for_event(
    events: Arc<Mutex<Vec<(String, Value)>>>,
    event_name: &str,
    server_name: &str,
) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let has_event = events
                .lock()
                .expect("events lock")
                .iter()
                .any(|(name, payload)| name == event_name && payload["server_name"] == server_name);
            if has_event {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("event should be emitted");
}

fn oauth_browser_client() -> reqwest::Client {
    reqwest::Client::builder()
        .no_proxy()
        .build()
        .expect("OAuth browser client")
}

async fn test_oauth_metadata(
    State(state): State<TestOAuthProviderState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    require_test_header(&state, &headers);
    Json(serde_json::json!({
        "issuer": state.base_url,
        "authorization_endpoint": format!("{}/authorize", state.base_url),
        "token_endpoint": format!("{}/token", state.base_url),
        "registration_endpoint": format!("{}/register", state.base_url),
        "response_types_supported": ["code"],
        "scopes_supported": ["search.read", " search.write ", "search.read", ""]
    }))
}

async fn test_oauth_register(
    State(state): State<TestOAuthProviderState>,
    headers: HeaderMap,
) -> Json<serde_json::Value> {
    require_test_header(&state, &headers);
    Json(serde_json::json!({
        "client_id": "dynamic-client",
        "client_name": "Lime MCP Client",
        "redirect_uris": []
    }))
}

async fn test_oauth_authorize(
    State(state): State<TestOAuthProviderState>,
    Query(params): Query<HashMap<String, String>>,
) -> impl IntoResponse {
    state
        .authorize_queries
        .lock()
        .expect("authorize queries lock")
        .push(params.clone());
    let redirect_uri = params.get("redirect_uri").expect("redirect_uri");
    let state = params.get("state").expect("state");
    let mut callback_url = reqwest::Url::parse(redirect_uri).expect("redirect uri URL");
    callback_url
        .query_pairs_mut()
        .append_pair("code", "auth-code")
        .append_pair("state", state);
    let mut headers = HeaderMap::new();
    headers.insert(
        LOCATION,
        callback_url
            .as_str()
            .parse()
            .expect("callback location header"),
    );
    (StatusCode::FOUND, headers)
}

async fn test_oauth_token(
    State(state): State<TestOAuthProviderState>,
    _headers: HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let _ = state;
    assert_eq!(
        form.get("grant_type").map(String::as_str),
        Some("authorization_code")
    );
    assert_eq!(form.get("code").map(String::as_str), Some("auth-code"));
    Json(serde_json::json!({
        "access_token": "access-token",
        "token_type": "Bearer",
        "refresh_token": "refresh-token",
        "expires_in": 3600
    }))
}

#[test]
fn resolve_login_scopes_trims_empty_values() {
    let config = oauth_config_with_explicit_client_id();
    assert_eq!(
        resolve_login_scopes(None, &config, Some(vec!["discovered".to_string()])),
        vec!["search.read"]
    );
    assert_eq!(
        resolve_login_scopes(
            Some(vec![" explicit ".to_string()]),
            &config,
            Some(vec!["discovered".to_string()])
        ),
        vec!["explicit"]
    );
}

#[test]
fn resolve_login_scopes_uses_discovered_when_unconfigured() {
    let mut config = dynamic_oauth_config("http://127.0.0.1:1/mcp".to_string());
    config.scopes = None;

    assert_eq!(
        resolve_login_scopes(
            None,
            &config,
            Some(vec![
                "search.read".to_string(),
                " search.write ".to_string(),
                "".to_string()
            ])
        ),
        vec!["search.read", "search.write"]
    );
}

#[test]
fn merge_loopback_no_proxy_hosts_preserves_existing_entries() {
    let merged =
        merge_loopback_no_proxy_hosts("example.com, localhost").expect("NO_PROXY should change");
    let parts = merged.split(',').collect::<Vec<_>>();
    assert!(parts.iter().any(|part| *part == "example.com"));
    assert_eq!(parts.iter().filter(|part| **part == "localhost").count(), 1);
    assert!(parts.iter().any(|part| *part == "127.0.0.1"));
    assert!(parts.iter().any(|part| *part == "::1"));
}

#[test]
fn merge_loopback_no_proxy_hosts_returns_none_when_complete() {
    assert_eq!(
        merge_loopback_no_proxy_hosts("127.0.0.1,localhost,::1"),
        None
    );
}

#[test]
fn emit_oauth_completed_sends_server_event() {
    let emitter = RecordingEmitter::default();
    let events = emitter.events.clone();

    emit_oauth_completed(Some(DynEmitter::new(emitter)), "remote-docs");

    let events = events.lock().expect("events lock");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "mcp:oauth_completed");
    assert_eq!(events[0].1["server_name"], "remote-docs");
}

#[test]
fn emit_oauth_error_sends_server_error_event() {
    let emitter = RecordingEmitter::default();
    let events = emitter.events.clone();

    emit_oauth_error(
        Some(DynEmitter::new(emitter)),
        "remote-docs",
        "invalid_scope",
    );

    let events = events.lock().expect("events lock");
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].0, "mcp:server_error");
    assert_eq!(events[0].1["server_name"], "remote-docs");
    assert_eq!(events[0].1["error"], "invalid_scope");
}

#[tokio::test]
async fn credential_lookup_fails_closed_without_injected_store_root() {
    let registry = McpOAuthRegistry::new();
    let config = dynamic_oauth_config("https://example.com/mcp".to_string());

    let error = registry
        .has_credentials("remote-docs", &config)
        .await
        .expect_err("credential lookup without an injected root must fail closed");

    assert!(matches!(error, McpError::ConfigError(_)));
    assert!(error
        .to_string()
        .contains("MCP OAuth credential root is not injected"));
}

#[tokio::test]
async fn start_login_completes_against_local_oauth_provider_and_persists_token() {
    let provider = spawn_test_oauth_provider().await;
    let server_url = format!("{}/mcp", provider.base_url);
    let config = dynamic_oauth_config(server_url.clone());
    let temp = tempfile::tempdir().expect("tempdir");
    let store_root = temp.path().join("oauth-store");
    let registry = McpOAuthRegistry::new_in(&store_root);
    let emitter = RecordingEmitter::default();
    let events = emitter.events.clone();

    let login = registry
        .start_login(
            "remote-docs",
            &config,
            None,
            Some(30),
            Some(DynEmitter::new(emitter)),
        )
        .await
        .expect("start OAuth login");

    let response = oauth_browser_client()
        .get(login.authorization_url)
        .send()
        .await
        .expect("visit authorization URL");
    assert!(
        response.status().is_success(),
        "callback page should be successful"
    );

    let store = PersistentCredentialStore::new_in(&store_root, "remote-docs", &server_url);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let has_credentials = store
                .has_credentials()
                .await
                .expect("credential store should be readable");
            let has_event = events
                .lock()
                .expect("events lock")
                .iter()
                .any(|(name, payload)| {
                    name == "mcp:oauth_completed" && payload["server_name"] == "remote-docs"
                });
            if has_credentials && has_event {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("OAuth completion should persist token and emit event");

    let credentials = store
        .load()
        .await
        .expect("load credentials")
        .expect("credentials should be persisted");
    assert_eq!(credentials.client_id, "dynamic-client");
    let token_response = credentials
        .token_response
        .expect("token response should be persisted");
    let token_response_json =
        serde_json::to_value(&token_response).expect("token response should serialize");
    assert_eq!(token_response_json["access_token"], "access-token");
    assert_eq!(token_response_json["refresh_token"], "refresh-token");
}

#[tokio::test]
async fn start_login_uses_discovered_scopes_when_config_has_no_scopes() {
    let provider = spawn_test_oauth_provider().await;
    let server_url = format!("{}/mcp", provider.base_url);
    let mut config = dynamic_oauth_config(server_url);
    config.scopes = None;
    let temp = tempfile::tempdir().expect("tempdir");
    let registry = McpOAuthRegistry::new_in(temp.path().join("oauth-store"));

    let login = registry
        .start_login("remote-docs", &config, None, Some(30), None)
        .await
        .expect("start OAuth login");

    let response = oauth_browser_client()
        .get(login.authorization_url)
        .send()
        .await
        .expect("visit authorization URL");
    assert!(
        response.status().is_success(),
        "callback page should be successful"
    );

    let queries = provider
        .authorize_queries
        .lock()
        .expect("authorize queries lock");
    let scope = queries
        .last()
        .and_then(|query| query.get("scope"))
        .expect("authorize URL should include discovered scopes");
    assert_eq!(scope, "search.read search.write");
}

#[tokio::test]
async fn start_login_applies_configured_headers_to_oauth_requests() {
    let provider = spawn_test_oauth_provider_with_header(Some((
        "x-mcp-trace".to_string(),
        "trace-1".to_string(),
    )))
    .await;
    let server_url = format!("{}/mcp", provider.base_url);
    let mut config = dynamic_oauth_config(server_url.clone());
    if let McpServerTransport::StreamableHttp {
        http_headers,
        env_http_headers,
        ..
    } = &mut config.transport
    {
        *http_headers = Some(HashMap::from([(
            "X-MCP-Trace".to_string(),
            "trace-1".to_string(),
        )]));
        *env_http_headers = None;
    }
    let temp = tempfile::tempdir().expect("tempdir");
    let store_root = temp.path().join("oauth-store");
    let registry = McpOAuthRegistry::new_in(&store_root);

    let login = registry
        .start_login("remote-docs", &config, None, Some(30), None)
        .await
        .expect("start OAuth login");

    let response = oauth_browser_client()
        .get(login.authorization_url)
        .send()
        .await
        .expect("visit authorization URL");
    assert!(
        response.status().is_success(),
        "callback page should be successful"
    );

    let store = PersistentCredentialStore::new_in(&store_root, "remote-docs", &server_url);
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if store
                .has_credentials()
                .await
                .expect("credential store should be readable")
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .expect("OAuth completion should persist token");
}

#[tokio::test]
async fn oauth_callback_provider_error_emits_server_error_without_timeout() {
    let provider = spawn_test_oauth_provider().await;
    let server_url = format!("{}/mcp", provider.base_url);
    let config = dynamic_oauth_config(server_url);
    let temp = tempfile::tempdir().expect("tempdir");
    let registry = McpOAuthRegistry::new_in(temp.path().join("oauth-store"));
    let emitter = RecordingEmitter::default();
    let events = emitter.events.clone();

    let login = registry
        .start_login(
            "remote-docs",
            &config,
            None,
            Some(30),
            Some(DynEmitter::new(emitter)),
        )
        .await
        .expect("start OAuth login");
    let mut callback_url =
        reqwest::Url::parse(&login.authorization_url).expect("authorization URL should parse");
    let callback = {
        let query = callback_url
            .query_pairs()
            .into_owned()
            .collect::<HashMap<_, _>>();
        let redirect_uri = query
            .get("redirect_uri")
            .expect("authorization URL should include redirect_uri");
        let state = query
            .get("state")
            .expect("authorization URL should include state");
        let mut callback = reqwest::Url::parse(redirect_uri).expect("redirect URI should parse");
        callback
            .query_pairs_mut()
            .append_pair("error", "invalid_scope")
            .append_pair("error_description", "scope rejected")
            .append_pair("state", state);
        callback
    };
    callback_url.set_query(None);

    let response = oauth_browser_client()
        .get(callback)
        .send()
        .await
        .expect("visit callback URL");
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);

    wait_for_event(events.clone(), "mcp:server_error", "remote-docs").await;
    let events = events.lock().expect("events lock");
    let error = events
        .iter()
        .find(|(name, _)| name == "mcp:server_error")
        .map(|(_, payload)| payload["error"].as_str().unwrap_or_default())
        .expect("server error should be emitted");
    assert!(error.contains("invalid_scope"));
    assert!(error.contains("scope rejected"));
}
