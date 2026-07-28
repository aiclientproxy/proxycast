use std::sync::{Arc, Mutex};
use std::time::Duration;

use app_server::{AppServer, LocalAppDataSource, MockBackend, RuntimeCore};
use app_server_protocol::protocol::v2::METHOD_MODEL_LIST_UPDATED;
use app_server_protocol::{
    JsonRpcMessage, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_MODEL_LIST,
    METHOD_MODEL_PROVIDER_KEY_CREATE,
};
use lime_core::database::schema::create_tables;
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

struct CatalogRefreshServer {
    _temp: TempDir,
    server: AppServer,
}

#[tokio::test]
async fn credential_rotation_refreshes_and_keeps_last_success_on_failure() {
    let (api_host, fixture) = spawn_models_fixture().await;
    let app = catalog_refresh_server(&api_host).await;
    initialize_server(&app.server).await;
    let mut outbound = app.server.subscribe_outbound_messages();

    request(
        &app.server,
        2,
        METHOD_MODEL_PROVIDER_KEY_CREATE,
        json!({
            "providerId": "rotating-provider",
            "apiKey": "first-secret",
            "replaceExisting": false
        }),
    )
    .await;
    let captured = fixture.await.expect("models fixture task");
    assert!(captured
        .to_ascii_lowercase()
        .contains("authorization: bearer first-secret"));
    let first_generation = model_list_updated_generation(&mut outbound).await;
    assert_eq!(first_generation.1.as_deref(), Some("rotating-provider"));
    assert_catalog_model(&app.server, 3, "rotating-model").await;

    request(
        &app.server,
        4,
        METHOD_MODEL_PROVIDER_KEY_CREATE,
        json!({
            "providerId": "rotating-provider",
            "apiKey": "second-secret",
            "replaceExisting": true
        }),
    )
    .await;
    let second_generation = model_list_updated_generation(&mut outbound).await;
    assert!(second_generation.0 > first_generation.0);
    assert_eq!(second_generation.1.as_deref(), Some("rotating-provider"));
    assert_catalog_model(&app.server, 5, "rotating-model").await;
}

#[tokio::test]
async fn first_catalog_failure_recovers_in_one_background_retry() {
    let (api_host, fixture) = spawn_flaky_models_fixture().await;
    let app = catalog_refresh_server(&api_host).await;
    initialize_server(&app.server).await;
    let mut outbound = app.server.subscribe_outbound_messages();

    request(
        &app.server,
        2,
        METHOD_MODEL_PROVIDER_KEY_CREATE,
        json!({
            "providerId": "rotating-provider",
            "apiKey": "first-secret",
            "replaceExisting": false
        }),
    )
    .await;

    let first_generation = model_list_updated_generation(&mut outbound).await;
    let recovered_generation = model_list_updated_generation(&mut outbound).await;
    assert_eq!(first_generation.1.as_deref(), Some("rotating-provider"));
    assert_eq!(recovered_generation.1.as_deref(), Some("rotating-provider"));
    assert!(recovered_generation.0 > first_generation.0);
    assert_eq!(fixture.await.expect("flaky models fixture task"), 2);
    assert_catalog_model(&app.server, 3, "rotating-model").await;
}

async fn catalog_refresh_server(api_host: &str) -> CatalogRefreshServer {
    let temp = TempDir::new().expect("catalog refresh temp dir");
    let conn = Connection::open_in_memory().expect("catalog refresh product db");
    create_tables(&conn).expect("create product schema");
    conn.execute(
        "INSERT INTO api_key_providers (
            id, name, type, api_host, is_system, group_name, enabled, sort_order,
            models, created_at, updated_at
         ) VALUES (?1, ?2, 'openai', ?3, 0, 'cloud', 1, 0, '[]', ?4, ?4)",
        params![
            "rotating-provider",
            "Rotating Provider",
            api_host,
            "2026-07-28T00:00:00Z"
        ],
    )
    .expect("insert rotating provider");
    let db = Arc::new(Mutex::new(conn));
    let app_data_source =
        LocalAppDataSource::initialize_with_roots(db, temp.path(), temp.path().join("app-server"))
            .await
            .expect("local app data source");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_data_source(Arc::new(app_data_source));
    CatalogRefreshServer {
        _temp: temp,
        server: AppServer::with_runtime(runtime),
    }
}

async fn spawn_models_fixture() -> (String, tokio::task::JoinHandle<String>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind models fixture");
    let address = listener.local_addr().expect("models fixture address");
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.expect("accept models request");
        let mut request = [0_u8; 4096];
        let size = stream
            .read(&mut request)
            .await
            .expect("read models request");
        let body = json!({
            "data": [{
                "id": "rotating-model",
                "task_families": ["chat"],
                "input_modalities": ["text"],
                "output_modalities": ["text"],
                "runtime_features": ["streaming"],
                "capabilities": {"streaming": true}
            }]
        })
        .to_string();
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        stream
            .write_all(response.as_bytes())
            .await
            .expect("write models response");
        String::from_utf8_lossy(&request[..size]).into_owned()
    });
    (format!("http://{address}/v1"), server)
}

async fn spawn_flaky_models_fixture() -> (String, tokio::task::JoinHandle<usize>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind flaky models fixture");
    let address = listener.local_addr().expect("flaky models fixture address");
    let server = tokio::spawn(async move {
        for request_index in 0..2 {
            let (mut stream, _) = listener.accept().await.expect("accept models request");
            let mut request = [0_u8; 4096];
            stream
                .read(&mut request)
                .await
                .expect("read models request");
            if request_index == 0 {
                stream
                    .write_all(
                        b"HTTP/1.1 503 Service Unavailable\r\ncontent-length: 0\r\nconnection: close\r\n\r\n",
                    )
                    .await
                    .expect("write transient models failure");
                continue;
            }

            let body = json!({
                "data": [{
                    "id": "rotating-model",
                    "task_families": ["chat"],
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                    "runtime_features": ["streaming"],
                    "capabilities": {"streaming": true, "vision": true}
                }]
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream
                .write_all(response.as_bytes())
                .await
                .expect("write recovered models response");
        }
        2
    });
    (format!("http://{address}/v1"), server)
}

async fn initialize_server(server: &AppServer) {
    request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "model-catalog-refresh-test",
                "version": "1.0.0"
            }
        }),
    )
    .await;
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "method": METHOD_INITIALIZED,
                "params": {}
            })
            .to_string(),
        )
        .await
        .expect("handle initialized notification");
    assert!(lines.is_empty());
}

async fn assert_catalog_model(server: &AppServer, id: u64, expected: &str) {
    let response = request(server, id, METHOD_MODEL_LIST, json!({})).await;
    assert_eq!(
        response.pointer("/result/data/0/model"),
        Some(&json!(expected))
    );
}

async fn model_list_updated_generation(
    outbound: &mut tokio::sync::broadcast::Receiver<JsonRpcMessage>,
) -> (u64, Option<String>) {
    let message = tokio::time::timeout(Duration::from_secs(1), outbound.recv())
        .await
        .expect("model/list/updated timeout")
        .expect("model/list/updated broadcast");
    let JsonRpcMessage::Notification(notification) = message else {
        panic!("expected model/list/updated notification");
    };
    assert_eq!(notification.method, METHOD_MODEL_LIST_UPDATED);
    let params = notification.params.expect("model/list/updated params");
    (
        params
            .get("generation")
            .and_then(Value::as_u64)
            .expect("catalog generation"),
        params
            .get("providerId")
            .and_then(Value::as_str)
            .map(str::to_string),
    )
}

async fn request(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
    let lines = server
        .handle_json_line(
            &json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": method,
                "params": params
            })
            .to_string(),
        )
        .await
        .unwrap_or_else(|error| panic!("{method} request failed: {error}"));
    assert_eq!(lines.len(), 1, "{method} should return one response");
    let response: Value = serde_json::from_str(&lines[0]).expect("decode response");
    if let Some(error) = response.get("error") {
        panic!("{method} returned error: {error}");
    }
    response
}
