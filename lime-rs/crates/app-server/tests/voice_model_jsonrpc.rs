use app_server::{AppServer, LocalAppDataSource, MockBackend, RuntimeCore};
use app_server_protocol::{
    error_codes, METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_VOICE_MODEL_TEST_TRANSCRIBE_FILE,
    PROTOCOL_VERSION,
};
use lime_core::database::schema::create_tables;
use rusqlite::Connection;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

const MODEL_ID: &str = "sensevoice-small-int8-2024-07-17";

struct VoiceModelAppServer {
    _temp: TempDir,
    server: AppServer,
}

async fn voice_model_app_server() -> VoiceModelAppServer {
    let temp = TempDir::new().expect("create voice model fixture temp dir");
    let conn = Connection::open_in_memory().expect("open in-memory product db");
    create_tables(&conn).expect("create product schema");
    let app_data_source = LocalAppDataSource::initialize_with_roots(
        Arc::new(Mutex::new(conn)),
        temp.path(),
        temp.path().join("app-server"),
    )
    .await
    .expect("local app data source");
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_data_source(Arc::new(app_data_source));

    VoiceModelAppServer {
        _temp: temp,
        server: AppServer::with_runtime(runtime),
    }
}

#[tokio::test]
async fn test_transcribe_requires_install_dir_in_public_jsonrpc_contract() {
    let app = voice_model_app_server().await;
    initialize_server(&app.server).await;

    let response = request_error(
        &app.server,
        2,
        json!({
            "model_id": MODEL_ID,
            "file_path": "/tmp/interview.wav"
        }),
    )
    .await;

    assert_eq!(
        response.pointer("/error/code"),
        Some(&json!(error_codes::INVALID_PARAMS))
    );
}

#[tokio::test]
async fn test_transcribe_rejects_relative_install_dir_without_platform_fallback() {
    let app = voice_model_app_server().await;
    initialize_server(&app.server).await;

    let response = request_error(
        &app.server,
        2,
        json!({
            "model_id": MODEL_ID,
            "install_dir": "models/voice/sensevoice",
            "file_path": "/tmp/interview.wav"
        }),
    )
    .await;

    assert_eq!(
        response.pointer("/error/message").and_then(Value::as_str),
        Some("语音模型安装目录必须是绝对路径")
    );
}

async fn initialize_server(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "voice-model-storage-contract-test",
                "version": "1.0.0"
            }
        }),
    )
    .await;
    assert_eq!(
        response.pointer("/result/serverInfo/protocolVersion"),
        Some(&json!(PROTOCOL_VERSION))
    );
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
        .expect("handle JSON-RPC request");
    assert_eq!(lines.len(), 1, "{method} should return one response");
    let response: Value = serde_json::from_str(&lines[0]).expect("decode JSON-RPC response");
    assert_eq!(response.get("id"), Some(&json!(id)));
    response
}

async fn request_error(server: &AppServer, id: u64, params: Value) -> Value {
    let response = request(server, id, METHOD_VOICE_MODEL_TEST_TRANSCRIBE_FILE, params).await;
    assert!(
        response.get("error").is_some(),
        "voiceModel/testTranscribeFile should fail"
    );
    response
}
