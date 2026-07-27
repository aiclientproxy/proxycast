use std::sync::{Arc, Mutex};

use app_server::{
    ActionRespondRequest, AppServer, CancelExecutionRequest, ExecutionBackend, ExecutionRequest,
    LocalAppDataSource, MockBackend, ProjectionStore, RuntimeCore, RuntimeCoreError,
    RuntimeEventSink,
};
use app_server_protocol::protocol::v2::METHOD_THREAD_SETTINGS_UPDATED;
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_THREAD_READ, METHOD_THREAD_START,
    METHOD_TURN_START, PROTOCOL_VERSION,
};
use lime_core::database::dao::route_state::RouteStateDao;
use lime_core::database::schema::create_tables;
use lime_core::database::DbConnection;
use lime_core::models::model_registry::{
    ModelCapabilities, ModelModality, ModelRuntimeFeature, ModelTaskFamily,
    ProviderModelCapability, ProviderModelConfig,
};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tempfile::TempDir;

struct ModelSelectionBackend(MockBackend);

#[async_trait::async_trait]
impl ExecutionBackend for ModelSelectionBackend {
    fn requires_provider_selection(&self) -> bool {
        true
    }

    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.0.start_turn(request, sink).await
    }

    async fn cancel_turn(
        &self,
        request: CancelExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.0.cancel_turn(request, sink).await
    }

    async fn respond_action(
        &self,
        request: ActionRespondRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        self.0.respond_action(request, sink).await
    }
}

struct ModelSelectionServer {
    _temp: TempDir,
    db: DbConnection,
    server: AppServer,
}

#[tokio::test]
async fn turn_start_reselects_removed_model_before_admission() {
    let app = model_selection_server().await;
    initialize_server(&app.server).await;
    let started = request(
        &app.server,
        2,
        METHOD_THREAD_START,
        json!({
            "model": "model-a",
            "modelProvider": "provider-a"
        }),
    )
    .await;
    let thread_id = started
        .pointer("/result/thread/id")
        .and_then(Value::as_str)
        .expect("thread/start thread id")
        .to_string();

    {
        let conn = app.db.lock().expect("product db lock");
        conn.execute(
            "UPDATE api_key_providers SET models = ?1 WHERE id = 'provider-a'",
            [serde_json::to_string(&Vec::<ProviderModelConfig>::new())
                .expect("serialize empty provider catalog")],
        )
        .expect("remove current provider model");
        assert_eq!(RouteStateDao::advance_generation(&conn).unwrap(), 1);
    }

    let lines = request_lines(
        &app.server,
        3,
        METHOD_TURN_START,
        json!({
            "threadId": thread_id,
            "input": [{"type": "text", "text": "continue after catalog refresh"}]
        }),
    )
    .await;
    let response = response_for(&lines, 3);
    assert!(response.pointer("/result/turn/id").is_some());
    let changed_notifications = lines
        .iter()
        .filter(|message| message.get("method") == Some(&json!(METHOD_THREAD_SETTINGS_UPDATED)))
        .collect::<Vec<_>>();
    assert_eq!(
        changed_notifications.len(),
        1,
        "selection change must emit exactly once: {lines:#?}"
    );
    let changed = changed_notifications[0];
    assert_eq!(
        changed.pointer("/params/threadSettings/modelProvider"),
        Some(&json!("provider-b"))
    );
    assert_eq!(
        changed.pointer("/params/threadSettings/model"),
        Some(&json!("model-b"))
    );
    assert_eq!(
        changed.pointer("/params/threadSettings/collaborationMode/settings/model"),
        Some(&json!("model-b"))
    );
    assert_eq!(
        changed.pointer("/params/threadSettings/effort"),
        Some(&json!("medium"))
    );

    let read = request(
        &app.server,
        4,
        METHOD_THREAD_READ,
        json!({"threadId": thread_id}),
    )
    .await;
    assert_eq!(
        read.pointer("/result/thread/extra/providerSelector"),
        Some(&json!("provider-b"))
    );
    assert_eq!(
        read.pointer("/result/thread/extra/modelName"),
        Some(&json!("model-b"))
    );
}

async fn model_selection_server() -> ModelSelectionServer {
    let temp = TempDir::new().expect("model selection fixture temp dir");
    let conn = Connection::open_in_memory().expect("open in-memory product db");
    create_tables(&conn).expect("create product schema");
    for (provider_id, model_id) in [("provider-a", "model-a"), ("provider-b", "model-b")] {
        conn.execute(
            "INSERT INTO api_key_providers (
                id, name, type, api_host, is_system, group_name, enabled, sort_order,
                models, created_at, updated_at
             ) VALUES (?1, ?1, 'openai', ?2, 0, 'cloud', 1, 0, ?3, ?4, ?4)",
            params![
                provider_id,
                format!("https://{provider_id}.invalid/v1"),
                serde_json::to_string(&vec![provider_model(model_id)])
                    .expect("serialize provider model"),
                "2026-07-27T00:00:00Z",
            ],
        )
        .expect("insert provider fixture");
        conn.execute(
            "INSERT INTO api_keys (
                id, provider_id, api_key_encrypted, alias, enabled, usage_count,
                error_count, last_used_at, created_at
             ) VALUES (?1, ?2, ?3, NULL, 1, 0, 0, NULL, ?4)",
            params![
                format!("{provider_id}-key"),
                provider_id,
                "encrypted-test-key",
                "2026-07-27T00:00:00Z",
            ],
        )
        .expect("insert provider key fixture");
    }
    let db = Arc::new(Mutex::new(conn));
    let app_data_source = LocalAppDataSource::initialize_with_roots(
        Arc::clone(&db),
        temp.path(),
        temp.path().join("app-server"),
    )
    .await
    .expect("local app data source");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("projection store"),
    );
    let runtime = RuntimeCore::with_backend(Arc::new(ModelSelectionBackend(MockBackend)))
        .with_projection_store(projection_store)
        .with_app_data_source(Arc::new(app_data_source));

    ModelSelectionServer {
        _temp: temp,
        db,
        server: AppServer::with_runtime(runtime),
    }
}

fn provider_model(id: &str) -> ProviderModelConfig {
    ProviderModelConfig {
        id: id.to_string(),
        display_name: None,
        capability: Some(ProviderModelCapability {
            task_families: vec![ModelTaskFamily::Chat],
            input_modalities: vec![ModelModality::Text],
            output_modalities: vec![ModelModality::Text],
            runtime_features: vec![ModelRuntimeFeature::Streaming],
            capabilities: ModelCapabilities {
                streaming: true,
                reasoning: true,
                reasoning_effort: Some(
                    lime_core::models::model_registry::ModelReasoningEffortSupport {
                        supported: true,
                        levels: vec!["low".to_string(), "medium".to_string(), "high".to_string()],
                        options: Vec::new(),
                        default: Some("medium".to_string()),
                        source: None,
                    },
                ),
                ..Default::default()
            },
        }),
    }
}

async fn initialize_server(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "model-selection-refresh-test",
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
    let lines = request_lines(server, id, method, params).await;
    response_for(&lines, id).clone()
}

async fn request_lines(server: &AppServer, id: u64, method: &str, params: Value) -> Vec<Value> {
    server
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
        .unwrap_or_else(|error| panic!("{method} request failed: {error}"))
        .into_iter()
        .map(|line| serde_json::from_str(&line).expect("decode JSON-RPC line"))
        .collect()
}

fn response_for(lines: &[Value], id: u64) -> &Value {
    let response = lines
        .iter()
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("missing response {id}: {lines:#?}"));
    if let Some(error) = response.get("error") {
        panic!("request {id} failed: {error}");
    }
    response
}
