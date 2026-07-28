use std::sync::{Arc, Mutex};

use app_server::{AppServer, LocalAppDataSource, MockBackend, ProjectionStore, RuntimeCore};
use app_server_protocol::{
    METHOD_INITIALIZE, METHOD_INITIALIZED, METHOD_MODEL_LIST, METHOD_THREAD_START, PROTOCOL_VERSION,
};
use lime_core::database::schema::create_tables;
use lime_core::models::model_registry::{
    ModelCapabilities, ModelModality, ModelRuntimeFeature, ModelTaskFamily,
    ProviderModelCapability, ProviderModelConfig,
};
use rusqlite::{params, Connection};
use serde_json::{json, Value};
use tempfile::TempDir;

struct ModelListAppServer {
    _temp: TempDir,
    server: AppServer,
}

async fn model_list_app_server() -> ModelListAppServer {
    let temp = TempDir::new().expect("create model list fixture temp dir");
    let conn = Connection::open_in_memory().expect("open in-memory product db");
    create_tables(&conn).expect("create product schema");
    for (provider_id, enabled, models) in [
        (
            "enabled-provider",
            true,
            vec![
                chat_model("enabled-model"),
                ProviderModelConfig::hint("inferred-hint-model"),
                image_model("image-generation-model"),
                chat_model("enabled-model-2"),
            ],
        ),
        (
            "missing-key-provider",
            true,
            vec![chat_model("missing-key-model")],
        ),
        (
            "disabled-provider",
            false,
            vec![chat_model("disabled-model")],
        ),
    ] {
        conn.execute(
            "INSERT INTO api_key_providers (
                id, name, type, api_host, is_system, group_name, enabled, sort_order,
                models, created_at, updated_at
             ) VALUES (?1, ?1, 'openai', ?2, 0, 'cloud', ?3, 0, ?4, ?5, ?5)",
            params![
                provider_id,
                format!("https://{provider_id}.invalid/v1"),
                enabled,
                serde_json::to_string(&models).expect("serialize declared models"),
                "2026-07-25T00:00:00Z",
            ],
        )
        .expect("insert provider fixture");
    }
    conn.execute(
        "INSERT INTO api_keys (
            id, provider_id, api_key_encrypted, alias, enabled, usage_count,
            error_count, last_used_at, created_at
         ) VALUES (?1, ?2, ?3, NULL, 1, 0, 0, NULL, ?4)",
        params![
            "enabled-provider-key",
            "enabled-provider",
            "encrypted-test-key",
            "2026-07-25T00:00:00Z",
        ],
    )
    .expect("insert enabled provider key fixture");
    conn.execute(
        "INSERT INTO model_registry (
            id, display_name, provider_id, provider_name, created_at, updated_at
         ) VALUES (?1, ?2, ?3, ?4, 1, 1)",
        params![
            "stale-registry-model",
            "Stale Registry Model",
            "enabled-provider",
            "Enabled Provider",
        ],
    )
    .expect("insert retired registry fixture");
    let app_data_source = LocalAppDataSource::initialize_with_roots(
        Arc::new(Mutex::new(conn)),
        temp.path(),
        temp.path().join("app-server"),
    )
    .await
    .expect("local app data source");
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("model list projection store"),
    );
    let runtime = RuntimeCore::with_backend(Arc::new(MockBackend))
        .with_app_data_source(Arc::new(app_data_source))
        .with_projection_store(projection_store);

    ModelListAppServer {
        _temp: temp,
        server: AppServer::with_runtime(runtime),
    }
}

fn chat_model(id: &str) -> ProviderModelConfig {
    ProviderModelConfig {
        id: id.to_string(),
        display_name: None,
        capability: Some(ProviderModelCapability {
            task_families: vec![ModelTaskFamily::Chat, ModelTaskFamily::VisionUnderstanding],
            input_modalities: vec![
                ModelModality::Text,
                ModelModality::Image,
                ModelModality::Audio,
                ModelModality::Video,
                ModelModality::File,
            ],
            output_modalities: vec![ModelModality::Text, ModelModality::Json],
            runtime_features: vec![
                ModelRuntimeFeature::Streaming,
                ModelRuntimeFeature::ToolCalling,
                ModelRuntimeFeature::JsonSchema,
            ],
            capabilities: ModelCapabilities {
                vision: true,
                tools: true,
                streaming: true,
                json_mode: true,
                function_calling: true,
                ..ModelCapabilities::default()
            },
        }),
    }
}

fn image_model(id: &str) -> ProviderModelConfig {
    ProviderModelConfig {
        id: id.to_string(),
        display_name: None,
        capability: Some(ProviderModelCapability {
            task_families: vec![ModelTaskFamily::ImageGeneration],
            input_modalities: vec![ModelModality::Text],
            output_modalities: vec![ModelModality::Image],
            runtime_features: vec![],
            capabilities: ModelCapabilities::default(),
        }),
    }
}

#[tokio::test]
async fn model_list_uses_exact_v2_shape_and_runtime_ready_catalog() {
    let app = model_list_app_server().await;
    initialize_server(&app.server).await;

    let response = request(&app.server, 2, METHOD_MODEL_LIST, json!({ "limit": 1 })).await;
    let models = response
        .pointer("/result/data")
        .and_then(Value::as_array)
        .expect("model/list models");
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].get("model"), Some(&json!("enabled-model")));
    assert_eq!(
        models[0].get("providerId"),
        Some(&json!("enabled-provider"))
    );
    assert!(models[0]
        .get("id")
        .and_then(Value::as_str)
        .is_some_and(|id| id.starts_with("route:")));
    assert_eq!(models[0].get("hidden"), Some(&json!(false)));
    assert_eq!(
        models[0].get("defaultReasoningEffort"),
        Some(&json!("none"))
    );
    assert_eq!(
        models[0].get("inputModalities"),
        Some(&json!(["text", "image"]))
    );
    assert_eq!(
        models[0].pointer("/capabilitySnapshot/source"),
        Some(&json!("provider_explicit"))
    );
    assert_eq!(
        models[0].pointer("/capabilitySnapshot/taskFamilies"),
        Some(&json!(["chat", "vision_understanding"]))
    );
    assert_eq!(
        models[0].pointer("/capabilitySnapshot/outputModalities"),
        Some(&json!(["text", "json"]))
    );
    assert_eq!(
        models[0].pointer("/capabilitySnapshot/inputModalities"),
        Some(&json!(["text", "image"]))
    );
    assert_eq!(
        models[0].pointer("/capabilitySnapshot/runtimeFeatures"),
        Some(&json!(["streaming", "tool_calling", "json_schema"]))
    );
    assert_eq!(
        models[0].pointer("/capabilitySnapshot/capabilities/tools"),
        Some(&json!(true))
    );
    assert_eq!(models[0].get("contextWindow"), Some(&Value::Null));
    assert_eq!(models[0].get("maxOutputTokens"), Some(&Value::Null));
    assert_eq!(response.pointer("/result/nextCursor"), Some(&json!("1")));

    let second_page = request(
        &app.server,
        3,
        METHOD_MODEL_LIST,
        json!({ "cursor": "1", "limit": 1, "includeHidden": false }),
    )
    .await;
    assert_eq!(
        second_page.pointer("/result/data/0/model"),
        Some(&json!("enabled-model-2"))
    );
    assert_eq!(
        second_page.pointer("/result/nextCursor"),
        Some(&Value::Null)
    );

    let full_list = request(
        &app.server,
        4,
        METHOD_MODEL_LIST,
        json!({ "includeHidden": true }),
    )
    .await;
    let all_models = full_list
        .pointer("/result/data")
        .and_then(Value::as_array)
        .expect("full model/list data");
    assert_eq!(
        all_models
            .iter()
            .filter_map(|model| model.get("model").and_then(Value::as_str))
            .collect::<Vec<_>>(),
        vec!["enabled-model", "enabled-model-2"]
    );
    assert!(!all_models.iter().any(|model| {
        matches!(
            model.get("model").and_then(Value::as_str),
            Some(
                "disabled-model"
                    | "missing-key-model"
                    | "stale-registry-model"
                    | "inferred-hint-model"
                    | "image-generation-model"
            )
        )
    }));
}

#[tokio::test]
async fn model_list_rejects_invalid_cursor() {
    let app = model_list_app_server().await;
    initialize_server(&app.server).await;

    let response = request_raw(
        &app.server,
        2,
        METHOD_MODEL_LIST,
        json!({ "cursor": "invalid" }),
    )
    .await;
    assert_eq!(response.pointer("/error/code"), Some(&json!(-32600)));
    assert_eq!(
        response.pointer("/error/message"),
        Some(&json!("invalid cursor: invalid"))
    );
}

#[tokio::test]
async fn thread_start_without_explicit_route_uses_model_list_default() {
    let app = model_list_app_server().await;
    initialize_server(&app.server).await;

    let listed = request(
        &app.server,
        2,
        METHOD_MODEL_LIST,
        json!({ "includeHidden": false }),
    )
    .await;
    let default = listed
        .pointer("/result/data")
        .and_then(Value::as_array)
        .and_then(|models| models.iter().find(|model| model["isDefault"] == true))
        .expect("model/list default");
    let started = request(&app.server, 3, METHOD_THREAD_START, json!({})).await;

    assert_eq!(started.pointer("/result/model"), default.get("model"));
    assert_eq!(
        started.pointer("/result/modelProvider"),
        default.get("providerId")
    );
}

async fn initialize_server(server: &AppServer) {
    let response = request(
        server,
        1,
        METHOD_INITIALIZE,
        json!({
            "clientInfo": {
                "name": "model-list-jsonrpc-test",
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
    let response = request_raw(server, id, method, params).await;
    if let Some(error) = response.get("error") {
        panic!("{method} failed: {error}");
    }
    response
}

async fn request_raw(server: &AppServer, id: u64, method: &str, params: Value) -> Value {
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
    let response = lines
        .iter()
        .map(|line| serde_json::from_str::<Value>(line).expect("decode JSON-RPC message"))
        .find(|message| message.get("id") == Some(&json!(id)))
        .unwrap_or_else(|| panic!("{method} should return the matching response: {lines:#?}"));
    assert_eq!(response.get("id"), Some(&json!(id)));
    response
}
