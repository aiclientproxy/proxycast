use app_server::{AppServerRuntimeFactory, ProjectionStore, RuntimeCore};
use chrono::Utc;
use lime_core::database::dao::api_key_provider::{
    ApiKeyEntry, ApiKeyProvider, ApiKeyProviderDao, ApiProviderType, ProviderGroup,
};
use lime_core::database::schema::create_tables;
use lime_core::models::model_registry::{
    ModelCapabilities, ModelModality, ModelRuntimeFeature, ModelTaskFamily,
    ProviderModelCapability, ProviderModelConfig,
};
use lime_services::api_key_provider_service::ApiKeyProviderService;
use rusqlite::Connection;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

pub fn runtime_core_with_chat_provider(
    temp: &TempDir,
    provider_id: &str,
    model: &str,
) -> RuntimeCore {
    let conn = Connection::open_in_memory().expect("open in-memory product db");
    create_tables(&conn).expect("create product schema");
    let db = Arc::new(Mutex::new(conn));
    let now = Utc::now();
    let provider = ApiKeyProvider {
        id: provider_id.to_string(),
        name: provider_id.to_string(),
        provider_type: ApiProviderType::Openai,
        api_host: "https://api.openai.com/v1".to_string(),
        is_system: false,
        group: ProviderGroup::Custom,
        enabled: true,
        sort_order: 1,
        api_version: None,
        project: None,
        location: None,
        region: None,
        models: vec![ProviderModelConfig {
            id: model.to_string(),
            display_name: None,
            capability: Some(ProviderModelCapability {
                task_families: vec![ModelTaskFamily::Chat],
                input_modalities: vec![ModelModality::Text],
                output_modalities: vec![ModelModality::Text],
                runtime_features: vec![
                    ModelRuntimeFeature::Streaming,
                    ModelRuntimeFeature::ToolCalling,
                ],
                capabilities: ModelCapabilities {
                    tools: true,
                    streaming: true,
                    function_calling: true,
                    ..ModelCapabilities::default()
                },
            }),
        }],
        prompt_cache_mode: None,
        created_at: now,
        updated_at: now,
    };
    let key = ApiKeyEntry {
        id: format!("{provider_id}-key"),
        provider_id: provider_id.to_string(),
        api_key_encrypted: ApiKeyProviderService::new().encrypt_api_key("test-key"),
        alias: None,
        enabled: true,
        usage_count: 0,
        error_count: 0,
        last_used_at: None,
        created_at: now,
    };
    {
        let conn = db.lock().expect("lock product db");
        ApiKeyProviderDao::insert_provider(&conn, &provider).expect("insert chat provider");
        ApiKeyProviderDao::insert_api_key(&conn, &key).expect("insert chat provider api key");
    }
    let projection_store = Arc::new(
        ProjectionStore::initialize(temp.path().join("projection.sqlite"))
            .expect("initialize projection store"),
    );

    AppServerRuntimeFactory::runtime_backend_core_with_db(db)
        .with_projection_store(projection_store)
}
