use super::{model_registry_metadata, model_route_contract, model_route_credential, model_routing};
use crate::model_route_assembly::{self, ModelRouteSelection};
use crate::model_task_contract::{build_model_task_request, ModelTaskRequestInput};
use crate::RuntimeCoreError;
use app_server_protocol::{ModelRefSource, ModelTaskKind, ModelTaskSource};
use async_trait::async_trait;
use lime_agent::{
    KnowledgeBuilderSkillRequest, KnowledgeBuilderSkillRunner, ModelRouteProviderConfiguration,
    SkillExecutionResult,
};
use lime_core::database::{self, DbConnection};
use lime_knowledge::{KnowledgeBuilderRuntimeExecution, KnowledgeBuilderRuntimePlan};
use lime_services::api_key_provider_service::ApiKeyProviderService;
use runtime_core::RuntimeModelSelection;

#[async_trait]
pub trait KnowledgeBuilderRuntimeExecutor: Send + Sync {
    async fn execute(
        &self,
        plan: KnowledgeBuilderRuntimePlan,
    ) -> Result<KnowledgeBuilderRuntimeExecution, RuntimeCoreError>;
}

#[derive(Default)]
pub struct NativeKnowledgeBuilderRuntimeExecutor {
    runner: KnowledgeBuilderSkillRunner,
}

impl NativeKnowledgeBuilderRuntimeExecutor {
    pub fn new() -> Self {
        Self {
            runner: KnowledgeBuilderSkillRunner::new(),
        }
    }
}

#[async_trait]
impl KnowledgeBuilderRuntimeExecutor for NativeKnowledgeBuilderRuntimeExecutor {
    async fn execute(
        &self,
        plan: KnowledgeBuilderRuntimePlan,
    ) -> Result<KnowledgeBuilderRuntimeExecution, RuntimeCoreError> {
        let db = database::init_database().map_err(RuntimeCoreError::Backend)?;
        let (selection, provider_configuration) =
            resolve_builder_provider_configuration(&db, &plan).await?;
        Ok(execute_native_knowledge_builder_skill(
            &self.runner,
            &db,
            plan,
            provider_configuration,
            selection,
        )
        .await)
    }
}

async fn execute_native_knowledge_builder_skill(
    runner: &KnowledgeBuilderSkillRunner,
    db: &DbConnection,
    plan: KnowledgeBuilderRuntimePlan,
    provider_configuration: ModelRouteProviderConfiguration,
    selection: RuntimeModelSelection,
) -> KnowledgeBuilderRuntimeExecution {
    let result = runner
        .run(KnowledgeBuilderSkillRequest {
            db,
            skill_name: &plan.skill_name,
            execution_id: &plan.execution_id,
            session_id: &plan.session_id,
            user_input: &plan.user_input,
            request_context: &plan.request_context,
            provider_configuration,
        })
        .await;
    knowledge_builder_execution_from_skill_result(plan, selection, result)
}

async fn resolve_builder_provider_configuration(
    db: &DbConnection,
    plan: &KnowledgeBuilderRuntimePlan,
) -> Result<(RuntimeModelSelection, ModelRouteProviderConfiguration), RuntimeCoreError> {
    let selection = explicit_builder_model_selection(plan)?;
    let service = ApiKeyProviderService::new();
    let readiness = model_routing::resolve_provider_readiness(db, &service, &selection, None)
        .map_err(RuntimeCoreError::Backend)?;
    let provider = service
        .get_provider(db, &selection.provider)
        .map_err(RuntimeCoreError::Backend)?;
    let route_credential = if readiness.ready {
        model_route_credential::resolve_route_credential(
            db,
            &service,
            &selection.provider,
            provider.as_ref(),
            None,
            None,
            &[],
        )
        .await
        .map_err(RuntimeCoreError::Backend)?
    } else {
        model_route_credential::RouteCredential::unavailable()
    };
    let registry = model_registry_metadata::resolve_runtime_model_registry_metadata(
        db,
        &service,
        &selection,
        None,
        route_credential.runtime_credential(),
    )
    .await
    .map_err(RuntimeCoreError::Backend)?;
    let task_request = builder_model_task_request(plan, &selection);
    let routing_payload = builder_route_payload(&task_request, &selection, &readiness, &registry);
    let mut resolved_route = model_route_assembly::resolved_route_from_task_with_credential(
        &task_request,
        ModelRouteSelection {
            provider_id: &selection.provider,
            model_id: &selection.model,
            model_ref_source: ModelRefSource::Task,
            reasoning_effort: selection.reasoning_effort.as_deref(),
        },
        &routing_payload,
        provider.as_ref(),
        route_credential.credential_ref(),
        None,
    );
    if resolved_route.auth.kind == app_server_protocol::AuthKind::ApiKeyRef {
        resolved_route.auth.credential_ref =
            route_credential.credential_ref().map(ToString::to_string);
        if resolved_route.failure.is_none() && resolved_route.auth.credential_ref.is_none() {
            return Err(RuntimeCoreError::Backend(
                "resolved_credential_unavailable".to_string(),
            ));
        }
    }
    if let Some(failure) = resolved_route.failure.as_ref() {
        return Err(RuntimeCoreError::Backend(format!(
            "Knowledge Builder model route unavailable: {}",
            failure.reason_code
        )));
    }
    let provider_configuration = model_route_contract::provider_configuration_from_runtime(
        &selection,
        &resolved_route,
        None,
        None,
        None,
    );
    Ok((selection, provider_configuration))
}

fn builder_route_payload(
    task_request: &app_server_protocol::ModelTaskRequest,
    selection: &RuntimeModelSelection,
    readiness: &runtime_core::ProviderReadiness,
    registry: &model_registry_metadata::RuntimeModelRegistryMetadata,
) -> serde_json::Value {
    serde_json::json!({
        "backend": "knowledge_builder",
        "routingMode": "task_route",
        "routing_mode": "task_route",
        "decisionSource": selection.source,
        "decision_source": selection.source,
        "decisionReason": "explicit_task_model",
        "decision_reason": "explicit_task_model",
        "settingsSource": "knowledge_builder",
        "settings_source": "knowledge_builder",
        "serviceModelSlot": task_request.routing_slot,
        "service_model_slot": task_request.routing_slot,
        "selectedProvider": selection.provider,
        "selected_provider": selection.provider,
        "selectedModel": selection.model,
        "selected_model": selection.model,
        "provider": selection.provider,
        "model": selection.model,
        "requiredCapabilities": task_request.requirements.capabilities,
        "required_capabilities": task_request.requirements.capabilities,
        "providerReadiness": readiness.to_payload(),
        "provider_readiness": readiness.to_payload(),
        "modelRegistry": registry.payload(),
        "model_registry": registry.payload(),
    })
}

fn explicit_builder_model_selection(
    plan: &KnowledgeBuilderRuntimePlan,
) -> Result<RuntimeModelSelection, RuntimeCoreError> {
    let provider = non_empty(plan.provider_override.as_deref());
    let model = non_empty(plan.model_override.as_deref());
    let (Some(provider), Some(model)) = (provider, model) else {
        return Err(RuntimeCoreError::Backend(
            "Knowledge Builder requires providerOverride and modelOverride from the current model route selection"
                .to_string(),
        ));
    };
    Ok(RuntimeModelSelection {
        provider,
        model,
        source: "knowledge_builder_override",
        reasoning_effort: None,
    })
}

fn builder_model_task_request(
    plan: &KnowledgeBuilderRuntimePlan,
    selection: &RuntimeModelSelection,
) -> app_server_protocol::ModelTaskRequest {
    build_model_task_request(ModelTaskRequestInput {
        task_kind: ModelTaskKind::Chat,
        source: ModelTaskSource::Automation,
        provider_id: Some(selection.provider.clone()),
        model_id: Some(selection.model.clone()),
        model_ref_source: ModelRefSource::Task,
        modality_contract_key: Some("knowledge_builder".to_string()),
        routing_slot: Some("knowledge_builder".to_string()),
        task_families: vec!["chat".to_string()],
        input_modalities: vec!["text".to_string()],
        output_modalities: vec!["text".to_string()],
        runtime_features: vec!["streaming".to_string()],
        capabilities: vec!["streaming".to_string()],
        session_id: Some(plan.session_id.clone()),
        thread_id: None,
        turn_id: None,
        content_id: None,
        trace_id: Some(plan.execution_id.clone()),
    })
}

fn non_empty(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

fn knowledge_builder_execution_from_skill_result(
    plan: KnowledgeBuilderRuntimePlan,
    selection: RuntimeModelSelection,
    result: Result<SkillExecutionResult, String>,
) -> KnowledgeBuilderRuntimeExecution {
    match result {
        Ok(output) if output.success => KnowledgeBuilderRuntimeExecution {
            skill_name: plan.skill_name,
            execution_id: plan.execution_id,
            session_id: Some(plan.session_id),
            status: "succeeded".to_string(),
            provider: Some(selection.provider.clone()),
            model: Some(selection.model.clone()),
            output: output.output,
            error: None,
        },
        Ok(output) => KnowledgeBuilderRuntimeExecution {
            skill_name: plan.skill_name,
            execution_id: plan.execution_id,
            session_id: Some(plan.session_id),
            status: "failed".to_string(),
            provider: Some(selection.provider.clone()),
            model: Some(selection.model.clone()),
            output: output.output,
            error: output
                .error
                .or_else(|| Some("Builder Skill 执行失败".to_string())),
        },
        Err(error) => KnowledgeBuilderRuntimeExecution {
            skill_name: plan.skill_name,
            execution_id: plan.execution_id,
            session_id: Some(plan.session_id),
            status: "failed".to_string(),
            provider: Some(selection.provider),
            model: Some(selection.model),
            output: None,
            error: Some(error),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use app_server_protocol::ProtocolKind;
    use lime_core::database::dao::api_key_provider::ApiProviderType;
    use lime_core::database::schema::create_tables;
    use lime_core::models::runtime_api_key_credential_uuid;
    use rusqlite::Connection;
    use serde_json::json;
    use std::sync::{Arc, Mutex};

    fn test_db() -> DbConnection {
        let connection = Connection::open_in_memory().expect("open in-memory database");
        create_tables(&connection).expect("create schema");
        Arc::new(Mutex::new(connection))
    }

    fn plan(provider: Option<&str>, model: Option<&str>) -> KnowledgeBuilderRuntimePlan {
        KnowledgeBuilderRuntimePlan {
            skill_name: "personal-ip-knowledge-builder".to_string(),
            execution_id: "builder-execution".to_string(),
            session_id: "builder-session".to_string(),
            user_input: "build".to_string(),
            request_context: json!({}),
            provider_override: provider.map(str::to_string),
            model_override: model.map(str::to_string),
        }
    }

    #[test]
    fn builder_route_requires_explicit_provider_and_model() {
        for plan in [
            plan(None, None),
            plan(Some("openai"), None),
            plan(None, Some("gpt-5")),
        ] {
            let error = explicit_builder_model_selection(&plan)
                .expect_err("partial model override must fail closed");
            assert!(error
                .to_string()
                .contains("providerOverride and modelOverride"));
        }
    }

    #[test]
    fn builder_route_uses_automation_chat_task_contract() {
        let plan = plan(Some(" openai "), Some(" gpt-5 "));
        let selection = explicit_builder_model_selection(&plan).expect("selection");
        let task = builder_model_task_request(&plan, &selection);

        assert_eq!(selection.provider, "openai");
        assert_eq!(selection.model, "gpt-5");
        assert_eq!(task.task_kind, ModelTaskKind::Chat);
        assert_eq!(task.source, ModelTaskSource::Automation);
        assert_eq!(
            task.modality_contract_key.as_deref(),
            Some("knowledge_builder")
        );
        assert_eq!(task.routing_slot.as_deref(), Some("knowledge_builder"));
        assert_eq!(task.requirements.capabilities, vec!["streaming"]);
    }

    #[tokio::test]
    async fn builder_route_resolves_protocol_and_exact_credential() {
        let db = test_db();
        let service = ApiKeyProviderService::new();
        service
            .initialize_system_providers(&db)
            .expect("initialize providers");
        let provider = service
            .update_provider(
                &db,
                "openai",
                Some("Knowledge Builder Gateway".to_string()),
                Some(ApiProviderType::Openai),
                Some("https://builder.example.com/v1".to_string()),
                Some(true),
                None,
                None,
                None,
                None,
                None,
                None,
                Some(vec![
                    lime_core::models::model_registry::ProviderModelConfig::hint("gpt-4.1-mini"),
                ]),
            )
            .expect("configure provider");
        let key = service
            .add_api_key(&db, &provider.id, "sk-builder", None, false)
            .expect("add credential");
        let plan = plan(Some(&provider.id), Some("gpt-4.1-mini"));

        let (selection, configuration) = resolve_builder_provider_configuration(&db, &plan)
            .await
            .expect("resolve route");

        assert_eq!(selection.provider, provider.id);
        assert_eq!(selection.model, "gpt-4.1-mini");
        assert_eq!(configuration.route_protocol, Some(ProtocolKind::OpenaiChat));
        assert_eq!(
            configuration
                .turn_provider
                .route
                .metadata
                .get("serviceModelSlot")
                .and_then(serde_json::Value::as_str),
            Some("knowledge_builder")
        );
        assert_eq!(
            configuration.credential_ref.as_deref(),
            Some(runtime_api_key_credential_uuid(&key.id).as_str())
        );
        assert!(configuration.direct_provider_config.is_none());
    }
}
