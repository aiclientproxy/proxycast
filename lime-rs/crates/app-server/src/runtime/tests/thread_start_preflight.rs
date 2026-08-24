use super::support::TestSessionDataSource;
use super::*;
use app_server_protocol::protocol::v2::ThreadSettings;
use async_trait::async_trait;
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

struct ThreadStartPreflightBackend {
    reason_code: &'static str,
    succeed_after_first_call: bool,
    calls: AtomicUsize,
}

impl ThreadStartPreflightBackend {
    fn new(reason_code: &'static str, succeed_after_first_call: bool) -> Self {
        Self {
            reason_code,
            succeed_after_first_call,
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl ExecutionBackend for ThreadStartPreflightBackend {
    fn requires_provider_selection(&self) -> bool {
        true
    }

    async fn preflight_thread_settings(
        &self,
        session: &AgentSession,
        settings: &ThreadSettings,
    ) -> Result<(), RuntimeCoreError> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if self.succeed_after_first_call && call > 0 {
            return Ok(());
        }
        Err(RuntimeCoreError::PendingRoute {
            session_id: session.session_id.clone(),
            provider: Some(settings.model_provider.clone()),
            model: Some(settings.model.clone()),
            reason_code: self.reason_code.to_string(),
        })
    }

    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Ok(())
    }
}

#[tokio::test]
async fn thread_start_refreshes_missing_model_catalog_and_retries_once() {
    let backend = Arc::new(ThreadStartPreflightBackend::new(
        "model_registry_metadata_missing",
        true,
    ));
    let data_source = Arc::new(TestSessionDataSource::new().with_model_fetch_response(Ok(
        ModelProviderFetchModelsResponse {
            source: "Api".to_string(),
            ..ModelProviderFetchModelsResponse::default()
        },
    )));
    let core = RuntimeCore::with_backend(backend.clone()).with_app_data_source(data_source.clone());

    core.preflight_thread_start(&thread_start_params())
        .await
        .expect("refreshed model catalog should make the route executable");

    assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        data_source
            .model_fetch_requests()
            .into_iter()
            .map(|request| request.provider_id)
            .collect::<Vec<_>>(),
        vec!["provider-a"]
    );
}

#[tokio::test]
async fn thread_start_does_not_refresh_for_other_pending_route_errors() {
    let backend = Arc::new(ThreadStartPreflightBackend::new(
        "provider_models_cache_missing_requested_model",
        false,
    ));
    let data_source = Arc::new(TestSessionDataSource::new());
    let core = RuntimeCore::with_backend(backend.clone()).with_app_data_source(data_source.clone());

    let error = core
        .preflight_thread_start(&thread_start_params())
        .await
        .expect_err("unrelated pending route must remain fail closed");

    assert!(matches!(
        error,
        RuntimeCoreError::PendingRoute { reason_code, .. }
            if reason_code == "provider_models_cache_missing_requested_model"
    ));
    assert_eq!(backend.calls.load(Ordering::SeqCst), 1);
    assert!(data_source.model_fetch_requests().is_empty());
}

#[tokio::test]
async fn thread_start_remains_fail_closed_when_route_is_missing_after_refresh() {
    let backend = Arc::new(ThreadStartPreflightBackend::new(
        "model_registry_metadata_missing",
        false,
    ));
    let data_source = Arc::new(TestSessionDataSource::new().with_model_fetch_response(Ok(
        ModelProviderFetchModelsResponse {
            source: "Api".to_string(),
            ..ModelProviderFetchModelsResponse::default()
        },
    )));
    let core = RuntimeCore::with_backend(backend.clone()).with_app_data_source(data_source.clone());

    let error = core
        .preflight_thread_start(&thread_start_params())
        .await
        .expect_err("missing route after refresh must remain rejected");

    assert!(matches!(
        error,
        RuntimeCoreError::PendingRoute { reason_code, .. }
            if reason_code == "model_registry_metadata_missing"
    ));
    assert_eq!(backend.calls.load(Ordering::SeqCst), 2);
    assert_eq!(data_source.model_fetch_requests().len(), 1);
}

fn thread_start_params() -> AgentSessionStartParams {
    AgentSessionStartParams {
        session_id: Some("session-thread-start-preflight".to_string()),
        thread_id: Some("thread-thread-start-preflight".to_string()),
        app_id: "agent-chat".to_string(),
        workspace_id: None,
        business_object_ref: Some(BusinessObjectRef {
            kind: "agent.thread".to_string(),
            id: "thread-thread-start-preflight".to_string(),
            title: None,
            uri: None,
            metadata: Some(json!({
                "providerSelector": "provider-a",
                "providerName": "provider-a",
                "modelName": "model-a"
            })),
        }),
        locale: None,
    }
}
