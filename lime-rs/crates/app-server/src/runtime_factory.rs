use crate::execution_process::ExecutionProcessServer;
use crate::AppServer;
use crate::ArtifactContentProvider;
use crate::CapabilitySource;
use crate::ExternalBackend;
use crate::ExternalBackendConfig;
use crate::MockBackend;
use crate::RuntimeBackend;
use crate::RuntimeCore;
use crate::UnavailableBackend;
use crate::WorkspaceArtifactContentProvider;
use lime_core::config::load_config;
use lime_core::database::DbConnection;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppServerBackendMode {
    External,
    Runtime,
    Mock,
    Unavailable,
}

impl AppServerBackendMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::External => "external",
            Self::Runtime => "runtime",
            Self::Mock => "mock",
            Self::Unavailable => "unavailable",
        }
    }

    pub fn parse(value: &str) -> Result<Self, UnsupportedBackendMode> {
        let normalized = value.trim().to_ascii_lowercase();
        match normalized.as_str() {
            "external" => Ok(Self::External),
            "runtime" => Ok(Self::Runtime),
            "mock" => Ok(Self::Mock),
            "unavailable" => Ok(Self::Unavailable),
            _ => Err(UnsupportedBackendMode {
                value: value.to_string(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("unsupported app-server backend mode: {value}")]
pub struct UnsupportedBackendMode {
    value: String,
}

impl UnsupportedBackendMode {
    pub fn value(&self) -> &str {
        &self.value
    }
}

pub struct AppServerRuntimeFactory;

impl AppServerRuntimeFactory {
    pub fn mock_runtime_core() -> RuntimeCore {
        RuntimeCore::with_backend(Arc::new(MockBackend))
    }

    pub fn external_runtime_core(config: ExternalBackendConfig) -> RuntimeCore {
        RuntimeCore::with_backend(Arc::new(ExternalBackend::new(config)))
    }

    pub fn external_runtime_core_with_capability_source(
        config: ExternalBackendConfig,
        capability_source: Arc<dyn CapabilitySource>,
    ) -> RuntimeCore {
        RuntimeCore::with_backend_and_capability_source(
            Arc::new(ExternalBackend::new(config)),
            capability_source,
        )
    }

    pub fn external_app_server(config: ExternalBackendConfig) -> AppServer {
        AppServer::with_runtime(Self::external_runtime_core(config))
    }

    pub fn external_app_server_with_capability_source(
        config: ExternalBackendConfig,
        capability_source: Arc<dyn CapabilitySource>,
    ) -> AppServer {
        AppServer::with_runtime(Self::external_runtime_core_with_capability_source(
            config,
            capability_source,
        ))
    }

    pub fn runtime_backend_core() -> RuntimeCore {
        let execution_process = ExecutionProcessServer::default();
        let runtime = RuntimeCore::with_backend_capability_source_and_artifact_content_provider(
            Arc::new(RuntimeBackend::with_execution_process_server(
                execution_process.clone(),
            )),
            Arc::new(crate::CapabilityInventorySource::default()),
            Arc::new(WorkspaceArtifactContentProvider),
        )
        .with_code_mode_factory(
            agent_runtime::code_mode::RuntimeCodeModeServiceFactory::production(),
        )
        .with_execution_process_server(execution_process);
        Self::inject_rollout_budget(runtime)
    }

    fn inject_rollout_budget(runtime: RuntimeCore) -> RuntimeCore {
        match load_config()
            .ok()
            .and_then(|config| config.agent.rollout_budget)
        {
            Some(config) => runtime
                .with_rollout_budget_config(Some(config))
                .expect("runtime factory requires valid rollout budget config"),
            None => runtime,
        }
    }

    pub fn runtime_backend_core_with_db(db: DbConnection) -> RuntimeCore {
        let execution_process = ExecutionProcessServer::default();
        let runtime = RuntimeCore::with_backend_capability_source_and_artifact_content_provider(
            Arc::new(RuntimeBackend::with_db_and_execution_process_server(
                db,
                execution_process.clone(),
            )),
            Arc::new(crate::CapabilityInventorySource::default()),
            Arc::new(WorkspaceArtifactContentProvider),
        )
        .with_code_mode_factory(
            agent_runtime::code_mode::RuntimeCodeModeServiceFactory::production(),
        )
        .with_execution_process_server(execution_process);
        Self::inject_rollout_budget(runtime)
    }

    pub fn runtime_backend_core_with_capability_source(
        capability_source: Arc<dyn CapabilitySource>,
    ) -> RuntimeCore {
        let execution_process = ExecutionProcessServer::default();
        let runtime = RuntimeCore::with_backend_capability_source_and_artifact_content_provider(
            Arc::new(RuntimeBackend::with_execution_process_server(
                execution_process.clone(),
            )),
            capability_source,
            Arc::new(WorkspaceArtifactContentProvider),
        )
        .with_code_mode_factory(
            agent_runtime::code_mode::RuntimeCodeModeServiceFactory::production(),
        )
        .with_execution_process_server(execution_process);
        Self::inject_rollout_budget(runtime)
    }

    pub fn runtime_backend_core_with_db_and_capability_source(
        db: DbConnection,
        capability_source: Arc<dyn CapabilitySource>,
    ) -> RuntimeCore {
        let execution_process = ExecutionProcessServer::default();
        let runtime = RuntimeCore::with_backend_capability_source_and_artifact_content_provider(
            Arc::new(RuntimeBackend::with_db_and_execution_process_server(
                db,
                execution_process.clone(),
            )),
            capability_source,
            Arc::new(WorkspaceArtifactContentProvider),
        )
        .with_code_mode_factory(
            agent_runtime::code_mode::RuntimeCodeModeServiceFactory::production(),
        )
        .with_execution_process_server(execution_process);
        Self::inject_rollout_budget(runtime)
    }

    pub fn runtime_app_server() -> AppServer {
        AppServer::with_runtime(Self::runtime_backend_core())
    }

    pub fn runtime_app_server_with_capability_source(
        capability_source: Arc<dyn CapabilitySource>,
    ) -> AppServer {
        AppServer::with_runtime(Self::runtime_backend_core_with_capability_source(
            capability_source,
        ))
    }

    pub fn mock_runtime_core_with_capability_source(
        capability_source: Arc<dyn CapabilitySource>,
    ) -> RuntimeCore {
        RuntimeCore::with_backend_and_capability_source(Arc::new(MockBackend), capability_source)
    }

    pub fn mock_runtime_core_with_sources(
        capability_source: Arc<dyn CapabilitySource>,
        artifact_content_provider: Arc<dyn ArtifactContentProvider>,
    ) -> RuntimeCore {
        RuntimeCore::with_backend_capability_source_and_artifact_content_provider(
            Arc::new(MockBackend),
            capability_source,
            artifact_content_provider,
        )
    }

    pub fn mock_app_server() -> AppServer {
        AppServer::with_runtime(Self::mock_runtime_core())
    }

    pub fn mock_app_server_with_capability_source(
        capability_source: Arc<dyn CapabilitySource>,
    ) -> AppServer {
        AppServer::with_runtime(Self::mock_runtime_core_with_capability_source(
            capability_source,
        ))
    }

    pub fn unavailable_runtime_core() -> RuntimeCore {
        RuntimeCore::with_backend(Arc::new(UnavailableBackend))
    }

    pub fn unavailable_runtime_core_with_capability_source(
        capability_source: Arc<dyn CapabilitySource>,
    ) -> RuntimeCore {
        RuntimeCore::with_backend_and_capability_source(
            Arc::new(UnavailableBackend),
            capability_source,
        )
    }

    pub fn unavailable_app_server() -> AppServer {
        AppServer::with_runtime(Self::unavailable_runtime_core())
    }

    pub fn unavailable_app_server_with_capability_source(
        capability_source: Arc<dyn CapabilitySource>,
    ) -> AppServer {
        AppServer::with_runtime(Self::unavailable_runtime_core_with_capability_source(
            capability_source,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CapabilityInventoryRecord;
    use crate::CapabilityInventorySource;
    use app_server_protocol::CapabilityDescriptor;
    use app_server_protocol::CapabilityListParams;

    #[test]
    fn backend_mode_is_explicit_for_standalone_binary() {
        assert_eq!(AppServerBackendMode::External.as_str(), "external");
        assert_eq!(AppServerBackendMode::Runtime.as_str(), "runtime");
        assert_eq!(AppServerBackendMode::Mock.as_str(), "mock");
        assert_eq!(AppServerBackendMode::Unavailable.as_str(), "unavailable");
        assert_eq!(
            AppServerBackendMode::parse(" external ").expect("external mode"),
            AppServerBackendMode::External
        );
        assert_eq!(
            AppServerBackendMode::parse(" runtime ").expect("runtime mode"),
            AppServerBackendMode::Runtime
        );
        assert_eq!(
            AppServerBackendMode::parse(" mock ").expect("mock mode"),
            AppServerBackendMode::Mock
        );
        assert_eq!(
            AppServerBackendMode::parse(" unavailable ").expect("unavailable mode"),
            AppServerBackendMode::Unavailable
        );

        let error = AppServerBackendMode::parse("agent").expect_err("unsupported mode");
        assert_eq!(error.value(), "agent");
    }

    #[test]
    fn factory_builds_mock_runtime_without_host_dependencies() {
        let _server = AppServerRuntimeFactory::mock_app_server();
    }

    #[test]
    fn factory_builds_external_runtime_without_host_dependencies() {
        let _server =
            AppServerRuntimeFactory::external_app_server(ExternalBackendConfig::new("backend"));
    }

    #[test]
    fn factory_builds_runtime_backend_without_host_dependencies() {
        let _server = AppServerRuntimeFactory::runtime_app_server();
    }

    #[test]
    fn factory_builds_runtime_backend_with_injected_capability_source() {
        let runtime = AppServerRuntimeFactory::runtime_backend_core_with_capability_source(
            Arc::new(CapabilityInventorySource::new(vec![
                CapabilityInventoryRecord::new(CapabilityDescriptor {
                    id: "content.draft.generate".to_string(),
                    title: "Generate Draft".to_string(),
                    description: None,
                    methods: vec!["turn/start".to_string()],
                })
                .for_apps(["content-studio"]),
            ])),
        );

        let response = runtime
            .list_capabilities(CapabilityListParams {
                app_id: Some("content-studio".to_string()),
                workspace_id: None,
                session_id: None,
                cursor: None,
                limit: None,
            })
            .expect("capability list");

        assert_eq!(response.capabilities.len(), 1);
        assert_eq!(response.capabilities[0].id, "content.draft.generate");
    }

    #[test]
    fn factory_builds_external_runtime_with_injected_capability_source() {
        let runtime = AppServerRuntimeFactory::external_runtime_core_with_capability_source(
            ExternalBackendConfig::new("backend"),
            Arc::new(CapabilityInventorySource::new(vec![
                CapabilityInventoryRecord::new(CapabilityDescriptor {
                    id: "content.draft.generate".to_string(),
                    title: "Generate Draft".to_string(),
                    description: None,
                    methods: vec!["turn/start".to_string()],
                })
                .for_apps(["content-studio"]),
            ])),
        );

        let response = runtime
            .list_capabilities(CapabilityListParams {
                app_id: Some("content-studio".to_string()),
                workspace_id: None,
                session_id: None,
                cursor: None,
                limit: None,
            })
            .expect("capability list");

        assert_eq!(response.capabilities.len(), 1);
        assert_eq!(response.capabilities[0].id, "content.draft.generate");
    }

    #[test]
    fn factory_builds_unavailable_runtime_without_host_dependencies() {
        let _server = AppServerRuntimeFactory::unavailable_app_server();
    }

    #[test]
    fn factory_builds_unavailable_runtime_with_injected_capability_source() {
        let runtime = AppServerRuntimeFactory::unavailable_runtime_core_with_capability_source(
            Arc::new(CapabilityInventorySource::new(vec![
                CapabilityInventoryRecord::new(CapabilityDescriptor {
                    id: "content.draft.generate".to_string(),
                    title: "Generate Draft".to_string(),
                    description: None,
                    methods: vec!["turn/start".to_string()],
                })
                .for_apps(["content-studio"]),
            ])),
        );

        let response = runtime
            .list_capabilities(CapabilityListParams {
                app_id: Some("content-studio".to_string()),
                workspace_id: None,
                session_id: None,
                cursor: None,
                limit: None,
            })
            .expect("capability list");

        assert_eq!(response.capabilities.len(), 1);
        assert_eq!(response.capabilities[0].id, "content.draft.generate");
    }

    #[test]
    fn factory_default_mock_runtime_uses_inventory_capability_source() {
        let runtime = AppServerRuntimeFactory::mock_runtime_core();

        let response = runtime
            .list_capabilities(CapabilityListParams::default())
            .expect("capability list");

        let ids = response
            .capabilities
            .iter()
            .map(|capability| capability.id.as_str())
            .collect::<Vec<_>>();

        assert!(ids.contains(&"agent.session"));
        assert!(ids.contains(&"tool.WebFetch"));
        assert!(ids.contains(&"tool.WebSearch"));
    }

    #[test]
    fn factory_can_inject_scoped_inventory_capability_source() {
        let runtime = AppServerRuntimeFactory::mock_runtime_core_with_capability_source(Arc::new(
            CapabilityInventorySource::new(vec![CapabilityInventoryRecord::new(
                CapabilityDescriptor {
                    id: "content.draft.generate".to_string(),
                    title: "Generate Draft".to_string(),
                    description: None,
                    methods: vec!["turn/start".to_string()],
                },
            )
            .for_apps(["content-studio"])]),
        ));

        let matched = runtime
            .list_capabilities(CapabilityListParams {
                app_id: Some("content-studio".to_string()),
                workspace_id: None,
                session_id: None,
                cursor: None,
                limit: None,
            })
            .expect("matched capability list");
        let unmatched = runtime
            .list_capabilities(CapabilityListParams {
                app_id: Some("other-app".to_string()),
                workspace_id: None,
                session_id: None,
                cursor: None,
                limit: None,
            })
            .expect("unmatched capability list");

        assert_eq!(matched.capabilities.len(), 1);
        assert_eq!(matched.capabilities[0].id, "content.draft.generate");
        assert!(unmatched.capabilities.is_empty());
    }
}
