use super::super::*;
use crate::MemoryBackend;

pub(in crate::runtime::tests) struct TestSessionDataSource {
    workspace: Option<serde_json::Value>,
    memory_store_read_response: Mutex<Option<Result<MemoryStoreReadResponse, String>>>,
    memory_store_read_requests: Mutex<Vec<MemoryStoreReadParams>>,
    memory_backend: Option<crate::LocalMemoryBackend>,
    plugin_installed_states: Mutex<Vec<serde_json::Value>>,
    knowledge_compile_requests: Mutex<Vec<lime_knowledge::KnowledgeCompilePackRequest>>,
    right_surface_pending: Mutex<Vec<WorkspaceRightSurfacePendingRequest>>,
    object_canvas_snapshots: Mutex<Vec<WorkspaceObjectCanvasSnapshot>>,
    media_task_artifacts: Mutex<Vec<MediaTaskArtifactResponse>>,
    media_task_list_requests: Mutex<Vec<MediaTaskArtifactListParams>>,
    model_catalogs: Vec<ProviderModelCatalog>,
}

impl TestSessionDataSource {
    pub(in crate::runtime::tests) fn new() -> Self {
        Self {
            workspace: None,
            memory_store_read_response: Mutex::new(None),
            memory_store_read_requests: Mutex::new(Vec::new()),
            memory_backend: None,
            plugin_installed_states: Mutex::new(Vec::new()),
            knowledge_compile_requests: Mutex::new(Vec::new()),
            right_surface_pending: Mutex::new(Vec::new()),
            object_canvas_snapshots: Mutex::new(Vec::new()),
            media_task_artifacts: Mutex::new(Vec::new()),
            media_task_list_requests: Mutex::new(Vec::new()),
            model_catalogs: Vec::new(),
        }
    }

    pub(in crate::runtime::tests) fn with_workspace(self, workspace: serde_json::Value) -> Self {
        Self {
            workspace: Some(workspace),
            ..self
        }
    }

    pub(in crate::runtime::tests) fn with_memory_data_root(
        self,
        data_root: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self {
            memory_backend: Some(crate::LocalMemoryBackend::new(data_root)),
            ..self
        }
    }

    pub(in crate::runtime::tests) fn with_memory_store_read_response(
        self,
        response: Result<MemoryStoreReadResponse, String>,
    ) -> Self {
        *self
            .memory_store_read_response
            .lock()
            .expect("test memory response mutex poisoned") = Some(response);
        self
    }

    pub(in crate::runtime::tests) fn with_plugin_installed_states(
        self,
        states: Vec<serde_json::Value>,
    ) -> Self {
        *self
            .plugin_installed_states
            .lock()
            .expect("test plugin installed states mutex poisoned") = states;
        self
    }

    pub(in crate::runtime::tests) fn with_media_task_artifacts(
        self,
        tasks: Vec<MediaTaskArtifactResponse>,
    ) -> Self {
        *self
            .media_task_artifacts
            .lock()
            .expect("test media task artifacts mutex poisoned") = tasks;
        self
    }

    pub(in crate::runtime::tests) fn with_model_catalogs(
        self,
        model_catalogs: Vec<ProviderModelCatalog>,
    ) -> Self {
        Self {
            model_catalogs,
            ..self
        }
    }

    pub(in crate::runtime::tests) fn knowledge_compile_requests(
        &self,
    ) -> Vec<lime_knowledge::KnowledgeCompilePackRequest> {
        self.knowledge_compile_requests
            .lock()
            .expect("test knowledge compile requests mutex poisoned")
            .clone()
    }

    pub(in crate::runtime::tests) fn memory_store_read_requests(
        &self,
    ) -> Vec<MemoryStoreReadParams> {
        self.memory_store_read_requests
            .lock()
            .expect("test memory requests mutex poisoned")
            .clone()
    }

    pub(in crate::runtime::tests) fn object_canvas_snapshots(
        &self,
    ) -> Vec<WorkspaceObjectCanvasSnapshot> {
        self.object_canvas_snapshots
            .lock()
            .expect("test object canvas snapshot mutex poisoned")
            .clone()
    }

    pub(in crate::runtime::tests) fn media_task_list_requests(
        &self,
    ) -> Vec<MediaTaskArtifactListParams> {
        self.media_task_list_requests
            .lock()
            .expect("test media task list requests mutex poisoned")
            .clone()
    }
}

#[async_trait]
impl SessionAppDataSource for TestSessionDataSource {}

#[async_trait]
impl KnowledgeAppDataSource for TestSessionDataSource {
    async fn compile_knowledge_pack(
        &self,
        request: lime_knowledge::KnowledgeCompilePackRequest,
    ) -> Result<KnowledgeCompilePackResponse, RuntimeCoreError> {
        self.knowledge_compile_requests
            .lock()
            .expect("test knowledge compile requests mutex poisoned")
            .push(request.clone());
        let response =
            lime_knowledge::compile_knowledge_pack(request).map_err(RuntimeCoreError::Backend)?;
        Ok(KnowledgeCompilePackResponse {
            pack: serde_json::to_value(response.pack)
                .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?,
            selected_source_count: response.selected_source_count,
            compiled_view: serde_json::to_value(response.compiled_view)
                .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?,
            run: serde_json::to_value(response.run)
                .map_err(|error| RuntimeCoreError::Backend(error.to_string()))?,
            warnings: response.warnings,
        })
    }
}

#[async_trait]
impl WorkspaceAppDataSource for TestSessionDataSource {
    async fn read_workspace(
        &self,
        _params: WorkspaceReadParams,
    ) -> Result<WorkspaceReadResponse, RuntimeCoreError> {
        Ok(WorkspaceReadResponse {
            workspace: self.workspace.clone(),
        })
    }
}
impl SkillAppDataSource for TestSessionDataSource {}
impl WorkspaceSkillBindingAppDataSource for TestSessionDataSource {}
impl GatewayAppDataSource for TestSessionDataSource {}
#[async_trait]
impl MediaAppDataSource for TestSessionDataSource {
    async fn list_media_task_artifacts(
        &self,
        params: MediaTaskArtifactListParams,
    ) -> Result<MediaTaskArtifactListResponse, RuntimeCoreError> {
        self.media_task_list_requests
            .lock()
            .expect("test media task list requests mutex poisoned")
            .push(params.clone());
        let status_filter = optional_trimmed(params.status.clone());
        let task_family_filter = optional_trimmed(params.task_family.clone());
        let task_type_filter = optional_trimmed(params.task_type.clone());
        let mut tasks = self
            .media_task_artifacts
            .lock()
            .expect("test media task artifacts mutex poisoned")
            .iter()
            .filter(|task| {
                status_filter.as_ref().is_none_or(|status| {
                    task.status == *status || task.normalized_status == *status
                })
            })
            .filter(|task| {
                optional_filter_matches(&task_family_filter, Some(task.task_family.as_str()))
            })
            .filter(|task| {
                optional_filter_matches(&task_type_filter, Some(task.task_type.as_str()))
            })
            .cloned()
            .collect::<Vec<_>>();
        if let Some(limit) = params.limit {
            tasks.truncate(limit);
        }
        Ok(MediaTaskArtifactListResponse {
            success: true,
            workspace_root: params.project_root_path.clone(),
            artifact_root: String::new(),
            filters: MediaTaskArtifactListFilters {
                status: status_filter,
                task_family: task_family_filter,
                task_type: task_type_filter,
                modality_contract_key: params.modality_contract_key,
                routing_outcome: params.routing_outcome,
                limit: params.limit,
            },
            total: tasks.len(),
            modality_runtime_contracts: serde_json::Value::Null,
            tasks,
        })
    }
}
impl VoiceAppDataSource for TestSessionDataSource {}
#[async_trait]
impl PluginDataSource for TestSessionDataSource {
    async fn list_plugin_installed(&self) -> Result<PluginInstalledListResponse, RuntimeCoreError> {
        Ok(PluginInstalledListResponse {
            states: self
                .plugin_installed_states
                .lock()
                .expect("test plugin installed states mutex poisoned")
                .clone(),
            issues: Vec::new(),
        })
    }
}
impl AutomationOverviewAppDataSource for TestSessionDataSource {}
impl McpAppDataSource for TestSessionDataSource {}
impl AutomationManagementAppDataSource for TestSessionDataSource {}
#[async_trait]
impl MemoryAppDataSource for TestSessionDataSource {
    async fn read_memory_store(
        &self,
        params: MemoryStoreReadParams,
    ) -> Result<MemoryStoreReadResponse, RuntimeCoreError> {
        self.memory_store_read_requests
            .lock()
            .expect("test memory requests mutex poisoned")
            .push(params);
        let response = self
            .memory_store_read_response
            .lock()
            .expect("test memory response mutex poisoned")
            .clone();
        match response {
            Some(Ok(response)) => Ok(response),
            Some(Err(message)) => Err(RuntimeCoreError::Backend(message)),
            None => Err(RuntimeCoreError::Backend(
                "memoryStore/read unavailable in test data source".to_string(),
            )),
        }
    }

    async fn write_memory_rollout_summary(
        &self,
        params: crate::RolloutSummaryWriteParams,
    ) -> Result<MemoryStoreAddNoteResponse, RuntimeCoreError> {
        let Some(memory_backend) = &self.memory_backend else {
            return Err(RuntimeCoreError::Backend(
                "memory rollout summary write unavailable in test data source".to_string(),
            ));
        };
        memory_backend.write_rollout_summary(params).await
    }

    async fn consolidate_memory_store(
        &self,
        params: MemoryStoreConsolidateParams,
    ) -> Result<MemoryStoreConsolidateResponse, RuntimeCoreError> {
        let Some(memory_backend) = &self.memory_backend else {
            return Err(RuntimeCoreError::Backend(
                "memoryStore/consolidate unavailable in test data source".to_string(),
            ));
        };
        memory_backend.consolidate(params).await
    }
}
impl DiagnosticsAppDataSource for TestSessionDataSource {}
impl UsageStatsAppDataSource for TestSessionDataSource {}
#[async_trait]
impl ModelProviderAppDataSource for TestSessionDataSource {
    fn model_catalog_reconciliation_enabled(&self) -> bool {
        true
    }

    async fn read_model_route_generation(&self) -> Result<u64, RuntimeCoreError> {
        Ok(1)
    }

    async fn model_catalog(
        &self,
        _query: ModelCatalogQuery,
    ) -> Result<Vec<ProviderModelCatalog>, RuntimeCoreError> {
        Ok(self.model_catalogs.clone())
    }
}
impl ConnectAppDataSource for TestSessionDataSource {}

#[async_trait]
impl RightSurfaceAppDataSource for TestSessionDataSource {
    fn workspace_right_surface_pending_persistence_enabled(&self) -> bool {
        true
    }

    async fn save_workspace_right_surface_pending(
        &self,
        request: WorkspaceRightSurfacePendingRequest,
    ) -> Result<(), RuntimeCoreError> {
        let mut pending = self
            .right_surface_pending
            .lock()
            .expect("test right surface pending mutex poisoned");
        pending.retain(|item| item.request_id != request.request_id);
        pending.push(request);
        Ok(())
    }

    async fn list_workspace_right_surface_pending(
        &self,
        params: WorkspaceRightSurfacePendingListParams,
    ) -> Result<Vec<WorkspaceRightSurfacePendingRequest>, RuntimeCoreError> {
        let workspace_id = optional_trimmed(params.workspace_id);
        let workspace_root = optional_trimmed(params.workspace_root);
        let session_id = optional_trimmed(params.session_id);
        let surface_kind = optional_trimmed(params.surface_kind);
        let mut pending = self
            .right_surface_pending
            .lock()
            .expect("test right surface pending mutex poisoned")
            .iter()
            .filter(|request| {
                optional_filter_matches(&workspace_id, request.workspace_id.as_deref())
            })
            .filter(|request| {
                optional_filter_matches(&workspace_root, request.workspace_root.as_deref())
            })
            .filter(|request| optional_filter_matches(&session_id, request.session_id.as_deref()))
            .filter(|request| {
                surface_kind
                    .as_ref()
                    .is_none_or(|value| request.surface_kind == *value)
            })
            .cloned()
            .collect::<Vec<_>>();
        if let Some(limit) = params.limit.map(|value| value as usize) {
            pending.truncate(limit);
        }
        Ok(pending)
    }

    async fn delete_workspace_right_surface_pending(
        &self,
        request_ids: Vec<String>,
    ) -> Result<Vec<String>, RuntimeCoreError> {
        let mut deleted = Vec::new();
        let mut pending = self
            .right_surface_pending
            .lock()
            .expect("test right surface pending mutex poisoned");
        pending.retain(|request| {
            if request_ids.contains(&request.request_id) {
                deleted.push(request.request_id.clone());
                false
            } else {
                true
            }
        });
        Ok(deleted)
    }

    async fn save_workspace_object_canvas_snapshot(
        &self,
        snapshot: WorkspaceObjectCanvasSnapshot,
    ) -> Result<(), RuntimeCoreError> {
        let mut snapshots = self
            .object_canvas_snapshots
            .lock()
            .expect("test object canvas snapshot mutex poisoned");
        snapshots.retain(|item| {
            item.persistence_key != snapshot.persistence_key || item.revision != snapshot.revision
        });
        snapshots.push(snapshot);
        Ok(())
    }

    async fn list_workspace_object_canvas_snapshots(
        &self,
        params: WorkspaceObjectCanvasSnapshotListParams,
    ) -> Result<Vec<WorkspaceObjectCanvasSnapshot>, RuntimeCoreError> {
        let workspace_id = optional_trimmed(params.workspace_id);
        let workspace_root = optional_trimmed(params.workspace_root);
        let session_id = optional_trimmed(params.session_id);
        let board_id = optional_trimmed(params.board_id);
        let persistence_key = optional_trimmed(params.persistence_key);
        let mut snapshots = self
            .object_canvas_snapshots
            .lock()
            .expect("test object canvas snapshot mutex poisoned")
            .iter()
            .filter(|snapshot| {
                optional_filter_matches(&workspace_id, snapshot.workspace_id.as_deref())
            })
            .filter(|snapshot| {
                optional_filter_matches(&workspace_root, snapshot.workspace_root.as_deref())
            })
            .filter(|snapshot| optional_filter_matches(&session_id, snapshot.session_id.as_deref()))
            .filter(|snapshot| optional_filter_matches(&board_id, Some(snapshot.board_id.as_str())))
            .filter(|snapshot| {
                optional_filter_matches(&persistence_key, Some(snapshot.persistence_key.as_str()))
            })
            .cloned()
            .collect::<Vec<_>>();
        if let Some(limit) = params.limit.map(|value| value as usize) {
            snapshots.truncate(limit);
        }
        Ok(snapshots)
    }
}

fn optional_trimmed(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn optional_filter_matches(filter: &Option<String>, value: Option<&str>) -> bool {
    filter
        .as_ref()
        .is_none_or(|filter| value == Some(filter.as_str()))
}
