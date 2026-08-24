//! project domain handlers for the App Server processor.

use super::{dispatch_result, parse_params, to_jsonrpc_error, RequestProcessor, RpcDispatch};
use app_server_protocol::protocol::v2;
use app_server_protocol::{
    JsonRpcError, ProjectMaterialImportFromUrlParams, ProjectMaterialListParams,
    ProjectMaterialLookupParams, ProjectMaterialUpdateParams, ProjectMaterialUploadParams,
    ProjectMemoryReadParams,
};
use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use thread_store::{
    CreateProjectParams as StoreCreateProjectParams, ListProjectsParams as StoreListProjectsParams,
    MoveProjectParams as StoreMoveProjectParams, ProjectMoveOutcome, StoredProject,
    StoredProjectRoot, ThreadStoreError, ThreadStoreErrorKind,
    UpdateProjectParams as StoreUpdateProjectParams,
};

impl RequestProcessor {
    pub(super) async fn handle_project_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: v2::ProjectListParams = parse_params(params)?;
        let page = self
            .runtime
            .canonical_thread_store()
            .map_err(to_jsonrpc_error)?
            .list_projects(StoreListProjectsParams {
                cursor: params.cursor,
                limit: params.limit.unwrap_or(50).clamp(1, 500),
            })
            .await
            .map_err(|error| project_store_error("project/list", error))?;
        dispatch_result(v2::ProjectListResponse {
            data: page
                .projects
                .into_iter()
                .map(api_project)
                .collect::<Result<_, _>>()?,
            next_cursor: page.next_cursor,
        })
    }

    pub(super) async fn handle_project_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: v2::ProjectReadParams = parse_params(params)?;
        let project = self
            .runtime
            .canonical_thread_store()
            .map_err(to_jsonrpc_error)?
            .read_project(params.project_id.clone())
            .await
            .map_err(|error| project_store_error("project/read", error))?
            .ok_or_else(|| invalid_params(format!("project not found: {}", params.project_id)))?;
        dispatch_result(v2::ProjectReadResponse {
            project: api_project(project)?,
        })
    }

    pub(super) async fn handle_project_create_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: v2::ProjectCreateParams = parse_params(params)?;
        let (project, created) = self
            .create_project(
                params.name,
                params.roots,
                params.metadata,
                Vec::new(),
                params.idempotency_key,
                "project/create",
            )
            .await?;
        if created {
            self.notify_project_changed(&project.id, v2::ProjectChangeType::Created)
                .await;
        }
        dispatch_result(v2::ProjectCreateResponse { project })
    }

    pub(super) async fn handle_project_import_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: v2::ProjectImportParams = parse_params(params)?;
        let thread_ids = validate_thread_ids(params.threads.unwrap_or_default())?;
        let (project, created) = self
            .create_project(
                params.name,
                params.roots,
                params.metadata,
                thread_ids.clone(),
                params.idempotency_key,
                "project/import",
            )
            .await?;
        if created {
            self.notify_project_changed(&project.id, v2::ProjectChangeType::Created)
                .await;
            self.notify_thread_projects(thread_ids, Some(project.id.clone()))
                .await;
        }
        dispatch_result(v2::ProjectImportResponse { project })
    }

    pub(super) async fn handle_project_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: v2::ProjectUpdateParams = parse_params(params)?;
        let name = params.name.map(validate_name).transpose()?;
        let roots = params.roots.map(validate_roots).transpose()?;
        let updated = self
            .runtime
            .canonical_thread_store()
            .map_err(to_jsonrpc_error)?
            .update_project(StoreUpdateProjectParams {
                project_id: params.project_id.clone(),
                name,
                roots,
                metadata: params.metadata,
            })
            .await
            .map_err(|error| project_store_error("project/update", error))?
            .ok_or_else(|| invalid_params(format!("project not found: {}", params.project_id)))?;
        let project = api_project(updated.project)?;
        if updated.changed {
            self.notify_project_changed(&project.id, v2::ProjectChangeType::Updated)
                .await;
        }
        dispatch_result(v2::ProjectUpdateResponse { project })
    }

    pub(super) async fn handle_project_move_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: v2::ProjectMoveParams = parse_params(params)?;
        let project_id = params.project_id;
        let outcome = self
            .runtime
            .canonical_thread_store()
            .map_err(to_jsonrpc_error)?
            .move_project(StoreMoveProjectParams {
                project_id: project_id.clone(),
                before_project_id: params.before_project_id,
            })
            .await
            .map_err(|error| project_store_error("project/move", error))?
            .ok_or_else(|| invalid_params(format!("project not found: {project_id}")))?;
        if outcome == ProjectMoveOutcome::Moved {
            self.notify_project_changed(&project_id, v2::ProjectChangeType::Updated)
                .await;
        }
        dispatch_result(v2::ProjectMoveResponse {})
    }

    pub(super) async fn handle_project_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: v2::ProjectDeleteParams = parse_params(params)?;
        let deleted = self
            .runtime
            .canonical_thread_store()
            .map_err(to_jsonrpc_error)?
            .delete_project(params.project_id.clone())
            .await
            .map_err(|error| project_store_error("project/delete", error))?
            .ok_or_else(|| invalid_params(format!("project not found: {}", params.project_id)))?;
        self.notify_project_changed(&params.project_id, v2::ProjectChangeType::Deleted)
            .await;
        self.notify_thread_projects(deleted.affected_active_thread_ids, None)
            .await;
        dispatch_result(v2::ProjectDeleteResponse {})
    }

    async fn create_project(
        &self,
        name: String,
        roots: Vec<v2::ProjectRoot>,
        metadata: Option<BTreeMap<String, String>>,
        thread_ids: Vec<String>,
        idempotency_key: String,
        operation: &'static str,
    ) -> Result<(v2::Project, bool), JsonRpcError> {
        let created = self
            .runtime
            .canonical_thread_store()
            .map_err(to_jsonrpc_error)?
            .create_project(StoreCreateProjectParams {
                name: validate_name(name)?,
                roots: validate_roots(roots)?,
                metadata: metadata.unwrap_or_default(),
                thread_ids,
                idempotency_key: validate_idempotency_key(idempotency_key)?,
            })
            .await
            .map_err(|error| project_store_error(operation, error))?;
        Ok((api_project(created.project)?, created.created))
    }

    pub(super) async fn ensure_project_exists(
        &self,
        project_id: &str,
        operation: &'static str,
    ) -> Result<(), JsonRpcError> {
        if project_id.is_empty() {
            return Err(invalid_params("projectId must not be empty"));
        }
        let project = self
            .runtime
            .canonical_thread_store()
            .map_err(to_jsonrpc_error)?
            .read_project(project_id.to_string())
            .await
            .map_err(|error| project_store_error(operation, error))?;
        if project.is_none() {
            return Err(invalid_params(format!("project not found: {project_id}")));
        }
        Ok(())
    }

    async fn notify_project_changed(&self, project_id: &str, change_type: v2::ProjectChangeType) {
        self.publish_server_notification(v2::ServerNotification::ProjectChanged(
            v2::ProjectChangedNotification {
                project_id: project_id.to_string(),
                change_type,
            },
        ))
        .await;
    }

    async fn notify_thread_projects(&self, thread_ids: Vec<String>, project_id: Option<String>) {
        for thread_id in thread_ids {
            self.publish_server_notification(v2::ServerNotification::ThreadProjectUpdated(
                v2::ThreadProjectUpdatedNotification {
                    thread_id,
                    project_id: project_id.clone(),
                },
            ))
            .await;
        }
    }

    pub(super) async fn handle_project_material_list_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ProjectMaterialListParams = parse_params(params)?;
        let response = self
            .runtime
            .list_project_materials(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_project_material_get_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ProjectMaterialLookupParams = parse_params(params)?;
        let response = self
            .runtime
            .get_project_material(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_project_material_count_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ProjectMaterialListParams = parse_params(params)?;
        let response = self
            .runtime
            .count_project_materials(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_project_material_upload_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ProjectMaterialUploadParams = parse_params(params)?;
        let response = self
            .runtime
            .upload_project_material(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_project_material_import_from_url_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ProjectMaterialImportFromUrlParams = parse_params(params)?;
        let response = self
            .runtime
            .import_project_material_from_url(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_project_material_update_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ProjectMaterialUpdateParams = parse_params(params)?;
        let response = self
            .runtime
            .update_project_material(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_project_material_delete_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ProjectMaterialLookupParams = parse_params(params)?;
        let response = self
            .runtime
            .delete_project_material(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }

    pub(super) async fn handle_project_material_content_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ProjectMaterialLookupParams = parse_params(params)?;
        let response = self
            .runtime
            .read_project_material_content(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
    // voice handlers 已提取到 processor/voice.rs

    // plugin handlers 已提取到 processor/plugin.rs

    pub(super) async fn handle_project_memory_read_impl(
        &self,
        params: Option<serde_json::Value>,
    ) -> Result<RpcDispatch, JsonRpcError> {
        self.ensure_initialized()?;
        let params: ProjectMemoryReadParams = parse_params(params)?;
        let response = self
            .runtime
            .read_project_memory(params)
            .await
            .map_err(to_jsonrpc_error)?;
        dispatch_result(response)
    }
}

fn validate_name(name: String) -> Result<String, JsonRpcError> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err(invalid_params("project name must not be empty"));
    }
    Ok(name)
}

fn validate_idempotency_key(key: String) -> Result<String, JsonRpcError> {
    if key.trim().is_empty() {
        return Err(invalid_params("idempotencyKey must not be empty"));
    }
    if key.len() > 512 {
        return Err(invalid_params("idempotencyKey must be at most 512 bytes"));
    }
    Ok(key)
}

fn validate_roots(roots: Vec<v2::ProjectRoot>) -> Result<Vec<StoredProjectRoot>, JsonRpcError> {
    let mut logical = HashSet::new();
    let mut canonical = HashSet::new();
    roots
        .into_iter()
        .map(|root| {
            let path = PathBuf::from(root.path);
            if !path.is_absolute() {
                return Err(invalid_params(format!(
                    "invalid project root: path is not absolute: {}",
                    path.display()
                )));
            }
            if !logical.insert(path.clone()) {
                return Err(invalid_params(format!(
                    "duplicate project root: {}",
                    path.display()
                )));
            }
            if let Ok(resolved) = std::fs::canonicalize(&path) {
                if !canonical.insert(resolved) {
                    return Err(invalid_params(format!(
                        "duplicate resolved project root: {}",
                        path.display()
                    )));
                }
            }
            Ok(StoredProjectRoot {
                path: path.to_string_lossy().into_owned(),
            })
        })
        .collect()
}

fn validate_thread_ids(thread_ids: Vec<String>) -> Result<Vec<String>, JsonRpcError> {
    let mut seen = HashSet::new();
    for thread_id in &thread_ids {
        if !seen.insert(thread_id.clone()) {
            return Err(invalid_params(format!("duplicate thread id: {thread_id}")));
        }
    }
    Ok(thread_ids)
}

fn api_project(project: StoredProject) -> Result<v2::Project, JsonRpcError> {
    let roots = project
        .roots
        .into_iter()
        .map(|root| {
            let path = PathBuf::from(&root.path);
            if !path.is_absolute() {
                return Err(JsonRpcError::new(
                    app_server_protocol::error_codes::RUNTIME_ERROR,
                    format!("stored project root is not absolute: {}", path.display()),
                ));
            }
            Ok(v2::ProjectRoot { path: root.path })
        })
        .collect::<Result<_, _>>()?;
    Ok(v2::Project {
        id: project.id,
        name: project.name,
        roots,
        metadata: project.metadata,
        position: project.position,
        created_at: project.created_at_ms / 1000,
        updated_at: project.updated_at_ms / 1000,
    })
}

pub(super) fn project_store_error(
    operation: &'static str,
    error: ThreadStoreError,
) -> JsonRpcError {
    match error.kind() {
        ThreadStoreErrorKind::InvalidRequest | ThreadStoreErrorKind::ThreadNotFound => {
            invalid_params(error.to_string())
        }
        ThreadStoreErrorKind::Unsupported => JsonRpcError::new(
            app_server_protocol::error_codes::METHOD_NOT_FOUND,
            format!("{operation} is unavailable without sqlite state"),
        ),
        ThreadStoreErrorKind::Internal => JsonRpcError::new(
            app_server_protocol::error_codes::RUNTIME_ERROR,
            format!("failed to run {operation}: {error}"),
        ),
    }
}

fn invalid_params(message: impl Into<String>) -> JsonRpcError {
    JsonRpcError::new(app_server_protocol::error_codes::INVALID_PARAMS, message)
}
