use super::new_id;
use super::RuntimeCore;
use super::RuntimeCoreError;
use app_server_protocol::WorkspaceRightSurfacePendingConsumeParams;
use app_server_protocol::WorkspaceRightSurfacePendingConsumeResponse;
use app_server_protocol::WorkspaceRightSurfacePendingDismissParams;
use app_server_protocol::WorkspaceRightSurfacePendingDismissResponse;
use app_server_protocol::WorkspaceRightSurfacePendingListParams;
use app_server_protocol::WorkspaceRightSurfacePendingListResponse;
use app_server_protocol::WorkspaceRightSurfacePendingRequest;
use app_server_protocol::WorkspaceRightSurfaceRequestParams;
use app_server_protocol::WorkspaceRightSurfaceRequestResponse;
use chrono::Duration;
use chrono::SecondsFormat;
use chrono::Utc;
use serde_json::Map;
use serde_json::Value;
use std::collections::HashMap;
use std::collections::HashSet;

const DEFAULT_PRIORITY: &str = "normal";
const STATUS_CONSUMED: &str = "consumed";
const STATUS_DISMISSED: &str = "dismissed";
const STATUS_PENDING: &str = "pending";
const BROWSER_SURFACE_KIND: &str = "browser";
const BROWSER_ACTION_OPEN: &str = "open";
const BROWSER_ACTION_CREATE_TAB: &str = "createTab";

#[derive(Debug)]
pub(in crate::runtime) struct BrowserWorkspaceIdentityRecord {
    browser_session_id: String,
    primary_tab_id: String,
    runtime_session_id: Option<String>,
    tab_origins: HashMap<String, String>,
}

impl RuntimeCore {
    pub async fn request_workspace_right_surface(
        &self,
        params: WorkspaceRightSurfaceRequestParams,
    ) -> Result<WorkspaceRightSurfaceRequestResponse, RuntimeCoreError> {
        let surface_kind = required_string(
            params.surface_kind,
            "surfaceKind is required for workspaceRightSurface/request",
        )?;
        let origin = required_string(
            params.origin,
            "origin is required for workspaceRightSurface/request",
        )?;
        let priority = optional_trimmed(params.priority).unwrap_or_else(|| DEFAULT_PRIORITY.into());
        let runtime_session_id = optional_trimmed(params.session_id);
        let metadata = if surface_kind == BROWSER_SURFACE_KIND {
            self.resolve_browser_identity_metadata(runtime_session_id.as_deref(), params.metadata)
                .await?
        } else {
            params.metadata
        };
        let now = Utc::now();
        let requested_at = now.to_rfc3339_opts(SecondsFormat::Millis, true);
        let expires_at = params.ttl_ms.and_then(|ttl_ms| {
            Duration::try_milliseconds(ttl_ms as i64)
                .map(|duration| (now + duration).to_rfc3339_opts(SecondsFormat::Millis, true))
        });
        let pending = WorkspaceRightSurfacePendingRequest {
            request_id: new_id("right_surface"),
            workspace_id: optional_trimmed(params.workspace_id),
            workspace_root: optional_trimmed(params.workspace_root),
            session_id: runtime_session_id,
            surface_kind,
            origin,
            reason: optional_trimmed(params.reason),
            priority,
            candidate_id: optional_trimmed(params.candidate_id),
            ttl_ms: params.ttl_ms,
            metadata,
            requested_at,
            expires_at,
            status: STATUS_PENDING.to_string(),
        };
        self.app_data_source
            .save_workspace_right_surface_pending(pending.clone())
            .await?;
        let mut state = self.state.lock().map_err(|_| {
            RuntimeCoreError::Backend(
                "failed to lock runtime state for workspaceRightSurface/request".to_string(),
            )
        })?;
        prune_expired_pending(&mut state.right_surface_pending);
        state.right_surface_pending.push(pending.clone());
        Ok(WorkspaceRightSurfaceRequestResponse {
            status: STATUS_PENDING.to_string(),
            request_id: pending.request_id.clone(),
            pending,
        })
    }

    async fn resolve_browser_identity_metadata(
        &self,
        runtime_session_id: Option<&str>,
        metadata: Option<Value>,
    ) -> Result<Option<Value>, RuntimeCoreError> {
        let mut metadata = value_object(metadata);
        let mut browser = value_object(metadata.remove("browser"));
        let Some(action) = value_string(browser.get("action")) else {
            if !browser.is_empty() {
                metadata.insert("browser".to_string(), Value::Object(browser));
            }
            return Ok((!metadata.is_empty()).then_some(Value::Object(metadata)));
        };
        if action != BROWSER_ACTION_OPEN && action != BROWSER_ACTION_CREATE_TAB {
            return Err(RuntimeCoreError::Backend(
                "Browser identity action must be open or createTab".to_string(),
            ));
        }
        let thread_id = value_string(browser.get("threadId")).ok_or_else(|| {
            RuntimeCoreError::Backend(
                "Browser identity requires metadata.browser.threadId".to_string(),
            )
        })?;
        let runtime_session_id = runtime_session_id.ok_or_else(|| {
            RuntimeCoreError::Backend(
                "Browser identity requires a canonical runtime session".to_string(),
            )
        })?;
        let canonical_thread_id = self
            .resolve_session_thread_id_current(runtime_session_id)
            .await?;
        if canonical_thread_id != thread_id {
            return Err(RuntimeCoreError::Backend(
                "Browser identity thread does not belong to the runtime session".to_string(),
            ));
        }

        let mut state = self.state.lock().map_err(|_| {
            RuntimeCoreError::Backend(
                "failed to lock runtime state for Browser identity".to_string(),
            )
        })?;
        let identity = state
            .browser_workspaces
            .entry(thread_id)
            .or_insert_with(|| {
                let primary_tab_id = new_id("browser_tab");
                BrowserWorkspaceIdentityRecord {
                    browser_session_id: new_id("browser_session"),
                    primary_tab_id: primary_tab_id.clone(),
                    runtime_session_id: Some(runtime_session_id.to_string()),
                    tab_origins: HashMap::from([(primary_tab_id, "user".to_string())]),
                }
            });
        if identity.runtime_session_id.as_deref() != Some(runtime_session_id) {
            return Err(RuntimeCoreError::Backend(
                "Browser identity runtime session mismatch".to_string(),
            ));
        }

        let tab_id = if action == BROWSER_ACTION_CREATE_TAB {
            let requested_session_id = value_string(browser.get("browserSessionId"));
            if requested_session_id.as_deref() != Some(identity.browser_session_id.as_str()) {
                return Err(RuntimeCoreError::Backend(
                    "Browser identity session is stale or unavailable".to_string(),
                ));
            }
            let tab_id = new_id("browser_tab");
            identity
                .tab_origins
                .insert(tab_id.clone(), "user".to_string());
            tab_id
        } else {
            identity.primary_tab_id.clone()
        };
        browser.insert(
            "browserSessionId".to_string(),
            Value::String(identity.browser_session_id.clone()),
        );
        browser.insert("tabId".to_string(), Value::String(tab_id));
        metadata.insert("browser".to_string(), Value::Object(browser));
        Ok(Some(Value::Object(metadata)))
    }

    pub async fn list_workspace_right_surface_pending(
        &self,
        params: WorkspaceRightSurfacePendingListParams,
    ) -> Result<WorkspaceRightSurfacePendingListResponse, RuntimeCoreError> {
        let normalized_params = normalize_pending_list_params(params);
        let persistence_enabled = self
            .app_data_source
            .workspace_right_surface_pending_persistence_enabled();
        let persisted_pending = if persistence_enabled {
            let mut persistence_params = normalized_params.clone();
            persistence_params.limit = None;
            self.app_data_source
                .list_workspace_right_surface_pending(persistence_params)
                .await?
        } else {
            Vec::new()
        };
        let mut state = self.state.lock().map_err(|_| {
            RuntimeCoreError::Backend(
                "failed to lock runtime state for workspaceRightSurface/pending/list".to_string(),
            )
        })?;
        prune_expired_pending(&mut state.right_surface_pending);

        if persistence_enabled {
            let persisted_ids = persisted_pending
                .iter()
                .map(|request| request.request_id.as_str())
                .collect::<HashSet<_>>();
            state.right_surface_pending.retain(|request| {
                !pending_matches_params(&normalized_params, request)
                    || persisted_ids.contains(request.request_id.as_str())
            });
        }

        let mut pending = state
            .right_surface_pending
            .iter()
            .filter(|request| pending_matches_params(&normalized_params, request))
            .cloned()
            .collect::<Vec<_>>();
        merge_pending_requests(&mut pending, persisted_pending);
        if let Some(limit) = normalized_params.limit.map(|value| value as usize) {
            pending.truncate(limit);
        }
        Ok(WorkspaceRightSurfacePendingListResponse { pending })
    }

    pub async fn consume_workspace_right_surface_pending(
        &self,
        params: WorkspaceRightSurfacePendingConsumeParams,
    ) -> Result<WorkspaceRightSurfacePendingConsumeResponse, RuntimeCoreError> {
        let request_ids = normalize_pending_request_ids(
            params.request_id,
            params.request_ids,
            "requestId is required for workspaceRightSurface/pending/consume",
        )?;
        let persisted_request_ids = self
            .app_data_source
            .delete_workspace_right_surface_pending(request_ids.clone())
            .await?;
        let mut state = self.state.lock().map_err(|_| {
            RuntimeCoreError::Backend(
                "failed to lock runtime state for workspaceRightSurface/pending/consume"
                    .to_string(),
            )
        })?;
        prune_expired_pending(&mut state.right_surface_pending);

        let memory_request_ids =
            remove_pending_requests(&mut state.right_surface_pending, &request_ids);
        let consumed_request_ids = merge_request_ids(memory_request_ids, persisted_request_ids);
        let missing_request_ids = missing_request_ids(&request_ids, &consumed_request_ids);

        Ok(WorkspaceRightSurfacePendingConsumeResponse {
            status: STATUS_CONSUMED.to_string(),
            consumed_request_ids,
            missing_request_ids,
        })
    }

    pub async fn dismiss_workspace_right_surface_pending(
        &self,
        params: WorkspaceRightSurfacePendingDismissParams,
    ) -> Result<WorkspaceRightSurfacePendingDismissResponse, RuntimeCoreError> {
        let request_ids = normalize_pending_request_ids(
            params.request_id,
            params.request_ids,
            "requestId is required for workspaceRightSurface/pending/dismiss",
        )?;
        let persisted_request_ids = self
            .app_data_source
            .delete_workspace_right_surface_pending(request_ids.clone())
            .await?;
        let mut state = self.state.lock().map_err(|_| {
            RuntimeCoreError::Backend(
                "failed to lock runtime state for workspaceRightSurface/pending/dismiss"
                    .to_string(),
            )
        })?;
        prune_expired_pending(&mut state.right_surface_pending);

        let memory_request_ids =
            remove_pending_requests(&mut state.right_surface_pending, &request_ids);
        let dismissed_request_ids = merge_request_ids(memory_request_ids, persisted_request_ids);
        let missing_request_ids = missing_request_ids(&request_ids, &dismissed_request_ids);

        Ok(WorkspaceRightSurfacePendingDismissResponse {
            status: STATUS_DISMISSED.to_string(),
            dismissed_request_ids,
            missing_request_ids,
        })
    }
}

fn required_string(value: String, message: &str) -> Result<String, RuntimeCoreError> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        Err(RuntimeCoreError::Backend(message.to_string()))
    } else {
        Ok(trimmed.to_string())
    }
}

fn optional_trimmed(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn value_object(value: Option<Value>) -> Map<String, Value> {
    value
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_default()
}

fn value_string(value: Option<&Value>) -> Option<String> {
    value
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
}

fn optional_filter_matches(filter: &Option<String>, value: Option<&str>) -> bool {
    filter
        .as_ref()
        .is_none_or(|filter| value == Some(filter.as_str()))
}

fn normalize_pending_list_params(
    params: WorkspaceRightSurfacePendingListParams,
) -> WorkspaceRightSurfacePendingListParams {
    WorkspaceRightSurfacePendingListParams {
        workspace_id: optional_trimmed(params.workspace_id),
        workspace_root: optional_trimmed(params.workspace_root),
        session_id: optional_trimmed(params.session_id),
        surface_kind: optional_trimmed(params.surface_kind),
        limit: params.limit,
    }
}

fn pending_matches_params(
    params: &WorkspaceRightSurfacePendingListParams,
    request: &WorkspaceRightSurfacePendingRequest,
) -> bool {
    optional_filter_matches(&params.workspace_id, request.workspace_id.as_deref())
        && optional_filter_matches(&params.workspace_root, request.workspace_root.as_deref())
        && optional_filter_matches(&params.session_id, request.session_id.as_deref())
        && params
            .surface_kind
            .as_ref()
            .is_none_or(|value| request.surface_kind == *value)
}

fn normalize_pending_request_ids(
    request_id: Option<String>,
    request_ids: Vec<String>,
    empty_message: &str,
) -> Result<Vec<String>, RuntimeCoreError> {
    let mut normalized_request_ids = Vec::new();
    if let Some(request_id) = optional_trimmed(request_id) {
        normalized_request_ids.push(request_id);
    }
    normalized_request_ids.extend(
        request_ids
            .into_iter()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    );
    normalized_request_ids.sort();
    normalized_request_ids.dedup();

    if normalized_request_ids.is_empty() {
        Err(RuntimeCoreError::Backend(empty_message.to_string()))
    } else {
        Ok(normalized_request_ids)
    }
}

fn remove_pending_requests(
    pending: &mut Vec<WorkspaceRightSurfacePendingRequest>,
    request_ids: &[String],
) -> Vec<String> {
    let mut removed_request_ids = Vec::new();
    pending.retain(|request| {
        if request_ids.contains(&request.request_id) {
            removed_request_ids.push(request.request_id.clone());
            false
        } else {
            true
        }
    });

    removed_request_ids
}

fn merge_pending_requests(
    pending: &mut Vec<WorkspaceRightSurfacePendingRequest>,
    persisted_pending: Vec<WorkspaceRightSurfacePendingRequest>,
) {
    let mut seen_request_ids = pending
        .iter()
        .map(|request| request.request_id.clone())
        .collect::<HashSet<_>>();
    for request in persisted_pending {
        if seen_request_ids.insert(request.request_id.clone()) {
            pending.push(request);
        }
    }
}

fn merge_request_ids(mut primary: Vec<String>, secondary: Vec<String>) -> Vec<String> {
    for request_id in secondary {
        if !primary.contains(&request_id) {
            primary.push(request_id);
        }
    }
    primary
}

fn missing_request_ids(request_ids: &[String], removed_request_ids: &[String]) -> Vec<String> {
    request_ids
        .iter()
        .filter(|request_id| !removed_request_ids.contains(request_id))
        .cloned()
        .collect()
}

fn prune_expired_pending(pending: &mut Vec<WorkspaceRightSurfacePendingRequest>) {
    let now = Utc::now();
    pending.retain(|request| {
        request
            .expires_at
            .as_deref()
            .and_then(|value| chrono::DateTime::parse_from_rfc3339(value).ok())
            .is_none_or(|expires_at| expires_at.with_timezone(&Utc) > now)
    });
}
