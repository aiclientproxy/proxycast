use super::*;
use crate::capability::capability_descriptor_allows_agent_turn_start;
use crate::CapabilityListContext;
use app_server_protocol::*;
use serde_json::json;

fn paginate_capabilities(
    capabilities: Vec<app_server_protocol::CapabilityDescriptor>,
    cursor: Option<String>,
    limit: Option<u32>,
) -> (
    Vec<app_server_protocol::CapabilityDescriptor>,
    Option<String>,
) {
    let start = cursor
        .as_deref()
        .and_then(|cursor| cursor.parse::<usize>().ok())
        .unwrap_or(0)
        .min(capabilities.len());
    let Some(limit) = limit
        .filter(|limit| *limit > 0)
        .and_then(|limit| usize::try_from(limit).ok())
    else {
        return (capabilities.into_iter().skip(start).collect(), None);
    };

    let end = start.saturating_add(limit).min(capabilities.len());
    let next_cursor = (end < capabilities.len()).then(|| end.to_string());
    (
        capabilities
            .into_iter()
            .skip(start)
            .take(end.saturating_sub(start))
            .collect(),
        next_cursor,
    )
}

fn runtime_capability_manifest_from_descriptors(
    capabilities: &[app_server_protocol::CapabilityDescriptor],
    context: &CapabilityListContext,
) -> RuntimeCapabilityManifest {
    RuntimeCapabilityManifest {
        schema_version: RUNTIME_CAPABILITY_MANIFEST_SCHEMA_VERSION.to_string(),
        runtime_id: "app-server".to_string(),
        provider_id: None,
        session_id: context.session_id.clone(),
        generated_at: timestamp(),
        capabilities: capabilities
            .iter()
            .map(runtime_capability_entry_from_descriptor)
            .collect(),
    }
}

fn runtime_capability_entry_from_descriptor(
    descriptor: &app_server_protocol::CapabilityDescriptor,
) -> RuntimeCapabilityEntry {
    RuntimeCapabilityEntry {
        id: runtime_capability_id_from_descriptor_id(&descriptor.id),
        status: "supported".to_string(),
        scope: runtime_capability_scope_from_descriptor_id(&descriptor.id).to_string(),
        title: descriptor.title.clone(),
        detail: descriptor.description.clone(),
        version: None,
        metadata: Some(json!({
            "appServerCapabilityId": descriptor.id,
            "methods": descriptor.methods,
        })),
    }
}

fn runtime_capability_id_from_descriptor_id(id: &str) -> String {
    if id == "agent.session" {
        return "transport.jsonrpc".to_string();
    }
    if id.contains("state.delta") {
        return "state.delta".to_string();
    }
    if id.contains("snapshot") || id.contains("session") {
        return "state.snapshot".to_string();
    }
    if id.contains("action") || id.contains("hitl") {
        return "hitl.actions".to_string();
    }
    if id.contains("subagent") {
        return "subagents.handoff".to_string();
    }
    if id.contains("evidence") {
        return "evidence.export".to_string();
    }
    if id.contains("tool") {
        return "tools.native".to_string();
    }
    id.to_string()
}

fn runtime_capability_scope_from_descriptor_id(id: &str) -> &'static str {
    if id.starts_with("session.") || id.contains(".session") {
        return "session";
    }
    if id.starts_with("turn.") || id.contains(".turn") {
        return "turn";
    }
    if id.starts_with("tool.") || id.contains(".tool") {
        return "tool";
    }
    if id.starts_with("provider.") || id.contains(".provider") {
        return "provider";
    }
    "runtime"
}

impl RuntimeCore {
    pub fn list_capabilities(
        &self,
        params: CapabilityListParams,
    ) -> Result<CapabilityListResponse, RuntimeCoreError> {
        let cursor = params.cursor.clone();
        let limit = params.limit;
        let context = self.capability_list_context(params)?;
        let capabilities = self.capability_source.list_capabilities(&context);
        let (capabilities, next_cursor) = paginate_capabilities(capabilities, cursor, limit);
        let runtime_capability_manifest = Some(runtime_capability_manifest_from_descriptors(
            &capabilities,
            &context,
        ));
        Ok(CapabilityListResponse {
            capabilities,
            runtime_capability_manifest,
            next_cursor,
        })
    }

    #[allow(dead_code)]
    fn ensure_capability_allowed(
        &self,
        session_id: &str,
        capability_id: &str,
    ) -> Result<(), RuntimeCoreError> {
        let context = self.capability_list_context(CapabilityListParams {
            session_id: Some(session_id.to_string()),
            ..CapabilityListParams::default()
        })?;
        self.ensure_capability_allowed_with_context(&context, capability_id)
    }

    pub(in crate::runtime) fn ensure_capability_allowed_with_context(
        &self,
        context: &CapabilityListContext,
        capability_id: &str,
    ) -> Result<(), RuntimeCoreError> {
        let allowed = self
            .capability_source
            .list_capabilities(&context)
            .iter()
            .any(|capability| {
                capability.id == capability_id
                    && capability_descriptor_allows_agent_turn_start(capability)
            });
        if allowed {
            Ok(())
        } else {
            Err(RuntimeCoreError::CapabilityDenied(
                capability_id.to_string(),
            ))
        }
    }

    pub(in crate::runtime) fn capability_list_context(
        &self,
        params: CapabilityListParams,
    ) -> Result<CapabilityListContext, RuntimeCoreError> {
        let CapabilityListParams {
            app_id,
            workspace_id,
            session_id,
            cursor: _,
            limit: _,
        } = params;

        let Some(session_id) = session_id else {
            return Ok(CapabilityListContext {
                app_id,
                workspace_id,
                session_id: None,
            });
        };

        let (session_app_id, session_workspace_id) = {
            let state = self
                .state
                .lock()
                .expect("runtime core state mutex poisoned");
            let stored = state
                .sessions
                .get(&session_id)
                .ok_or_else(|| RuntimeCoreError::SessionNotFound(session_id.clone()))?;
            (
                stored.session.app_id.clone(),
                stored.session.workspace_id.clone(),
            )
        };

        Ok(CapabilityListContext {
            app_id: Some(session_app_id),
            workspace_id: session_workspace_id,
            session_id: Some(session_id),
        })
    }

    pub async fn read_agent_session_tool_inventory(
        &self,
        params: AgentSessionToolInventoryReadParams,
    ) -> Result<AgentSessionToolInventoryReadResponse, RuntimeCoreError> {
        let inventory = self
            .backend
            .read_tool_inventory(ToolInventoryReadRequest {
                caller: params.caller,
                workbench: params.workbench,
                metadata: params.metadata,
            })
            .await?;
        Ok(AgentSessionToolInventoryReadResponse { inventory })
    }
}
