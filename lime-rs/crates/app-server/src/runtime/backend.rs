use super::ActionRespondRequest;
use super::CancelExecutionRequest;
use super::ExecutionBackend;
use super::ExecutionRequest;
use super::RuntimeCoreError;
use super::RuntimeEvent;
use super::RuntimeEventSink;
use super::ToolInventoryReadRequest;
use async_trait::async_trait;
use serde_json::json;

#[derive(Debug, Default)]
pub struct MockBackend;

#[async_trait]
impl ExecutionBackend for MockBackend {
    async fn start_turn(
        &self,
        request: ExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new(
            "turn.accepted",
            json!({
                "inputTextLength": request.input.concat_text().len(),
                "backend": "mock",
                "clientName": request.host.client_name,
            }),
        ))
    }

    async fn cancel_turn(
        &self,
        request: CancelExecutionRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new(
            "turn.canceled",
            json!({
                "backend": "mock",
                "clientName": request.host.client_name,
            }),
        ))
    }

    async fn respond_action(
        &self,
        request: ActionRespondRequest,
        sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        sink.emit(RuntimeEvent::new(
            "action.resolved",
            json!({
                "backend": "mock",
                "clientName": request.host.client_name,
                "requestId": request.request_id,
                "actionType": request.action_type,
                "confirmed": request.confirmed,
                "response": request.response,
            }),
        ))
    }

    async fn read_tool_inventory(
        &self,
        request: ToolInventoryReadRequest,
    ) -> Result<serde_json::Value, RuntimeCoreError> {
        Ok(json!({
            "request": {
                "caller": request.caller.unwrap_or_else(|| "assistant".to_string()),
                "surface": {
                    "workbench": request.workbench,
                },
            },
            "agent_initialized": false,
            "warnings": ["mock backend does not expose runtime tool inventory"],
            "mcp_servers": [],
            "default_allowed_tools": [],
            "counts": {
                "catalog_total": 0,
                "catalog_current_total": 0,
                "catalog_compat_total": 0,
                "catalog_deprecated_total": 0,
                "default_allowed_total": 0,
                "runtime_total": 0,
                "runtime_visible_total": 0,
                "native_total": 0,
                "native_visible_total": 0,
                "native_catalog_unmapped_total": 0,
                "extension_surface_total": 0,
                "extension_mcp_bridge_total": 0,
                "extension_runtime_total": 0,
                "extension_tool_total": 0,
                "extension_tool_visible_total": 0,
                "mcp_server_total": 0,
                "mcp_tool_total": 0,
                "mcp_tool_visible_total": 0,
            },
            "catalog_tools": [],
            "native_tools": [],
            "runtime_tools": [],
            "extension_surfaces": [],
            "extension_tools": [],
            "mcp_tools": [],
        }))
    }
}

#[derive(Debug, Default)]
pub struct UnavailableBackend;

#[async_trait]
impl ExecutionBackend for UnavailableBackend {
    async fn start_turn(
        &self,
        _request: ExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Err(RuntimeCoreError::Backend(
            "standalone app-server backend is not configured".to_string(),
        ))
    }

    async fn cancel_turn(
        &self,
        _request: CancelExecutionRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Err(RuntimeCoreError::Backend(
            "standalone app-server backend is not configured".to_string(),
        ))
    }

    async fn respond_action(
        &self,
        _request: ActionRespondRequest,
        _sink: &mut dyn RuntimeEventSink,
    ) -> Result<(), RuntimeCoreError> {
        Err(RuntimeCoreError::Backend(
            "standalone app-server backend is not configured".to_string(),
        ))
    }

    async fn read_tool_inventory(
        &self,
        _request: ToolInventoryReadRequest,
    ) -> Result<serde_json::Value, RuntimeCoreError> {
        Err(RuntimeCoreError::Backend(
            "standalone app-server backend is not configured".to_string(),
        ))
    }
}
