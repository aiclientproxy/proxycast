use super::backend_error;
use super::provider_config::current_agent_runtime_config_metadata;
use super::provider_config::initialize_runtime_database;
use super::request_context::{
    resolve_runtime_model_selection, runtime_request_from_request, session_scope_from_request,
    RuntimeModelSelection,
};
use super::workspace_patch_host_tools;
use super::RuntimeBackend;
use crate::runtime::ensure_workspace_patch_artifact_paths;
use crate::ExecutionRequest;
use crate::RuntimeCoreError;
use crate::RuntimeEvent;
use lime_agent::agent_tools::workspace_patch_host::{
    execute_workspace_patch_host_tool_plan, WorkspacePatchHostToolExecutionInput,
};
use std::path::PathBuf;

pub(super) async fn prepare_runtime_worker_artifact_events(
    runtime_backend: &RuntimeBackend,
    request: &ExecutionRequest,
    events: &mut Vec<RuntimeEvent>,
) -> Result<(), RuntimeCoreError> {
    ensure_workspace_patch_artifact_paths(events.as_mut_slice());
    let Some(host_tool_plan) =
        workspace_patch_host_tools::workspace_patch_host_tool_plan_from_events(events.as_slice())
    else {
        return Ok(());
    };
    if host_tool_plan.requests.is_empty() {
        return Ok(());
    }

    let db = initialize_runtime_database(runtime_backend.db.as_ref())?;
    runtime_backend
        .agent_state
        .init_agent_with_db(&db)
        .await
        .map_err(backend_error)?;
    runtime_backend
        .install_live_execution_process_hook_if_available()
        .await?;
    runtime_backend
        .register_current_native_tools_if_available()
        .await?;

    let host_request = runtime_request_from_request(request);
    let scope = session_scope_from_request(request)?;
    let selection = resolve_runtime_model_selection(request).unwrap_or(RuntimeModelSelection {
        provider: "host-tool-execution".to_string(),
        model: "host-tool-execution".to_string(),
        source: workspace_patch_host_tools::WORKSPACE_PATCH_HOST_TOOL_EVENT_SOURCE,
        reasoning_effort: None,
    });
    let turn_context = workspace_patch_host_tools::workspace_patch_host_tool_turn_context(
        request,
        host_request.as_ref(),
        &scope,
        &selection,
        current_agent_runtime_config_metadata(),
    );
    let working_directory = turn_context
        .cwd
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."));
    let execution = execute_workspace_patch_host_tool_plan(
        &host_tool_plan,
        WorkspacePatchHostToolExecutionInput {
            session_id: scope.session_id.clone(),
            thread_id: scope.thread_id.clone(),
            turn_id: scope.turn_id.clone(),
            working_directory,
            turn_context: Some(turn_context),
            parallelism: 2,
        },
    )
    .await
    .map_err(RuntimeCoreError::Backend)?;
    let host_tool_evidence = execution.host_tool_evidence;
    if host_tool_evidence.is_empty() {
        return Ok(());
    }
    workspace_patch_host_tools::update_workspace_patch_host_tool_artifact_events(
        events,
        &host_tool_evidence,
    );
    ensure_workspace_patch_artifact_paths(events.as_mut_slice());
    let mut tool_runtime_events = Vec::new();
    for event in execution.events {
        if let Some(runtime_event) =
            workspace_patch_host_tools::workspace_patch_host_tool_runtime_event(
                event,
                &execution.bound_requests,
            )
        {
            tool_runtime_events.push(runtime_event);
        }
    }
    let insert_at = events
        .iter()
        .position(|event| event.event_type == "artifact.snapshot")
        .unwrap_or(events.len());
    events.splice(insert_at..insert_at, tool_runtime_events);
    Ok(())
}
