use crate::protocol::{canonical_tool_item_event, AgentEvent, ToolItemLifecycleContext};
use crate::AgentTurnContext;
use agent_protocol::{SessionId, ThreadId};
use futures::{stream, StreamExt};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tool_runtime::tool_call::{ToolCall, ToolEnvironment};
use tool_runtime::tool_definition::{RuntimeToolDefinition, RuntimeToolExposure};
use tool_runtime::tool_executor::{
    RuntimeToolExecutionContext, RuntimeToolExecutionContextInput, RuntimeToolExecutorHandle,
};
use tool_runtime::tool_lifecycle::{
    ToolLifecycleEmissionFuture, ToolLifecycleEmitter, ToolLifecycleEvent, ToolLifecyclePhase,
};
use tool_runtime::tool_result_projection::NormalizedToolOutput;
use tool_runtime::web_search::{runtime_web_search_executor_handle, web_search_tool_definition};

pub const WORKSPACE_PATCH_HOST_TOOL_EVENT_SOURCE: &str = "workspace_patch_host_tool_requests";

const LEGACY_ARTICLE_WORKFLOW_KEY: &str = "content_article_workflow";
const WEB_SEARCH_TOOL_NAME: &str = "WebSearch";
const LOCAL_TOOL_ENVIRONMENT_ID: &str = "local";

#[derive(Debug, Clone)]
pub struct WorkspacePatchHostToolPlan {
    pub requests: Vec<WorkspacePatchHostToolRequest>,
}

#[derive(Debug, Clone)]
pub struct WorkspacePatchHostToolRequest {
    pub id: String,
    pub round_id: Option<String>,
    pub connector_ref: Option<String>,
    pub purpose: Option<String>,
    pub query: Option<String>,
    pub tool_name: String,
    pub arguments: String,
    pub params: Value,
    pub workflow_key: Option<String>,
    ordinal: u64,
}

#[derive(Debug, Clone)]
pub struct BoundWorkspacePatchHostToolRequest {
    pub request: WorkspacePatchHostToolRequest,
    pub tool_id: String,
}

#[derive(Debug, Clone)]
pub struct WorkspacePatchHostToolExecutionInput {
    pub session_id: String,
    pub thread_id: String,
    pub turn_id: String,
    pub working_directory: PathBuf,
    pub turn_context: Option<AgentTurnContext>,
    pub parallelism: usize,
}

#[derive(Debug, Clone)]
pub struct WorkspacePatchHostToolExecutionResult {
    pub events: Vec<AgentEvent>,
    pub host_tool_evidence: Vec<Value>,
    pub bound_requests: Vec<BoundWorkspacePatchHostToolRequest>,
}

#[derive(Debug, Clone)]
struct WorkspacePatchHostToolOutcome {
    output: NormalizedToolOutput,
}

struct WorkspacePatchToolLifecycleEmitter {
    session_id: SessionId,
    thread_id: ThreadId,
    state: Mutex<WorkspacePatchToolLifecycleState>,
}

#[derive(Default)]
struct WorkspacePatchToolLifecycleState {
    next_sequence: u64,
    next_ordinal: u64,
    items: HashMap<String, WorkspacePatchToolItemState>,
    events: Vec<AgentEvent>,
}

#[derive(Clone, Copy)]
struct WorkspacePatchToolItemState {
    ordinal: u64,
    created_at_ms: i64,
}

impl WorkspacePatchHostToolPlan {
    pub fn from_patch(patch: &Value) -> Option<Self> {
        let requests = workspace_patch_host_tool_request_candidates(patch)
            .into_iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                WorkspacePatchHostToolRequest::from_candidate(index, candidate)
            })
            .collect::<Vec<_>>();
        (!requests.is_empty()).then_some(Self { requests })
    }
}

pub async fn execute_workspace_patch_host_tool_plan(
    plan: &WorkspacePatchHostToolPlan,
    input: WorkspacePatchHostToolExecutionInput,
) -> Result<WorkspacePatchHostToolExecutionResult, String> {
    execute_workspace_patch_host_tool_plan_with_runtime(
        plan,
        input,
        runtime_web_search_executor_handle(),
        web_search_tool_definition(),
    )
    .await
}

async fn execute_workspace_patch_host_tool_plan_with_runtime(
    plan: &WorkspacePatchHostToolPlan,
    input: WorkspacePatchHostToolExecutionInput,
    executor: RuntimeToolExecutorHandle,
    definition: RuntimeToolDefinition,
) -> Result<WorkspacePatchHostToolExecutionResult, String> {
    let emitter = Arc::new(WorkspacePatchToolLifecycleEmitter::new(
        input.session_id.clone(),
        input.thread_id.clone(),
    ));
    let bound_requests = bind_workspace_patch_host_tool_requests(plan, &input);
    let mut indexed_outcomes = stream::iter(bound_requests.iter().cloned().enumerate().map(
        |(index, request)| {
            let input = input.clone();
            let executor = executor.clone();
            let definition = definition.clone();
            let emitter = emitter.clone();
            async move {
                let outcome = execute_workspace_patch_host_tool_request(
                    request, input, executor, definition, emitter,
                )
                .await;
                (index, outcome)
            }
        },
    ))
    .buffer_unordered(input.parallelism.max(1))
    .collect::<Vec<_>>()
    .await;
    indexed_outcomes.sort_by_key(|(index, _)| *index);
    let outcomes = indexed_outcomes
        .into_iter()
        .map(|(_, outcome)| outcome)
        .collect::<Vec<_>>();
    let host_tool_evidence =
        build_workspace_patch_host_tool_evidence(&bound_requests, outcomes.as_slice());

    Ok(WorkspacePatchHostToolExecutionResult {
        events: emitter.events(),
        host_tool_evidence,
        bound_requests,
    })
}

fn bind_workspace_patch_host_tool_requests(
    plan: &WorkspacePatchHostToolPlan,
    input: &WorkspacePatchHostToolExecutionInput,
) -> Vec<BoundWorkspacePatchHostToolRequest> {
    plan.requests
        .iter()
        .cloned()
        .map(|request| {
            let ordinal_bytes = request.ordinal.to_le_bytes();
            let identity_hash = stable_identity_hash(&[
                input.session_id.as_bytes(),
                input.thread_id.as_bytes(),
                input.turn_id.as_bytes(),
                &ordinal_bytes,
                request.id.as_bytes(),
            ]);
            BoundWorkspacePatchHostToolRequest {
                tool_id: format!(
                    "workspace-patch-host-tool-{}-{:04}-{:016x}",
                    sanitize_tool_id(&request.tool_name.to_ascii_lowercase()),
                    request.ordinal.saturating_add(1),
                    identity_hash,
                ),
                request,
            }
        })
        .collect()
}

async fn execute_workspace_patch_host_tool_request(
    request: BoundWorkspacePatchHostToolRequest,
    input: WorkspacePatchHostToolExecutionInput,
    executor: RuntimeToolExecutorHandle,
    definition: RuntimeToolDefinition,
    emitter: Arc<dyn ToolLifecycleEmitter>,
) -> WorkspacePatchHostToolOutcome {
    let context = RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
        working_directory: input.working_directory.clone(),
        session_id: input.session_id,
        cancel_token: None,
        workspace_sandbox: None,
    });
    let call = ToolCall::new(
        input.turn_id,
        request.tool_id.clone(),
        request.request.tool_name.clone(),
        request.request.params,
        vec![ToolEnvironment::new(
            LOCAL_TOOL_ENVIRONMENT_ID,
            input.working_directory,
        )],
        emitter,
    );
    let runtime = executor.bind(definition, RuntimeToolExposure::Direct);
    let output = runtime
        .execute_call(&call, &context, input.turn_context.as_ref())
        .await;
    WorkspacePatchHostToolOutcome { output }
}

impl WorkspacePatchToolLifecycleEmitter {
    fn new(session_id: impl Into<String>, thread_id: impl Into<String>) -> Self {
        Self {
            session_id: SessionId::new(session_id),
            thread_id: ThreadId::new(thread_id),
            state: Mutex::new(WorkspacePatchToolLifecycleState::default()),
        }
    }

    fn project(&self, event: ToolLifecycleEvent) {
        let terminal = matches!(event.phase, ToolLifecyclePhase::Completed);
        if terminal && event.output.is_none() {
            return;
        }
        let now = chrono::Utc::now().timestamp_millis();
        let key = format!("{}\0{}", event.turn_id, event.call_id);
        let mut state = self
            .state
            .lock()
            .expect("workspace patch tool lifecycle mutex poisoned");
        let item_state = state.items.get(&key).copied().unwrap_or_else(|| {
            state.next_ordinal += 1;
            let item_state = WorkspacePatchToolItemState {
                ordinal: state.next_ordinal,
                created_at_ms: now,
            };
            state.items.insert(key.clone(), item_state);
            item_state
        });
        if terminal {
            state.items.remove(&key);
        }
        state.next_sequence += 1;
        if let Some(event) = canonical_tool_item_event(
            event,
            ToolItemLifecycleContext {
                session_id: self.session_id.clone(),
                thread_id: self.thread_id.clone(),
                sequence: state.next_sequence,
                ordinal: item_state.ordinal,
                created_at_ms: item_state.created_at_ms,
                updated_at_ms: now,
            },
        ) {
            state.events.push(event);
        }
    }

    fn events(&self) -> Vec<AgentEvent> {
        self.state
            .lock()
            .expect("workspace patch tool lifecycle mutex poisoned")
            .events
            .clone()
    }
}

impl ToolLifecycleEmitter for WorkspacePatchToolLifecycleEmitter {
    fn emit<'a>(&'a self, event: ToolLifecycleEvent) -> ToolLifecycleEmissionFuture<'a> {
        Box::pin(async move { self.project(event) })
    }
}

impl WorkspacePatchHostToolRequest {
    fn from_candidate(
        index: usize,
        candidate: WorkspacePatchHostToolRequestCandidate,
    ) -> Option<Self> {
        let value = candidate.value;
        let request_query = value_string(&value, &["query"]).map(ToString::to_string);
        let tool_name = value_string(&value, &["toolName", "tool_name", "tool"])
            .map(ToString::to_string)
            .unwrap_or_else(|| WEB_SEARCH_TOOL_NAME.to_string());
        if tool_name != WEB_SEARCH_TOOL_NAME {
            return None;
        }
        let mut params = host_tool_params_from_value(&value, request_query.as_deref())?;
        let query = value_string(&params, &["query"]).map(ToString::to_string);
        if query.is_none() {
            return None;
        }
        if let (Some(params_object), Some(query)) = (params.as_object_mut(), query.as_deref()) {
            params_object
                .entry("query".to_string())
                .or_insert_with(|| Value::String(query.to_string()));
        }
        let arguments = serde_json::to_string(&params).ok()?;
        let id = value_string(&value, &["id"])
            .map(ToString::to_string)
            .unwrap_or_else(|| format!("host-tool-request-{}", index + 1));
        let round_id = value_string(&value, &["roundId", "round_id"]).map(ToString::to_string);
        let connector_ref =
            value_string(&value, &["connectorRef", "connector_ref"]).map(ToString::to_string);
        let purpose = value_string(&value, &["purpose"]).map(ToString::to_string);
        let workflow_key = value_string(&value, &["workflowKey", "workflow_key"])
            .map(ToString::to_string)
            .or(candidate.workflow_key);
        Some(Self {
            id,
            round_id,
            connector_ref,
            purpose,
            query,
            tool_name: tool_name.clone(),
            arguments,
            params,
            workflow_key,
            ordinal: u64::try_from(index).unwrap_or(u64::MAX),
        })
    }
}

pub fn enrich_workspace_patch_host_tool_payload(
    payload: &mut Value,
    requests: &[BoundWorkspacePatchHostToolRequest],
) {
    let workflow_key = payload_tool_call_id(payload)
        .and_then(|tool_call_id| {
            requests
                .iter()
                .find(|request| request.tool_id == tool_call_id)
        })
        .and_then(|request| request.request.workflow_key.as_deref())
        .map(ToString::to_string);
    let Some(metadata) = payload
        .pointer_mut("/item/metadata")
        .and_then(Value::as_object_mut)
    else {
        return;
    };
    metadata.insert(
        "source".to_string(),
        Value::String(WORKSPACE_PATCH_HOST_TOOL_EVENT_SOURCE.to_string()),
    );
    if let Some(workflow_key) = workflow_key {
        metadata.insert(
            "workflowKey".to_string(),
            Value::String(workflow_key.clone()),
        );
        metadata.insert("workflow_key".to_string(), Value::String(workflow_key));
    }
}

fn build_workspace_patch_host_tool_evidence(
    requests: &[BoundWorkspacePatchHostToolRequest],
    outcomes: &[WorkspacePatchHostToolOutcome],
) -> Vec<Value> {
    requests
        .iter()
        .zip(outcomes)
        .map(|(request, outcome)| {
            let source = &request.request;
            let output_ref = outcome
                .output
                .sidecar_reference
                .as_ref()
                .map(|reference| reference.reference.as_str());
            json!({
                "id": format!("host-tool-evidence-{}", sanitize_tool_id(&source.id)),
                "requestId": source.id,
                "roundId": source.round_id,
                "connectorRef": source.connector_ref,
                "tool": source.tool_name,
                "toolName": source.tool_name,
                "toolCallId": request.tool_id,
                "source": WORKSPACE_PATCH_HOST_TOOL_EVENT_SOURCE,
                "workflowKey": source.workflow_key,
                "status": if outcome.output.success { "completed" } else { "failed" },
                "query": source.query,
                "purpose": source.purpose,
                "summary": workspace_patch_search_output_summary(&outcome.output.text),
                "output": outcome.output.text,
                "structuredContent": outcome.output.structured_content,
                "error": outcome.output.error,
                "durationMs": outcome.output.duration_ms,
                "truncated": outcome.output.truncation.is_some(),
                "outputRef": output_ref,
                "metadata": outcome.output.metadata,
                "confidence": if outcome.output.success { "host_verified" } else { "needs_review" },
            })
        })
        .collect()
}

pub fn update_workspace_patch_with_host_tool_evidence(
    patch: &mut Value,
    host_tool_evidence: &[Value],
) {
    let evidence = Value::Array(host_tool_evidence.to_vec());
    let host_tool_status = if host_tool_evidence
        .iter()
        .all(|evidence| evidence.get("status").and_then(Value::as_str) == Some("completed"))
    {
        "completed"
    } else if host_tool_evidence
        .iter()
        .all(|evidence| evidence.get("status").and_then(Value::as_str) == Some("failed"))
    {
        "failed"
    } else {
        "partial"
    };
    let Some(objects) = patch.get_mut("objects").and_then(Value::as_array_mut) else {
        return;
    };
    let only_web_search_evidence = host_tool_evidence
        .iter()
        .all(|evidence| evidence.get("tool").and_then(Value::as_str) == Some(WEB_SEARCH_TOOL_NAME));
    for object in objects {
        let Some(kind) = article_object_kind(object) else {
            continue;
        };
        let has_host_tool_request = object
            .get("source")
            .is_some_and(source_has_host_tool_request);
        if !has_host_tool_request && kind != "articleDraft" && kind != "imageGenerationSet" {
            continue;
        }
        {
            let Some(source) = object.get_mut("source").and_then(Value::as_object_mut) else {
                continue;
            };
            source.insert("hostToolEvidence".to_string(), evidence.clone());
            source.insert("hostToolStatus".to_string(), json!(host_tool_status));
            if only_web_search_evidence {
                source.insert("searchEvidence".to_string(), evidence.clone());
                source.insert("hostSearchEvidence".to_string(), evidence.clone());
                source.insert("hostSearchStatus".to_string(), json!(host_tool_status));
            }
        }
        if kind == "articleDraft" && host_tool_status == "failed" {
            if let Some(object_map) = object.as_object_mut() {
                object_map.insert("status".to_string(), json!("failed"));
                object_map.insert(
                    "summary".to_string(),
                    json!("检索失败，文章草稿未达到可交付状态"),
                );
            }
        }
    }
}

#[derive(Debug, Clone)]
struct WorkspacePatchHostToolRequestCandidate {
    value: Value,
    workflow_key: Option<String>,
}

fn workspace_patch_host_tool_request_candidates(
    patch: &Value,
) -> Vec<WorkspacePatchHostToolRequestCandidate> {
    let patch_workflow_key = workflow_key_from_value(patch);
    let mut candidates = Vec::new();
    push_host_tool_request_candidates(&mut candidates, patch, patch_workflow_key.clone(), false);

    for object in patch
        .get("objects")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let object_workflow_key = workflow_key_from_value(object).or_else(|| {
            object
                .get("source")
                .and_then(workflow_key_from_value)
                .or_else(|| patch_workflow_key.clone())
        });
        if let Some(source) = object.get("source") {
            push_host_tool_request_candidates(
                &mut candidates,
                source,
                object_workflow_key.clone(),
                false,
            );
            if article_object_kind(object).as_deref() == Some("articleDraft") {
                push_host_tool_request_candidates(
                    &mut candidates,
                    source,
                    object_workflow_key
                        .clone()
                        .or_else(|| Some(LEGACY_ARTICLE_WORKFLOW_KEY.to_string())),
                    true,
                );
            }
        }
    }
    candidates
}

fn push_host_tool_request_candidates(
    candidates: &mut Vec<WorkspacePatchHostToolRequestCandidate>,
    value: &Value,
    workflow_key: Option<String>,
    legacy_search_requests: bool,
) {
    let request_keys = if legacy_search_requests {
        &["searchRequests"][..]
    } else {
        &["hostToolRequests", "host_tool_requests"][..]
    };
    for request in request_keys
        .iter()
        .filter_map(|key| value.get(*key))
        .filter_map(Value::as_array)
        .flat_map(|requests| requests.iter().cloned())
    {
        candidates.push(WorkspacePatchHostToolRequestCandidate {
            value: request,
            workflow_key: workflow_key.clone(),
        });
    }
}

fn workspace_patch_search_output_summary(output: &str) -> String {
    let normalized = output
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(8)
        .collect::<Vec<_>>()
        .join("\n");
    if normalized.chars().count() <= 1_200 {
        normalized
    } else {
        normalized.chars().take(1_200).collect::<String>()
    }
}

fn host_tool_params_from_value(value: &Value, request_query: Option<&str>) -> Option<Value> {
    let mut params = value
        .get("params")
        .or_else(|| value.get("arguments"))
        .or_else(|| value.get("args"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    if let Some(params_string) = params.as_str() {
        params = serde_json::from_str(params_string)
            .unwrap_or_else(|_| json!({ "query": params_string }));
    }
    if !params.is_object() {
        params = json!({});
    }
    if let Some(query) = request_query {
        if let Some(params_object) = params.as_object_mut() {
            params_object
                .entry("query".to_string())
                .or_insert_with(|| Value::String(query.to_string()));
        }
    }
    params
        .as_object()
        .is_some_and(|object| object.contains_key("query"))
        .then_some(params)
}

fn article_object_kind(object: &Value) -> Option<String> {
    value_string(object, &["kind"])
        .map(ToString::to_string)
        .or_else(|| {
            object
                .get("ref")
                .or_else(|| object.get("objectRef"))
                .and_then(|reference| value_string(reference, &["kind"]))
                .map(ToString::to_string)
        })
}

fn workflow_key_from_value(value: &Value) -> Option<String> {
    value_string(value, &["workflowKey", "workflow_key", "workflow"]).map(ToString::to_string)
}

fn source_has_host_tool_request(source: &Value) -> bool {
    ["hostToolRequests", "host_tool_requests", "searchRequests"]
        .iter()
        .any(|key| {
            source
                .get(*key)
                .and_then(Value::as_array)
                .is_some_and(|items| !items.is_empty())
        })
}

fn payload_tool_call_id(payload: &Value) -> Option<&str> {
    payload
        .pointer("/item/payload/call_id")
        .or_else(|| payload.pointer("/item/payload/callId"))
        .and_then(Value::as_str)
}

fn value_string<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .find_map(|key| value.get(*key))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

fn sanitize_tool_id(value: &str) -> String {
    let sanitized = value
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '-' || character == '_' {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    sanitized.trim_matches('-').to_string()
}

fn stable_identity_hash(fields: &[&[u8]]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for field in fields {
        let field_len = u64::try_from(field.len()).unwrap_or(u64::MAX).to_le_bytes();
        for byte in field_len.iter().chain(field.iter()) {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_protocol::thread::ThreadItemPayload;
    use serde_json::json;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tool_runtime::tool_executor::{
        RuntimeToolExecutionFuture, RuntimeToolExecutionRequest, RuntimeToolExecutionResult,
        RuntimeToolExecutor,
    };
    use tool_runtime::tool_io::{
        ToolIoPayloadStats, ToolOutputReference, ToolOutputTruncation, ToolOutputTruncationReason,
    };

    #[test]
    fn reads_host_tool_requests_from_workspace_patch_source() {
        let patch = json!({
            "objects": [
                {
                    "ref": { "kind": "articleDraft" },
                    "source": {
                        "workflowKey": "content_article_workflow",
                        "hostToolRequests": [
                            {
                                "id": "research-query-1",
                                "toolName": "WebSearch",
                                "params": { "query": "Lime 写文章" },
                                "purpose": "验证依据"
                            }
                        ]
                    }
                }
            ]
        });

        let plan = WorkspacePatchHostToolPlan::from_patch(&patch).expect("host tool plan");

        assert_eq!(plan.requests.len(), 1);
        assert_eq!(plan.requests[0].tool_name, WEB_SEARCH_TOOL_NAME);
        assert_eq!(plan.requests[0].query.as_deref(), Some("Lime 写文章"));
        assert_eq!(
            plan.requests[0].workflow_key.as_deref(),
            Some("content_article_workflow")
        );
        assert_eq!(plan.requests[0].params["query"], "Lime 写文章");
    }

    #[test]
    fn reads_legacy_article_search_requests_as_host_tool_requests() {
        let patch = json!({
            "objects": [
                {
                    "ref": { "kind": "articleDraft" },
                    "source": {
                        "searchRequests": [
                            {
                                "id": "search-request-1",
                                "query": "Lime 写文章",
                                "purpose": "历史检索请求"
                            }
                        ]
                    }
                }
            ]
        });

        let plan = WorkspacePatchHostToolPlan::from_patch(&patch).expect("host tool plan");

        assert_eq!(plan.requests.len(), 1);
        assert_eq!(plan.requests[0].tool_name, WEB_SEARCH_TOOL_NAME);
        assert_eq!(plan.requests[0].query.as_deref(), Some("Lime 写文章"));
        assert_eq!(
            plan.requests[0].workflow_key.as_deref(),
            Some(LEGACY_ARTICLE_WORKFLOW_KEY)
        );
    }

    #[tokio::test]
    async fn runtime_execution_emits_bounded_canonical_items_and_preserves_output() {
        let patch = json!({
            "workflowKey": "content_article_workflow",
            "hostToolRequests": [
                { "id": "request-1", "toolName": "WebSearch", "query": "alpha" },
                { "id": "request-2", "toolName": "WebSearch", "query": "beta" },
                { "id": "request-3", "toolName": "WebSearch", "query": "gamma" }
            ]
        });
        let plan = WorkspacePatchHostToolPlan::from_patch(&patch).expect("host tool plan");
        let executor = Arc::new(CapturingExecutor::default());

        let execution = execute_workspace_patch_host_tool_plan_with_runtime(
            &plan,
            WorkspacePatchHostToolExecutionInput {
                session_id: "session-1".to_string(),
                thread_id: "thread-1".to_string(),
                turn_id: "turn-1".to_string(),
                working_directory: PathBuf::from("/tmp/workspace"),
                turn_context: None,
                parallelism: 2,
            },
            RuntimeToolExecutorHandle::new(executor.clone()),
            web_search_tool_definition(),
        )
        .await
        .expect("workspace patch host tools should execute");

        assert_eq!(executor.max_active.load(Ordering::SeqCst), 2);
        assert_eq!(
            executor
                .sessions
                .lock()
                .expect("captured sessions mutex poisoned")
                .as_slice(),
            ["session-1", "session-1", "session-1"]
        );
        assert_eq!(execution.events.len(), 6);
        assert_eq!(execution.bound_requests.len(), 3);
        assert_eq!(
            execution
                .bound_requests
                .iter()
                .map(|request| request.tool_id.as_str())
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );
        let completed_items = execution
            .events
            .iter()
            .filter_map(|event| match event {
                AgentEvent::ItemCompleted { item } => Some(item),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(completed_items.len(), 3);
        for item in &completed_items {
            assert_eq!(item.session_id.as_str(), "session-1");
            assert_eq!(item.thread_id.as_str(), "thread-1");
            assert_eq!(item.turn_id.as_str(), "turn-1");
            assert_eq!(item.metadata["source"], "fake_web_search");
            assert_eq!(item.metadata["truncation"]["reason"], "byte_limit");
            assert_eq!(
                item.metadata["sidecar_reference"]["reference"],
                "sidecar://workspace-patch"
            );
            let ThreadItemPayload::Tool {
                call_id, output, ..
            } = &item.payload
            else {
                panic!("expected canonical tool item");
            };
            assert!(call_id.starts_with("workspace-patch-host-tool-websearch-"));
            let output = output.as_ref().expect("terminal tool output");
            assert!(output.duration_ms.is_some_and(|duration| duration > 0));
            assert!(output.truncated);
            assert_eq!(
                output.output_ref.as_deref(),
                Some("sidecar://workspace-patch")
            );
            assert_eq!(output.structured_content.as_ref().unwrap()["matches"], 1);
        }

        assert_eq!(execution.host_tool_evidence.len(), 3);
        for evidence in &execution.host_tool_evidence {
            assert_eq!(evidence["status"], "completed");
            assert_eq!(evidence["structuredContent"]["matches"], 1);
            assert!(evidence["durationMs"]
                .as_u64()
                .is_some_and(|value| value > 0));
            assert_eq!(evidence["truncated"], true);
            assert_eq!(evidence["outputRef"], "sidecar://workspace-patch");
            assert_eq!(evidence["metadata"]["source"], "fake_web_search");
        }
    }

    #[tokio::test]
    async fn runtime_execution_binds_distinct_call_ids_across_turns_in_one_thread() {
        let patch = json!({
            "hostToolRequests": [
                { "id": "search-request-1", "toolName": "WebSearch", "query": "alpha" }
            ]
        });
        let plan = WorkspacePatchHostToolPlan::from_patch(&patch).expect("host tool plan");
        let executor = Arc::new(CapturingExecutor::default());
        let first = execute_workspace_patch_host_tool_plan_with_runtime(
            &plan,
            execution_input("turn-1"),
            RuntimeToolExecutorHandle::new(executor.clone()),
            web_search_tool_definition(),
        )
        .await
        .expect("first turn execution");
        let second = execute_workspace_patch_host_tool_plan_with_runtime(
            &plan,
            execution_input("turn-2"),
            RuntimeToolExecutorHandle::new(executor),
            web_search_tool_definition(),
        )
        .await
        .expect("second turn execution");

        let first_id = &first.bound_requests[0].tool_id;
        let second_id = &second.bound_requests[0].tool_id;
        assert_ne!(first_id, second_id);
        assert!(first
            .events
            .iter()
            .all(|event| event_tool_call_id(event).as_deref() == Some(first_id)));
        assert!(second
            .events
            .iter()
            .all(|event| event_tool_call_id(event).as_deref() == Some(second_id)));
    }

    #[tokio::test]
    async fn runtime_execution_disambiguates_duplicate_and_sanitized_request_ids() {
        let patch = json!({
            "hostToolRequests": [
                { "id": "request/a", "toolName": "WebSearch", "query": "alpha" },
                { "id": "request a", "toolName": "WebSearch", "query": "beta" },
                { "id": "request/a", "toolName": "WebSearch", "query": "gamma" }
            ]
        });
        let plan = WorkspacePatchHostToolPlan::from_patch(&patch).expect("host tool plan");
        let input = execution_input("turn-collision");
        let expected = bind_workspace_patch_host_tool_requests(&plan, &input);
        let rebound = bind_workspace_patch_host_tool_requests(&plan, &input);
        assert_eq!(
            expected
                .iter()
                .map(|request| request.tool_id.as_str())
                .collect::<Vec<_>>(),
            rebound
                .iter()
                .map(|request| request.tool_id.as_str())
                .collect::<Vec<_>>()
        );

        let execution = execute_workspace_patch_host_tool_plan_with_runtime(
            &plan,
            input,
            RuntimeToolExecutorHandle::new(Arc::new(CapturingExecutor::default())),
            web_search_tool_definition(),
        )
        .await
        .expect("collision execution");

        let call_ids = execution
            .bound_requests
            .iter()
            .map(|request| request.tool_id.as_str())
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(call_ids.len(), 3);
        assert_eq!(execution.host_tool_evidence[0]["requestId"], "request/a");
        assert_eq!(execution.host_tool_evidence[0]["query"], "alpha");
        assert_eq!(
            execution.host_tool_evidence[0]["output"],
            "result=\"alpha\""
        );
        assert_eq!(execution.host_tool_evidence[1]["requestId"], "request a");
        assert_eq!(execution.host_tool_evidence[1]["query"], "beta");
        assert_eq!(execution.host_tool_evidence[1]["output"], "result=\"beta\"");
        assert_eq!(execution.host_tool_evidence[2]["requestId"], "request/a");
        assert_eq!(execution.host_tool_evidence[2]["query"], "gamma");
        assert_eq!(
            execution.host_tool_evidence[2]["output"],
            "result=\"gamma\""
        );
    }

    #[test]
    fn canonical_payload_enrichment_only_updates_nested_item_metadata() {
        let patch = json!({
            "workflowKey": "content_article_workflow",
            "hostToolRequests": [
                { "id": "request-1", "toolName": "WebSearch", "query": "alpha" }
            ]
        });
        let plan = WorkspacePatchHostToolPlan::from_patch(&patch).expect("host tool plan");
        let bound_requests =
            bind_workspace_patch_host_tool_requests(&plan, &execution_input("turn-1"));
        let mut payload = json!({
            "item": {
                "payload": {
                    "type": "tool",
                    "call_id": bound_requests[0].tool_id
                },
                "metadata": { "existing": true }
            }
        });

        enrich_workspace_patch_host_tool_payload(&mut payload, &bound_requests);

        assert!(payload.get("source").is_none());
        assert!(payload.get("workflowKey").is_none());
        assert_eq!(
            payload["item"]["metadata"]["source"],
            WORKSPACE_PATCH_HOST_TOOL_EVENT_SOURCE
        );
        assert_eq!(
            payload["item"]["metadata"]["workflowKey"],
            "content_article_workflow"
        );
        assert_eq!(
            payload["item"]["metadata"]["workflow_key"],
            "content_article_workflow"
        );
        assert_eq!(payload["item"]["metadata"]["existing"], true);
    }

    #[test]
    fn workspace_patch_host_evidence_updates_article_search_fields_on_success() {
        let mut patch = article_workspace_patch_with_host_search_request();
        let plan = WorkspacePatchHostToolPlan::from_patch(&patch).expect("host tool plan");
        let bound_requests =
            bind_workspace_patch_host_tool_requests(&plan, &execution_input("turn-1"));
        let evidence = build_workspace_patch_host_tool_evidence(
            &bound_requests,
            &[tool_outcome(
                true,
                "session=session-1 query=Lime 写文章 result=found",
                None,
            )],
        );

        update_workspace_patch_with_host_tool_evidence(&mut patch, &evidence);

        let source = &patch["objects"][0]["source"];
        let search_evidence = source["searchEvidence"]
            .as_array()
            .expect("search evidence");
        assert_eq!(search_evidence.len(), 1);
        assert_eq!(search_evidence[0]["tool"], "WebSearch");
        assert_eq!(search_evidence[0]["status"], "completed");
        assert_eq!(
            search_evidence[0]["summary"],
            "session=session-1 query=Lime 写文章 result=found"
        );
        assert_eq!(source["hostSearchStatus"], "completed");
        assert_eq!(source["hostToolStatus"], "completed");
        assert_eq!(
            source["hostSearchEvidence"],
            Value::Array(search_evidence.clone())
        );
        assert_eq!(
            source["hostToolEvidence"],
            Value::Array(search_evidence.clone())
        );
    }

    #[test]
    fn workspace_patch_host_evidence_marks_article_failed_on_tool_failure() {
        let mut patch = article_workspace_patch_with_host_search_request();
        let plan = WorkspacePatchHostToolPlan::from_patch(&patch).expect("host tool plan");
        let bound_requests =
            bind_workspace_patch_host_tool_requests(&plan, &execution_input("turn-1"));
        let evidence = build_workspace_patch_host_tool_evidence(
            &bound_requests,
            &[tool_outcome(false, "", Some("web search unavailable"))],
        );

        update_workspace_patch_with_host_tool_evidence(&mut patch, &evidence);

        let source = &patch["objects"][0]["source"];
        let search_evidence = source["searchEvidence"]
            .as_array()
            .expect("search evidence");
        assert_eq!(search_evidence.len(), 1);
        assert_eq!(search_evidence[0]["tool"], "WebSearch");
        assert_eq!(search_evidence[0]["status"], "failed");
        assert_eq!(search_evidence[0]["summary"], "");
        assert_eq!(source["hostSearchStatus"], "failed");
        assert_eq!(source["hostToolStatus"], "failed");
        assert_eq!(
            source["hostSearchEvidence"],
            Value::Array(search_evidence.clone())
        );
        assert_eq!(
            source["hostToolEvidence"],
            Value::Array(search_evidence.clone())
        );
        assert_eq!(patch["objects"][0]["status"], "failed");
        assert_eq!(
            patch["objects"][0]["summary"],
            "检索失败，文章草稿未达到可交付状态"
        );
    }

    fn article_workspace_patch_with_host_search_request() -> Value {
        json!({
            "schemaVersion": 1,
            "appId": "content-factory-app",
            "sessionId": "session-1",
            "objects": [
                {
                    "ref": {
                        "appId": "content-factory-app",
                        "kind": "articleDraft",
                        "id": "article-draft-1",
                        "sessionId": "session-1"
                    },
                    "title": "公众号文章草稿",
                    "status": "ready",
                    "source": {
                        "taskKind": "content.article.generate",
                        "taskId": "task-article-draft-1",
                        "documentText": "# 草稿\n\n正文。",
                        "finalMarkdown": "# 草稿\n\n正文。",
                        "workflowKey": "content_article_workflow",
                        "hostToolRequests": [
                            {
                                "id": "host-tool-request-1",
                                "toolName": "WebSearch",
                                "query": "Lime 写文章",
                                "params": {
                                    "query": "Lime 写文章"
                                },
                                "purpose": "验证宿主真实检索回填"
                            }
                        ]
                    }
                }
            ]
        })
    }

    fn tool_outcome(
        success: bool,
        output: &str,
        error: Option<&str>,
    ) -> WorkspacePatchHostToolOutcome {
        WorkspacePatchHostToolOutcome {
            output: NormalizedToolOutput {
                success,
                text: output.to_string(),
                structured_content: None,
                error: error.map(ToString::to_string),
                duration_ms: 0,
                truncation: None,
                sidecar_reference: None,
                metadata: HashMap::from([("source".to_string(), json!("fixed_web_search"))]),
                agent_control_projection_facts: Vec::new(),
                agent_control_state_facts: Vec::new(),
            },
        }
    }

    fn execution_input(turn_id: &str) -> WorkspacePatchHostToolExecutionInput {
        WorkspacePatchHostToolExecutionInput {
            session_id: "session-1".to_string(),
            thread_id: "thread-1".to_string(),
            turn_id: turn_id.to_string(),
            working_directory: PathBuf::from("/tmp/workspace"),
            turn_context: None,
            parallelism: 2,
        }
    }

    fn event_tool_call_id(event: &AgentEvent) -> Option<String> {
        let item = match event {
            AgentEvent::ItemStarted { item }
            | AgentEvent::ItemUpdated { item }
            | AgentEvent::ItemCompleted { item } => item,
            _ => return None,
        };
        let ThreadItemPayload::Tool { call_id, .. } = &item.payload else {
            return None;
        };
        Some(call_id.clone())
    }

    #[derive(Default)]
    struct CapturingExecutor {
        active: AtomicUsize,
        max_active: AtomicUsize,
        sessions: Mutex<Vec<String>>,
    }

    impl RuntimeToolExecutor for CapturingExecutor {
        fn execute<'a>(
            &'a self,
            request: RuntimeToolExecutionRequest<'a>,
        ) -> RuntimeToolExecutionFuture<'a> {
            Box::pin(async move {
                self.sessions
                    .lock()
                    .expect("captured sessions mutex poisoned")
                    .push(request.context.session_id().to_string());
                let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_active.fetch_max(active, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                self.active.fetch_sub(1, Ordering::SeqCst);

                Ok(RuntimeToolExecutionResult::new(
                    true,
                    format!("result={}", request.params["query"]),
                    None,
                    HashMap::from([("source".to_string(), json!("fake_web_search"))]),
                )
                .with_structured_content(json!({ "matches": 1 }))
                .with_truncation(ToolOutputTruncation::new(
                    ToolOutputTruncationReason::ByteLimit,
                    ToolIoPayloadStats {
                        chars: 32,
                        bytes: 32,
                        tokens: 8,
                    },
                ))
                .with_sidecar_reference(ToolOutputReference::new(
                    "sidecar://workspace-patch",
                    Some("preview".to_string()),
                )))
            })
        }
    }
}
