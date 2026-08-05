use agent_protocol::{
    CollabAgentOperation, ItemStatus, ThreadItem, ThreadItemPayload, ToolArgument, ToolOutput,
};
use app_server_protocol::AgentEvent;
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CanonicalToolKind {
    Tool,
    Mcp,
    Collab,
}

#[derive(Debug)]
pub(super) struct CanonicalTool {
    pub call_id: String,
    pub name: String,
    pub server_name: Option<String>,
    pub kind: CanonicalToolKind,
    pub status: ItemStatus,
    pub arguments: Map<String, Value>,
    pub output: Option<ToolOutput>,
    pub metadata: Value,
}

impl CanonicalTool {
    pub fn status_label(&self) -> &'static str {
        match self.status {
            ItemStatus::Pending | ItemStatus::InProgress => "started",
            ItemStatus::Completed => "completed",
            ItemStatus::Failed => "failed",
            ItemStatus::Interrupted => "interrupted",
            ItemStatus::Cancelled => "cancelled",
        }
    }

    pub fn structured_content(&self) -> Option<&Value> {
        self.output
            .as_ref()
            .and_then(|output| output.structured_content.as_ref())
    }

    pub fn evidence_value(&self) -> Value {
        let mut value = Map::new();
        value.insert(
            "toolCallId".to_string(),
            Value::String(self.call_id.clone()),
        );
        value.insert("toolName".to_string(), Value::String(self.name.clone()));
        value.insert(
            "arguments".to_string(),
            Value::Object(self.arguments.clone()),
        );
        value.insert("metadata".to_string(), self.metadata.clone());
        if let Some(output) = &self.output {
            if let Some(structured_content) = &output.structured_content {
                value.insert("result".to_string(), structured_content.clone());
            }
            if let Some(output_ref) = &output.output_ref {
                value.insert("outputRef".to_string(), Value::String(output_ref.clone()));
            }
            if let Some(error) = &output.error {
                value.insert("error".to_string(), Value::String(error.clone()));
            }
        }
        Value::Object(value)
    }
}

pub(super) fn canonical_tool(event: &AgentEvent) -> Option<CanonicalTool> {
    if !matches!(
        event.event_type.as_str(),
        "item.started" | "item.updated" | "item.completed"
    ) {
        return None;
    }
    let item = serde_json::from_value::<ThreadItem>(event.payload.get("item")?.clone()).ok()?;
    let status = item.status;
    let metadata = item.metadata;
    let (call_id, name, server_name, kind, arguments, output) = match item.payload {
        ThreadItemPayload::Tool {
            call_id,
            name,
            arguments,
            output,
        } => (
            call_id,
            name,
            None,
            CanonicalToolKind::Tool,
            arguments,
            output,
        ),
        ThreadItemPayload::McpToolCall {
            call_id,
            server_name,
            tool_name,
            arguments,
            output,
            ..
        } => (
            call_id,
            tool_name,
            Some(server_name),
            CanonicalToolKind::Mcp,
            arguments,
            output,
        ),
        ThreadItemPayload::CollabAgentToolCall {
            call_id,
            operation,
            output,
            ..
        } => (
            call_id,
            collab_tool_name(operation).to_string(),
            None,
            CanonicalToolKind::Collab,
            Vec::new(),
            output,
        ),
        _ => return None,
    };

    Some(CanonicalTool {
        call_id,
        name,
        server_name,
        kind,
        status,
        arguments: arguments_map(arguments),
        output,
        metadata,
    })
}

pub(super) fn is_retired_raw_tool_lifecycle(event_type: &str) -> bool {
    matches!(
        event_type,
        "tool.started" | "tool.result" | "tool.failed" | "tool.completed"
    )
}

pub(super) fn canonical_tool_or_side_channel(event: &AgentEvent) -> Option<Option<CanonicalTool>> {
    if is_retired_raw_tool_lifecycle(&event.event_type) {
        return None;
    }
    if event.event_type.starts_with("item.") {
        return canonical_tool(event).map(Some);
    }
    Some(None)
}

fn arguments_map(arguments: Vec<ToolArgument>) -> Map<String, Value> {
    arguments
        .into_iter()
        .map(|argument| {
            let value = serde_json::from_str(&argument.value)
                .unwrap_or_else(|_| Value::String(argument.value));
            (argument.name, value)
        })
        .collect()
}

fn collab_tool_name(operation: CollabAgentOperation) -> &'static str {
    match operation {
        CollabAgentOperation::Spawn => "collab.spawn",
        CollabAgentOperation::SendMessage => "collab.send_message",
        CollabAgentOperation::FollowUp => "collab.follow_up",
        CollabAgentOperation::Wait => "collab.wait",
        CollabAgentOperation::Interrupt => "collab.interrupt",
        CollabAgentOperation::Resume => "collab.resume",
        CollabAgentOperation::Close => "collab.close",
    }
}
