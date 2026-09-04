use app_server_protocol::protocol::v2::{
    CollabAgentToolCallStatus, CommandExecutionStatus, DynamicToolCallStatus, McpToolCallStatus,
    PatchApplyStatus, PatchChangeKind, ServerNotification, Thread, ThreadItem, TurnStatus,
    UserInput,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryKind {
    User,
    Assistant,
    Reasoning,
    Command,
    Patch,
    Mcp,
    Plan,
    MultiAgent,
    Tool,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EntryStatus {
    Running,
    Completed,
    Failed,
    Declined,
    Interrupted,
}

impl EntryStatus {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Declined => "declined",
            Self::Interrupted => "interrupted",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TranscriptEntry {
    pub(crate) id: String,
    pub(crate) kind: EntryKind,
    pub(crate) text: String,
    pub(crate) streaming: bool,
    pub(crate) status: Option<EntryStatus>,
    /// Stable, display-ready facts derived from the canonical item payload.
    pub(crate) summary: Vec<String>,
}

#[derive(Debug, Default)]
pub(crate) struct ConversationProjection {
    entries: Vec<TranscriptEntry>,
    active_turn_id: Option<String>,
    status: String,
}

impl ConversationProjection {
    pub(crate) fn entries(&self) -> &[TranscriptEntry] {
        &self.entries
    }

    pub(crate) fn active_turn_id(&self) -> Option<&str> {
        self.active_turn_id.as_deref()
    }

    pub(crate) fn status(&self) -> &str {
        &self.status
    }

    pub(crate) fn set_status(&mut self, status: impl Into<String>) {
        self.status = status.into();
    }

    pub(crate) fn start_turn(&mut self, turn_id: String) {
        self.active_turn_id = Some(turn_id);
        self.status = "running".to_string();
    }

    pub(crate) fn final_answer(&self) -> String {
        self.entries
            .iter()
            .rev()
            .find(|entry| entry.kind == EntryKind::Assistant && !entry.text.is_empty())
            .map(|entry| entry.text.clone())
            .unwrap_or_default()
    }

    pub(crate) fn hydrate_thread(&mut self, thread: Thread) {
        self.entries.clear();
        self.active_turn_id = None;
        self.status = "ready".to_string();

        for turn in thread.turns {
            if turn.status == TurnStatus::InProgress {
                self.active_turn_id = Some(turn.id.clone());
                self.status = "running".to_string();
            }
            for item in turn.items {
                if let Some(entry) = project_item(&item, false) {
                    self.replace_entry(entry);
                }
            }
            if self.active_turn_id.is_none() {
                self.status = turn_status(turn.status).to_string();
            }
        }
    }

    pub(crate) fn apply(&mut self, notification: ServerNotification) {
        match notification {
            ServerNotification::TurnStarted(params) => {
                self.active_turn_id = Some(params.turn.id);
                self.status = "running".to_string();
            }
            ServerNotification::TurnCompleted(params) => {
                self.active_turn_id = None;
                self.settle_running_entries(params.turn.status);
                self.status = turn_status(params.turn.status).to_string();
            }
            ServerNotification::ItemStarted(params) => {
                if let Some(entry) = project_item(&params.item, true) {
                    self.replace_entry(entry);
                }
            }
            ServerNotification::ItemCompleted(params) => {
                if let Some(entry) = project_item(&params.item, false) {
                    self.replace_entry(entry);
                }
            }
            ServerNotification::AgentMessageDelta(params) => {
                self.append_delta(params.item_id, EntryKind::Assistant, params.delta);
            }
            ServerNotification::ReasoningSummaryTextDelta(params) => {
                self.append_delta(params.item_id, EntryKind::Reasoning, params.delta);
            }
            ServerNotification::ReasoningTextDelta(params) => {
                self.append_delta(params.item_id, EntryKind::Reasoning, params.delta);
            }
            ServerNotification::PlanDelta(params) => {
                self.append_delta(params.item_id, EntryKind::Plan, params.delta);
            }
            ServerNotification::CommandExecutionOutputDelta(params) => {
                self.append_delta(params.item_id, EntryKind::Command, params.delta);
            }
            ServerNotification::FileChangePatchUpdated(params) => {
                self.replace_entry(TranscriptEntry {
                    id: params.item_id,
                    kind: EntryKind::Patch,
                    text: format_patch(&params.changes),
                    streaming: true,
                    status: Some(EntryStatus::Running),
                    summary: Vec::new(),
                });
            }
            ServerNotification::TurnDiffUpdated(params) => {
                self.replace_entry(TranscriptEntry {
                    id: format!("turn-{}-diff", params.turn_id),
                    kind: EntryKind::Patch,
                    text: params.diff,
                    streaming: true,
                    status: Some(EntryStatus::Running),
                    summary: Vec::new(),
                });
            }
            ServerNotification::TurnPlanUpdated(params) => {
                let text = params
                    .plan
                    .iter()
                    .map(|step| format!("{} {}", plan_marker(step.status), step.step))
                    .collect::<Vec<_>>()
                    .join("\n");
                self.replace_entry(TranscriptEntry {
                    id: format!("turn-{}-plan", params.turn_id),
                    kind: EntryKind::Plan,
                    text,
                    streaming: true,
                    status: Some(EntryStatus::Running),
                    summary: Vec::new(),
                });
            }
            ServerNotification::Warning(params) => {
                self.push_system(format!("warning: {}", params.message));
            }
            ServerNotification::Error(params) => {
                self.status = if params.will_retry {
                    "retrying".to_string()
                } else {
                    "failed".to_string()
                };
                self.push_system(params.error.message);
            }
            _ => {}
        }
    }

    fn append_delta(&mut self, id: String, kind: EntryKind, delta: String) {
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) {
            entry.text.push_str(&delta);
            entry.streaming = true;
            return;
        }
        self.entries.push(TranscriptEntry {
            id,
            kind,
            text: delta,
            streaming: true,
            status: (kind == EntryKind::Command || kind == EntryKind::Plan)
                .then_some(EntryStatus::Running),
            summary: Vec::new(),
        });
    }

    fn replace_entry(&mut self, entry: TranscriptEntry) {
        if let Some(current) = self
            .entries
            .iter_mut()
            .find(|current| current.id == entry.id)
        {
            *current = entry;
        } else {
            self.entries.push(entry);
        }
    }

    fn push_system(&mut self, text: String) {
        self.entries.push(TranscriptEntry {
            id: format!("system-{}", self.entries.len()),
            kind: EntryKind::System,
            text,
            streaming: false,
            status: None,
            summary: Vec::new(),
        });
    }

    fn settle_running_entries(&mut self, status: TurnStatus) {
        let Some(entry_status) = (match status {
            TurnStatus::Completed => Some(EntryStatus::Completed),
            TurnStatus::Failed => Some(EntryStatus::Failed),
            TurnStatus::Interrupted => Some(EntryStatus::Interrupted),
            TurnStatus::InProgress => None,
        }) else {
            return;
        };

        for entry in &mut self.entries {
            if entry.status == Some(EntryStatus::Running) {
                entry.status = Some(entry_status);
            }
        }
    }
}

fn turn_status(status: TurnStatus) -> &'static str {
    match status {
        TurnStatus::Completed => "ready",
        TurnStatus::Interrupted => "interrupted",
        TurnStatus::Failed => "failed",
        TurnStatus::InProgress => "running",
    }
}

fn plan_marker(status: app_server_protocol::protocol::v2::TurnPlanStepStatus) -> &'static str {
    match status {
        app_server_protocol::protocol::v2::TurnPlanStepStatus::Pending => "[ ]",
        app_server_protocol::protocol::v2::TurnPlanStepStatus::InProgress => "[~]",
        app_server_protocol::protocol::v2::TurnPlanStepStatus::Completed => "[x]",
    }
}

fn project_item(item: &ThreadItem, streaming: bool) -> Option<TranscriptEntry> {
    let (id, kind, text, status, summary) = match item {
        ThreadItem::UserMessage {
            id,
            client_id,
            content,
            ..
        } => (
            client_id.as_ref().unwrap_or(id).clone(),
            EntryKind::User,
            user_input_text(content),
            None,
            Vec::new(),
        ),
        ThreadItem::HookPrompt { id, fragments, .. } => (
            id.clone(),
            EntryKind::System,
            fragments
                .iter()
                .map(|fragment| fragment.text.as_str())
                .collect::<Vec<_>>()
                .join("\n"),
            None,
            Vec::new(),
        ),
        ThreadItem::AgentMessage { id, text, .. } => (
            id.clone(),
            EntryKind::Assistant,
            text.clone(),
            None,
            Vec::new(),
        ),
        ThreadItem::Plan { id, text, .. } => (
            id.clone(),
            EntryKind::Plan,
            text.clone(),
            Some(EntryStatus::Completed),
            Vec::new(),
        ),
        ThreadItem::Reasoning {
            id,
            summary,
            content,
            ..
        } => {
            let text = if summary.is_empty() { content } else { summary }.join("\n");
            (id.clone(), EntryKind::Reasoning, text, None, Vec::new())
        }
        ThreadItem::CommandExecution {
            id,
            command,
            status,
            aggregated_output,
            exit_code,
            duration_ms,
            ..
        } => {
            let mut summary = Vec::new();
            if let Some(exit_code) = exit_code {
                summary.push(format!("exit {exit_code}"));
            }
            if let Some(duration_ms) = duration_ms {
                summary.push(format!("duration {duration_ms}ms"));
            }
            (
                id.clone(),
                EntryKind::Command,
                command_summary(command, aggregated_output.as_deref()),
                Some(command_entry_status(*status)),
                summary,
            )
        }
        ThreadItem::FileChange {
            id,
            changes,
            status,
            ..
        } => (
            id.clone(),
            EntryKind::Patch,
            format_patch(changes),
            Some(patch_entry_status(*status)),
            patch_summary(changes),
        ),
        ThreadItem::McpToolCall {
            id,
            server,
            tool,
            status,
            result,
            error,
            duration_ms,
            ..
        } => (
            id.clone(),
            EntryKind::Mcp,
            format!("{server}.{tool}"),
            Some(mcp_entry_status(*status)),
            mcp_summary(result.as_deref(), error.as_ref(), *duration_ms),
        ),
        ThreadItem::DynamicToolCall {
            id,
            tool,
            status,
            content_items,
            success,
            duration_ms,
            ..
        } => (
            id.clone(),
            EntryKind::Tool,
            tool.clone(),
            Some(dynamic_entry_status(*status)),
            dynamic_summary(content_items.as_deref(), *success, *duration_ms),
        ),
        ThreadItem::CollabAgentToolCall {
            id,
            tool,
            status,
            agents_states,
            ..
        } => (
            id.clone(),
            EntryKind::MultiAgent,
            format!("{tool:?}"),
            Some(collab_entry_status(*status)),
            collab_summary(agents_states),
        ),
        ThreadItem::SubAgentActivity {
            id,
            kind,
            agent_path,
            ..
        } => (
            id.clone(),
            EntryKind::MultiAgent,
            format!("{agent_path} [{kind:?}]"),
            sub_agent_status(*kind),
            Vec::new(),
        ),
        ThreadItem::WebSearch(item) => (
            item.id.clone(),
            EntryKind::Tool,
            format!("web search: {}", item.query.as_deref().unwrap_or("")),
            None,
            Vec::new(),
        ),
        ThreadItem::ImageView { id, path, .. } => (
            id.clone(),
            EntryKind::Tool,
            format!("view image: {path}"),
            None,
            Vec::new(),
        ),
        ThreadItem::Sleep(item) => (
            item.id.clone(),
            EntryKind::Tool,
            format!("sleep: {}ms", item.duration_ms.unwrap_or_default()),
            None,
            Vec::new(),
        ),
        ThreadItem::ImageGeneration(item) => {
            let mut summary = Vec::new();
            if !item.result.trim().is_empty() {
                summary.push(format!("result: {}", compact_text(&item.result)));
            }
            if let Some(path) = item.saved_path.as_deref() {
                summary.push(format!("saved: {path}"));
            }
            if let Some(prompt) = item.revised_prompt.as_deref() {
                summary.push(format!("revised prompt: {}", compact_text(prompt)));
            }
            (
                item.id.clone(),
                EntryKind::Tool,
                "image generation".to_string(),
                image_generation_status(&item.status),
                summary,
            )
        }
        ThreadItem::EnteredReviewMode { id, review, .. } => (
            id.clone(),
            EntryKind::System,
            format!("review started: {review}"),
            Some(EntryStatus::Running),
            Vec::new(),
        ),
        ThreadItem::ExitedReviewMode { id, review, .. } => (
            id.clone(),
            EntryKind::System,
            format!("review completed: {review}"),
            Some(EntryStatus::Completed),
            Vec::new(),
        ),
        ThreadItem::ContextCompaction { id, .. } => (
            id.clone(),
            EntryKind::System,
            "context compacted".to_string(),
            None,
            Vec::new(),
        ),
        ThreadItem::UnknownItem {
            id, upstream_type, ..
        } => (
            id.clone(),
            EntryKind::System,
            format!("unsupported item: {upstream_type}"),
            Some(EntryStatus::Failed),
            Vec::new(),
        ),
    };

    Some(TranscriptEntry {
        id,
        kind,
        text,
        streaming,
        status,
        summary,
    })
}

fn command_summary(command: &str, output: Option<&str>) -> String {
    let mut summary = command.to_string();
    if let Some(output) = output.filter(|output| !output.is_empty()) {
        summary.push('\n');
        summary.push_str(output);
    }
    summary
}

fn patch_summary(changes: &[app_server_protocol::protocol::v2::FileUpdateChange]) -> Vec<String> {
    let mut added = 0;
    let mut deleted = 0;
    let mut updated = 0;
    for change in changes {
        match &change.kind {
            PatchChangeKind::Add => added += 1,
            PatchChangeKind::Delete => deleted += 1,
            PatchChangeKind::Update { .. } => updated += 1,
        }
    }
    let mut details = vec![format!("files: {}", changes.len())];
    if added > 0 {
        details.push(format!("added: {added}"));
    }
    if deleted > 0 {
        details.push(format!("deleted: {deleted}"));
    }
    if updated > 0 {
        details.push(format!("updated: {updated}"));
    }
    details
}

fn mcp_summary(
    result: Option<&app_server_protocol::protocol::v2::McpToolCallResult>,
    error: Option<&app_server_protocol::protocol::v2::McpToolCallError>,
    duration_ms: Option<i64>,
) -> Vec<String> {
    let mut details = Vec::new();
    if let Some(result) = result {
        details.push(format!("result items: {}", result.content.len()));
    }
    if let Some(error) = error {
        details.push(format!("error: {}", compact_text(&error.message)));
    }
    if let Some(duration_ms) = duration_ms {
        details.push(format!("duration {duration_ms}ms"));
    }
    details
}

fn dynamic_summary(
    content_items: Option<&[app_server_protocol::protocol::v2::DynamicToolCallOutputContentItem]>,
    success: Option<bool>,
    duration_ms: Option<i64>,
) -> Vec<String> {
    let mut details = Vec::new();
    if let Some(success) = success {
        details.push(format!("success: {success}"));
    }
    if let Some(content_items) = content_items {
        details.push(format!("content items: {}", content_items.len()));
    }
    if let Some(duration_ms) = duration_ms {
        details.push(format!("duration {duration_ms}ms"));
    }
    details
}

fn collab_summary(
    agents_states: &std::collections::HashMap<
        String,
        app_server_protocol::protocol::v2::CollabAgentState,
    >,
) -> Vec<String> {
    let mut details = vec![format!("agents: {}", agents_states.len())];
    let mut counts = std::collections::BTreeMap::<&'static str, usize>::new();
    for state in agents_states.values() {
        *counts
            .entry(collab_agent_status_label(state.status))
            .or_default() += 1;
    }
    details.extend(
        counts
            .into_iter()
            .map(|(label, count)| format!("{label}: {count}")),
    );
    details
}

fn sub_agent_status(
    kind: app_server_protocol::protocol::v2::SubAgentActivityKind,
) -> Option<EntryStatus> {
    use app_server_protocol::protocol::v2::SubAgentActivityKind;
    Some(match kind {
        SubAgentActivityKind::Started | SubAgentActivityKind::Interacted => EntryStatus::Running,
        SubAgentActivityKind::Interrupted => EntryStatus::Interrupted,
    })
}

fn collab_agent_status_label(
    status: app_server_protocol::protocol::v2::CollabAgentStatus,
) -> &'static str {
    use app_server_protocol::protocol::v2::CollabAgentStatus;
    match status {
        CollabAgentStatus::PendingInit => "pending",
        CollabAgentStatus::Running => "running",
        CollabAgentStatus::Interrupted => "interrupted",
        CollabAgentStatus::Completed => "completed",
        CollabAgentStatus::Errored => "errored",
        CollabAgentStatus::Shutdown => "shutdown",
        CollabAgentStatus::NotFound => "not-found",
    }
}

fn compact_text(value: &str) -> String {
    let value = value.lines().next().unwrap_or(value);
    let mut text = value.chars().take(160).collect::<String>();
    if value.chars().count() > 160 {
        text.push_str("...");
    }
    text
}

fn command_entry_status(status: CommandExecutionStatus) -> EntryStatus {
    match status {
        CommandExecutionStatus::InProgress => EntryStatus::Running,
        CommandExecutionStatus::Completed => EntryStatus::Completed,
        CommandExecutionStatus::Failed => EntryStatus::Failed,
        CommandExecutionStatus::Declined => EntryStatus::Declined,
    }
}

fn patch_entry_status(status: PatchApplyStatus) -> EntryStatus {
    match status {
        PatchApplyStatus::InProgress => EntryStatus::Running,
        PatchApplyStatus::Completed => EntryStatus::Completed,
        PatchApplyStatus::Failed => EntryStatus::Failed,
        PatchApplyStatus::Declined => EntryStatus::Declined,
    }
}

fn mcp_entry_status(status: McpToolCallStatus) -> EntryStatus {
    match status {
        McpToolCallStatus::InProgress => EntryStatus::Running,
        McpToolCallStatus::Completed => EntryStatus::Completed,
        McpToolCallStatus::Failed => EntryStatus::Failed,
    }
}

fn dynamic_entry_status(status: DynamicToolCallStatus) -> EntryStatus {
    match status {
        DynamicToolCallStatus::InProgress => EntryStatus::Running,
        DynamicToolCallStatus::Completed => EntryStatus::Completed,
        DynamicToolCallStatus::Failed => EntryStatus::Failed,
    }
}

fn collab_entry_status(status: CollabAgentToolCallStatus) -> EntryStatus {
    match status {
        CollabAgentToolCallStatus::InProgress => EntryStatus::Running,
        CollabAgentToolCallStatus::Completed => EntryStatus::Completed,
        CollabAgentToolCallStatus::Failed => EntryStatus::Failed,
    }
}

fn image_generation_status(status: &str) -> Option<EntryStatus> {
    match status.to_ascii_lowercase().as_str() {
        "in_progress" | "in-progress" | "running" => Some(EntryStatus::Running),
        "completed" | "succeeded" | "success" => Some(EntryStatus::Completed),
        "failed" | "error" => Some(EntryStatus::Failed),
        "declined" => Some(EntryStatus::Declined),
        _ => None,
    }
}

fn format_patch(changes: &[app_server_protocol::protocol::v2::FileUpdateChange]) -> String {
    changes
        .iter()
        .map(|change| {
            let kind = match &change.kind {
                app_server_protocol::protocol::v2::PatchChangeKind::Add => "added",
                app_server_protocol::protocol::v2::PatchChangeKind::Delete => "deleted",
                app_server_protocol::protocol::v2::PatchChangeKind::Update { .. } => "updated",
            };
            if change.diff.trim().is_empty() {
                format!("{kind} {}", change.path)
            } else {
                format!("{kind} {}\n{}", change.path, change.diff)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn user_input_text(content: &[UserInput]) -> String {
    content
        .iter()
        .map(|input| match input {
            UserInput::Text { text, .. } => text.clone(),
            UserInput::Image { url, .. } => format!("[image: {url}]"),
            UserInput::LocalImage { path, .. } => format!("[image: {path}]"),
            UserInput::Skill { name, .. } => format!("[skill: {name}]"),
            UserInput::Mention { name, .. } => format!("[@{name}]"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
#[path = "projection_tests.rs"]
mod tests;
