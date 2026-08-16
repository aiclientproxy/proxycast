use super::super::soul::locale_copy::HandoffCopy;
use super::super::soul::locale_copy::RuntimeExportCopy;
use super::super::status::agent_turn_status_label;
use super::super::RuntimeCoreError;
use super::metrics::HandoffMetrics;
use super::metrics::HandoffRecentArtifact;
use app_server_protocol::AgentSessionReadResponse;
use app_server_protocol::AgentSessionReviewDecision;
use app_server_protocol::AgentSessionReviewDecisionSaveParams;
use serde_json::json;
use std::fmt::Write as _;
use std::path::Component;
use std::path::Path;

pub(super) fn sanitized_workspace_root(workspace_root: &Path) -> String {
    let mut components = workspace_root
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => value.to_str().map(ToString::to_string),
            _ => None,
        })
        .collect::<Vec<_>>();
    if components.is_empty() {
        return workspace_root.to_string_lossy().to_string();
    }
    if components.len() > 3 {
        components = components.split_off(components.len() - 3);
    }
    components.join("/")
}

fn json_pretty(value: serde_json::Value, label: &str) -> Result<String, RuntimeCoreError> {
    serde_json::to_string_pretty(&value)
        .map_err(|error| RuntimeCoreError::Backend(format!("failed to serialize {label}: {error}")))
}

pub(super) fn build_replay_input_json(
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    recent_artifacts: &[HandoffRecentArtifact],
    exported_at: &str,
) -> Result<String, RuntimeCoreError> {
    json_pretty(
        json!({
            "schemaVersion": "agent-session-replay-case.v1",
            "sessionId": read.session.session_id,
            "threadId": read.session.thread_id,
            "exportedAt": exported_at,
            "threadStatus": metrics.thread_status.as_str(),
            "latestTurnStatus": metrics.latest_turn_status,
            "turns": read.turns.iter().map(|turn| {
                json!({
                    "turnId": turn.turn_id,
                    "status": agent_turn_status_label(turn.status),
                    "startedAt": turn.started_at,
                    "completedAt": turn.completed_at,
                })
            }).collect::<Vec<_>>(),
            "detail": read.detail,
            "recentArtifacts": recent_artifacts.iter().map(|artifact| {
                json!({
                    "title": artifact.title,
                    "kind": artifact.kind,
                    "path": artifact.path,
                })
            }).collect::<Vec<_>>(),
        }),
        "replay input",
    )
}

pub(super) fn build_replay_expected_json(
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    exported_at: &str,
) -> Result<String, RuntimeCoreError> {
    json_pretty(
        json!({
            "schemaVersion": "agent-session-replay-expected.v1",
            "sessionId": read.session.session_id,
            "threadId": read.session.thread_id,
            "exportedAt": exported_at,
            "expected": {
                "terminalThreadStatus": metrics.thread_status.as_str(),
                "latestTurnStatus": metrics.latest_turn_status,
                "pendingRequestCount": metrics.pending_request_count,
                "queuedTurnCount": metrics.queued_turn_count,
            }
        }),
        "replay expected",
    )
}

pub(super) fn build_replay_grader_markdown(
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    exported_at: &str,
    copy: &RuntimeExportCopy,
) -> String {
    let mut content = String::new();
    let _ = writeln!(content, "# {}", copy.replay.grader_heading);
    let _ = writeln!(content);
    let _ = writeln!(content, "- sessionId: `{}`", read.session.session_id);
    let _ = writeln!(content, "- threadId: `{}`", read.session.thread_id);
    let _ = writeln!(content, "- exportedAt: `{exported_at}`");
    let _ = writeln!(
        content,
        "- threadStatus: `{}`",
        metrics.thread_status.as_str()
    );
    let _ = writeln!(content);
    write_generation_brief_boundary(&mut content, copy);
    let _ = writeln!(content);
    let _ = writeln!(content, "## {}", copy.replay.checks_heading);
    let pending_request_check = copy
        .replay
        .pending_request_check
        .replace("{count}", &metrics.pending_request_count.to_string());
    let queued_turn_check = copy
        .replay
        .queued_turn_check
        .replace("{count}", &metrics.queued_turn_count.to_string());
    let _ = writeln!(content, "- {pending_request_check}");
    let _ = writeln!(content, "- {queued_turn_check}");
    let _ = writeln!(content, "- {}", copy.replay.read_model_check);
    content
}

pub(super) fn build_replay_evidence_links_json(
    session_id: &str,
    handoff_relative_root: &str,
    evidence_relative_root: &str,
    recent_artifacts: &[HandoffRecentArtifact],
    exported_at: &str,
) -> Result<String, RuntimeCoreError> {
    json_pretty(
        json!({
            "schemaVersion": "agent-session-replay-evidence-links.v1",
            "sessionId": session_id,
            "exportedAt": exported_at,
            "handoffBundleRelativeRoot": handoff_relative_root,
            "evidencePackRelativeRoot": evidence_relative_root,
            "recentArtifacts": recent_artifacts.iter().map(|artifact| {
                json!({
                    "title": artifact.title,
                    "kind": artifact.kind,
                    "path": artifact.path,
                })
            }).collect::<Vec<_>>(),
        }),
        "replay evidence links",
    )
}

pub(super) fn build_analysis_brief_markdown(
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    recent_artifacts: &[HandoffRecentArtifact],
    exported_at: &str,
    copy: &RuntimeExportCopy,
) -> String {
    let mut content = String::new();
    let _ = writeln!(content, "# {}", copy.analysis.markdown_heading);
    let _ = writeln!(content);
    let _ = writeln!(content, "- sessionId: `{}`", read.session.session_id);
    let _ = writeln!(content, "- threadId: `{}`", read.session.thread_id);
    let _ = writeln!(
        content,
        "- threadStatus: `{}`",
        metrics.thread_status.as_str()
    );
    if let Some(latest_turn_status) = metrics.latest_turn_status.as_deref() {
        let _ = writeln!(content, "- latestTurnStatus: `{latest_turn_status}`");
    }
    let _ = writeln!(content, "- exportedAt: `{exported_at}`");
    let _ = writeln!(content);
    write_generation_brief_boundary(&mut content, copy);
    let _ = writeln!(content);
    let _ = writeln!(content, "## {}", copy.analysis.focus_heading);
    let _ = writeln!(content);
    let _ = writeln!(content, "- {}", copy.analysis.focus_current_read_model);
    let _ = writeln!(content, "- {}", copy.analysis.focus_no_legacy);
    let _ = writeln!(content);
    write_handoff_todo_summary(&mut content, metrics, copy.handoff);
    let _ = writeln!(content);
    write_handoff_recent_artifacts(&mut content, recent_artifacts, copy.handoff);
    content
}

pub(super) fn build_analysis_context_json(
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    workspace_root: &Path,
    replay_relative_root: &str,
    handoff_relative_root: &str,
    evidence_relative_root: &str,
    exported_at: &str,
) -> Result<String, RuntimeCoreError> {
    json_pretty(
        json!({
            "schemaVersion": "agent-session-analysis-handoff.v1",
            "sessionId": read.session.session_id,
            "threadId": read.session.thread_id,
            "workspaceId": read.session.workspace_id,
            "workspaceRoot": workspace_root.to_string_lossy(),
            "sanitizedWorkspaceRoot": sanitized_workspace_root(workspace_root),
            "exportedAt": exported_at,
            "threadStatus": metrics.thread_status.as_str(),
            "latestTurnStatus": metrics.latest_turn_status,
            "pendingRequestCount": metrics.pending_request_count,
            "queuedTurnCount": metrics.queued_turn_count,
            "handoffBundleRelativeRoot": handoff_relative_root,
            "evidencePackRelativeRoot": evidence_relative_root,
            "replayCaseRelativeRoot": replay_relative_root,
            "detail": read.detail,
        }),
        "analysis context",
    )
}

pub(super) fn build_analysis_copy_prompt(
    read: &AgentSessionReadResponse,
    analysis_relative_root: &str,
    replay_relative_root: &str,
    copy: &RuntimeExportCopy,
) -> String {
    copy.analysis_copy_prompt(
        &read.session.session_id,
        analysis_relative_root,
        replay_relative_root,
    )
}

pub(super) fn default_review_decision(copy: &RuntimeExportCopy) -> AgentSessionReviewDecision {
    AgentSessionReviewDecision {
        decision_status: "pending_review".to_string(),
        decision_summary: String::new(),
        chosen_fix_strategy: String::new(),
        risk_level: "unknown".to_string(),
        risk_tags: Vec::new(),
        human_reviewer: String::new(),
        followup_actions: Vec::new(),
        regression_requirements: vec![copy
            .review
            .default_decision_regression_requirement
            .to_string()],
        notes: String::new(),
    }
}

pub(super) fn review_decision_from_save_params(
    params: &AgentSessionReviewDecisionSaveParams,
) -> AgentSessionReviewDecision {
    AgentSessionReviewDecision {
        decision_status: normalize_review_decision_status(params.decision_status.as_str()),
        decision_summary: params.decision_summary.trim().to_string(),
        chosen_fix_strategy: params.chosen_fix_strategy.trim().to_string(),
        risk_level: normalize_review_risk_level(params.risk_level.as_str()),
        risk_tags: trim_string_vec(&params.risk_tags),
        human_reviewer: params.human_reviewer.trim().to_string(),
        followup_actions: trim_string_vec(&params.followup_actions),
        regression_requirements: trim_string_vec(&params.regression_requirements),
        notes: params.notes.trim().to_string(),
    }
}

fn normalize_review_decision_status(value: &str) -> String {
    match value.trim() {
        "accepted" | "deferred" | "rejected" | "needs_more_evidence" | "pending_review" => {
            value.trim().to_string()
        }
        _ => "pending_review".to_string(),
    }
}

fn normalize_review_risk_level(value: &str) -> String {
    match value.trim() {
        "low" | "medium" | "high" | "unknown" => value.trim().to_string(),
        _ => "unknown".to_string(),
    }
}

fn trim_string_vec(values: &[String]) -> Vec<String> {
    values
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(ToString::to_string)
        .collect()
}

pub(super) fn build_review_decision_markdown(
    read: &AgentSessionReadResponse,
    decision: &AgentSessionReviewDecision,
    analysis_relative_root: &str,
    replay_relative_root: &str,
    exported_at: &str,
    copy: &RuntimeExportCopy,
) -> String {
    let mut content = String::new();
    let _ = writeln!(content, "# {}", copy.review.markdown_heading);
    let _ = writeln!(content);
    let _ = writeln!(content, "- sessionId: `{}`", read.session.session_id);
    let _ = writeln!(content, "- threadId: `{}`", read.session.thread_id);
    let _ = writeln!(content, "- exportedAt: `{exported_at}`");
    let _ = writeln!(content, "- decisionStatus: `{}`", decision.decision_status);
    let _ = writeln!(content, "- riskLevel: `{}`", decision.risk_level);
    let _ = writeln!(content, "- analysis: `{analysis_relative_root}`");
    let _ = writeln!(content, "- replay: `{replay_relative_root}`");
    let _ = writeln!(content);
    write_generation_brief_boundary(&mut content, copy);
    let _ = writeln!(content);
    let _ = writeln!(content, "## {}", copy.review.summary_heading);
    let _ = writeln!(
        content,
        "{}",
        if decision.decision_summary.is_empty() {
            copy.review.pending_summary
        } else {
            decision.decision_summary.as_str()
        }
    );
    let _ = writeln!(content);
    let _ = writeln!(content, "## {}", copy.review.followup_heading);
    if decision.followup_actions.is_empty() {
        let _ = writeln!(content, "- {}", copy.review.no_followup);
    } else {
        for action in &decision.followup_actions {
            let _ = writeln!(content, "- {action}");
        }
    }
    let _ = writeln!(content);
    let _ = writeln!(content, "## {}", copy.review.regression_heading);
    if decision.regression_requirements.is_empty() {
        let _ = writeln!(content, "- {}", copy.review.default_regression_requirement);
    } else {
        for item in &decision.regression_requirements {
            let _ = writeln!(content, "- {item}");
        }
    }
    if !decision.notes.is_empty() {
        let _ = writeln!(content);
        let _ = writeln!(content, "## {}", copy.review.notes_heading);
        let _ = writeln!(content, "{}", decision.notes);
    }
    content
}

pub(super) fn build_review_decision_json(
    read: &AgentSessionReadResponse,
    decision: &AgentSessionReviewDecision,
    analysis_relative_root: &str,
    replay_relative_root: &str,
    exported_at: &str,
) -> Result<String, RuntimeCoreError> {
    json_pretty(
        json!({
            "schemaVersion": "agent-session-review-decision.v1",
            "sessionId": read.session.session_id,
            "threadId": read.session.thread_id,
            "exportedAt": exported_at,
            "analysisRelativeRoot": analysis_relative_root,
            "replayCaseRelativeRoot": replay_relative_root,
            "decision": decision,
        }),
        "review decision",
    )
}

pub(super) fn build_handoff_plan_markdown(
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    recent_artifacts: &[HandoffRecentArtifact],
    exported_at: &str,
    copy: HandoffCopy,
) -> String {
    let mut content = String::new();
    let _ = writeln!(content, "# {}", copy.plan_title);
    let _ = writeln!(content);
    write_handoff_header(&mut content, read, metrics, exported_at, copy);
    let _ = writeln!(content);
    write_handoff_todo_summary(&mut content, metrics, copy);
    let _ = writeln!(content);
    write_handoff_recent_artifacts(&mut content, recent_artifacts, copy);
    let _ = writeln!(content);
    let _ = writeln!(content, "## {}", copy.next_step_title);
    let _ = writeln!(content);
    let _ = writeln!(content, "- {}", copy.next_step_body);
    content
}

pub(super) fn build_handoff_markdown(
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    recent_artifacts: &[HandoffRecentArtifact],
    exported_at: &str,
    copy: HandoffCopy,
) -> String {
    let mut content = String::new();
    let _ = writeln!(content, "# {}", copy.handoff_title);
    let _ = writeln!(content);
    write_handoff_header(&mut content, read, metrics, exported_at, copy);
    let _ = writeln!(content);
    let _ = writeln!(content, "## {}", copy.next_step_title);
    let _ = writeln!(content);
    let _ = writeln!(content, "- {}", copy.next_step_body);
    let _ = writeln!(content);
    write_handoff_recent_artifacts(&mut content, recent_artifacts, copy);
    content
}

pub(super) fn build_handoff_review_summary_markdown(
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    recent_artifacts: &[HandoffRecentArtifact],
    exported_at: &str,
    copy: HandoffCopy,
) -> String {
    let mut content = String::new();
    let _ = writeln!(content, "# {}", copy.review_summary_title);
    let _ = writeln!(content);
    write_handoff_header(&mut content, read, metrics, exported_at, copy);
    let _ = writeln!(content);
    let _ = writeln!(content, "{}", copy.review_note);
    let _ = writeln!(content);
    write_handoff_todo_summary(&mut content, metrics, copy);
    let _ = writeln!(content);
    write_handoff_recent_artifacts(&mut content, recent_artifacts, copy);
    content
}

pub(super) fn build_rollout_summary_candidate_markdown(
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    recent_artifacts: &[HandoffRecentArtifact],
    exported_at: &str,
    export_relative_root: &str,
    export_kind: &str,
    copy: &RuntimeExportCopy,
) -> String {
    let mut content = String::new();
    let _ = writeln!(
        content,
        "{}",
        copy.rollout_opening(&read.session.session_id, export_kind)
    );
    let _ = writeln!(content);
    let _ = writeln!(content, "## {}", copy.rollout.runtime_facts_heading);
    let _ = writeln!(content, "- sessionId: `{}`", read.session.session_id);
    let _ = writeln!(content, "- threadId: `{}`", read.session.thread_id);
    let _ = writeln!(content, "- exportKind: `{export_kind}`");
    let _ = writeln!(content, "- exportRoot: `{export_relative_root}`");
    let _ = writeln!(
        content,
        "- threadStatus: `{}`",
        metrics.thread_status.as_str()
    );
    if let Some(latest_turn_status) = metrics.latest_turn_status.as_deref() {
        let _ = writeln!(content, "- latestTurnStatus: `{latest_turn_status}`");
    }
    let _ = writeln!(content, "- exportedAt: `{exported_at}`");
    let _ = writeln!(content);
    write_generation_brief_boundary(&mut content, copy);
    let _ = writeln!(content);
    let _ = writeln!(content, "## {}", copy.rollout.candidate_memory_heading);
    let _ = writeln!(
        content,
        "- {} `{export_relative_root}`",
        copy.rollout.review_before_promoting
    );
    if metrics.todo_pending > 0 || metrics.todo_in_progress > 0 {
        let _ = writeln!(
            content,
            "- {}: {} {}, {} {}.",
            copy.rollout.pending_work_label,
            metrics.todo_pending,
            copy.rollout.pending_label,
            metrics.todo_in_progress,
            copy.rollout.in_progress_label
        );
    }
    if !recent_artifacts.is_empty() {
        let _ = writeln!(content);
        let _ = writeln!(content, "## {}", copy.rollout.referenced_artifacts_heading);
        for artifact in recent_artifacts {
            let _ = writeln!(
                content,
                "- {} `{}` ({})",
                artifact.title, artifact.path, artifact.kind
            );
        }
    }
    content
}

pub(super) fn build_handoff_progress_json(
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    recent_artifacts: &[HandoffRecentArtifact],
    workspace_root: &Path,
    exported_at: &str,
) -> Result<String, RuntimeCoreError> {
    let recent_artifacts = recent_artifacts
        .iter()
        .map(|artifact| {
            json!({
                "title": artifact.title,
                "kind": artifact.kind,
                "path": artifact.path,
            })
        })
        .collect::<Vec<_>>();
    let turns = read
        .turns
        .iter()
        .map(|turn| {
            json!({
                "turnId": turn.turn_id,
                "status": agent_turn_status_label(turn.status),
                "startedAt": turn.started_at,
                "completedAt": turn.completed_at,
            })
        })
        .collect::<Vec<_>>();
    serde_json::to_string_pretty(&json!({
        "schemaVersion": "agent-session-handoff-bundle.v1",
        "sessionId": read.session.session_id,
        "threadId": read.session.thread_id,
        "workspaceId": read.session.workspace_id,
        "workspaceRoot": workspace_root.to_string_lossy(),
        "exportedAt": exported_at,
        "status": {
            "thread": metrics.thread_status.as_str(),
            "latestTurn": metrics.latest_turn_status,
        },
        "counts": {
            "pendingRequest": metrics.pending_request_count,
            "queuedTurn": metrics.queued_turn_count,
            "activeSubagent": metrics.active_subagent_count,
        },
        "todos": {
            "total": metrics.todo_total,
            "pending": metrics.todo_pending,
            "inProgress": metrics.todo_in_progress,
            "completed": metrics.todo_completed,
        },
        "turns": turns,
        "recentArtifacts": recent_artifacts,
    }))
    .map_err(|error| {
        RuntimeCoreError::Backend(format!("failed to serialize handoff progress: {error}"))
    })
}

fn write_handoff_header(
    content: &mut String,
    read: &AgentSessionReadResponse,
    metrics: &HandoffMetrics,
    exported_at: &str,
    copy: HandoffCopy,
) {
    let _ = writeln!(
        content,
        "- {}: `{}`",
        copy.session_label, read.session.session_id
    );
    let _ = writeln!(
        content,
        "- {}: `{}`",
        copy.thread_label, read.session.thread_id
    );
    let _ = writeln!(
        content,
        "- {}: `{}`",
        copy.status_label,
        metrics.thread_status.as_str()
    );
    if let Some(latest_turn_status) = metrics.latest_turn_status.as_deref() {
        let _ = writeln!(content, "- latestTurnStatus: `{latest_turn_status}`");
    }
    let _ = writeln!(content, "- {}: `{}`", copy.exported_at_label, exported_at);
}

fn write_generation_brief_boundary(content: &mut String, copy: &RuntimeExportCopy) {
    let _ = writeln!(content, "## {}", copy.generation_brief.heading);
    let _ = writeln!(
        content,
        "- {}: `{}`",
        copy.generation_brief.voice_source_label,
        copy.formal_artifact_voice_source()
    );
    let _ = writeln!(content, "- {}", copy.generation_brief.default_voice);
    let _ = writeln!(content, "- {}", copy.generation_brief.explicit_voice);
    let _ = writeln!(content, "- {}", copy.generation_brief.fidelity_rule);
}

fn write_handoff_todo_summary(content: &mut String, metrics: &HandoffMetrics, copy: HandoffCopy) {
    let _ = writeln!(content, "## {}", copy.todo_summary_title);
    let _ = writeln!(content);
    let _ = writeln!(content, "- total: {}", metrics.todo_total);
    let _ = writeln!(content, "- pending: {}", metrics.todo_pending);
    let _ = writeln!(content, "- inProgress: {}", metrics.todo_in_progress);
    let _ = writeln!(content, "- completed: {}", metrics.todo_completed);
    let _ = writeln!(
        content,
        "- pendingRequests: {}",
        metrics.pending_request_count
    );
    let _ = writeln!(content, "- queuedTurns: {}", metrics.queued_turn_count);
    let _ = writeln!(
        content,
        "- activeSubagents: {}",
        metrics.active_subagent_count
    );
}

fn write_handoff_recent_artifacts(
    content: &mut String,
    recent_artifacts: &[HandoffRecentArtifact],
    copy: HandoffCopy,
) {
    let _ = writeln!(content, "## {}", copy.recent_artifacts_title);
    let _ = writeln!(content);
    if recent_artifacts.is_empty() {
        let _ = writeln!(content, "- {}", copy.no_recent_artifacts);
        return;
    }
    for artifact in recent_artifacts {
        let _ = writeln!(
            content,
            "- `{}` {} ({})",
            artifact.path, artifact.title, artifact.kind
        );
    }
}
