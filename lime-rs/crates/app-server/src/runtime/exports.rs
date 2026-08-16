mod builders;
mod files;
mod metrics;
mod rollout;

use self::builders::*;
use self::files::*;
use self::metrics::*;
use super::soul::locale_copy::runtime_export_copy;
use super::timestamp;
use super::RuntimeCore;
use super::RuntimeCoreError;
use app_server_protocol::AgentSessionAnalysisHandoffExportParams;
use app_server_protocol::AgentSessionAnalysisHandoffExportResponse;
use app_server_protocol::AgentSessionHandoffBundleExportParams;
use app_server_protocol::AgentSessionHandoffBundleExportResponse;
use app_server_protocol::AgentSessionReadParams;
use app_server_protocol::AgentSessionReplayCaseExportParams;
use app_server_protocol::AgentSessionReplayCaseExportResponse;
use app_server_protocol::AgentSessionReviewDecision;
use app_server_protocol::AgentSessionReviewDecisionSaveParams;
use app_server_protocol::AgentSessionReviewDecisionTemplateExportParams;
use app_server_protocol::AgentSessionReviewDecisionTemplateExportResponse;
use std::fs;

const HANDOFF_BUNDLE_RELATIVE_ROOT: &str = ".lime/harness/sessions";
const HANDOFF_PLAN_FILE_NAME: &str = "plan.md";
const HANDOFF_PROGRESS_FILE_NAME: &str = "progress.json";
const HANDOFF_FILE_NAME: &str = "handoff.md";
const HANDOFF_REVIEW_SUMMARY_FILE_NAME: &str = "review-summary.md";
const HANDOFF_RECENT_ARTIFACT_LIMIT: usize = 8;
const REPLAY_CASE_INPUT_FILE_NAME: &str = "input.json";
const REPLAY_CASE_EXPECTED_FILE_NAME: &str = "expected.json";
const REPLAY_CASE_GRADER_FILE_NAME: &str = "grader.md";
const REPLAY_CASE_EVIDENCE_LINKS_FILE_NAME: &str = "evidence-links.json";
const ANALYSIS_BRIEF_FILE_NAME: &str = "analysis-brief.md";
const ANALYSIS_CONTEXT_FILE_NAME: &str = "analysis-context.json";
const REVIEW_DECISION_MARKDOWN_FILE_NAME: &str = "review-decision.md";
const REVIEW_DECISION_JSON_FILE_NAME: &str = "review-decision.json";

impl RuntimeCore {
    pub async fn export_handoff_bundle(
        &self,
        params: AgentSessionHandoffBundleExportParams,
    ) -> Result<AgentSessionHandoffBundleExportResponse, RuntimeCoreError> {
        let session_id = params.session_id.trim().to_string();
        if session_id.is_empty() {
            return Err(RuntimeCoreError::Backend(
                "sessionId is required for agentSession/handoffBundle/export".to_string(),
            ));
        }
        validate_handoff_session_id(&session_id)?;

        let read = self
            .read_session_current(AgentSessionReadParams {
                session_id: session_id.clone(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .await?;
        let workspace_root = resolve_handoff_workspace_root(&read)?;
        let workspace_root = workspace_root
            .canonicalize()
            .map_err(|error| RuntimeCoreError::Backend(format!(
                "workspaceRoot must be an existing directory for agentSession/handoffBundle/export: {} ({error})",
                workspace_root.display()
            )))?;
        if !workspace_root.is_dir() {
            return Err(RuntimeCoreError::Backend(format!(
                "workspaceRoot must be a directory for agentSession/handoffBundle/export: {}",
                workspace_root.display()
            )));
        }

        let copy = runtime_export_copy(params.locale.as_deref());
        let exported_at = timestamp();
        let bundle_relative_root = format!("{HANDOFF_BUNDLE_RELATIVE_ROOT}/{session_id}");
        let bundle_absolute_root = workspace_root
            .join(".lime")
            .join("harness")
            .join("sessions")
            .join(&session_id);
        fs::create_dir_all(&bundle_absolute_root).map_err(|error| {
            RuntimeCoreError::Backend(format!(
                "failed to create handoff bundle directory {}: {error}",
                bundle_absolute_root.display()
            ))
        })?;

        let metrics = handoff_metrics(self, &read).await?;
        let recent_artifacts = handoff_recent_artifacts(&read);
        let artifacts = vec![
            write_handoff_bundle_file(
                &bundle_absolute_root,
                &bundle_relative_root,
                HANDOFF_PLAN_FILE_NAME,
                "plan",
                copy.handoff.plan_title,
                build_handoff_plan_markdown(
                    &read,
                    &metrics,
                    &recent_artifacts,
                    &exported_at,
                    copy.handoff,
                ),
            )?,
            write_handoff_bundle_file(
                &bundle_absolute_root,
                &bundle_relative_root,
                HANDOFF_PROGRESS_FILE_NAME,
                "progress",
                copy.handoff.progress_title,
                build_handoff_progress_json(
                    &read,
                    &metrics,
                    &recent_artifacts,
                    &workspace_root,
                    &exported_at,
                )?,
            )?,
            write_handoff_bundle_file(
                &bundle_absolute_root,
                &bundle_relative_root,
                HANDOFF_FILE_NAME,
                "handoff",
                copy.handoff.handoff_title,
                build_handoff_markdown(
                    &read,
                    &metrics,
                    &recent_artifacts,
                    &exported_at,
                    copy.handoff,
                ),
            )?,
            write_handoff_bundle_file(
                &bundle_absolute_root,
                &bundle_relative_root,
                HANDOFF_REVIEW_SUMMARY_FILE_NAME,
                "review_summary",
                copy.handoff.review_summary_title,
                build_handoff_review_summary_markdown(
                    &read,
                    &metrics,
                    &recent_artifacts,
                    &exported_at,
                    copy.handoff,
                ),
            )?,
        ];

        self.write_export_rollout_summary_candidate(
            &read,
            &metrics,
            &recent_artifacts,
            &workspace_root,
            &exported_at,
            &bundle_relative_root,
            "handoff_bundle",
            "agentSession/handoffBundle/export",
            &copy,
        )
        .await?;

        Ok(AgentSessionHandoffBundleExportResponse {
            session_id: read.session.session_id,
            thread_id: read.session.thread_id,
            workspace_id: read.session.workspace_id,
            workspace_root: workspace_root.to_string_lossy().to_string(),
            bundle_relative_root,
            bundle_absolute_root: bundle_absolute_root.to_string_lossy().to_string(),
            exported_at,
            thread_status: metrics.thread_status,
            latest_turn_status: metrics.latest_turn_status,
            pending_request_count: metrics.pending_request_count,
            queued_turn_count: metrics.queued_turn_count,
            active_subagent_count: metrics.active_subagent_count,
            todo_total: metrics.todo_total,
            todo_pending: metrics.todo_pending,
            todo_in_progress: metrics.todo_in_progress,
            todo_completed: metrics.todo_completed,
            artifacts,
        })
    }

    pub async fn export_replay_case(
        &self,
        params: AgentSessionReplayCaseExportParams,
    ) -> Result<AgentSessionReplayCaseExportResponse, RuntimeCoreError> {
        const METHOD: &str = "agentSession/replayCase/export";
        let session_id = required_runtime_export_session_id(&params.session_id, METHOD)?;
        let read = self
            .read_session_current(AgentSessionReadParams {
                session_id: session_id.clone(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .await?;
        let workspace_root = canonical_runtime_export_workspace_root(&read, METHOD)?;
        let copy = runtime_export_copy(params.locale.as_deref());
        let exported_at = timestamp();
        let metrics = handoff_metrics(self, &read).await?;
        let recent_artifacts = handoff_recent_artifacts(&read);
        let (handoff_relative_root, evidence_relative_root, _) =
            runtime_export_base_roots(&session_id);
        let (replay_relative_root, replay_absolute_root) =
            runtime_export_root(&workspace_root, &session_id, "replay");
        ensure_runtime_export_root(&replay_absolute_root)?;

        let artifacts = vec![
            write_runtime_export_file(
                &replay_absolute_root,
                &replay_relative_root,
                REPLAY_CASE_INPUT_FILE_NAME,
                "input",
                copy.replay.input_title,
                build_replay_input_json(&read, &metrics, &recent_artifacts, &exported_at)?,
            )?,
            write_runtime_export_file(
                &replay_absolute_root,
                &replay_relative_root,
                REPLAY_CASE_EXPECTED_FILE_NAME,
                "expected",
                copy.replay.expected_title,
                build_replay_expected_json(&read, &metrics, &exported_at)?,
            )?,
            write_runtime_export_file(
                &replay_absolute_root,
                &replay_relative_root,
                REPLAY_CASE_GRADER_FILE_NAME,
                "grader",
                copy.replay.grader_title,
                build_replay_grader_markdown(&read, &metrics, &exported_at, &copy),
            )?,
            write_runtime_export_file(
                &replay_absolute_root,
                &replay_relative_root,
                REPLAY_CASE_EVIDENCE_LINKS_FILE_NAME,
                "evidence_links",
                copy.replay.evidence_links_title,
                build_replay_evidence_links_json(
                    &session_id,
                    &handoff_relative_root,
                    &evidence_relative_root,
                    &recent_artifacts,
                    &exported_at,
                )?,
            )?,
        ];

        self.write_export_rollout_summary_candidate(
            &read,
            &metrics,
            &recent_artifacts,
            &workspace_root,
            &exported_at,
            &replay_relative_root,
            "replay_case",
            METHOD,
            &copy,
        )
        .await?;

        Ok(AgentSessionReplayCaseExportResponse {
            session_id: read.session.session_id,
            thread_id: read.session.thread_id,
            workspace_id: read.session.workspace_id,
            workspace_root: workspace_root.to_string_lossy().to_string(),
            replay_relative_root,
            replay_absolute_root: replay_absolute_root.to_string_lossy().to_string(),
            handoff_bundle_relative_root: handoff_relative_root,
            evidence_pack_relative_root: evidence_relative_root,
            exported_at,
            thread_status: metrics.thread_status,
            latest_turn_status: metrics.latest_turn_status,
            pending_request_count: metrics.pending_request_count,
            queued_turn_count: metrics.queued_turn_count,
            linked_handoff_artifact_count: 0,
            linked_evidence_artifact_count: recent_artifacts.len(),
            recent_artifact_count: recent_artifacts.len(),
            artifacts,
        })
    }

    pub async fn export_analysis_handoff(
        &self,
        params: AgentSessionAnalysisHandoffExportParams,
    ) -> Result<AgentSessionAnalysisHandoffExportResponse, RuntimeCoreError> {
        const METHOD: &str = "agentSession/analysisHandoff/export";
        let session_id = required_runtime_export_session_id(&params.session_id, METHOD)?;
        let read = self
            .read_session_current(AgentSessionReadParams {
                session_id: session_id.clone(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .await?;
        let workspace_root = canonical_runtime_export_workspace_root(&read, METHOD)?;
        let copy = runtime_export_copy(params.locale.as_deref());
        let exported_at = timestamp();
        let metrics = handoff_metrics(self, &read).await?;
        let recent_artifacts = handoff_recent_artifacts(&read);
        let (handoff_relative_root, evidence_relative_root, replay_relative_root) =
            runtime_export_base_roots(&session_id);
        let (analysis_relative_root, analysis_absolute_root) =
            runtime_export_root(&workspace_root, &session_id, "analysis");
        ensure_runtime_export_root(&analysis_absolute_root)?;

        let title = copy.analysis.response_title.to_string();
        let copy_prompt = build_analysis_copy_prompt(
            &read,
            &analysis_relative_root,
            &replay_relative_root,
            &copy,
        );
        let artifacts = vec![
            write_runtime_export_file(
                &analysis_absolute_root,
                &analysis_relative_root,
                ANALYSIS_BRIEF_FILE_NAME,
                "analysis_brief",
                copy.analysis.brief_artifact_title,
                build_analysis_brief_markdown(
                    &read,
                    &metrics,
                    &recent_artifacts,
                    &exported_at,
                    &copy,
                ),
            )?,
            write_runtime_export_file(
                &analysis_absolute_root,
                &analysis_relative_root,
                ANALYSIS_CONTEXT_FILE_NAME,
                "analysis_context",
                copy.analysis.context_artifact_title,
                build_analysis_context_json(
                    &read,
                    &metrics,
                    &workspace_root,
                    &replay_relative_root,
                    &handoff_relative_root,
                    &evidence_relative_root,
                    &exported_at,
                )?,
            )?,
        ];

        self.write_export_rollout_summary_candidate(
            &read,
            &metrics,
            &recent_artifacts,
            &workspace_root,
            &exported_at,
            &analysis_relative_root,
            "analysis_handoff",
            METHOD,
            &copy,
        )
        .await?;

        Ok(AgentSessionAnalysisHandoffExportResponse {
            session_id: read.session.session_id,
            thread_id: read.session.thread_id,
            workspace_id: read.session.workspace_id,
            workspace_root: workspace_root.to_string_lossy().to_string(),
            sanitized_workspace_root: sanitized_workspace_root(&workspace_root),
            analysis_relative_root,
            analysis_absolute_root: analysis_absolute_root.to_string_lossy().to_string(),
            handoff_bundle_relative_root: handoff_relative_root,
            evidence_pack_relative_root: evidence_relative_root,
            replay_case_relative_root: replay_relative_root,
            exported_at,
            thread_status: metrics.thread_status,
            latest_turn_status: metrics.latest_turn_status,
            pending_request_count: metrics.pending_request_count,
            queued_turn_count: metrics.queued_turn_count,
            title,
            copy_prompt,
            artifacts,
        })
    }

    pub async fn export_review_decision_template(
        &self,
        params: AgentSessionReviewDecisionTemplateExportParams,
    ) -> Result<AgentSessionReviewDecisionTemplateExportResponse, RuntimeCoreError> {
        let copy = runtime_export_copy(params.locale.as_deref());
        self.sync_review_decision(
            params.session_id,
            params.locale,
            default_review_decision(&copy),
            false,
        )
        .await
    }

    pub async fn save_review_decision(
        &self,
        params: AgentSessionReviewDecisionSaveParams,
    ) -> Result<AgentSessionReviewDecisionTemplateExportResponse, RuntimeCoreError> {
        let decision = review_decision_from_save_params(&params);
        self.sync_review_decision(params.session_id, params.locale, decision, true)
            .await
    }

    async fn sync_review_decision(
        &self,
        session_id: String,
        locale: Option<String>,
        decision: AgentSessionReviewDecision,
        saving: bool,
    ) -> Result<AgentSessionReviewDecisionTemplateExportResponse, RuntimeCoreError> {
        let method = if saving {
            "agentSession/reviewDecision/save"
        } else {
            "agentSession/reviewDecisionTemplate/export"
        };
        let session_id = required_runtime_export_session_id(&session_id, method)?;
        let read = self
            .read_session_current(AgentSessionReadParams {
                session_id: session_id.clone(),
                history_limit: None,
                history_offset: None,
                history_before_message_id: None,
            })
            .await?;
        let workspace_root = canonical_runtime_export_workspace_root(&read, method)?;
        let copy = runtime_export_copy(locale.as_deref());
        let exported_at = timestamp();
        let metrics = handoff_metrics(self, &read).await?;
        let recent_artifacts = handoff_recent_artifacts(&read);
        let (handoff_relative_root, evidence_relative_root, replay_relative_root) =
            runtime_export_base_roots(&session_id);
        let (analysis_relative_root, analysis_absolute_root) =
            runtime_export_root(&workspace_root, &session_id, "analysis");
        ensure_runtime_export_root(&analysis_absolute_root)?;
        let (review_relative_root, review_absolute_root) =
            runtime_export_root(&workspace_root, &session_id, "review");
        ensure_runtime_export_root(&review_absolute_root)?;

        let analysis_artifacts = vec![
            write_runtime_export_file(
                &analysis_absolute_root,
                &analysis_relative_root,
                ANALYSIS_CONTEXT_FILE_NAME,
                "analysis_context",
                copy.analysis.context_artifact_title,
                build_analysis_context_json(
                    &read,
                    &metrics,
                    &workspace_root,
                    &replay_relative_root,
                    &handoff_relative_root,
                    &evidence_relative_root,
                    &exported_at,
                )?,
            )?,
            write_runtime_export_file(
                &analysis_absolute_root,
                &analysis_relative_root,
                ANALYSIS_BRIEF_FILE_NAME,
                "analysis_brief",
                copy.analysis.brief_artifact_title,
                build_analysis_brief_markdown(
                    &read,
                    &metrics,
                    &recent_artifacts,
                    &exported_at,
                    &copy,
                ),
            )?,
        ];
        let artifacts = vec![
            write_runtime_export_file(
                &review_absolute_root,
                &review_relative_root,
                REVIEW_DECISION_MARKDOWN_FILE_NAME,
                "review_decision_markdown",
                copy.review.markdown_artifact_title,
                build_review_decision_markdown(
                    &read,
                    &decision,
                    &analysis_relative_root,
                    &replay_relative_root,
                    &exported_at,
                    &copy,
                ),
            )?,
            write_runtime_export_file(
                &review_absolute_root,
                &review_relative_root,
                REVIEW_DECISION_JSON_FILE_NAME,
                "review_decision_json",
                copy.review.json_artifact_title,
                build_review_decision_json(
                    &read,
                    &decision,
                    &analysis_relative_root,
                    &replay_relative_root,
                    &exported_at,
                )?,
            )?,
        ];

        if saving {
            self.write_export_rollout_summary_candidate(
                &read,
                &metrics,
                &recent_artifacts,
                &workspace_root,
                &exported_at,
                &review_relative_root,
                "review_decision",
                method,
                &copy,
            )
            .await?;
        }

        Ok(AgentSessionReviewDecisionTemplateExportResponse {
            session_id: read.session.session_id,
            thread_id: read.session.thread_id,
            workspace_id: read.session.workspace_id,
            workspace_root: workspace_root.to_string_lossy().to_string(),
            review_relative_root,
            review_absolute_root: review_absolute_root.to_string_lossy().to_string(),
            analysis_relative_root,
            analysis_absolute_root: analysis_absolute_root.to_string_lossy().to_string(),
            handoff_bundle_relative_root: handoff_relative_root,
            evidence_pack_relative_root: evidence_relative_root,
            replay_case_relative_root: replay_relative_root,
            exported_at,
            thread_status: metrics.thread_status,
            latest_turn_status: metrics.latest_turn_status,
            pending_request_count: metrics.pending_request_count,
            queued_turn_count: metrics.queued_turn_count,
            title: copy.review.response_title.to_string(),
            default_decision_status: "pending_review".to_string(),
            decision,
            decision_status_options: vec![
                "pending_review".to_string(),
                "accepted".to_string(),
                "deferred".to_string(),
                "rejected".to_string(),
                "needs_more_evidence".to_string(),
            ],
            risk_level_options: vec![
                "unknown".to_string(),
                "low".to_string(),
                "medium".to_string(),
                "high".to_string(),
            ],
            review_checklist: copy
                .review
                .checklist
                .iter()
                .map(|item| (*item).to_string())
                .collect(),
            analysis_artifacts,
            artifacts,
        })
    }
}
