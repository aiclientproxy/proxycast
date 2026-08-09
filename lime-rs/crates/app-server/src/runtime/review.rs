use super::status::agent_turn_is_active;
use super::turn_start::{ReviewContext, TurnStartInputKind, TurnStartRequest};
use super::*;
use agent_protocol::AgentInput;
use app_server_protocol::protocol::v2::{ReviewDelivery, ReviewTarget};
use app_server_protocol::AgentSessionTurnStartResponse;

impl RuntimeCore {
    pub async fn start_review(
        &self,
        thread_id: &str,
        target: ReviewTarget,
        delivery: Option<ReviewDelivery>,
    ) -> Result<RuntimeCoreOutput<AgentSessionTurnStartResponse>, RuntimeCoreError> {
        let thread_id = thread_id.trim();
        if thread_id.is_empty() {
            return Err(RuntimeCoreError::InvalidRequest(
                "review/start requires threadId".to_string(),
            ));
        }
        if matches!(delivery, Some(ReviewDelivery::Detached)) {
            return Err(RuntimeCoreError::InvalidRequest(
                "detached review is not supported by Lime Desktop".to_string(),
            ));
        }
        let session_id = self
            .loaded_session_id_for_thread(thread_id)
            .ok_or_else(|| RuntimeCoreError::SessionNotFound(thread_id.to_string()))?;
        let (_, turns) = self.session_snapshot(&session_id)?;
        if let Some(active_turn) = turns.iter().find(|turn| agent_turn_is_active(turn.status)) {
            return Err(RuntimeCoreError::TurnAlreadyActive(
                active_turn.turn_id.clone(),
            ));
        }

        let (prompt_target, target) = match target {
            ReviewTarget::UncommittedChanges => (
                lime_agent::ReviewPromptTarget::UncommittedChanges,
                ReviewTarget::UncommittedChanges,
            ),
            ReviewTarget::BaseBranch { branch } => {
                let branch = non_empty_review_field(branch, "branch")?;
                (
                    lime_agent::ReviewPromptTarget::BaseBranch {
                        branch: branch.clone(),
                        merge_base_sha: None,
                    },
                    ReviewTarget::BaseBranch { branch },
                )
            }
            ReviewTarget::Commit { sha, title } => {
                let sha = non_empty_review_field(sha, "sha")?;
                let title = title
                    .map(|value| value.trim().to_string())
                    .filter(|value| !value.is_empty());
                (
                    lime_agent::ReviewPromptTarget::Commit {
                        sha: sha.clone(),
                        title: title.clone(),
                    },
                    ReviewTarget::Commit { sha, title },
                )
            }
            ReviewTarget::Custom { instructions } => {
                let instructions = non_empty_review_field(instructions, "instructions")?;
                (
                    lime_agent::ReviewPromptTarget::Custom {
                        instructions: instructions.clone(),
                    },
                    ReviewTarget::Custom { instructions },
                )
            }
        };
        let target_wire = serde_json::to_value(target)
            .map_err(|error| RuntimeCoreError::InvalidRequest(error.to_string()))?;
        let resolved_prompt = lime_agent::resolve_review_prompt(prompt_target)
            .map_err(RuntimeCoreError::InvalidRequest)?;
        let review_context = ReviewContext {
            user_facing_hint: resolved_prompt.user_facing_hint,
            target: target_wire,
        };
        self.start_turn_inner_with_review_context(
            TurnStartRequest {
                session_id,
                turn_id: None,
                input: vec![AgentInput::text(resolved_prompt.prompt)],
                runtime_options: None,
                queue_if_busy: false,
                skip_pre_submit_resume: true,
            },
            RuntimeHostContext::default(),
            None,
            false,
            true,
            TurnStartInputKind::Review,
            Some(review_context),
        )
        .await
    }
}

fn non_empty_review_field(value: String, field: &str) -> Result<String, RuntimeCoreError> {
    let value = value.trim().to_string();
    if value.is_empty() {
        return Err(RuntimeCoreError::InvalidRequest(format!(
            "{field} must not be empty"
        )));
    }
    Ok(value)
}
