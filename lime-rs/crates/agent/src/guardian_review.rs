use crate::credential_bridge::ConfiguredReplyProvider;
use crate::protocol::{
    AgentEvent, GuardianReviewStatus, GuardianRiskLevel, GuardianUserAuthorization,
};
use agent_protocol::ThreadId;
use futures::StreamExt;
use model_provider::current_client::CanonicalLlmEvent;
use model_provider::current_client::{
    CurrentProviderContent, CurrentProviderMessage, CurrentProviderRequest,
    CurrentProviderRequestMetadata, GenerationOptions,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const REVIEW_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub(crate) struct GuardianReviewRequest {
    pub(crate) session_id: String,
    pub(crate) thread_id: ThreadId,
    pub(crate) turn_id: String,
    pub(crate) target_item_id: Option<String>,
    pub(crate) tool_name: String,
    pub(crate) command: String,
    pub(crate) cwd: String,
    pub(crate) started_at_ms: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct GuardianReviewResult {
    pub(crate) status: GuardianReviewStatus,
    pub(crate) risk_level: Option<GuardianRiskLevel>,
    pub(crate) user_authorization: Option<GuardianUserAuthorization>,
    pub(crate) rationale: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GuardianAssessment {
    decision: String,
    risk_level: Option<String>,
    user_authorization: Option<String>,
    rationale: Option<String>,
}

pub(crate) fn review_id() -> String {
    format!("guardian-review-{}", Uuid::new_v4())
}

pub(crate) fn action_value(request: &GuardianReviewRequest) -> Value {
    json!({
        "type": "command",
        "source": "shell",
        "toolName": request.tool_name,
        "command": request.command,
        "cwd": request.cwd,
    })
}

pub(crate) fn started_event(
    request: &GuardianReviewRequest,
    review_id: &str,
    action: Value,
) -> AgentEvent {
    AgentEvent::GuardianReviewStarted {
        review_id: review_id.to_string(),
        target_item_id: request.target_item_id.clone(),
        turn_id: request.turn_id.clone(),
        action,
        started_at_ms: request.started_at_ms,
    }
}

pub(crate) fn completed_event(
    request: &GuardianReviewRequest,
    review_id: &str,
    action: Value,
    result: GuardianReviewResult,
    completed_at_ms: i64,
) -> AgentEvent {
    AgentEvent::GuardianReviewCompleted {
        review_id: review_id.to_string(),
        target_item_id: request.target_item_id.clone(),
        turn_id: request.turn_id.clone(),
        status: result.status,
        risk_level: result.risk_level,
        user_authorization: result.user_authorization,
        rationale: (!result.rationale.trim().is_empty()).then_some(result.rationale),
        decision_source: "agent".to_string(),
        action,
        started_at_ms: request.started_at_ms,
        completed_at_ms,
    }
}

pub(crate) async fn run(
    provider: ConfiguredReplyProvider,
    request: &GuardianReviewRequest,
    cancel_token: CancellationToken,
) -> GuardianReviewResult {
    let action = action_value(request);
    let system_prompt = "You are Lime Guardian, a strict security reviewer for a desktop coding agent. Evaluate exactly one shell command before execution. Return only one JSON object with keys decision (approved or denied), riskLevel (low, medium, high, critical), userAuthorization (unknown, low, medium, high), and rationale (short plain text). Deny when uncertain, destructive, credential-exfiltrating, persistence-related, or outside the requested workspace.";
    let user_prompt = format!(
        "Review this requested action. Do not execute it and do not suggest alternatives.\n{}",
        serde_json::to_string(&action).unwrap_or_else(|_| "{}".to_string())
    );
    let metadata = CurrentProviderRequestMetadata::new(
        request.session_id.clone(),
        request.thread_id.as_str().to_string(),
        request.turn_id.clone(),
        None,
    );
    let provider_request = CurrentProviderRequest::new(vec![CurrentProviderMessage::user(vec![
        CurrentProviderContent::Text(user_prompt),
    ])])
    .with_system_prompt(Some(system_prompt.to_string()))
    .with_generation(GenerationOptions {
        max_tokens: Some(512),
        temperature: Some(0.0),
        ..Default::default()
    })
    .with_metadata(metadata);

    let stream = match tokio::time::timeout(
        REVIEW_TIMEOUT,
        provider.client().stream(provider_request),
    )
    .await
    {
        Ok(Ok(stream)) => stream,
        Ok(Err(error)) => return denied(format!("Guardian provider request failed: {error}")),
        Err(_) => return timed_out(),
    };
    let mut stream = Box::pin(stream);
    let mut output = String::new();
    let review = async {
        while let Some(event) = stream.next().await {
            match event {
                Ok(CanonicalLlmEvent::TextDelta { text, .. }) => output.push_str(&text),
                Ok(CanonicalLlmEvent::ProviderError { message, .. }) => {
                    return Err(format!("Guardian provider stream failed: {message}"));
                }
                Ok(CanonicalLlmEvent::Finish { .. } | CanonicalLlmEvent::StepFinish { .. }) => {
                    break;
                }
                Ok(_) => {}
                Err(error) => return Err(format!("Guardian provider stream failed: {error}")),
            }
        }
        parse_assessment(&output)
    };
    let parsed = tokio::select! {
        _ = cancel_token.cancelled() => return GuardianReviewResult {
            status: GuardianReviewStatus::Aborted,
            risk_level: None,
            user_authorization: None,
            rationale: "Guardian review was cancelled.".to_string(),
        },
        result = tokio::time::timeout(REVIEW_TIMEOUT, review) => result,
    };
    match parsed {
        Ok(Ok(assessment)) => assessment,
        Ok(Err(error)) => denied(error),
        Err(_) => timed_out(),
    }
}

fn parse_assessment(output: &str) -> Result<GuardianReviewResult, String> {
    let candidate = output
        .find('{')
        .and_then(|start| output.rfind('}').map(|end| &output[start..=end]))
        .ok_or_else(|| "Guardian response did not contain a JSON assessment".to_string())?;
    let assessment: GuardianAssessment = serde_json::from_str(candidate)
        .map_err(|error| format!("Guardian response was not valid JSON: {error}"))?;
    let risk_level = assessment
        .risk_level
        .as_deref()
        .and_then(parse_risk_level)
        .ok_or_else(|| "Guardian response omitted a valid riskLevel".to_string())?;
    let user_authorization = assessment
        .user_authorization
        .as_deref()
        .and_then(parse_user_authorization)
        .ok_or_else(|| "Guardian response omitted a valid userAuthorization".to_string())?;
    let rationale = assessment
        .rationale
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| "Guardian response omitted rationale".to_string())?;
    let status = match assessment.decision.trim().to_ascii_lowercase().as_str() {
        "approved" | "allow" => GuardianReviewStatus::Approved,
        "denied" | "deny" => GuardianReviewStatus::Denied,
        other => return Err(format!("Guardian response used unknown decision '{other}'")),
    };
    Ok(GuardianReviewResult {
        status,
        risk_level: Some(risk_level),
        user_authorization: Some(user_authorization),
        rationale,
    })
}

fn parse_risk_level(value: &str) -> Option<GuardianRiskLevel> {
    match value.trim().to_ascii_lowercase().as_str() {
        "low" => Some(GuardianRiskLevel::Low),
        "medium" => Some(GuardianRiskLevel::Medium),
        "high" => Some(GuardianRiskLevel::High),
        "critical" => Some(GuardianRiskLevel::Critical),
        _ => None,
    }
}

fn parse_user_authorization(value: &str) -> Option<GuardianUserAuthorization> {
    match value.trim().to_ascii_lowercase().as_str() {
        "unknown" => Some(GuardianUserAuthorization::Unknown),
        "low" => Some(GuardianUserAuthorization::Low),
        "medium" => Some(GuardianUserAuthorization::Medium),
        "high" => Some(GuardianUserAuthorization::High),
        _ => None,
    }
}

fn denied(rationale: impl Into<String>) -> GuardianReviewResult {
    GuardianReviewResult {
        status: GuardianReviewStatus::Denied,
        risk_level: Some(GuardianRiskLevel::High),
        user_authorization: Some(GuardianUserAuthorization::Unknown),
        rationale: rationale.into(),
    }
}

fn timed_out() -> GuardianReviewResult {
    GuardianReviewResult {
        status: GuardianReviewStatus::TimedOut,
        risk_level: None,
        user_authorization: None,
        rationale: "Guardian review timed out; the action was denied.".to_string(),
    }
}
