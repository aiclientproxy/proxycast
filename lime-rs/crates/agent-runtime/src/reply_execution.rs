//! Reply execution 的 current result contract。
//!
//! 具体后端的错误类型留在 adapter 边界转换；这里仅描述 Lime runtime 主链
//! 对外可传递的执行结果和可恢复错误状态。

use crate::reply_host::RuntimeReplyStartError;
use model_provider::current_client::FailureClassification;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RuntimeReplyAttemptErrorKind {
    #[default]
    Execution,
    UsageLimitExceeded,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeReplyAttemptError {
    pub message: String,
    pub emitted_any: bool,
    classification: Option<FailureClassification>,
    retryable: bool,
    reroute_safe: bool,
    kind: RuntimeReplyAttemptErrorKind,
}

impl RuntimeReplyAttemptError {
    pub fn new(message: impl Into<String>, emitted_any: bool) -> Self {
        Self {
            message: message.into(),
            emitted_any,
            classification: None,
            retryable: false,
            reroute_safe: false,
            kind: RuntimeReplyAttemptErrorKind::Execution,
        }
    }

    pub fn usage_limit_exceeded(message: impl Into<String>, emitted_any: bool) -> Self {
        Self {
            message: message.into(),
            emitted_any,
            classification: None,
            retryable: false,
            reroute_safe: false,
            kind: RuntimeReplyAttemptErrorKind::UsageLimitExceeded,
        }
    }

    pub fn provider_failure(
        message: impl Into<String>,
        emitted_any: bool,
        classification: Option<FailureClassification>,
        retryable: bool,
    ) -> Self {
        Self {
            message: message.into(),
            emitted_any,
            classification,
            retryable,
            reroute_safe: true,
            kind: if classification == Some(FailureClassification::Quota) {
                RuntimeReplyAttemptErrorKind::UsageLimitExceeded
            } else {
                RuntimeReplyAttemptErrorKind::Execution
            },
        }
    }

    pub fn kind(&self) -> RuntimeReplyAttemptErrorKind {
        self.kind
    }

    pub fn is_usage_limit_exceeded(&self) -> bool {
        self.kind == RuntimeReplyAttemptErrorKind::UsageLimitExceeded
    }

    pub fn classification(&self) -> Option<FailureClassification> {
        self.classification
    }

    pub fn retryable(&self) -> bool {
        self.retryable
    }

    pub fn suppress_reroute(mut self) -> Self {
        self.reroute_safe = false;
        self
    }

    pub fn is_reroutable_provider_failure(&self) -> bool {
        !self.emitted_any
            && self.reroute_safe
            && self.retryable
            && matches!(
                self.classification,
                Some(
                    FailureClassification::RateLimit
                        | FailureClassification::ProviderInternal
                        | FailureClassification::Transport
                )
            )
    }
}

impl From<RuntimeReplyStartError> for RuntimeReplyAttemptError {
    fn from(error: RuntimeReplyStartError) -> Self {
        Self::new(error.message, error.emitted_any)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeReplyAttemptState {
    text_output: String,
    event_errors: Vec<String>,
    emitted_any: bool,
}

impl RuntimeReplyAttemptState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn emitted_any(&self) -> bool {
        self.emitted_any
    }

    pub fn emitted_any_mut(&mut self) -> &mut bool {
        &mut self.emitted_any
    }

    pub fn mark_emitted(&mut self) {
        self.emitted_any = true;
    }

    pub fn push_text(&mut self, text: impl AsRef<str>) {
        self.text_output.push_str(text.as_ref());
    }

    pub fn text_output(&self) -> &str {
        &self.text_output
    }

    pub fn trimmed_text_output(&self) -> String {
        self.text_output.trim().to_string()
    }

    pub fn push_error(&mut self, error: impl Into<String>) {
        self.event_errors.push(error.into());
    }

    pub fn event_errors(&self) -> &[String] {
        &self.event_errors
    }

    pub fn last_error(&self) -> Option<&str> {
        self.event_errors.last().map(String::as_str)
    }

    pub fn error(&self, message: impl Into<String>) -> RuntimeReplyAttemptError {
        RuntimeReplyAttemptError::new(message, self.emitted_any)
    }

    pub fn into_execution(
        self,
        attempts_summary: impl Into<String>,
        cancelled: bool,
    ) -> RuntimeReplyExecution {
        RuntimeReplyExecution::new(
            self.text_output,
            self.event_errors,
            self.emitted_any,
            attempts_summary.into(),
            cancelled,
        )
    }

    pub fn into_execution_with_text(
        self,
        text_output: impl Into<String>,
        attempts_summary: impl Into<String>,
        cancelled: bool,
    ) -> RuntimeReplyExecution {
        RuntimeReplyExecution::new(
            text_output.into(),
            self.event_errors,
            self.emitted_any,
            attempts_summary.into(),
            cancelled,
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeReplyExecution {
    pub text_output: String,
    pub event_errors: Vec<String>,
    pub emitted_any: bool,
    pub attempts_summary: String,
    pub cancelled: bool,
}

impl RuntimeReplyExecution {
    pub fn new(
        text_output: String,
        event_errors: Vec<String>,
        emitted_any: bool,
        attempts_summary: String,
        cancelled: bool,
    ) -> Self {
        Self {
            text_output,
            event_errors,
            emitted_any,
            attempts_summary,
            cancelled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reply_attempt_error_carries_emission_state() {
        let error = RuntimeReplyAttemptError::new("provider failed", true);

        assert_eq!(error.message, "provider failed");
        assert!(error.emitted_any);
        assert_eq!(error.kind(), RuntimeReplyAttemptErrorKind::Execution);
    }

    #[test]
    fn reply_attempt_error_preserves_usage_limit_kind() {
        let error = RuntimeReplyAttemptError::usage_limit_exceeded("quota exhausted", false);

        assert_eq!(error.message, "quota exhausted");
        assert!(!error.emitted_any);
        assert!(error.is_usage_limit_exceeded());
    }

    #[test]
    fn reply_attempt_error_preserves_reroutable_provider_failure() {
        let error = RuntimeReplyAttemptError::provider_failure(
            "provider unavailable",
            false,
            Some(FailureClassification::ProviderInternal),
            true,
        );

        assert_eq!(
            error.classification(),
            Some(FailureClassification::ProviderInternal)
        );
        assert!(error.retryable());
        assert!(error.is_reroutable_provider_failure());
    }

    #[test]
    fn reply_attempt_error_rejects_unsafe_provider_reroute() {
        for error in [
            RuntimeReplyAttemptError::provider_failure(
                "auth failed",
                false,
                Some(FailureClassification::Authentication),
                false,
            ),
            RuntimeReplyAttemptError::provider_failure(
                "partial stream failed",
                true,
                Some(FailureClassification::Transport),
                true,
            ),
            RuntimeReplyAttemptError::provider_failure(
                "unknown failure",
                false,
                Some(FailureClassification::Unknown),
                true,
            ),
            RuntimeReplyAttemptError::provider_failure(
                "provider failed after consuming steer",
                false,
                Some(FailureClassification::Transport),
                true,
            )
            .suppress_reroute(),
        ] {
            assert!(!error.is_reroutable_provider_failure());
        }
    }

    #[test]
    fn reply_attempt_error_converts_from_start_error() {
        let error =
            RuntimeReplyAttemptError::from(RuntimeReplyStartError::new("backend failed", true));

        assert_eq!(error.message, "backend failed");
        assert!(error.emitted_any);
    }

    #[test]
    fn reply_execution_preserves_attempt_summary_and_cancel_state() {
        let execution = RuntimeReplyExecution::new(
            "done".to_string(),
            vec!["tool warning".to_string()],
            true,
            "attempts=1".to_string(),
            false,
        );

        assert_eq!(execution.text_output, "done");
        assert_eq!(execution.event_errors, vec!["tool warning"]);
        assert!(execution.emitted_any);
        assert_eq!(execution.attempts_summary, "attempts=1");
        assert!(!execution.cancelled);
    }

    #[test]
    fn reply_attempt_state_accumulates_runtime_output() {
        let mut state = RuntimeReplyAttemptState::new();

        state.push_text("hel");
        state.push_text("lo");
        state.push_error("provider warning");
        state.mark_emitted();

        assert_eq!(state.text_output(), "hello");
        assert_eq!(state.trimmed_text_output(), "hello");
        assert_eq!(state.event_errors(), &["provider warning".to_string()]);
        assert_eq!(state.last_error(), Some("provider warning"));
        assert!(state.emitted_any());
    }

    #[test]
    fn reply_attempt_state_builds_error_with_current_emission_state() {
        let mut state = RuntimeReplyAttemptState::new();

        state.mark_emitted();

        let error = state.error("provider failed");

        assert_eq!(error.message, "provider failed");
        assert!(error.emitted_any);
    }
}
