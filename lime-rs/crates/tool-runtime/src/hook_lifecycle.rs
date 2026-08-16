//! Hook lifecycle 决策 owner。
//!
//! `turn_snapshot` 只固化某个 sampling step 可见的 Hook 注册快照；本模块决定这些
//! Hook 在一次事件上的实际生命周期结果：是否参与、是否放行、是否阻断、是否改写工具
//! 输入、以及注入哪些额外上下文。
//!
//! 边界：
//! - 只消费不可变 `RuntimeHookSnapshot`，不自己发现或加载配置。
//! - 不执行 handler；执行由持有进程/沙箱写集的 owner 完成，本模块只裁决与归集结果。
//! - 未实现的 handler 类型和不可信来源一律 fail closed，不静默放行。

use crate::turn_snapshot::{
    RuntimeHookEventName, RuntimeHookExecutionMode, RuntimeHookHandlerType, RuntimeHookSnapshot,
    RuntimeHookTrustStatus, RuntimeTurnSnapshot,
};
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;

/// 单个 Hook 在一次事件上的裁决结果。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeHookOutcome {
    /// 放行，可选注入额外上下文。
    Allow { additional_context: Option<String> },
    /// 阻断当前操作，携带用户可见原因。
    Block { reason: String },
    /// 改写工具输入后放行。
    Rewrite { arguments: String },
    /// 中止整个回合。
    Abort { reason: String },
    /// 未参与：matcher 不匹配、未启用或事件不符。
    Skipped { reason: RuntimeHookSkipReason },
    /// fail closed：无法安全裁决。
    Failed { reason: RuntimeHookFailure },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeHookSkipReason {
    Disabled,
    MatcherMismatch,
    EventMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeHookFailure {
    /// upstream 尚未形成真实执行能力的 handler 类型。
    UnsupportedHandler(RuntimeHookHandlerType),
    /// 不可信或被改写的来源不得参与决策。
    UntrustedSource(RuntimeHookTrustStatus),
    /// 与 `RuntimeTurnSnapshot::try_new` 一致：非 Sync 执行模式尚未形成可裁决语义。
    UnsupportedExecutionMode(RuntimeHookExecutionMode),
    /// 非法 matcher 正则。
    InvalidMatcher { matcher: String },
    /// 超时必须为正值，0 无法表达“立即超时”与“无限等待”的差异。
    InvalidTimeout,
    /// handler 报告了终态之外的结果。
    MissingDecision,
    /// handler 进程或输出无法形成安全裁决。
    HandlerFailed { reason: String },
}

/// 一次事件的输入事实。
#[derive(Debug, Clone, Default)]
pub struct RuntimeHookEventContext {
    pub session_id: Option<String>,
    pub turn_id: Option<String>,
    pub tool_call_id: Option<String>,
    pub working_directory: PathBuf,
    pub tool_name: Option<String>,
    pub tool_arguments: Option<String>,
    pub tool_output: Option<String>,
    pub content: Option<String>,
}

/// handler 侧回报的原始结果，由执行 owner 填充。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeHookHandlerReport {
    Allow { additional_context: Option<String> },
    Block { reason: String },
    Rewrite { arguments: String },
    Abort { reason: String },
    Failed { reason: String },
}

pub type RuntimeHookReportFuture<'a> =
    Pin<Box<dyn Future<Output = Option<RuntimeHookHandlerReport>> + Send + 'a>>;

pub trait RuntimeHookReporter: Send + Sync {
    fn report<'a>(
        &'a self,
        hook: &'a RuntimeHookSnapshot,
        event_name: RuntimeHookEventName,
        context: &'a RuntimeHookEventContext,
    ) -> RuntimeHookReportFuture<'a>;
}

/// 单个 Hook 的裁决记录，保留稳定 identity 便于投影和 evidence。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeHookDecision {
    pub key: String,
    pub run_id: String,
    pub event_name: RuntimeHookEventName,
    pub outcome: RuntimeHookOutcome,
}

impl RuntimeHookDecision {
    pub fn is_blocking(&self) -> bool {
        matches!(
            self.outcome,
            RuntimeHookOutcome::Block { .. }
                | RuntimeHookOutcome::Abort { .. }
                | RuntimeHookOutcome::Failed { .. }
        )
    }
}

/// 一次事件的聚合结果。
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeHookEvaluation {
    pub decisions: Vec<RuntimeHookDecision>,
}

impl RuntimeHookEvaluation {
    /// 是否阻断当前操作。任一 block/abort/fail 即阻断，fail closed。
    pub fn is_blocked(&self) -> bool {
        self.decisions.iter().any(RuntimeHookDecision::is_blocking)
    }

    /// 是否要求中止整个回合。
    pub fn is_aborted(&self) -> bool {
        self.decisions
            .iter()
            .any(|decision| matches!(decision.outcome, RuntimeHookOutcome::Abort { .. }))
    }

    /// 第一个阻断原因，按 display order 顺序稳定。
    pub fn blocking_reason(&self) -> Option<String> {
        self.decisions
            .iter()
            .find_map(|decision| match &decision.outcome {
                RuntimeHookOutcome::Block { reason } | RuntimeHookOutcome::Abort { reason } => {
                    Some(reason.clone())
                }
                RuntimeHookOutcome::Failed { reason } => {
                    Some(format!("hook {} fail closed: {reason:?}", decision.key))
                }
                RuntimeHookOutcome::Allow { .. }
                | RuntimeHookOutcome::Rewrite { .. }
                | RuntimeHookOutcome::Skipped { .. } => None,
            })
    }

    /// 最终生效的工具输入改写。后序 Hook 覆盖前序，保持 display order 语义。
    pub fn rewritten_arguments(&self) -> Option<String> {
        self.decisions
            .iter()
            .filter_map(|decision| match &decision.outcome {
                RuntimeHookOutcome::Rewrite { arguments } => Some(arguments.clone()),
                _ => None,
            })
            .next_back()
    }

    /// 按 display order 收集注入上下文。
    pub fn injected_context(&self) -> Vec<String> {
        self.decisions
            .iter()
            .filter_map(|decision| match &decision.outcome {
                RuntimeHookOutcome::Allow {
                    additional_context: Some(context),
                } if !context.trim().is_empty() => Some(context.clone()),
                _ => None,
            })
            .collect()
    }

    /// 参与裁决（未被 skip）的 Hook 数量。
    pub fn participating_count(&self) -> usize {
        self.decisions
            .iter()
            .filter(|decision| !matches!(decision.outcome, RuntimeHookOutcome::Skipped { .. }))
            .count()
    }
}

/// 判定某个 Hook 是否可以参与本次事件；返回 `Err` 表示直接得出终态。
fn admit(
    hook: &RuntimeHookSnapshot,
    context: &RuntimeHookEventContext,
) -> Result<(), RuntimeHookOutcome> {
    if !hook.enabled {
        return Err(RuntimeHookOutcome::Skipped {
            reason: RuntimeHookSkipReason::Disabled,
        });
    }
    match hook.trust_status {
        RuntimeHookTrustStatus::Untrusted | RuntimeHookTrustStatus::Modified => {
            return Err(RuntimeHookOutcome::Failed {
                reason: RuntimeHookFailure::UntrustedSource(hook.trust_status),
            });
        }
        RuntimeHookTrustStatus::Managed | RuntimeHookTrustStatus::Trusted => {}
    }
    if hook.handler_type != RuntimeHookHandlerType::Command {
        return Err(RuntimeHookOutcome::Failed {
            reason: RuntimeHookFailure::UnsupportedHandler(hook.handler_type),
        });
    }
    if hook.timeout_sec == 0 {
        return Err(RuntimeHookOutcome::Failed {
            reason: RuntimeHookFailure::InvalidTimeout,
        });
    }
    if let Some(matcher) = hook.matcher.as_deref() {
        if !matcher_matches(matcher, context)? {
            return Err(RuntimeHookOutcome::Skipped {
                reason: RuntimeHookSkipReason::MatcherMismatch,
            });
        }
    }
    Ok(())
}

fn matcher_matches(
    matcher: &str,
    context: &RuntimeHookEventContext,
) -> Result<bool, RuntimeHookOutcome> {
    let target = context
        .tool_name
        .as_deref()
        .or(context.content.as_deref())
        .unwrap_or_default();
    if matcher.is_empty() || matcher == "*" {
        return Ok(true);
    }
    if matcher
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || ch == '_' || ch == '|')
    {
        return Ok(matcher.split('|').any(|candidate| candidate == target));
    }
    let pattern = regex::Regex::new(matcher).map_err(|_| RuntimeHookOutcome::Failed {
        reason: RuntimeHookFailure::InvalidMatcher {
            matcher: matcher.to_string(),
        },
    })?;
    Ok(pattern.is_match(target))
}

/// 对一个事件求值。`report` 由执行 owner 提供；返回 `None` 表示 handler 没有给出终态，
/// 按 fail closed 处理，不当作放行。
pub fn evaluate_hook_event<F>(
    snapshot: &RuntimeTurnSnapshot,
    event_name: RuntimeHookEventName,
    context: &RuntimeHookEventContext,
    mut report: F,
) -> RuntimeHookEvaluation
where
    F: FnMut(&RuntimeHookSnapshot) -> Option<RuntimeHookHandlerReport>,
{
    let mut decisions = Vec::new();
    for hook in snapshot.hooks_for(event_name) {
        let outcome = match admit(hook, context) {
            Err(outcome) => outcome,
            Ok(()) if hook.execution_mode == RuntimeHookExecutionMode::Async => {
                RuntimeHookOutcome::Allow {
                    additional_context: None,
                }
            }
            Ok(()) => outcome_from_report(report(hook)),
        };
        decisions.push(RuntimeHookDecision {
            key: hook.key.clone(),
            run_id: hook.run_id(),
            event_name,
            outcome,
        });
    }
    RuntimeHookEvaluation { decisions }
}

pub async fn evaluate_hook_event_async(
    snapshot: &RuntimeTurnSnapshot,
    event_name: RuntimeHookEventName,
    context: &RuntimeHookEventContext,
    reporter: &dyn RuntimeHookReporter,
) -> RuntimeHookEvaluation {
    let mut decisions = Vec::new();
    for hook in snapshot.hooks_for(event_name) {
        let outcome = match admit(hook, context) {
            Err(outcome) => outcome,
            Ok(()) => outcome_from_report(reporter.report(hook, event_name, context).await),
        };
        decisions.push(RuntimeHookDecision {
            key: hook.key.clone(),
            run_id: hook.run_id(),
            event_name,
            outcome,
        });
    }
    RuntimeHookEvaluation { decisions }
}

fn outcome_from_report(report: Option<RuntimeHookHandlerReport>) -> RuntimeHookOutcome {
    match report {
        Some(RuntimeHookHandlerReport::Allow { additional_context }) => {
            RuntimeHookOutcome::Allow { additional_context }
        }
        Some(RuntimeHookHandlerReport::Block { reason }) => RuntimeHookOutcome::Block { reason },
        Some(RuntimeHookHandlerReport::Rewrite { arguments }) => {
            RuntimeHookOutcome::Rewrite { arguments }
        }
        Some(RuntimeHookHandlerReport::Abort { reason }) => RuntimeHookOutcome::Abort { reason },
        Some(RuntimeHookHandlerReport::Failed { reason }) => RuntimeHookOutcome::Failed {
            reason: RuntimeHookFailure::HandlerFailed { reason },
        },
        None => RuntimeHookOutcome::Failed {
            reason: RuntimeHookFailure::MissingDecision,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_definition::{RuntimeToolDefinition, RuntimeToolExposure};
    use crate::turn_snapshot::{RuntimeHookSource, RuntimeToolIdentity, RuntimeToolSnapshot};
    use std::path::PathBuf;

    fn hook(
        key: &str,
        display_order: i64,
        event_name: RuntimeHookEventName,
    ) -> RuntimeHookSnapshot {
        RuntimeHookSnapshot {
            key: key.to_string(),
            event_name,
            handler_type: RuntimeHookHandlerType::Command,
            execution_mode: RuntimeHookExecutionMode::Sync,
            matcher: None,
            command: Some("true".to_string()),
            timeout_sec: 10,
            status_message: None,
            additional_context_limit: None,
            source_path: PathBuf::from("/etc/lime/hooks.json"),
            source: RuntimeHookSource::Project,
            plugin_id: None,
            display_order,
            enabled: true,
            is_managed: false,
            current_hash: "sha256:test".to_string(),
            trust_status: RuntimeHookTrustStatus::Trusted,
        }
    }

    fn snapshot(hooks: Vec<RuntimeHookSnapshot>) -> RuntimeTurnSnapshot {
        let tool = RuntimeToolSnapshot::new(
            RuntimeToolIdentity::plain("shell"),
            RuntimeToolDefinition::new("shell", "run shell", serde_json::json!({"type": "object"})),
            RuntimeToolExposure::Direct,
            false,
            true,
        );
        RuntimeTurnSnapshot::try_new(vec![tool], hooks).expect("valid snapshot")
    }

    fn tool_context(tool_name: &str) -> RuntimeHookEventContext {
        RuntimeHookEventContext {
            tool_name: Some(tool_name.to_string()),
            tool_arguments: Some("{}".to_string()),
            content: None,
            ..RuntimeHookEventContext::default()
        }
    }

    #[test]
    fn block_from_any_hook_blocks_the_event() {
        let snapshot = snapshot(vec![
            hook("allow", 0, RuntimeHookEventName::PreToolUse),
            hook("deny", 1, RuntimeHookEventName::PreToolUse),
        ]);

        let evaluation = evaluate_hook_event(
            &snapshot,
            RuntimeHookEventName::PreToolUse,
            &tool_context("shell"),
            |hook| match hook.key.as_str() {
                "deny" => Some(RuntimeHookHandlerReport::Block {
                    reason: "denied by policy".to_string(),
                }),
                _ => Some(RuntimeHookHandlerReport::Allow {
                    additional_context: None,
                }),
            },
        );

        assert!(evaluation.is_blocked());
        assert!(!evaluation.is_aborted());
        assert_eq!(
            evaluation.blocking_reason().as_deref(),
            Some("denied by policy")
        );
        assert_eq!(evaluation.participating_count(), 2);
    }

    #[test]
    fn missing_handler_decision_fails_closed_instead_of_allowing() {
        let snapshot = snapshot(vec![hook("silent", 0, RuntimeHookEventName::PreToolUse)]);

        let evaluation = evaluate_hook_event(
            &snapshot,
            RuntimeHookEventName::PreToolUse,
            &tool_context("shell"),
            |_| None,
        );

        assert!(evaluation.is_blocked());
        assert_eq!(
            evaluation.decisions[0].outcome,
            RuntimeHookOutcome::Failed {
                reason: RuntimeHookFailure::MissingDecision
            }
        );
    }

    #[test]
    fn unsupported_handler_and_untrusted_source_fail_closed() {
        let mut prompt_hook = hook("prompt", 0, RuntimeHookEventName::PreToolUse);
        prompt_hook.handler_type = RuntimeHookHandlerType::Prompt;
        let mut untrusted = hook("untrusted", 1, RuntimeHookEventName::PreToolUse);
        untrusted.trust_status = RuntimeHookTrustStatus::Untrusted;
        // Prompt handler 无法进入 snapshot 校验，因此单独验证 admit 语义。
        let context = tool_context("shell");

        assert_eq!(
            admit(&prompt_hook, &context).unwrap_err(),
            RuntimeHookOutcome::Failed {
                reason: RuntimeHookFailure::UnsupportedHandler(RuntimeHookHandlerType::Prompt)
            }
        );
        assert_eq!(
            admit(&untrusted, &context).unwrap_err(),
            RuntimeHookOutcome::Failed {
                reason: RuntimeHookFailure::UntrustedSource(RuntimeHookTrustStatus::Untrusted)
            }
        );
    }

    #[test]
    fn disabled_and_mismatched_hooks_are_skipped_without_blocking() {
        let mut disabled = hook("disabled", 0, RuntimeHookEventName::PreToolUse);
        disabled.enabled = false;
        let mut scoped = hook("scoped", 1, RuntimeHookEventName::PreToolUse);
        scoped.matcher = Some("^apply_patch$".to_string());
        let snapshot = snapshot(vec![disabled, scoped]);

        let evaluation = evaluate_hook_event(
            &snapshot,
            RuntimeHookEventName::PreToolUse,
            &tool_context("shell"),
            |_| {
                panic!("skipped hooks must not reach the handler");
            },
        );

        assert!(!evaluation.is_blocked());
        assert_eq!(evaluation.participating_count(), 0);
        assert_eq!(
            evaluation.decisions[0].outcome,
            RuntimeHookOutcome::Skipped {
                reason: RuntimeHookSkipReason::Disabled
            }
        );
        assert_eq!(
            evaluation.decisions[1].outcome,
            RuntimeHookOutcome::Skipped {
                reason: RuntimeHookSkipReason::MatcherMismatch
            }
        );
    }

    #[test]
    fn invalid_matcher_fails_closed() {
        let mut broken = hook("broken", 0, RuntimeHookEventName::PreToolUse);
        broken.matcher = Some("([".to_string());
        let snapshot = snapshot(vec![broken]);

        let evaluation = evaluate_hook_event(
            &snapshot,
            RuntimeHookEventName::PreToolUse,
            &tool_context("shell"),
            |_| {
                panic!("invalid matcher must not reach the handler");
            },
        );

        assert!(evaluation.is_blocked());
        assert_eq!(
            evaluation.decisions[0].outcome,
            RuntimeHookOutcome::Failed {
                reason: RuntimeHookFailure::InvalidMatcher {
                    matcher: "([".to_string()
                }
            }
        );
    }

    #[test]
    fn rewrite_uses_last_hook_in_display_order_and_collects_context() {
        let snapshot = snapshot(vec![
            hook("first", 0, RuntimeHookEventName::PreToolUse),
            hook("second", 1, RuntimeHookEventName::PreToolUse),
            hook("third", 2, RuntimeHookEventName::PreToolUse),
        ]);

        let evaluation = evaluate_hook_event(
            &snapshot,
            RuntimeHookEventName::PreToolUse,
            &tool_context("shell"),
            |hook| match hook.key.as_str() {
                "first" => Some(RuntimeHookHandlerReport::Allow {
                    additional_context: Some("first note".to_string()),
                }),
                "second" => Some(RuntimeHookHandlerReport::Rewrite {
                    arguments: "{\"safe\":true}".to_string(),
                }),
                _ => Some(RuntimeHookHandlerReport::Rewrite {
                    arguments: "{\"safest\":true}".to_string(),
                }),
            },
        );

        assert!(!evaluation.is_blocked());
        assert_eq!(
            evaluation.rewritten_arguments().as_deref(),
            Some("{\"safest\":true}")
        );
        assert_eq!(
            evaluation.injected_context(),
            vec!["first note".to_string()]
        );
    }

    #[test]
    fn abort_is_distinguished_from_block() {
        let snapshot = snapshot(vec![hook("stop", 0, RuntimeHookEventName::PostToolUse)]);

        let evaluation = evaluate_hook_event(
            &snapshot,
            RuntimeHookEventName::PostToolUse,
            &tool_context("shell"),
            |_| {
                Some(RuntimeHookHandlerReport::Abort {
                    reason: "turn aborted".to_string(),
                })
            },
        );

        assert!(evaluation.is_blocked());
        assert!(evaluation.is_aborted());
        assert_eq!(
            evaluation.blocking_reason().as_deref(),
            Some("turn aborted")
        );
    }

    #[test]
    fn other_events_are_not_evaluated() {
        let snapshot = snapshot(vec![hook("pre", 0, RuntimeHookEventName::PreToolUse)]);

        let evaluation = evaluate_hook_event(
            &snapshot,
            RuntimeHookEventName::SessionEnd,
            &tool_context("shell"),
            |_| {
                panic!("unrelated events must not reach the handler");
            },
        );

        assert!(evaluation.decisions.is_empty());
        assert!(!evaluation.is_blocked());
    }

    #[test]
    fn async_execution_mode_is_admitted_for_out_of_band_reporting() {
        let mut background = hook("background", 0, RuntimeHookEventName::PostToolUse);
        background.execution_mode = RuntimeHookExecutionMode::Async;

        assert!(admit(&background, &tool_context("shell")).is_ok());
        let snapshot = RuntimeTurnSnapshot::try_new(Vec::new(), vec![background])
            .expect("async hook is valid");
        let evaluation = evaluate_hook_event(
            &snapshot,
            RuntimeHookEventName::PostToolUse,
            &tool_context("shell"),
            |_| {
                panic!("async hook must not run in the synchronous evaluator");
            },
        );
        assert_eq!(evaluation.injected_context(), Vec::<String>::new());
        assert!(!evaluation.is_blocked());
    }

    #[test]
    fn decision_keeps_stable_run_identity() {
        let snapshot = snapshot(vec![hook("keyed", 7, RuntimeHookEventName::PreToolUse)]);

        let evaluation = evaluate_hook_event(
            &snapshot,
            RuntimeHookEventName::PreToolUse,
            &tool_context("shell"),
            |_| {
                Some(RuntimeHookHandlerReport::Allow {
                    additional_context: None,
                })
            },
        );

        assert_eq!(evaluation.decisions[0].key, "keyed");
        assert_eq!(
            evaluation.decisions[0].run_id,
            "pre-tool-use:7:/etc/lime/hooks.json"
        );
    }
}
