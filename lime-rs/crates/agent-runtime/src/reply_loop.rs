//! Reply loop 的 current Turn 规则骨架。
//!
//! 这里只保存 provider/reply loop 的纯状态和退出文案，不引入具体
//! provider、tool、session store 或 Agent 事件类型。

pub const DEFAULT_MAX_REPLY_TURNS: u32 = 1000;
pub const DEFAULT_MAX_EMPTY_RESPONSE_RETRIES: u32 = 2;
pub const MAX_REPLY_TURNS_REACHED_MESSAGE: &str =
    "I've reached the maximum number of actions I can do without user input. Would you like me to continue?";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeReplyLoop {
    attempts_taken: u32,
    reply_turns_taken: u32,
    empty_response_retries: u32,
    max_turns: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeReplyLoopStep {
    Continue { attempt: u32 },
    MaxTurnsReached { attempt: u32, max_turns: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeEmptyResponseStep {
    Retry { retry: u32, max_retries: u32 },
    Exhausted { retries: u32, max_retries: u32 },
}

impl RuntimeReplyLoop {
    pub fn new(max_turns: Option<u32>) -> Self {
        Self {
            attempts_taken: 0,
            reply_turns_taken: 0,
            empty_response_retries: 0,
            max_turns: max_turns.unwrap_or(DEFAULT_MAX_REPLY_TURNS),
        }
    }

    pub fn max_turns(&self) -> u32 {
        self.max_turns
    }

    pub fn attempts_taken(&self) -> u32 {
        self.attempts_taken
    }

    pub fn next_attempt(&mut self) -> RuntimeReplyLoopStep {
        self.empty_response_retries = 0;
        self.attempts_taken = self.attempts_taken.saturating_add(1);
        self.reply_turns_taken = self.reply_turns_taken.saturating_add(1);
        if self.reply_turns_taken > self.max_turns {
            return RuntimeReplyLoopStep::MaxTurnsReached {
                attempt: self.attempts_taken,
                max_turns: self.max_turns,
            };
        }

        RuntimeReplyLoopStep::Continue {
            attempt: self.attempts_taken,
        }
    }

    pub fn next_retry_attempt(&mut self) -> u32 {
        self.attempts_taken = self.attempts_taken.saturating_add(1);
        self.attempts_taken
    }

    pub fn request_empty_response_retry(&mut self) -> RuntimeEmptyResponseStep {
        if self.empty_response_retries >= DEFAULT_MAX_EMPTY_RESPONSE_RETRIES {
            return RuntimeEmptyResponseStep::Exhausted {
                retries: self.empty_response_retries,
                max_retries: DEFAULT_MAX_EMPTY_RESPONSE_RETRIES,
            };
        }
        self.empty_response_retries = self.empty_response_retries.saturating_add(1);
        RuntimeEmptyResponseStep::Retry {
            retry: self.empty_response_retries,
            max_retries: DEFAULT_MAX_EMPTY_RESPONSE_RETRIES,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_default_max_turns() {
        let loop_state = RuntimeReplyLoop::new(None);

        assert_eq!(loop_state.max_turns(), DEFAULT_MAX_REPLY_TURNS);
        assert_eq!(loop_state.attempts_taken(), 0);
    }

    #[test]
    fn yields_attempt_until_max_is_reached() {
        let mut loop_state = RuntimeReplyLoop::new(Some(2));

        assert_eq!(
            loop_state.next_attempt(),
            RuntimeReplyLoopStep::Continue { attempt: 1 }
        );
        assert_eq!(
            loop_state.next_attempt(),
            RuntimeReplyLoopStep::Continue { attempt: 2 }
        );
        assert_eq!(
            loop_state.next_attempt(),
            RuntimeReplyLoopStep::MaxTurnsReached {
                attempt: 3,
                max_turns: 2
            }
        );
    }

    #[test]
    fn empty_response_retries_are_bounded_without_spending_reply_turns() {
        let mut loop_state = RuntimeReplyLoop::new(Some(1));

        assert_eq!(
            loop_state.next_attempt(),
            RuntimeReplyLoopStep::Continue { attempt: 1 }
        );
        assert_eq!(
            loop_state.request_empty_response_retry(),
            RuntimeEmptyResponseStep::Retry {
                retry: 1,
                max_retries: 2
            }
        );
        assert_eq!(loop_state.next_retry_attempt(), 2);
        assert_eq!(
            loop_state.request_empty_response_retry(),
            RuntimeEmptyResponseStep::Retry {
                retry: 2,
                max_retries: 2
            }
        );
        assert_eq!(loop_state.next_retry_attempt(), 3);
        assert_eq!(
            loop_state.request_empty_response_retry(),
            RuntimeEmptyResponseStep::Exhausted {
                retries: 2,
                max_retries: 2
            }
        );
        assert_eq!(loop_state.attempts_taken(), 3);
        assert_eq!(
            loop_state.next_attempt(),
            RuntimeReplyLoopStep::MaxTurnsReached {
                attempt: 4,
                max_turns: 1
            }
        );
    }
}
