//! Code Mode V8 执行 runtime。
//!
//! V8 cell actor/session runtime 的实现仍复用已经经过 Lime Gate B 验证的实现，
//! 但通过独立 crate 暴露，构建图和职责与 Codex 对齐。

#![allow(hidden_glob_reexports)]

pub use code_mode_protocol::*;
mod cell_actor;
mod runtime;
mod service;
mod session_runtime;
mod v8_init;

pub(crate) type TaskFailureHandler = std::sync::Arc<dyn Fn(String) + Send + Sync>;

pub use service::{InProcessCodeModeSession, V8CodeModeSessionProvider};

/// 初始化 V8 运行时。
pub fn initialize_v8() -> Result<(), String> {
    v8_init::ensure_v8_initialized()
}

/// V8 初始化模式占位类型，保留 Codex runtime 的显式扩展点。
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum V8JitMode {
    #[default]
    Default,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_exposes_in_process_provider() {
        let _provider = V8CodeModeSessionProvider;
        let _ = V8JitMode::Default;
    }
}
