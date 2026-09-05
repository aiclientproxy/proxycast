//! Code Mode session facade。
//!
//! 该 crate 负责 process-owned session provider 的生命周期和公开 facade；host
//! transport 与 V8 runtime 分别归 `code-mode-host`、`code-mode-runtime`。

pub use code_mode_protocol::*;

mod grpc_session;
mod remote_session;

pub use grpc_session::GrpcCodeModeSessionProvider;
pub use remote_session::ProcessCodeModeSessionProvider;

/// Host executable lookup used by the process-owned provider.
pub fn default_code_mode_host_path() -> std::path::PathBuf {
    const BINARY_NAME: &str = "code-mode-host";
    if let Some(path) = std::env::var_os("CODE_MODE_HOST_BIN").filter(|path| !path.is_empty()) {
        return std::path::PathBuf::from(path);
    }
    let binary_name = if cfg!(windows) {
        format!("{BINARY_NAME}.exe")
    } else {
        BINARY_NAME.to_string()
    };
    std::env::current_exe()
        .ok()
        .and_then(|path| {
            let parent = path.parent()?;
            let sibling = parent.join(&binary_name);
            if sibling.is_file() {
                return Some(sibling);
            }
            if parent.file_name().is_some_and(|name| name == "deps") {
                return parent.parent().map(|target| target.join(&binary_name));
            }
            Some(sibling)
        })
        .unwrap_or_else(|| std::path::PathBuf::from(binary_name))
}

/// Codex-compatible process-owned provider 名称。
pub type ProcessOwnedCodeModeSessionProvider = ProcessCodeModeSessionProvider;
/// 公开 handle 作为 facade session，避免暴露 transport 私有实现。
pub type ProcessOwnedCodeModeSession = RuntimeCodeModeSessionHandle;

/// 在能力探测或显式禁用时使用的 fail-closed provider。
#[derive(Clone, Copy, Debug, Default)]
pub struct DisabledCodeModeSessionProvider;

impl RuntimeCodeModeSessionProvider for DisabledCodeModeSessionProvider {
    fn availability(&self) -> Result<(), String> {
        Err("code mode is disabled".to_string())
    }

    fn create_session<'a>(
        &'a self,
        _delegate: std::sync::Arc<dyn RuntimeCodeModeSessionDelegate>,
    ) -> RuntimeCodeModeSessionProviderFuture<'a> {
        Box::pin(async { Err("code mode is disabled".to_string()) })
    }
}

/// 统一的 facade provider 构造别名。
pub type CodeModeSessionProviderHandle = std::sync::Arc<dyn RuntimeCodeModeSessionProvider>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_provider_fails_closed() {
        assert!(DisabledCodeModeSessionProvider.availability().is_err());
    }
}
