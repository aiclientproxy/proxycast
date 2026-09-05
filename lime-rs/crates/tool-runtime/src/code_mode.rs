//! Code Mode compatibility surface.
//!
//! The current owners are the standalone `code-mode-protocol`,
//! `code-mode-runtime`, `code-mode-host`, and `code-mode` crates. Existing
//! tool-runtime consumers can migrate without a type conversion through these
//! explicit re-exports; no execution logic lives here.

pub use code_mode::{
    default_code_mode_host_path, DisabledCodeModeSessionProvider, ProcessCodeModeSessionProvider,
    ProcessOwnedCodeModeSession, ProcessOwnedCodeModeSessionProvider,
};
pub use code_mode_protocol::*;
pub use code_mode_runtime::V8CodeModeSessionProvider;

/// Compatibility namespace for the former process implementation.
pub mod process {
    pub use code_mode::default_code_mode_host_path;
    pub use code_mode::ProcessCodeModeSessionProvider;
    pub use code_mode_host::run_stdio;

    pub mod protocol {
        pub use code_mode_protocol::host::*;
    }
}

/// Compatibility namespace for the former in-process implementation.
pub mod v8 {
    pub use code_mode_runtime::V8CodeModeSessionProvider;
}

#[doc(hidden)]
pub async fn run_code_mode_host_stdio() -> Result<(), String> {
    code_mode_host::run_stdio().await
}
