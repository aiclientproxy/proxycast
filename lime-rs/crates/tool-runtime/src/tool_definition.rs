//! Tool metadata is part of the Code Mode protocol contract.
//!
//! This module remains as a compatibility path for existing tool-runtime
//! consumers; the definitions themselves are owned by `code-mode-protocol`.

pub use code_mode_protocol::{RuntimeToolDefinition, RuntimeToolExposure};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_tool_exposure_distinguishes_direct_model_surfaces() {
        assert!(RuntimeToolExposure::Direct.is_direct());
        assert!(RuntimeToolExposure::DirectModelOnly.is_direct());
        assert!(!RuntimeToolExposure::Deferred.is_direct());
        assert!(!RuntimeToolExposure::DeferredModelOnly.is_direct());
        assert!(!RuntimeToolExposure::CodeModeOnly.is_direct());
        assert!(!RuntimeToolExposure::Hidden.is_direct());
        assert!(RuntimeToolExposure::Deferred.is_deferred());
        assert!(RuntimeToolExposure::DeferredModelOnly.is_deferred());
        assert!(RuntimeToolExposure::Direct.is_available_in_code_mode());
        assert!(RuntimeToolExposure::Deferred.is_available_in_code_mode());
        assert!(RuntimeToolExposure::CodeModeOnly.is_available_in_code_mode());
        assert!(!RuntimeToolExposure::DirectModelOnly.is_available_in_code_mode());
    }
}
