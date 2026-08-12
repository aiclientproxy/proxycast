mod client;
mod client_state;
mod host;
mod protocol;

use std::path::PathBuf;

pub use client::ProcessCodeModeSessionProvider;

const CODE_MODE_HOST_BINARY_NAME: &str = "code-mode-host";
const CODE_MODE_HOST_BIN_ENV: &str = "CODE_MODE_HOST_BIN";

pub fn default_code_mode_host_path() -> PathBuf {
    if let Some(path) = std::env::var_os(CODE_MODE_HOST_BIN_ENV).filter(|path| !path.is_empty()) {
        return PathBuf::from(path);
    }

    let binary_name = if cfg!(windows) {
        format!("{CODE_MODE_HOST_BINARY_NAME}.exe")
    } else {
        CODE_MODE_HOST_BINARY_NAME.to_string()
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
        .unwrap_or_else(|| PathBuf::from(binary_name))
}

pub async fn run_stdio() -> Result<(), String> {
    host::run_stdio().await
}

#[cfg(test)]
mod tests;
