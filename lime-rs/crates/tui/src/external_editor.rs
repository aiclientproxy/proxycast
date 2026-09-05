use std::env;
use std::fs;
use std::path::Path;
use std::process::Stdio;

use anyhow::{Context, Result, bail};
use tokio::process::Command;

/// Open the user's configured editor and return the edited draft.
///
/// `VISUAL` takes precedence over `EDITOR`, matching common Unix terminal
/// conventions. The editor command is intentionally split only on shell
/// whitespace; the file path is appended as a separate argument so user input
/// cannot be interpreted as editor flags.
pub(crate) async fn edit_draft(initial: &str, cwd: &Path) -> Result<Option<String>> {
    let command = env::var("VISUAL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            env::var("EDITOR")
                .ok()
                .filter(|value| !value.trim().is_empty())
        })
        .ok_or_else(|| anyhow::anyhow!("cannot open external editor: set $VISUAL or $EDITOR"))?;
    edit_draft_with_command(initial, cwd, &command).await
}

async fn edit_draft_with_command(
    initial: &str,
    cwd: &Path,
    command: &str,
) -> Result<Option<String>> {
    let parts = command_parts(command)?;
    let executable = parts
        .first()
        .ok_or_else(|| anyhow::anyhow!("external editor command is empty"))?;
    let temporary = tempfile::Builder::new()
        .prefix("lime-tui-")
        .suffix(".md")
        .tempfile_in(cwd)
        .or_else(|_| {
            tempfile::Builder::new()
                .prefix("lime-tui-")
                .suffix(".md")
                .tempfile()
        })?
        .into_temp_path();
    // Close the temporary file before launching the editor so Windows shims can open it.
    fs::write(&temporary, initial).context("failed to write external editor draft")?;
    let mut editor = Command::new(resolve_editor_executable(executable));
    editor
        .args(&parts[1..])
        .arg(&temporary)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    let status = editor
        .status()
        .await
        .with_context(|| format!("failed to start external editor {executable:?}"))?;
    if !status.success() {
        bail!("external editor exited with status {status}");
    }
    let edited = fs::read_to_string(&temporary).context("failed to read external editor draft")?;
    if edited.trim().is_empty() {
        return Ok(None);
    }
    Ok(Some(edited))
}

#[cfg(not(windows))]
fn resolve_editor_executable(executable: &str) -> &str {
    executable
}

#[cfg(windows)]
fn resolve_editor_executable(executable: &str) -> std::path::PathBuf {
    // `Command::new` does not resolve PATH shims such as `code.cmd` on Windows.
    which::which(executable).unwrap_or_else(|_| std::path::PathBuf::from(executable))
}

fn command_parts(command: &str) -> Result<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    let mut escaped = false;
    for character in command.chars() {
        if escaped {
            current.push(character);
            escaped = false;
            continue;
        }
        match (quote, character) {
            (_, '\\') => escaped = true,
            (Some(active), value) if value == active => quote = None,
            (Some(_), value) => current.push(value),
            (None, '\'' | '"') => quote = Some(character),
            (None, value) if value.is_whitespace() => {
                if !current.is_empty() {
                    parts.push(std::mem::take(&mut current));
                }
            }
            (None, value) => current.push(value),
        }
    }
    if escaped || quote.is_some() {
        bail!("external editor command has an unterminated escape or quote");
    }
    if !current.is_empty() {
        parts.push(current);
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn splits_editor_command_without_losing_quotes() {
        assert_eq!(
            command_parts("code --wait 'workspace file'").expect("parts"),
            vec!["code", "--wait", "workspace file"]
        );
    }

    #[test]
    fn rejects_unterminated_quotes() {
        assert!(command_parts("vim '").is_err());
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn edits_draft_after_closing_the_tempfile_handle() {
        let directory = tempfile::tempdir().expect("temp directory");
        let script = directory.path().join("editor.sh");
        fs::write(&script, "#!/bin/sh\nprintf 'edited' > \"$1\"\n").expect("script");
        let mut permissions = fs::metadata(&script)
            .expect("script metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).expect("script permissions");

        let edited = edit_draft_with_command(
            "seed",
            directory.path(),
            script.to_str().expect("script path"),
        )
        .await
        .expect("editor");

        assert_eq!(edited.as_deref(), Some("edited"));
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn editor_failure_is_returned_without_leaking_the_tempfile() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().expect("temp directory");
        let script = directory.path().join("editor-fails.sh");
        fs::write(&script, "#!/bin/sh\nexit 7\n").expect("script");
        let mut permissions = fs::metadata(&script)
            .expect("script metadata")
            .permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&script, permissions).expect("script permissions");

        let error = edit_draft_with_command(
            "seed",
            directory.path(),
            script.to_str().expect("script path"),
        )
        .await
        .expect_err("non-zero editor must fail");

        assert!(error.to_string().contains("exited with status"));
        assert!(
            fs::read_dir(directory.path())
                .expect("directory listing")
                .filter_map(Result::ok)
                .all(|entry| !entry.file_name().to_string_lossy().starts_with("lime-tui-"))
        );
    }
}
