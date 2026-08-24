use crate::tool_definition::RuntimeToolDefinition;
use crate::tool_executor::{
    RuntimeToolExecutionError, RuntimeToolExecutionRequest, RuntimeToolExecutionResult,
    RuntimeToolExecutor, RuntimeToolExecutorHandle, RuntimeToolPolicyErrorKind,
};
use app_server_protocol::protocol::v2::{FileSystemAccessMode, FileSystemPath};
use patch_apply::{
    apply_hunks_to_workdir, apply_patch_to_workdir, parse_patch, AppliedPatchFileChange,
    ApplyPatchReport, Hunk,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, OnceLock};

pub const APPLY_PATCH_TOOL_NAME: &str = "apply_patch";
pub const TURN_DIFF_DELTA_METADATA_KEY: &str = "turnDiffDelta";

#[derive(Debug, Default)]
pub struct RuntimeApplyPatchExecutor;

impl RuntimeApplyPatchExecutor {
    pub fn new() -> Self {
        Self
    }

    pub fn handle() -> RuntimeToolExecutorHandle {
        RuntimeToolExecutorHandle::new(Arc::new(Self::new()))
    }

    async fn execute_apply_patch(
        &self,
        request: RuntimeToolExecutionRequest<'_>,
    ) -> Result<RuntimeToolExecutionResult, RuntimeToolExecutionError> {
        if request.tool_name != APPLY_PATCH_TOOL_NAME {
            return Err(runtime_apply_patch_error(format!(
                "apply_patch executor cannot run tool '{}'",
                request.tool_name
            )));
        }
        if request
            .context
            .cancel_token()
            .is_some_and(|token| token.is_cancelled())
        {
            return Err(RuntimeToolExecutionError::new(
                "apply_patch cancelled",
                Some(RuntimeToolPolicyErrorKind::Canceled(
                    "apply_patch_cancelled".to_string(),
                )),
            )
            .before_handler());
        }

        match request.context.execution_attempt() {
            Some(attempt) => check_runtime_apply_patch_attempt_permissions(
                request.params,
                request.context.working_directory(),
                attempt,
            )?,
            None => check_runtime_apply_patch_permissions(
                request.params,
                request.context.working_directory(),
            )?,
        }

        let patch = patch_text_from_params(request.params)?;
        if let Some(environment_id) = request
            .context
            .environment_id()
            .filter(|environment_id| *environment_id != "local")
        {
            let gateway = request.context.filesystem_gateway().ok_or_else(|| {
                runtime_apply_patch_error(format!(
                    "remote Environment '{environment_id}' filesystem gateway is unavailable"
                ))
            })?;
            let result = gateway
                .apply_patch(
                    environment_id,
                    request.context.working_directory(),
                    &patch,
                    request
                        .turn_context
                        .and_then(|turn| turn.sandbox_policy.as_deref()),
                )
                .await
                .map_err(runtime_apply_patch_error)?;
            return Ok(RuntimeToolExecutionResult::new(
                true,
                format!("Applied patch to {} file(s)", result.modified_paths.len()),
                None,
                HashMap::from([
                    ("environment_id".to_string(), json!(environment_id)),
                    (
                        "modified_paths".to_string(),
                        json!(result
                            .modified_paths
                            .iter()
                            .map(|path| path.display().to_string())
                            .collect::<Vec<_>>()),
                    ),
                ]),
            ));
        }
        let workdir = request.context.working_directory().clone();
        let canonical_workdir = workdir.canonicalize().unwrap_or_else(|_| workdir.clone());
        let apply_root = request
            .context
            .execution_attempt()
            .filter(|attempt| {
                attempt.effective_sandbox_policy()
                    == crate::execution_orchestrator::RuntimeToolSandboxPolicy::DangerFullAccess
            })
            .map(|_| filesystem_root(&canonical_workdir))
            .transpose()?
            .unwrap_or_else(|| canonical_workdir.clone());
        let parsed =
            parse_patch(&patch).map_err(|error| runtime_apply_patch_error(error.to_string()))?;
        let hunks = if apply_root == canonical_workdir {
            None
        } else {
            Some(resolve_hunks_for_apply(&parsed.hunks, &canonical_workdir)?)
        };
        let patch_for_apply = patch.clone();
        let report = tokio::task::spawn_blocking(move || match hunks {
            Some(hunks) => apply_hunks_to_workdir(&hunks, apply_root),
            None => apply_patch_to_workdir(&patch_for_apply, apply_root),
        })
        .await
        .map_err(|error| runtime_apply_patch_error(format!("apply_patch task failed: {error}")))?
        .map_err(|error| runtime_apply_patch_error(error.to_string()))?;

        Ok(RuntimeToolExecutionResult::new(
            true,
            summary_text(&report),
            None,
            build_metadata(&patch, &canonical_workdir, &report),
        ))
    }
}

impl RuntimeToolExecutor for RuntimeApplyPatchExecutor {
    fn execute<'a>(
        &'a self,
        request: RuntimeToolExecutionRequest<'a>,
    ) -> crate::tool_executor::RuntimeToolExecutionFuture<'a> {
        Box::pin(async move { self.execute_apply_patch(request).await })
    }
}

pub fn runtime_apply_patch_executor_handle() -> RuntimeToolExecutorHandle {
    static HANDLE: OnceLock<RuntimeToolExecutorHandle> = OnceLock::new();
    HANDLE
        .get_or_init(RuntimeApplyPatchExecutor::handle)
        .clone()
}

pub fn apply_patch_tool_definition() -> RuntimeToolDefinition {
    RuntimeToolDefinition::new(
        APPLY_PATCH_TOOL_NAME,
        "Use this tool to edit files inside the current workspace. Send a structured patch in the patch field. For update hunks, every content line must start immediately with ' ' (context), '-' (remove), or '+' (add); do not add a space after '-' or '+' unless that space belongs in the file.",
        json!({
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Patch text with *** Begin Patch and *** End Patch markers. Example: *** Begin Patch\n*** Update File: probe.txt\n@@\n-before\n+after\n*** End Patch"
                }
            },
            "required": ["patch"]
        }),
    )
}

pub fn check_runtime_apply_patch_permissions(
    params: &Value,
    workdir: &Path,
) -> Result<(), RuntimeToolExecutionError> {
    let workdir = workdir
        .canonicalize()
        .unwrap_or_else(|_| workdir.to_path_buf());
    for candidate in resolved_patch_write_paths(params, &workdir)? {
        if !candidate.starts_with(&workdir) {
            return Err(runtime_apply_patch_permission_error(format!(
                "Patch path '{}' must stay inside workspace '{}'",
                candidate.display(),
                workdir.display()
            )));
        }
    }
    Ok(())
}

fn check_runtime_apply_patch_attempt_permissions(
    params: &Value,
    workdir: &Path,
    attempt: &crate::execution_orchestrator::RuntimeToolExecutionAttempt,
) -> Result<(), RuntimeToolExecutionError> {
    use crate::execution_orchestrator::RuntimeToolSandboxPolicy;

    if attempt.is_cancelled() {
        return Err(RuntimeToolExecutionError::new(
            "apply_patch cancelled",
            Some(RuntimeToolPolicyErrorKind::Canceled(
                "apply_patch_cancelled".to_string(),
            )),
        )
        .before_handler());
    }
    if attempt.effective_sandbox_policy() == RuntimeToolSandboxPolicy::ReadOnly {
        return Err(runtime_apply_patch_sandbox_error(
            "read-only sandbox denies apply_patch writes",
        ));
    }
    let workdir = workdir
        .canonicalize()
        .unwrap_or_else(|_| workdir.to_path_buf());
    for candidate in resolved_patch_write_paths(params, &workdir)? {
        if candidate.starts_with(&workdir)
            || attempt.effective_sandbox_policy() == RuntimeToolSandboxPolicy::DangerFullAccess
            || granted_write_path(attempt.granted_permissions(), &candidate)
        {
            continue;
        }
        return Err(runtime_apply_patch_sandbox_error(format!(
            "filesystem sandbox denies patch write to '{}'",
            candidate.display()
        )));
    }
    Ok(())
}

fn resolved_patch_write_paths(
    params: &Value,
    workdir: &Path,
) -> Result<Vec<PathBuf>, RuntimeToolExecutionError> {
    let patch = patch_text_from_params(params)?;
    let parsed = parse_patch(&patch)
        .map_err(|error| runtime_apply_patch_permission_error(error.to_string()))?;
    let mut paths = Vec::new();
    for path in parsed.hunks.iter().flat_map(hunk_write_paths) {
        let candidate = resolve_patch_path_for_permission(&workdir, &path)
            .map_err(runtime_apply_patch_permission_error)?;
        paths.push(candidate);
    }
    Ok(paths)
}

fn resolve_hunks_for_apply(
    hunks: &[Hunk],
    workdir: &Path,
) -> Result<Vec<Hunk>, RuntimeToolExecutionError> {
    hunks
        .iter()
        .map(|hunk| match hunk {
            Hunk::AddFile { path, contents } => Ok(Hunk::AddFile {
                path: resolve_patch_path_for_permission(workdir, path)
                    .map_err(runtime_apply_patch_error)?,
                contents: contents.clone(),
            }),
            Hunk::DeleteFile { path } => Ok(Hunk::DeleteFile {
                path: resolve_patch_path_for_permission(workdir, path)
                    .map_err(runtime_apply_patch_error)?,
            }),
            Hunk::UpdateFile {
                path,
                move_path,
                chunks,
            } => Ok(Hunk::UpdateFile {
                path: resolve_patch_path_for_permission(workdir, path)
                    .map_err(runtime_apply_patch_error)?,
                move_path: move_path
                    .as_ref()
                    .map(|path| {
                        resolve_patch_path_for_permission(workdir, path)
                            .map_err(runtime_apply_patch_error)
                    })
                    .transpose()?,
                chunks: chunks.clone(),
            }),
        })
        .collect()
}

fn filesystem_root(path: &Path) -> Result<PathBuf, RuntimeToolExecutionError> {
    let mut root = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => root.push(prefix.as_os_str()),
            Component::RootDir => {
                root.push(component.as_os_str());
                return Ok(root);
            }
            Component::CurDir | Component::ParentDir | Component::Normal(_) => break,
        }
    }
    Err(runtime_apply_patch_error(format!(
        "cannot resolve filesystem root for '{}'",
        path.display()
    )))
}

fn granted_write_path(
    permissions: &app_server_protocol::protocol::v2::GrantedPermissionProfile,
    candidate: &Path,
) -> bool {
    let Some(file_system) = permissions.file_system.as_ref() else {
        return false;
    };
    let explicitly_denied = file_system
        .entries
        .as_deref()
        .unwrap_or_default()
        .iter()
        .any(|entry| {
            entry.access == FileSystemAccessMode::Deny
                && file_system_entry_matches(&entry.path, candidate)
        });
    if explicitly_denied {
        return false;
    }
    file_system
        .write
        .as_deref()
        .unwrap_or_default()
        .iter()
        .any(|path| path_grants_candidate(Path::new(path), candidate))
        || file_system
            .entries
            .as_deref()
            .unwrap_or_default()
            .iter()
            .any(|entry| {
                entry.access == FileSystemAccessMode::Write
                    && file_system_entry_matches(&entry.path, candidate)
            })
}

fn file_system_entry_matches(path: &FileSystemPath, candidate: &Path) -> bool {
    match path {
        FileSystemPath::Path { path } => path_grants_candidate(Path::new(path), candidate),
        FileSystemPath::Special {
            value:
                app_server_protocol::protocol::v2::FileSystemSpecialPath::Unknown { path, subpath },
        } => {
            let root = subpath
                .as_deref()
                .map(|subpath| Path::new(path).join(subpath))
                .unwrap_or_else(|| PathBuf::from(path));
            path_grants_candidate(&root, candidate)
        }
        FileSystemPath::GlobPattern { .. } | FileSystemPath::Special { .. } => false,
    }
}

fn path_grants_candidate(root: &Path, candidate: &Path) -> bool {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    candidate == root || candidate.starts_with(root)
}

fn patch_text_from_params(params: &Value) -> Result<String, RuntimeToolExecutionError> {
    params
        .get("patch")
        .or_else(|| params.get("input"))
        .or_else(|| params.get("stdin"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| runtime_apply_patch_error("Missing required parameter: patch"))
}

fn hunk_write_paths(hunk: &Hunk) -> Vec<PathBuf> {
    match hunk {
        Hunk::AddFile { path, .. } | Hunk::DeleteFile { path } => vec![path.clone()],
        Hunk::UpdateFile {
            path, move_path, ..
        } => {
            let mut paths = vec![path.clone()];
            if let Some(move_path) = move_path {
                paths.push(move_path.clone());
            }
            paths
        }
    }
}

fn resolve_patch_path_for_permission(workdir: &Path, path: &Path) -> Result<PathBuf, String> {
    let lexical_path = if path.is_absolute() {
        normalize_path_lexically(path)?
    } else {
        normalize_path_lexically(&workdir.join(path))?
    };
    canonicalize_existing_prefix(&lexical_path)
}

fn canonicalize_existing_prefix(path: &Path) -> Result<PathBuf, String> {
    let mut existing = path;
    let mut missing_parts: Vec<PathBuf> = Vec::new();

    while !existing.exists() {
        let Some(parent) = existing.parent() else {
            return Ok(path.to_path_buf());
        };
        let Some(file_name) = existing.file_name() else {
            return Ok(path.to_path_buf());
        };
        missing_parts.push(PathBuf::from(file_name));
        existing = parent;
    }

    let mut canonical = existing.canonicalize().map_err(|error| {
        format!(
            "Failed to resolve patch path '{}': {error}",
            existing.display()
        )
    })?;
    for part in missing_parts.iter().rev() {
        canonical.push(part);
    }
    Ok(canonical)
}

fn normalize_path_lexically(path: &Path) -> Result<PathBuf, String> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => normalized.push(prefix.as_os_str()),
            Component::RootDir => normalized.push(component.as_os_str()),
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(format!(
                        "Patch path '{}' traverses above filesystem root",
                        path.display()
                    ));
                }
            }
        }
    }
    Ok(normalized)
}

fn build_metadata(
    patch: &str,
    workdir: &Path,
    report: &ApplyPatchReport,
) -> HashMap<String, Value> {
    let mut metadata = HashMap::new();
    let paths = report
        .changes
        .iter()
        .map(|change| display_path(workdir, &change.path))
        .collect::<Vec<_>>();
    metadata.insert("patch".to_string(), json!(patch));
    metadata.insert("paths".to_string(), json!(paths));
    metadata.insert("artifact_paths".to_string(), json!(paths));
    if let Some(first_path) = paths.first() {
        metadata.insert("path".to_string(), json!(first_path));
    }
    metadata.insert("file_changes".to_string(), report_to_value(workdir, report));
    metadata.insert(
        TURN_DIFF_DELTA_METADATA_KEY.to_string(),
        turn_diff_delta_to_value(workdir, report),
    );
    if let Some(first_change) = report.changes.first() {
        metadata.insert(
            "file_change".to_string(),
            change_to_value(workdir, first_change),
        );
    }
    metadata
}

fn turn_diff_delta_to_value(workdir: &Path, report: &ApplyPatchReport) -> Value {
    json!({
        "changes": report
            .changes
            .iter()
            .map(|change| turn_diff_change_to_value(workdir, change))
            .collect::<Vec<_>>(),
    })
}

fn turn_diff_change_to_value(workdir: &Path, change: &patch_apply::AppliedPatchChange) -> Value {
    let path = display_path(workdir, &change.path);
    match &change.change {
        AppliedPatchFileChange::Add {
            content,
            overwritten_content,
        } => compact_object(json!({
            "type": "add",
            "path": path,
            "content": content,
            "overwrittenContent": overwritten_content,
        })),
        AppliedPatchFileChange::Delete { content } => json!({
            "type": "delete",
            "path": path,
            "content": content,
        }),
        AppliedPatchFileChange::Update {
            move_path,
            old_content,
            overwritten_move_content,
            new_content,
        } => compact_object(json!({
            "type": "update",
            "path": path,
            "movePath": move_path.as_ref().map(|path| display_path(workdir, path)),
            "oldContent": old_content,
            "overwrittenMoveContent": overwritten_move_content,
            "newContent": new_content,
        })),
    }
}

fn report_to_value(workdir: &Path, report: &ApplyPatchReport) -> Value {
    json!({
        "added": report.added.iter().map(|path| path.to_string_lossy().to_string()).collect::<Vec<_>>(),
        "modified": report.modified.iter().map(|path| path.to_string_lossy().to_string()).collect::<Vec<_>>(),
        "deleted": report.deleted.iter().map(|path| path.to_string_lossy().to_string()).collect::<Vec<_>>(),
        "changes": report.changes.iter().map(|change| change_to_value(workdir, change)).collect::<Vec<_>>(),
    })
}

fn change_to_value(workdir: &Path, change: &patch_apply::AppliedPatchChange) -> Value {
    let path = display_path(workdir, &change.path);
    match &change.change {
        AppliedPatchFileChange::Add {
            content,
            overwritten_content,
        } => {
            let refs = file_refs(
                &path,
                overwritten_content.as_deref().unwrap_or_default(),
                content,
            );
            compact_object(json!({
                "kind": if overwritten_content.is_some() { "update" } else { "add" },
                "path": path,
                "content_size": content.len(),
                "overwrote_existing": overwritten_content.is_some(),
                "previousContent": overwritten_content,
                "contentPreview": preview_text(content),
                "diff": line_diff(overwritten_content.as_deref().unwrap_or_default(), content),
                "checkpointRef": refs.checkpoint_ref,
                "contentRef": refs.content_ref,
                "diffRef": refs.diff_ref,
            }))
        }
        AppliedPatchFileChange::Delete { content } => {
            let refs = file_refs(&path, content, "");
            json!({
                "kind": "delete",
                "path": path,
                "previous_content_size": content.len(),
                "previousContent": content,
                "diff": line_diff(content, ""),
                "checkpointRef": refs.checkpoint_ref,
                "contentRef": refs.content_ref,
                "diffRef": refs.diff_ref,
            })
        }
        AppliedPatchFileChange::Update {
            move_path,
            old_content,
            overwritten_move_content,
            new_content,
        } => {
            let destination_path = move_path
                .as_ref()
                .map(|path| display_path(workdir, path))
                .unwrap_or_else(|| path.clone());
            let destination_refs = file_refs(&destination_path, old_content, new_content);
            compact_object(json!({
                "kind": if move_path.is_some() { "move_update" } else { "update" },
                "path": destination_path,
                "sourcePath": if move_path.is_some() { Some(path) } else { None },
                "move_path": move_path.as_ref().map(|path| display_path(workdir, path)),
                "previous_content_size": old_content.len(),
                "content_size": new_content.len(),
                "previousContent": old_content,
                "contentPreview": preview_text(new_content),
                "diff": line_diff(old_content, new_content),
                "overwrote_move_destination": overwritten_move_content.is_some(),
                "overwrittenMovePreviousContent": overwritten_move_content,
                "checkpointRef": destination_refs.checkpoint_ref,
                "contentRef": destination_refs.content_ref,
                "diffRef": destination_refs.diff_ref,
            }))
        }
    }
}

struct FileRefs {
    checkpoint_ref: String,
    content_ref: String,
    diff_ref: String,
}

fn file_refs(path: &str, previous_content: &str, content: &str) -> FileRefs {
    let checkpoint_hash = stable_hash(format!("{path}\n{content}").as_str());
    let content_hash = stable_hash(content);
    let diff_hash = stable_hash(format!("{path}\n{previous_content}\n{content}").as_str());
    FileRefs {
        checkpoint_ref: format!("checkpoint:file:{checkpoint_hash:016x}"),
        content_ref: format!("content:file:{content_hash:016x}"),
        diff_ref: format!("diff:file:{diff_hash:016x}"),
    }
}

fn display_path(workdir: &Path, path: &Path) -> String {
    path.strip_prefix(workdir)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

fn preview_text(value: &str) -> String {
    value.chars().take(240).collect()
}

fn line_diff(previous: &str, content: &str) -> Value {
    const MAX_DIFF_ITEMS: usize = 240;
    let previous_lines = previous.lines().collect::<Vec<_>>();
    let content_lines = content.lines().collect::<Vec<_>>();
    let mut prefix_len = 0;
    while previous_lines.get(prefix_len) == content_lines.get(prefix_len)
        && prefix_len < previous_lines.len()
        && prefix_len < content_lines.len()
    {
        prefix_len += 1;
    }

    let mut suffix_len = 0;
    while suffix_len + prefix_len < previous_lines.len()
        && suffix_len + prefix_len < content_lines.len()
        && previous_lines[previous_lines.len() - 1 - suffix_len]
            == content_lines[content_lines.len() - 1 - suffix_len]
    {
        suffix_len += 1;
    }

    let mut diff = Vec::new();
    for line in previous_lines.iter().take(prefix_len) {
        push_diff_item(&mut diff, "context", line, MAX_DIFF_ITEMS);
    }
    for line in previous_lines
        .iter()
        .skip(prefix_len)
        .take(previous_lines.len().saturating_sub(prefix_len + suffix_len))
    {
        push_diff_item(&mut diff, "remove", line, MAX_DIFF_ITEMS);
    }
    for line in content_lines
        .iter()
        .skip(prefix_len)
        .take(content_lines.len().saturating_sub(prefix_len + suffix_len))
    {
        push_diff_item(&mut diff, "add", line, MAX_DIFF_ITEMS);
    }
    for line in previous_lines
        .iter()
        .skip(previous_lines.len() - suffix_len)
    {
        push_diff_item(&mut diff, "context", line, MAX_DIFF_ITEMS);
    }

    Value::Array(diff)
}

fn push_diff_item(diff: &mut Vec<Value>, kind: &str, line: &str, max_items: usize) {
    if diff.len() >= max_items {
        if diff
            .last()
            .and_then(|value| value.get("kind"))
            .and_then(Value::as_str)
            != Some("truncated")
        {
            diff.push(json!({ "kind": "truncated" }));
        }
        return;
    }
    diff.push(json!({
        "kind": kind,
        "value": line,
    }));
}

fn compact_object(value: Value) -> Value {
    match value {
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .filter_map(|(key, value)| {
                    let compacted = compact_object(value);
                    if compacted.is_null() {
                        None
                    } else {
                        Some((key, compacted))
                    }
                })
                .collect(),
        ),
        Value::Array(values) => Value::Array(values.into_iter().map(compact_object).collect()),
        other => other,
    }
}

fn stable_hash(value: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn summary_text(report: &ApplyPatchReport) -> String {
    let mut lines = vec!["Success. Updated the following files:".to_string()];
    lines.extend(
        report
            .added
            .iter()
            .map(|path| format!("A {}", path.display())),
    );
    lines.extend(
        report
            .modified
            .iter()
            .map(|path| format!("M {}", path.display())),
    );
    lines.extend(
        report
            .deleted
            .iter()
            .map(|path| format!("D {}", path.display())),
    );
    lines.join("\n")
}

fn runtime_apply_patch_error(message: impl Into<String>) -> RuntimeToolExecutionError {
    let message = message.into();
    RuntimeToolExecutionError::new(
        message.clone(),
        Some(RuntimeToolPolicyErrorKind::ExecutionFailed(message)),
    )
}

fn runtime_apply_patch_permission_error(message: impl Into<String>) -> RuntimeToolExecutionError {
    let message = message.into();
    RuntimeToolExecutionError::new(
        message.clone(),
        Some(RuntimeToolPolicyErrorKind::PermissionDenied(message)),
    )
}

fn runtime_apply_patch_sandbox_error(message: impl Into<String>) -> RuntimeToolExecutionError {
    RuntimeToolExecutionError::new(
        message,
        Some(RuntimeToolPolicyErrorKind::SandboxDenied(
            "apply_patch_sandbox_denied".to_string(),
        )),
    )
    .before_handler()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_executor::{RuntimeToolExecutionContext, RuntimeToolExecutionContextInput};
    use serde_json::json;
    use std::fs;
    use tempfile::tempdir;

    fn context(path: PathBuf) -> RuntimeToolExecutionContext {
        RuntimeToolExecutionContext::new(RuntimeToolExecutionContextInput {
            working_directory: path,
            session_id: "test-session".to_string(),
            cancel_token: None,
            workspace_sandbox: None,
        })
    }

    #[tokio::test]
    async fn runtime_apply_patch_applies_patch_inside_workspace() {
        let dir = tempdir().unwrap();
        let context = context(dir.path().to_path_buf());
        let params = json!({
            "patch": "*** Begin Patch\n*** Add File: notes/live.md\n+hello\n*** End Patch"
        });
        let result = runtime_apply_patch_executor_handle()
            .execute(RuntimeToolExecutionRequest {
                tool_name: APPLY_PATCH_TOOL_NAME,
                params: &params,
                context: &context,
                turn_context: None,
            })
            .await
            .unwrap();

        assert!(result.success);
        assert_eq!(
            fs::read_to_string(dir.path().join("notes/live.md")).unwrap(),
            "hello\n"
        );
        assert_eq!(
            result.metadata.get("path").and_then(Value::as_str),
            Some("notes/live.md")
        );
        let file_change = result
            .metadata
            .get("file_change")
            .expect("file_change metadata");
        assert_eq!(file_change["path"].as_str(), Some("notes/live.md"));
        assert_eq!(file_change["kind"].as_str(), Some("add"));
        assert!(file_change["checkpointRef"]
            .as_str()
            .is_some_and(|value| value.starts_with("checkpoint:file:")));
        assert_eq!(
            result.metadata[TURN_DIFF_DELTA_METADATA_KEY]["changes"][0],
            json!({
                "type": "add",
                "path": "notes/live.md",
                "content": "hello\n"
            })
        );
    }

    #[test]
    fn runtime_apply_patch_tool_contract_includes_an_exact_update_example() {
        let definition = apply_patch_tool_definition();
        let patch_description = definition.input_schema["properties"]["patch"]["description"]
            .as_str()
            .expect("patch description");

        assert!(patch_description.contains("\n-before\n+after\n"));
        assert!(!patch_description.contains("\n- before\n+ after\n"));
    }

    #[test]
    fn runtime_apply_patch_rejects_patch_path_outside_workspace() {
        let dir = tempdir().unwrap();
        let params = json!({
            "patch": "*** Begin Patch\n*** Add File: ../outside.md\n+blocked\n*** End Patch"
        });
        let result = check_runtime_apply_patch_permissions(&params, dir.path());

        assert!(result.is_err());
    }

    #[test]
    fn runtime_apply_patch_allows_absolute_path_inside_workspace() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("absolute.md");
        let params = json!({
            "patch": format!(
                "*** Begin Patch\n*** Add File: {}\n+absolute\n*** End Patch",
                path.display()
            )
        });
        let result = check_runtime_apply_patch_permissions(&params, dir.path());

        assert!(result.is_ok());
    }
}
