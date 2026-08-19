use super::ArtifactContentProvider;
use super::ArtifactContentRequest;
use serde_json::Value;
use std::fs;
use std::io::Read;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

pub const DEFAULT_ARTIFACT_CONTENT_MAX_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Default)]
pub struct InlineArtifactContentProvider;

impl ArtifactContentProvider for InlineArtifactContentProvider {
    fn read_content(&self, request: &ArtifactContentRequest) -> Option<String> {
        request.artifact.content.clone()
    }
}

/// Reads runtime-produced files from the workspace declared by the artifact.
/// The provider only accepts a relative path under an event-declared `cwd`, or
/// an absolute path that is itself the canonical artifact path. It never falls
/// back to the process working directory.
#[derive(Debug, Default)]
pub struct WorkspaceArtifactContentProvider;

impl ArtifactContentProvider for WorkspaceArtifactContentProvider {
    fn read_content(&self, request: &ArtifactContentRequest) -> Option<String> {
        let artifact = &request.artifact;
        if let Some(content) = artifact.content.clone() {
            return Some(content);
        }
        let path = artifact.path.as_deref()?;

        artifact_workspace_roots(artifact.metadata.as_ref())
            .into_iter()
            .find_map(|root| {
                let root = root.canonicalize().ok()?;
                let relative = Path::new(path);
                let candidate = if relative.is_absolute() {
                    relative.to_path_buf()
                } else {
                    if !is_safe_relative_path(relative) {
                        return None;
                    }
                    root.join(relative)
                };
                let canonical = candidate.canonicalize().ok()?;
                if !canonical.starts_with(&root) {
                    return None;
                }
                read_limited_utf8_file(&canonical, DEFAULT_ARTIFACT_CONTENT_MAX_BYTES)
            })
    }
}

fn artifact_workspace_roots(metadata: Option<&Value>) -> Vec<PathBuf> {
    let Some(metadata) = metadata else {
        return Vec::new();
    };
    let mut roots = Vec::new();
    if let Some(root) = metadata
        .get("cwd")
        .or_else(|| metadata.get("workingDir"))
        .or_else(|| metadata.get("working_dir"))
        .and_then(Value::as_str)
    {
        roots.push(PathBuf::from(root));
    }
    if let Some(environments) = metadata.get("environments").and_then(Value::as_array) {
        roots.extend(
            environments
                .iter()
                .filter_map(|environment| environment.get("cwd").and_then(Value::as_str))
                .map(PathBuf::from),
        );
    }
    roots
}

#[derive(Debug, Clone)]
pub struct FilesystemArtifactContentProvider {
    root: PathBuf,
    max_bytes: u64,
}

impl FilesystemArtifactContentProvider {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            max_bytes: DEFAULT_ARTIFACT_CONTENT_MAX_BYTES,
        }
    }

    pub fn with_max_bytes(mut self, max_bytes: u64) -> Self {
        self.max_bytes = max_bytes;
        self
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn max_bytes(&self) -> u64 {
        self.max_bytes
    }
}

impl ArtifactContentProvider for FilesystemArtifactContentProvider {
    fn read_content(&self, request: &ArtifactContentRequest) -> Option<String> {
        request
            .artifact
            .path
            .as_deref()
            .and_then(|path| read_limited_relative_utf8_file(&self.root, path, self.max_bytes))
            .or_else(|| request.artifact.content.clone())
    }
}

fn read_limited_relative_utf8_file(
    root: &Path,
    relative_path: &str,
    max_bytes: u64,
) -> Option<String> {
    if max_bytes == 0 {
        return None;
    }
    let relative = Path::new(relative_path);
    if relative.is_absolute() || !is_safe_relative_path(relative) {
        return None;
    }

    let root = root.canonicalize().ok()?;
    let path = root.join(relative);
    let canonical_path = path.canonicalize().ok()?;
    if !canonical_path.starts_with(&root) {
        return None;
    }

    read_limited_utf8_file(&canonical_path, max_bytes)
}

fn read_limited_utf8_file(path: &Path, max_bytes: u64) -> Option<String> {
    let metadata = fs::metadata(path).ok()?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        return None;
    }

    let mut file = fs::File::open(path).ok()?;
    let capacity = usize::try_from(metadata.len()).ok()?;
    let mut buffer = Vec::with_capacity(capacity);
    file.by_ref()
        .take(max_bytes.saturating_add(1))
        .read_to_end(&mut buffer)
        .ok()?;
    if u64::try_from(buffer.len()).ok()? > max_bytes {
        return None;
    }

    String::from_utf8(buffer).ok()
}

fn is_safe_relative_path(path: &Path) -> bool {
    path.components()
        .all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}
