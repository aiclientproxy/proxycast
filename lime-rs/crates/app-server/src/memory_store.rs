use app_server_protocol::*;
use async_trait::async_trait;
use chrono::Utc;
use serde::Serialize;
use std::collections::hash_map::DefaultHasher;
use std::ffi::OsStr;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Component, Path, PathBuf};

use crate::RuntimeCoreError;

mod audit;
mod consolidation;
mod review;
mod rollout;

pub use rollout::RolloutSummaryWriteParams;

const MEMORY_ROOT_DIR: &str = "memories";
const WORKSPACE_STATE_DIR: &str = ".lime";
const SUMMARY_FILE: &str = "memory_summary.md";
const MEMORY_FILE: &str = "MEMORY.md";
const NOTES_DIR: &str = "extensions/ad_hoc/notes";
const ROLLOUT_SUMMARIES_DIR: &str = "rollout_summaries";
const SKILLS_DIR: &str = "skills";
const INDEX_DIR: &str = "index";
const INDEX_MANIFEST_FILE: &str = "manifest.json";
const INDEX_SCHEMA_VERSION: &str = "memory-index-manifest/v1";
const DEFAULT_LIST_LIMIT: usize = 50;
const MAX_LIST_LIMIT: usize = 200;
const DEFAULT_READ_LINES: usize = 80;
const MAX_READ_LINES: usize = 500;
const DEFAULT_SEARCH_LIMIT: usize = 50;
const MAX_SEARCH_LIMIT: usize = 200;

#[async_trait]
pub trait MemoryBackend: Send + Sync {
    async fn list(
        &self,
        params: MemoryStoreListParams,
    ) -> Result<MemoryStoreListResponse, RuntimeCoreError>;

    async fn read(
        &self,
        params: MemoryStoreReadParams,
    ) -> Result<MemoryStoreReadResponse, RuntimeCoreError>;

    async fn search(
        &self,
        params: MemoryStoreSearchParams,
    ) -> Result<MemoryStoreSearchResponse, RuntimeCoreError>;

    async fn add_note(
        &self,
        params: MemoryStoreAddNoteParams,
    ) -> Result<MemoryStoreAddNoteResponse, RuntimeCoreError>;

    async fn write_rollout_summary(
        &self,
        params: RolloutSummaryWriteParams,
    ) -> Result<MemoryStoreAddNoteResponse, RuntimeCoreError>;

    async fn consolidate(
        &self,
        params: MemoryStoreConsolidateParams,
    ) -> Result<MemoryStoreConsolidateResponse, RuntimeCoreError>;

    async fn list_review(
        &self,
        params: MemoryStoreReviewListParams,
    ) -> Result<MemoryStoreReviewListResponse, RuntimeCoreError>;

    async fn resolve_review(
        &self,
        params: MemoryStoreReviewResolveParams,
    ) -> Result<MemoryStoreReviewResolveResponse, RuntimeCoreError>;

    async fn health(
        &self,
        params: MemoryStoreRootParams,
    ) -> Result<MemoryStoreHealthResponse, RuntimeCoreError>;

    async fn reset(&self) -> Result<(), RuntimeCoreError>;

    async fn rebuild_index(
        &self,
        params: MemoryStoreRootParams,
    ) -> Result<MemoryStoreIndexRebuildResponse, RuntimeCoreError>;
}

#[derive(Debug, Clone)]
pub struct LocalMemoryBackend {
    data_root: PathBuf,
}

impl LocalMemoryBackend {
    pub fn new(data_root: impl Into<PathBuf>) -> Self {
        Self {
            data_root: data_root.into(),
        }
    }

    fn resolve_root(&self, params: &MemoryStoreRootParams) -> Result<PathBuf, RuntimeCoreError> {
        match params.scope {
            MemoryStoreScope::Global => Ok(self.data_root.join(MEMORY_ROOT_DIR)),
            MemoryStoreScope::Workspace => {
                let workspace_root = params
                    .workspace_root
                    .as_deref()
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| backend_error("workspace memory requires workspaceRoot"))?;
                let workspace_root = PathBuf::from(workspace_root);
                if !workspace_root.is_absolute() {
                    return Err(backend_error("workspaceRoot must be absolute"));
                }
                Ok(workspace_root
                    .join(WORKSPACE_STATE_DIR)
                    .join(MEMORY_ROOT_DIR))
            }
        }
    }

    fn ensure_layout(&self, root: &Path) -> Result<(), RuntimeCoreError> {
        fs::create_dir_all(root).map_err(io_error)?;
        reject_symlink(root)?;
        fs::create_dir_all(root.join(ROLLOUT_SUMMARIES_DIR)).map_err(io_error)?;
        fs::create_dir_all(root.join(SKILLS_DIR)).map_err(io_error)?;
        fs::create_dir_all(root.join(NOTES_DIR)).map_err(io_error)?;
        fs::create_dir_all(root.join(audit::AUDIT_DIR)).map_err(io_error)?;
        fs::create_dir_all(root.join(INDEX_DIR)).map_err(io_error)?;
        ensure_file(root.join(SUMMARY_FILE))?;
        ensure_file(root.join(MEMORY_FILE))?;
        Ok(())
    }

    fn resolve_existing_path(
        &self,
        root: &Path,
        path: Option<&str>,
    ) -> Result<PathBuf, RuntimeCoreError> {
        let relative = validate_relative_path(path.unwrap_or_default())?;
        let resolved = root.join(relative);
        reject_outside_root(root, &resolved)?;
        reject_symlink_chain(root, &resolved)?;
        Ok(resolved)
    }

    fn relative_path(&self, root: &Path, path: &Path) -> Result<String, RuntimeCoreError> {
        let relative = path
            .strip_prefix(root)
            .map_err(|_| backend_error("memory path is outside root"))?;
        path_to_relative_string(relative)
    }
}

#[async_trait]
impl MemoryBackend for LocalMemoryBackend {
    async fn list(
        &self,
        params: MemoryStoreListParams,
    ) -> Result<MemoryStoreListResponse, RuntimeCoreError> {
        let root = self.resolve_root(&params.root)?;
        self.ensure_layout(&root)?;
        let requested_path = params.path.as_deref().unwrap_or_default();
        let directory = self.resolve_existing_path(&root, Some(requested_path))?;
        if !directory.is_dir() {
            return Err(backend_error("memoryStore/list path must be a directory"));
        }

        let mut entries = Vec::new();
        for entry in fs::read_dir(&directory).map_err(io_error)? {
            let entry = entry.map_err(io_error)?;
            let file_name = entry.file_name();
            if should_skip_segment(&file_name) {
                continue;
            }
            let path = entry.path();
            if is_symlink(&path)? {
                continue;
            }
            let metadata = fs::metadata(&path).map_err(io_error)?;
            let relative = match self.relative_path(&root, &path) {
                Ok(relative) => relative,
                Err(_) => continue,
            };
            entries.push(MemoryStoreEntry {
                path: relative,
                entry_type: if metadata.is_dir() {
                    "directory"
                } else {
                    "file"
                }
                .to_string(),
                size: metadata.len(),
                modified_at: modified_at_seconds(&metadata),
            });
        }
        entries.sort_by(|left, right| {
            left.entry_type
                .cmp(&right.entry_type)
                .then_with(|| left.path.cmp(&right.path))
        });

        let offset = parse_cursor(params.cursor.as_deref())?;
        let limit = bounded_limit(params.max_results, DEFAULT_LIST_LIMIT, MAX_LIST_LIMIT);
        let total = entries.len();
        let entries = entries
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect::<Vec<_>>();
        let next_offset = offset.saturating_add(entries.len());
        let truncated = next_offset < total;

        Ok(MemoryStoreListResponse {
            root_scope: params.root.scope,
            path: normalize_response_path(requested_path),
            entries,
            truncated,
            next_cursor: truncated.then(|| next_offset.to_string()),
        })
    }

    async fn read(
        &self,
        params: MemoryStoreReadParams,
    ) -> Result<MemoryStoreReadResponse, RuntimeCoreError> {
        let root = self.resolve_root(&params.root)?;
        self.ensure_layout(&root)?;
        let path = self.resolve_existing_path(&root, Some(&params.path))?;
        if !path.is_file() {
            return Err(backend_error("memoryStore/read path must be a file"));
        }
        let relative = self.relative_path(&root, &path)?;
        let content = fs::read_to_string(&path).map_err(|error| {
            if error.kind() == io::ErrorKind::InvalidData {
                backend_error("memoryStore/read only supports UTF-8 files")
            } else {
                io_error(error)
            }
        })?;
        let lines = content.lines().collect::<Vec<_>>();
        let offset = params.line_offset.unwrap_or(0);
        if offset >= lines.len() && !lines.is_empty() {
            return Err(backend_error("memoryStore/read lineOffset is out of range"));
        }
        let max_lines = bounded_limit(params.max_lines, DEFAULT_READ_LINES, MAX_READ_LINES);
        let end = lines.len().min(offset.saturating_add(max_lines));
        let mut selected = lines[offset..end].join("\n");
        if !selected.is_empty() {
            selected.push('\n');
        }
        let mut truncated = end < lines.len();
        if let Some(max_tokens) = params.max_tokens {
            let max_chars = max_tokens.saturating_mul(4).max(1);
            let (trimmed, was_truncated) = truncate_chars(&selected, max_chars);
            selected = trimmed;
            truncated |= was_truncated;
        }
        let start_line_number = offset + 1;
        let end_line_number = if selected.is_empty() {
            start_line_number
        } else {
            end.max(start_line_number)
        };

        Ok(MemoryStoreReadResponse {
            path: relative.clone(),
            start_line_number,
            content: selected,
            truncated,
            citation: MemoryStoreCitation {
                path: relative,
                start_line_number,
                end_line_number,
            },
        })
    }

    async fn search(
        &self,
        params: MemoryStoreSearchParams,
    ) -> Result<MemoryStoreSearchResponse, RuntimeCoreError> {
        let root = self.resolve_root(&params.root)?;
        self.ensure_layout(&root)?;
        let queries = normalize_queries(&params.queries, params.case_sensitive, params.normalized)?;
        let files = collect_searchable_files(&root)?;
        let offset = parse_cursor(params.cursor.as_deref())?;
        let limit = bounded_limit(params.max_results, DEFAULT_SEARCH_LIMIT, MAX_SEARCH_LIMIT);
        let mut hits = Vec::new();

        for file in files {
            let Ok(relative) = self.relative_path(&root, &file) else {
                continue;
            };
            let Ok(content) = fs::read_to_string(&file) else {
                continue;
            };
            let lines = content.lines().map(str::to_string).collect::<Vec<_>>();
            for hit in search_file(&relative, &lines, &queries, &params) {
                hits.push(hit);
            }
        }
        hits.sort_by(|left, right| {
            left.path
                .cmp(&right.path)
                .then_with(|| left.match_line_number.cmp(&right.match_line_number))
        });

        let total = hits.len();
        let hits = hits
            .into_iter()
            .skip(offset)
            .take(limit)
            .collect::<Vec<_>>();
        let next_offset = offset.saturating_add(hits.len());
        let truncated = next_offset < total;

        Ok(MemoryStoreSearchResponse {
            hits,
            truncated,
            next_cursor: truncated.then(|| next_offset.to_string()),
        })
    }

    async fn add_note(
        &self,
        params: MemoryStoreAddNoteParams,
    ) -> Result<MemoryStoreAddNoteResponse, RuntimeCoreError> {
        let root = self.resolve_root(&params.root)?;
        self.ensure_layout(&root)?;
        let content = params.content.trim();
        if content.is_empty() {
            return Err(backend_error(
                "memoryStore/addNote requires non-empty content",
            ));
        }
        let notes_dir = root.join(NOTES_DIR);
        reject_symlink_chain(&root, &notes_dir)?;
        let slug_source = params
            .slug
            .as_deref()
            .or(params.title.as_deref())
            .or_else(|| content.lines().next());
        let slug = sanitize_slug(slug_source.unwrap_or("note"));
        let timestamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
        let path = next_note_path(&notes_dir, &timestamp, &slug);
        let mut note_content = String::new();
        if let Some(title) = params
            .title
            .as_deref()
            .map(str::trim)
            .filter(|title| !title.is_empty())
        {
            note_content.push_str("# ");
            note_content.push_str(title);
            note_content.push_str("\n\n");
        }
        note_content.push_str(content);
        note_content.push('\n');
        fs::write(&path, &note_content).map_err(io_error)?;
        reject_symlink_chain(&root, &path)?;
        let relative = self.relative_path(&root, &path)?;
        let end_line_number = note_content.lines().count().max(1);

        Ok(MemoryStoreAddNoteResponse {
            path: relative.clone(),
            citation: MemoryStoreCitation {
                path: relative,
                start_line_number: 1,
                end_line_number,
            },
        })
    }

    async fn write_rollout_summary(
        &self,
        params: RolloutSummaryWriteParams,
    ) -> Result<MemoryStoreAddNoteResponse, RuntimeCoreError> {
        rollout::write_rollout_summary(self, params).await
    }

    async fn consolidate(
        &self,
        params: MemoryStoreConsolidateParams,
    ) -> Result<MemoryStoreConsolidateResponse, RuntimeCoreError> {
        consolidation::consolidate_memory_store(self, params).await
    }

    async fn list_review(
        &self,
        params: MemoryStoreReviewListParams,
    ) -> Result<MemoryStoreReviewListResponse, RuntimeCoreError> {
        review::list_review_notes(self, params).await
    }

    async fn resolve_review(
        &self,
        params: MemoryStoreReviewResolveParams,
    ) -> Result<MemoryStoreReviewResolveResponse, RuntimeCoreError> {
        review::resolve_review_note(self, params).await
    }

    async fn health(
        &self,
        params: MemoryStoreRootParams,
    ) -> Result<MemoryStoreHealthResponse, RuntimeCoreError> {
        let root = self.resolve_root(&params)?;
        self.ensure_layout(&root)?;
        let stats = collect_store_stats(&root)?;
        let summary = file_status(&root.join(SUMMARY_FILE))?;
        let memory = file_status(&root.join(MEMORY_FILE))?;
        let notes_count = count_markdown_files(&root.join(NOTES_DIR))?;

        Ok(MemoryStoreHealthResponse {
            root_scope: params.scope,
            root_path: path_to_display_string(&root)?,
            initialized: true,
            file_count: stats.file_count,
            total_bytes: stats.total_bytes,
            summary_exists: summary.exists,
            summary_bytes: summary.bytes,
            memory_exists: memory.exists,
            memory_bytes: memory.bytes,
            notes_count,
        })
    }

    async fn reset(&self) -> Result<(), RuntimeCoreError> {
        let root = self.resolve_root(&MemoryStoreRootParams::default())?;
        fs::create_dir_all(&root).map_err(io_error)?;
        reject_symlink(&root)?;
        remove_root_contents(&root)?;
        self.ensure_layout(&root)?;
        Ok(())
    }

    async fn rebuild_index(
        &self,
        params: MemoryStoreRootParams,
    ) -> Result<MemoryStoreIndexRebuildResponse, RuntimeCoreError> {
        let root = self.resolve_root(&params)?;
        self.ensure_layout(&root)?;
        let source_stats = collect_index_source_stats(&root)?;
        let indexed_at = Utc::now().to_rfc3339();
        let manifest_path = root.join(INDEX_DIR).join(INDEX_MANIFEST_FILE);
        reject_symlink_chain(&root, &manifest_path)?;
        let manifest = MemoryIndexManifest {
            schema_version: INDEX_SCHEMA_VERSION,
            indexed_at: &indexed_at,
            root_scope: params.scope,
            source_file_count: source_stats.file_count,
            source_total_bytes: source_stats.total_bytes,
            source_checksum: &source_stats.checksum,
        };
        let manifest_content = serde_json::to_string_pretty(&manifest).map_err(|error| {
            backend_error(format!("memory index manifest serialize failed: {error}"))
        })?;
        fs::write(&manifest_path, format!("{manifest_content}\n")).map_err(io_error)?;

        Ok(MemoryStoreIndexRebuildResponse {
            root_scope: params.scope,
            root_path: path_to_display_string(&root)?,
            manifest_path: path_to_relative_string(
                Path::new(INDEX_DIR).join(INDEX_MANIFEST_FILE).as_path(),
            )?,
            schema_version: INDEX_SCHEMA_VERSION.to_string(),
            source_file_count: source_stats.file_count,
            source_total_bytes: source_stats.total_bytes,
            source_checksum: source_stats.checksum,
            indexed_at,
            rebuilt: true,
        })
    }
}

fn ensure_file(path: PathBuf) -> Result<(), RuntimeCoreError> {
    if path.exists() {
        return Ok(());
    }
    fs::write(path, "").map_err(io_error)
}

fn validate_relative_path(path: &str) -> Result<PathBuf, RuntimeCoreError> {
    let trimmed = path.trim();
    if trimmed.is_empty() || trimmed == "." {
        return Ok(PathBuf::new());
    }
    let path = Path::new(trimmed);
    if path.is_absolute() {
        return Err(backend_error("memory path must be relative"));
    }
    let mut relative = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(segment) => {
                if should_skip_segment(segment) {
                    return Err(backend_error("hidden memory path segments are not allowed"));
                }
                relative.push(segment);
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(backend_error("memory path traversal is not allowed"));
            }
        }
    }
    Ok(relative)
}

fn reject_outside_root(root: &Path, path: &Path) -> Result<(), RuntimeCoreError> {
    let root = root.components().collect::<Vec<_>>();
    let path_components = path.components().collect::<Vec<_>>();
    if path_components.starts_with(&root) {
        Ok(())
    } else {
        Err(backend_error("memory path is outside root"))
    }
}

fn reject_symlink(path: &Path) -> Result<(), RuntimeCoreError> {
    if is_symlink(path)? {
        return Err(backend_error("memory symlink paths are not allowed"));
    }
    Ok(())
}

fn reject_symlink_chain(root: &Path, path: &Path) -> Result<(), RuntimeCoreError> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| backend_error("memory path is outside root"))?;
    reject_symlink(root)?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        if let Component::Normal(segment) = component {
            current.push(segment);
            if current.exists() {
                reject_symlink(&current)?;
            }
        }
    }
    Ok(())
}

fn is_symlink(path: &Path) -> Result<bool, RuntimeCoreError> {
    Ok(fs::symlink_metadata(path)
        .map_err(io_error)?
        .file_type()
        .is_symlink())
}

fn should_skip_segment(segment: &OsStr) -> bool {
    segment
        .to_str()
        .map(|value| value.starts_with('.') || value.is_empty())
        .unwrap_or(true)
}

fn path_to_relative_string(path: &Path) -> Result<String, RuntimeCoreError> {
    if path.as_os_str().is_empty() {
        return Ok(String::new());
    }
    let mut parts = Vec::new();
    for component in path.components() {
        let Component::Normal(segment) = component else {
            return Err(backend_error("invalid memory path component"));
        };
        let segment = segment
            .to_str()
            .ok_or_else(|| backend_error("memory path must be UTF-8"))?;
        parts.push(segment.to_string());
    }
    Ok(parts.join("/"))
}

fn collect_searchable_files(root: &Path) -> Result<Vec<PathBuf>, RuntimeCoreError> {
    let mut stack = vec![root.to_path_buf()];
    let mut files = Vec::new();
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(&directory).map_err(io_error)? {
            let entry = entry.map_err(io_error)?;
            if should_skip_segment(&entry.file_name()) {
                continue;
            }
            let path = entry.path();
            if is_symlink(&path)? {
                continue;
            }
            if path
                .strip_prefix(root)
                .ok()
                .is_some_and(is_under_derived_dir)
            {
                continue;
            }
            let metadata = fs::metadata(&path).map_err(io_error)?;
            if metadata.is_dir() {
                stack.push(path);
            } else if metadata.is_file() {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct StoreStats {
    file_count: usize,
    total_bytes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct IndexSourceStats {
    file_count: usize,
    total_bytes: u64,
    checksum: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MemoryIndexManifest<'a> {
    schema_version: &'a str,
    indexed_at: &'a str,
    root_scope: MemoryStoreScope,
    source_file_count: usize,
    source_total_bytes: u64,
    source_checksum: &'a str,
}

fn collect_store_stats(root: &Path) -> Result<StoreStats, RuntimeCoreError> {
    let mut stack = vec![root.to_path_buf()];
    let mut stats = StoreStats::default();
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(&directory).map_err(io_error)? {
            let entry = entry.map_err(io_error)?;
            if should_skip_segment(&entry.file_name()) {
                continue;
            }
            let path = entry.path();
            if is_symlink(&path)? {
                continue;
            }
            let metadata = fs::metadata(&path).map_err(io_error)?;
            if metadata.is_dir() {
                stack.push(path);
            } else if metadata.is_file() {
                stats.file_count += 1;
                stats.total_bytes = stats.total_bytes.saturating_add(metadata.len());
            }
        }
    }
    Ok(stats)
}

fn collect_index_source_stats(root: &Path) -> Result<IndexSourceStats, RuntimeCoreError> {
    let files = collect_searchable_files(root)?;
    let mut hasher = DefaultHasher::new();
    let mut total_bytes = 0_u64;
    for file in &files {
        let metadata = fs::metadata(file).map_err(io_error)?;
        total_bytes = total_bytes.saturating_add(metadata.len());
        let relative = file
            .strip_prefix(root)
            .map_err(|_| backend_error("memory path is outside root"))?;
        path_to_relative_string(relative)?.hash(&mut hasher);
        metadata.len().hash(&mut hasher);
        modified_at_seconds(&metadata).hash(&mut hasher);
    }
    Ok(IndexSourceStats {
        file_count: files.len(),
        total_bytes,
        checksum: format!("{:016x}", hasher.finish()),
    })
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct FileStatus {
    exists: bool,
    bytes: u64,
}

fn file_status(path: &Path) -> Result<FileStatus, RuntimeCoreError> {
    if !path.exists() {
        return Ok(FileStatus::default());
    }
    reject_symlink(path)?;
    let metadata = fs::metadata(path).map_err(io_error)?;
    Ok(FileStatus {
        exists: metadata.is_file(),
        bytes: if metadata.is_file() {
            metadata.len()
        } else {
            0
        },
    })
}

fn count_markdown_files(root: &Path) -> Result<usize, RuntimeCoreError> {
    if !root.exists() {
        return Ok(0);
    }
    reject_symlink(root)?;
    let mut stack = vec![root.to_path_buf()];
    let mut count = 0;
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(&directory).map_err(io_error)? {
            let entry = entry.map_err(io_error)?;
            if should_skip_segment(&entry.file_name()) {
                continue;
            }
            let path = entry.path();
            if is_symlink(&path)? {
                continue;
            }
            let metadata = fs::metadata(&path).map_err(io_error)?;
            if metadata.is_dir() {
                stack.push(path);
            } else if metadata.is_file()
                && path
                    .extension()
                    .and_then(OsStr::to_str)
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("md"))
            {
                count += 1;
            }
        }
    }
    Ok(count)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct RemovalStats {
    files: usize,
    directories: usize,
}

fn remove_root_contents(root: &Path) -> Result<RemovalStats, RuntimeCoreError> {
    let mut stats = RemovalStats::default();
    for entry in fs::read_dir(root).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        reject_outside_root(root, &path)?;
        reject_symlink_chain(root, &path)?;
        remove_entry(&path, &mut stats)?;
    }
    Ok(stats)
}

fn remove_entry(path: &Path, stats: &mut RemovalStats) -> Result<(), RuntimeCoreError> {
    reject_symlink(path)?;
    let metadata = fs::metadata(path).map_err(io_error)?;
    if metadata.is_dir() {
        for entry in fs::read_dir(path).map_err(io_error)? {
            let entry = entry.map_err(io_error)?;
            remove_entry(&entry.path(), stats)?;
        }
        fs::remove_dir(path).map_err(io_error)?;
        stats.directories += 1;
    } else if metadata.is_file() {
        fs::remove_file(path).map_err(io_error)?;
        stats.files += 1;
    }
    Ok(())
}

fn path_to_display_string(path: &Path) -> Result<String, RuntimeCoreError> {
    path.to_str()
        .map(str::to_string)
        .ok_or_else(|| backend_error("memory root path must be UTF-8"))
}

fn is_under_derived_dir(path: &Path) -> bool {
    path.components().next().is_some_and(|component| {
        let segment = component.as_os_str();
        segment == INDEX_DIR || segment == audit::AUDIT_DIR
    })
}

#[derive(Debug, Clone)]
struct PreparedQuery {
    original: String,
    comparable: String,
}

fn normalize_queries(
    queries: &[String],
    case_sensitive: bool,
    normalized: bool,
) -> Result<Vec<PreparedQuery>, RuntimeCoreError> {
    let mut prepared = Vec::new();
    for query in queries {
        let original = query.trim();
        if original.is_empty() {
            continue;
        }
        prepared.push(PreparedQuery {
            original: original.to_string(),
            comparable: comparable_text(original, case_sensitive, normalized),
        });
    }
    if prepared.is_empty() {
        return Err(backend_error(
            "memoryStore/search requires at least one query",
        ));
    }
    Ok(prepared)
}

fn search_file(
    relative_path: &str,
    lines: &[String],
    queries: &[PreparedQuery],
    params: &MemoryStoreSearchParams,
) -> Vec<MemoryStoreSearchHit> {
    let comparable_lines = lines
        .iter()
        .map(|line| comparable_text(line, params.case_sensitive, params.normalized))
        .collect::<Vec<_>>();
    let mut hits = Vec::new();
    for line_index in 0..lines.len() {
        let matched = match params.match_mode {
            MemoryStoreSearchMatchMode::Any => queries
                .iter()
                .filter(|query| comparable_lines[line_index].contains(&query.comparable))
                .collect::<Vec<_>>(),
            MemoryStoreSearchMatchMode::AllOnSameLine => {
                if queries
                    .iter()
                    .all(|query| comparable_lines[line_index].contains(&query.comparable))
                {
                    queries.iter().collect()
                } else {
                    Vec::new()
                }
            }
            MemoryStoreSearchMatchMode::AllWithinLines => {
                let within = params.within_lines.unwrap_or(3).max(1);
                let end = lines.len().min(line_index.saturating_add(within));
                let haystack = comparable_lines[line_index..end].join("\n");
                if queries
                    .iter()
                    .all(|query| haystack.contains(&query.comparable))
                {
                    queries.iter().collect()
                } else {
                    Vec::new()
                }
            }
        };
        if matched.is_empty() {
            continue;
        }
        let context = params.context_lines.min(20);
        let start = line_index.saturating_sub(context);
        let end = lines.len().min(line_index + context + 1);
        let mut content = lines[start..end].join("\n");
        if !content.is_empty() {
            content.push('\n');
        }
        hits.push(MemoryStoreSearchHit {
            path: relative_path.to_string(),
            matched_queries: matched
                .into_iter()
                .map(|query| query.original.clone())
                .collect(),
            match_line_number: line_index + 1,
            content_start_line_number: start + 1,
            content,
            citation: MemoryStoreCitation {
                path: relative_path.to_string(),
                start_line_number: line_index + 1,
                end_line_number: line_index + 1,
            },
        });
    }
    hits
}

fn comparable_text(value: &str, case_sensitive: bool, normalized: bool) -> String {
    let value = if normalized {
        value.split_whitespace().collect::<Vec<_>>().join(" ")
    } else {
        value.to_string()
    };
    if case_sensitive {
        value
    } else {
        value.to_lowercase()
    }
}

fn bounded_limit(value: Option<usize>, default: usize, max: usize) -> usize {
    value.unwrap_or(default).clamp(1, max)
}

fn parse_cursor(cursor: Option<&str>) -> Result<usize, RuntimeCoreError> {
    cursor
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|_| backend_error("memory cursor must be a decimal offset"))
        })
        .transpose()
        .map(|value| value.unwrap_or(0))
}

fn truncate_chars(content: &str, max_chars: usize) -> (String, bool) {
    if content.chars().count() <= max_chars {
        return (content.to_string(), false);
    }
    let truncated = content.chars().take(max_chars).collect::<String>();
    (truncated, true)
}

fn sanitize_slug(value: &str) -> String {
    let mut slug = String::new();
    let mut previous_dash = false;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            slug.push(character.to_ascii_lowercase());
            previous_dash = false;
        } else if !previous_dash {
            slug.push('-');
            previous_dash = true;
        }
        if slug.len() >= 48 {
            break;
        }
    }
    let slug = slug.trim_matches('-').to_string();
    if slug.is_empty() {
        "note".to_string()
    } else {
        slug
    }
}

fn next_note_path(notes_dir: &Path, timestamp: &str, slug: &str) -> PathBuf {
    let base = format!("{timestamp}-{slug}");
    let first = notes_dir.join(format!("{base}.md"));
    if !first.exists() {
        return first;
    }
    for index in 2..1000 {
        let candidate = notes_dir.join(format!("{base}-{index}.md"));
        if !candidate.exists() {
            return candidate;
        }
    }
    notes_dir.join(format!("{base}-overflow.md"))
}

fn normalize_response_path(path: &str) -> String {
    if path.trim().is_empty() || path.trim() == "." {
        String::new()
    } else {
        path.trim().replace('\\', "/")
    }
}

fn modified_at_seconds(metadata: &fs::Metadata) -> i64 {
    metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|value| value.as_secs() as i64)
        .unwrap_or_default()
}

fn io_error(error: io::Error) -> RuntimeCoreError {
    RuntimeCoreError::Backend(error.to_string())
}

fn backend_error(message: impl Into<String>) -> RuntimeCoreError {
    RuntimeCoreError::Backend(message.into())
}

#[cfg(test)]
mod tests;
