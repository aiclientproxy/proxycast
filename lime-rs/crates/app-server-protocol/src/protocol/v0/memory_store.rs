use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum MemoryStoreScope {
    #[default]
    Global,
    Workspace,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreRootParams {
    #[serde(default)]
    pub scope: MemoryStoreScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workspace_root: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreListParams {
    #[serde(flatten)]
    pub root: MemoryStoreRootParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreReadParams {
    #[serde(flatten)]
    pub root: MemoryStoreRootParams,
    pub path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_offset: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_lines: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum MemoryStoreSearchMatchMode {
    #[default]
    Any,
    AllOnSameLine,
    AllWithinLines,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreSearchParams {
    #[serde(flatten)]
    pub root: MemoryStoreRootParams,
    #[serde(default)]
    pub queries: Vec<String>,
    #[serde(default)]
    pub match_mode: MemoryStoreSearchMatchMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within_lines: Option<usize>,
    #[serde(default)]
    pub case_sensitive: bool,
    #[serde(default)]
    pub normalized: bool,
    #[serde(default)]
    pub context_lines: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreAddNoteParams {
    #[serde(flatten)]
    pub root: MemoryStoreRootParams,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slug: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreConsolidateParams {
    #[serde(flatten)]
    pub root: MemoryStoreRootParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_notes: Option<usize>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreReviewListParams {
    #[serde(flatten)]
    pub root: MemoryStoreRootParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub enum MemoryStoreReviewResolveAction {
    #[default]
    Accept,
    Reject,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreReviewResolveParams {
    #[serde(flatten)]
    pub root: MemoryStoreRootParams,
    pub path: String,
    pub action: MemoryStoreReviewResolveAction,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreEntry {
    pub path: String,
    pub entry_type: String,
    pub size: u64,
    pub modified_at: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreListResponse {
    pub root_scope: MemoryStoreScope,
    pub path: String,
    #[serde(default)]
    pub entries: Vec<MemoryStoreEntry>,
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreCitation {
    pub path: String,
    pub start_line_number: usize,
    pub end_line_number: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreReadResponse {
    pub path: String,
    pub start_line_number: usize,
    pub content: String,
    pub truncated: bool,
    pub citation: MemoryStoreCitation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreSearchHit {
    pub path: String,
    pub matched_queries: Vec<String>,
    pub match_line_number: usize,
    pub content_start_line_number: usize,
    pub content: String,
    pub citation: MemoryStoreCitation,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreSearchResponse {
    #[serde(default)]
    pub hits: Vec<MemoryStoreSearchHit>,
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreAddNoteResponse {
    pub path: String,
    pub citation: MemoryStoreCitation,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreConsolidateResponse {
    pub root_scope: MemoryStoreScope,
    pub root_path: String,
    pub processed_notes: usize,
    pub skipped_notes: usize,
    pub archived_notes: usize,
    pub memory_path: String,
    pub summary_path: String,
    #[serde(default)]
    pub warnings: Vec<String>,
    pub updated: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreReviewNote {
    pub path: String,
    pub size: u64,
    pub modified_at: i64,
    pub preview: String,
    pub citation: MemoryStoreCitation,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreReviewListResponse {
    pub root_scope: MemoryStoreScope,
    pub root_path: String,
    #[serde(default)]
    pub notes: Vec<MemoryStoreReviewNote>,
    pub truncated: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreReviewResolveResponse {
    pub root_scope: MemoryStoreScope,
    pub root_path: String,
    pub source_path: String,
    pub archived_path: String,
    pub action: MemoryStoreReviewResolveAction,
    pub memory_path: String,
    pub summary_path: String,
    pub updated: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreHealthResponse {
    pub root_scope: MemoryStoreScope,
    pub root_path: String,
    pub initialized: bool,
    pub file_count: usize,
    pub total_bytes: u64,
    pub summary_exists: bool,
    pub summary_bytes: u64,
    pub memory_exists: bool,
    pub memory_bytes: u64,
    pub notes_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "camelCase")]
pub struct MemoryStoreIndexRebuildResponse {
    pub root_scope: MemoryStoreScope,
    pub root_path: String,
    pub manifest_path: String,
    pub schema_version: String,
    pub source_file_count: usize,
    pub source_total_bytes: u64,
    pub source_checksum: String,
    pub indexed_at: String,
    pub rebuilt: bool,
}
