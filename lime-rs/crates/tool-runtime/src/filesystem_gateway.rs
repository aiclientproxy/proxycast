//! Environment-owned filesystem gateway used by workspace tools.
//!
//! Local execution keeps using the existing `std::fs` owner.  A gateway is
//! installed only by the App Server runtime and is selected by the execution
//! Environment identity carried on `RuntimeToolExecutionContext`.

use async_trait::async_trait;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeFileMetadata {
    pub is_directory: bool,
    pub is_file: bool,
    pub is_symlink: bool,
    pub size: u64,
    pub created_at_ms: i64,
    pub modified_at_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeFileEntry {
    pub path: PathBuf,
    pub is_directory: bool,
    pub is_file: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeFileWalkOptions {
    pub max_depth: usize,
    pub max_directories: usize,
    pub max_entries: usize,
    pub follow_directory_symlinks: bool,
    pub prune_hidden_directories: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RuntimeFilePatchResult {
    pub modified_paths: Vec<PathBuf>,
}

#[async_trait]
pub trait RuntimeFileSystemGateway: Send + Sync + std::fmt::Debug {
    async fn read_file(
        &self,
        environment_id: &str,
        path: &Path,
        sandbox_policy: Option<&str>,
    ) -> Result<Vec<u8>, String>;

    async fn write_file(
        &self,
        environment_id: &str,
        path: &Path,
        data: &[u8],
        sandbox_policy: Option<&str>,
    ) -> Result<(), String>;

    async fn metadata(
        &self,
        environment_id: &str,
        path: &Path,
        sandbox_policy: Option<&str>,
    ) -> Result<RuntimeFileMetadata, String>;

    async fn canonicalize(
        &self,
        environment_id: &str,
        path: &Path,
        sandbox_policy: Option<&str>,
    ) -> Result<PathBuf, String>;

    async fn read_directory(
        &self,
        environment_id: &str,
        path: &Path,
        sandbox_policy: Option<&str>,
    ) -> Result<Vec<RuntimeFileEntry>, String>;

    async fn walk(
        &self,
        environment_id: &str,
        path: &Path,
        options: RuntimeFileWalkOptions,
        sandbox_policy: Option<&str>,
    ) -> Result<Vec<RuntimeFileEntry>, String>;

    async fn apply_patch(
        &self,
        environment_id: &str,
        working_directory: &Path,
        patch: &str,
        sandbox_policy: Option<&str>,
    ) -> Result<RuntimeFilePatchResult, String>;
}
