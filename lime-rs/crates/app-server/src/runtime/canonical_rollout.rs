use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Component, Path, PathBuf};

use agent_protocol::{Thread, ThreadHistoryChangeSet};
use chrono::{DateTime, Datelike, Local, SecondsFormat, Timelike, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

mod append_guard;
mod delete;
mod tail_repair;
#[cfg(test)]
mod tests;

const ROLLOUT_SCHEMA_VERSION: u32 = 1;
const SESSIONS_DIR_NAME: &str = "sessions";
const ARCHIVED_SESSIONS_DIR_NAME: &str = "archived_sessions";
const MAX_THREAD_ID_FILE_CHARS: usize = 128;
const ROLLOUT_TAIL_READ_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct RolloutStore {
    agent_root: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RolloutRecord {
    SessionMeta {
        schema_version: u32,
        session_id: String,
        thread_id: String,
        created_at: String,
        created_at_ms: i64,
        rollout_path: String,
        thread: Thread,
    },
    ThreadHistory {
        schema_version: u32,
        session_id: String,
        thread_id: String,
        sequence: u64,
        fingerprint: String,
        content_digest: String,
        changes: ThreadHistoryChangeSet,
    },
    ThreadMetadata {
        schema_version: u32,
        session_id: String,
        thread_id: String,
        updated_at_ms: i64,
        previous_content_digest: String,
        content_digest: String,
        metadata: Value,
    },
}

#[derive(Debug)]
struct RolloutScan {
    session_id: String,
    thread_id: String,
    created_at_ms: i64,
    rollout_path: String,
    initial_thread: Thread,
    history: Vec<RolloutHistoryRecord>,
}

#[derive(Debug, Clone)]
pub(super) struct RolloutHistoryRecord {
    pub(super) sequence: u64,
    pub(super) fingerprint: String,
    pub(super) changes: ThreadHistoryChangeSet,
}

#[derive(Debug, Clone)]
pub(super) struct RolloutSnapshot {
    pub(super) relative_path: PathBuf,
    pub(super) archived: bool,
    pub(super) initial_thread: Thread,
    pub(super) history: Vec<RolloutHistoryRecord>,
}

impl RolloutStore {
    pub(super) fn new(agent_root: impl AsRef<Path>) -> Self {
        Self {
            agent_root: agent_root.as_ref().to_path_buf(),
        }
    }

    pub(super) fn path_for_thread(&self, thread: &Thread) -> Result<PathBuf, String> {
        validate_thread_id_for_file(thread.thread_id.as_str())?;
        let created_at =
            DateTime::<Utc>::from_timestamp_millis(thread.created_at_ms).ok_or_else(|| {
                format!(
                    "invalid thread creation timestamp: {}",
                    thread.created_at_ms
                )
            })?;
        let local = created_at.with_timezone(&Local);
        let timestamp = format!(
            "{:04}-{:02}-{:02}T{:02}-{:02}-{:02}",
            local.year(),
            local.month(),
            local.day(),
            local.hour(),
            local.minute(),
            local.second()
        );
        Ok(PathBuf::from(SESSIONS_DIR_NAME)
            .join(format!("{:04}", local.year()))
            .join(format!("{:02}", local.month()))
            .join(format!("{:02}", local.day()))
            .join(format!(
                "rollout-{timestamp}-{}.jsonl",
                thread.thread_id.as_str()
            )))
    }

    pub(super) fn storage_path(&self, relative_path: &Path) -> Result<String, String> {
        active_storage_path(relative_path)
    }

    pub(super) fn ensure_thread(
        &self,
        relative_path: &Path,
        thread: &Thread,
    ) -> Result<(), String> {
        let stored_path = active_storage_path(relative_path)?;
        let path = self.resolve_active(relative_path)?;
        if path.exists() {
            return self.validate_existing_thread(&path, thread, &stored_path);
        }
        let parent = path
            .parent()
            .ok_or_else(|| format!("rollout path has no parent: {}", path.display()))?;
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "failed to create rollout directory {}: {error}",
                parent.display()
            )
        })?;
        let record = RolloutRecord::SessionMeta {
            schema_version: ROLLOUT_SCHEMA_VERSION,
            session_id: thread.session_id.as_str().to_string(),
            thread_id: thread.thread_id.as_str().to_string(),
            created_at: DateTime::<Utc>::from_timestamp_millis(thread.created_at_ms)
                .ok_or_else(|| {
                    format!(
                        "invalid thread creation timestamp: {}",
                        thread.created_at_ms
                    )
                })?
                .to_rfc3339_opts(SecondsFormat::Millis, true),
            created_at_ms: thread.created_at_ms,
            rollout_path: stored_path.clone(),
            thread: thread.clone(),
        };
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => write_synced_record(&mut file, &path, &record),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                self.validate_existing_thread(&path, thread, &stored_path)
            }
            Err(error) => Err(format!(
                "failed to create rollout file {}: {error}",
                path.display()
            )),
        }
    }

    pub(super) fn append_history(
        &self,
        relative_path: &Path,
        session_id: &str,
        thread_id: &str,
        fingerprint: &str,
        changes: &ThreadHistoryChangeSet,
    ) -> Result<bool, String> {
        let path = self.resolve_active(relative_path)?;
        tail_repair::repair_crash_tail(&path)?;
        let latest =
            append_guard::latest_history_for_append(&path, relative_path, session_id, thread_id)?;
        if let Some(existing) = latest
            .as_ref()
            .filter(|record| record.sequence == changes.sequence)
        {
            if existing.fingerprint == fingerprint {
                return Ok(false);
            }
            return Err(format!(
                "rollout history sequence collision at {}",
                changes.sequence
            ));
        }
        if latest
            .as_ref()
            .is_some_and(|record| record.sequence > changes.sequence)
        {
            return Err(format!(
                "rollout history sequence {} is stale",
                changes.sequence
            ));
        }
        let record = RolloutRecord::ThreadHistory {
            schema_version: ROLLOUT_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            thread_id: thread_id.to_string(),
            sequence: changes.sequence,
            fingerprint: fingerprint.to_string(),
            content_digest: history_content_digest(session_id, thread_id, changes)?,
            changes: changes.clone(),
        };
        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|error| format!("failed to open rollout file {}: {error}", path.display()))?;
        write_synced_record(&mut file, &path, &record)?;
        Ok(true)
    }

    pub(super) fn append_metadata(
        &self,
        relative_path: &Path,
        expected: &Thread,
        next: &Thread,
    ) -> Result<i64, String> {
        validate_metadata_update(expected, next)?;
        let path = self.resolve_active(relative_path)?;
        let scan = scan_rollout_repaired(&path)?;
        validate_scan_identity(
            &scan,
            relative_path,
            expected.session_id.as_str(),
            expected.thread_id.as_str(),
            &path,
        )?;

        if scan.initial_thread.metadata == next.metadata
            && expected.updated_at_ms <= scan.initial_thread.updated_at_ms
        {
            return Ok(scan.initial_thread.updated_at_ms);
        }
        if scan.initial_thread.metadata != expected.metadata
            || scan.initial_thread.updated_at_ms != expected.updated_at_ms
        {
            return Err(format!(
                "rollout metadata state conflict for thread {}",
                expected.thread_id
            ));
        }
        if next.updated_at_ms < expected.updated_at_ms {
            return Err(format!(
                "rollout metadata timestamp moved backwards for thread {}",
                expected.thread_id
            ));
        }

        let previous_content_digest = metadata_content_digest(
            expected.session_id.as_str(),
            expected.thread_id.as_str(),
            expected.updated_at_ms,
            &expected.metadata,
        )?;
        let content_digest = metadata_content_digest(
            next.session_id.as_str(),
            next.thread_id.as_str(),
            next.updated_at_ms,
            &next.metadata,
        )?;
        let record = RolloutRecord::ThreadMetadata {
            schema_version: ROLLOUT_SCHEMA_VERSION,
            session_id: next.session_id.as_str().to_string(),
            thread_id: next.thread_id.as_str().to_string(),
            updated_at_ms: next.updated_at_ms,
            previous_content_digest,
            content_digest,
            metadata: next.metadata.clone(),
        };
        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .map_err(|error| format!("failed to open rollout file {}: {error}", path.display()))?;
        write_synced_record(&mut file, &path, &record)?;
        Ok(next.updated_at_ms)
    }

    pub(super) fn verify_history(
        &self,
        relative_path: &Path,
        session_id: &str,
        thread_id: &str,
        sequence: u64,
        fingerprint: &str,
    ) -> Result<(), String> {
        let path = self.resolve_active(relative_path)?;
        let scan = scan_rollout_repaired(&path)?;
        validate_scan_identity(&scan, relative_path, session_id, thread_id, &path)?;
        match scan
            .history
            .iter()
            .find(|record| record.sequence == sequence)
        {
            Some(record) if record.fingerprint == fingerprint => Ok(()),
            Some(_) => Err(format!("rollout history sequence collision at {sequence}")),
            None => Err(format!(
                "rollout is missing committed history sequence {sequence}: {}",
                path.display()
            )),
        }
    }

    pub(super) fn archive(
        &self,
        relative_path: &Path,
        session_id: &str,
        thread_id: &str,
    ) -> Result<PathBuf, String> {
        let source = self.resolve_active(relative_path)?;
        let archived_path = self.archive_path(relative_path)?;
        let destination = self.resolve_archived(&archived_path)?;
        self.move_rollout(&source, &destination, relative_path, session_id, thread_id)?;
        Ok(archived_path)
    }

    pub(super) fn archive_path(&self, relative_path: &Path) -> Result<PathBuf, String> {
        validate_scoped_rollout_path(relative_path, SESSIONS_DIR_NAME)?;
        let file_name = relative_path
            .file_name()
            .ok_or_else(|| format!("rollout path has no filename: {}", relative_path.display()))?;
        Ok(PathBuf::from(ARCHIVED_SESSIONS_DIR_NAME).join(file_name))
    }

    pub(super) fn unarchive(
        &self,
        relative_path: &Path,
        session_id: &str,
        thread_id: &str,
    ) -> Result<PathBuf, String> {
        let source = self.resolve_archived(relative_path)?;
        let restored_path = self.unarchive_path(relative_path)?;
        let destination = self.resolve_active(&restored_path)?;
        self.move_rollout(&source, &destination, &restored_path, session_id, thread_id)?;
        Ok(restored_path)
    }

    pub(super) fn unarchive_path(&self, relative_path: &Path) -> Result<PathBuf, String> {
        active_path_from_archived(relative_path)
    }

    pub(super) fn verify_location(
        &self,
        relative_path: &Path,
        session_id: &str,
        thread_id: &str,
        archived: bool,
    ) -> Result<(), String> {
        let (path, active_path) = if archived {
            (
                self.resolve_archived(relative_path)?,
                active_path_from_archived(relative_path)?,
            )
        } else {
            (
                self.resolve_active(relative_path)?,
                relative_path.to_path_buf(),
            )
        };
        let scan = scan_rollout_repaired(&path)?;
        validate_scan_identity(&scan, &active_path, session_id, thread_id, &path)
    }

    pub(super) fn location_storage_path(&self, relative_path: &Path) -> Result<String, String> {
        location_storage_path(relative_path)
    }

    pub(super) fn snapshots(&self) -> Result<Vec<RolloutSnapshot>, String> {
        let mut paths = Vec::new();
        self.collect_rollouts(SESSIONS_DIR_NAME, true, &mut paths)?;
        self.collect_rollouts(ARCHIVED_SESSIONS_DIR_NAME, false, &mut paths)?;
        paths.sort();
        let mut snapshots = Vec::with_capacity(paths.len());
        for relative_path in paths {
            let archived = relative_path.components().next()
                == Some(Component::Normal(ARCHIVED_SESSIONS_DIR_NAME.as_ref()));
            let path = if archived {
                self.resolve_archived(&relative_path)?
            } else {
                self.resolve_active(&relative_path)?
            };
            let scan = scan_rollout_repaired(&path)?;
            let active_path = if archived {
                active_path_from_archived(&relative_path)?
            } else {
                relative_path.clone()
            };
            validate_scan_identity(
                &scan,
                &active_path,
                scan.session_id.as_str(),
                scan.thread_id.as_str(),
                &path,
            )?;
            snapshots.push(RolloutSnapshot {
                relative_path,
                archived,
                initial_thread: scan.initial_thread,
                history: scan.history,
            });
        }
        Ok(snapshots)
    }

    fn validate_existing_thread(
        &self,
        path: &Path,
        thread: &Thread,
        stored_path: &str,
    ) -> Result<(), String> {
        let scan = scan_rollout_repaired(path)?;
        if scan.session_id == thread.session_id.as_str()
            && scan.thread_id == thread.thread_id.as_str()
            && scan.created_at_ms == thread.created_at_ms
            && scan.rollout_path == stored_path
            && &scan.initial_thread == thread
        {
            return Ok(());
        }
        Err(format!(
            "existing rollout metadata does not match thread {}",
            thread.thread_id
        ))
    }

    fn move_rollout(
        &self,
        source: &Path,
        destination: &Path,
        active_path: &Path,
        session_id: &str,
        thread_id: &str,
    ) -> Result<(), String> {
        let source_exists = source
            .try_exists()
            .map_err(|error| format!("failed to inspect rollout {}: {error}", source.display()))?;
        let destination_exists = destination.try_exists().map_err(|error| {
            format!(
                "failed to inspect rollout {}: {error}",
                destination.display()
            )
        })?;
        match (source_exists, destination_exists) {
            (true, false) => {
                let scan = scan_rollout_repaired(source)?;
                validate_scan_identity(&scan, active_path, session_id, thread_id, source)?;
                let parent = destination.parent().ok_or_else(|| {
                    format!("rollout path has no parent: {}", destination.display())
                })?;
                fs::create_dir_all(parent).map_err(|error| {
                    format!(
                        "failed to create rollout directory {}: {error}",
                        parent.display()
                    )
                })?;
                fs::rename(source, destination).map_err(|error| {
                    format!(
                        "failed to move rollout {} to {}: {error}",
                        source.display(),
                        destination.display()
                    )
                })
            }
            (false, true) => {
                let scan = scan_rollout_repaired(destination)?;
                validate_scan_identity(&scan, active_path, session_id, thread_id, destination)
            }
            (true, true) => Err(format!(
                "rollout move refused because source and destination both exist: {} -> {}",
                source.display(),
                destination.display()
            )),
            (false, false) => Err(format!(
                "rollout move source and destination are both missing: {} -> {}",
                source.display(),
                destination.display()
            )),
        }
    }

    fn collect_rollouts(
        &self,
        scope: &str,
        recursive: bool,
        output: &mut Vec<PathBuf>,
    ) -> Result<(), String> {
        let root = self.agent_root.join(scope);
        reject_symlink(&self.agent_root)?;
        reject_symlink(&root)?;
        if !root.try_exists().map_err(|error| {
            format!("failed to inspect rollout root {}: {error}", root.display())
        })? {
            return Ok(());
        }
        collect_rollout_files(&root, Path::new(scope), recursive, output)
    }

    fn resolve_active(&self, relative_path: &Path) -> Result<PathBuf, String> {
        self.resolve_scoped(relative_path, SESSIONS_DIR_NAME)
    }

    fn resolve_archived(&self, relative_path: &Path) -> Result<PathBuf, String> {
        self.resolve_scoped(relative_path, ARCHIVED_SESSIONS_DIR_NAME)
    }

    fn resolve_scoped(&self, relative_path: &Path, scope: &str) -> Result<PathBuf, String> {
        validate_scoped_rollout_path(relative_path, scope)?;
        let mut current = self.agent_root.clone();
        reject_symlink(&current)?;
        for component in relative_path.components() {
            let Component::Normal(component) = component else {
                return Err(format!(
                    "rollout path must be relative and normalized: {}",
                    relative_path.display()
                ));
            };
            current.push(component);
            reject_symlink(&current)?;
        }
        Ok(current)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct LatestRolloutHistory {
    sequence: u64,
    fingerprint: String,
}

fn validate_scan_identity(
    scan: &RolloutScan,
    relative_path: &Path,
    session_id: &str,
    thread_id: &str,
    path: &Path,
) -> Result<(), String> {
    let expected_path = active_storage_path(relative_path)?;
    if scan.session_id == session_id
        && scan.thread_id == thread_id
        && scan.rollout_path == expected_path
    {
        return Ok(());
    }
    Err(format!("rollout identity mismatch for {}", path.display()))
}

fn scan_rollout_repaired(path: &Path) -> Result<RolloutScan, String> {
    tail_repair::repair_crash_tail(path)?;
    scan_rollout(path)
}

fn scan_rollout(path: &Path) -> Result<RolloutScan, String> {
    let file = File::open(path)
        .map_err(|error| format!("failed to read rollout file {}: {error}", path.display()))?;
    scan_rollout_reader(BufReader::new(file), path)
}

fn validate_rollout_prefix(path: &Path, byte_len: u64) -> Result<(), String> {
    let file = File::open(path)
        .map_err(|error| format!("failed to read rollout file {}: {error}", path.display()))?;
    scan_rollout_reader(BufReader::new(file.take(byte_len)), path).map(|_| ())
}

fn scan_rollout_reader(reader: impl BufRead, path: &Path) -> Result<RolloutScan, String> {
    let mut lines = reader.lines();
    let first = lines
        .next()
        .ok_or_else(|| format!("rollout file is empty: {}", path.display()))?
        .map_err(|error| format!("failed to read rollout file {}: {error}", path.display()))?;
    let RolloutRecord::SessionMeta {
        schema_version,
        session_id,
        thread_id,
        created_at_ms,
        rollout_path,
        thread: initial_thread,
        ..
    } = parse_record(&first, path)?
    else {
        return Err(format!(
            "rollout first line must be session metadata: {}",
            path.display()
        ));
    };
    if schema_version != ROLLOUT_SCHEMA_VERSION {
        return Err(format!(
            "unsupported rollout schema version {schema_version}: {}",
            path.display()
        ));
    }
    let mut initial_thread = initial_thread;
    let mut metadata_digest = metadata_content_digest(
        &session_id,
        &thread_id,
        initial_thread.updated_at_ms,
        &initial_thread.metadata,
    )?;
    let mut history = Vec::new();
    for line in lines {
        let line = line
            .map_err(|error| format!("failed to read rollout file {}: {error}", path.display()))?;
        match parse_record(&line, path)? {
            RolloutRecord::ThreadHistory {
                schema_version,
                session_id: record_session_id,
                thread_id: record_thread_id,
                sequence,
                fingerprint,
                content_digest,
                changes,
            } => {
                if schema_version != ROLLOUT_SCHEMA_VERSION
                    || record_session_id != session_id
                    || record_thread_id != thread_id
                    || changes.sequence != sequence
                    || !is_sha256_hex(&fingerprint)
                    || history_content_digest(&session_id, &thread_id, &changes)? != content_digest
                {
                    return Err(format!(
                        "invalid rollout history record: {}",
                        path.display()
                    ));
                }
                if history
                    .last()
                    .is_some_and(|record: &RolloutHistoryRecord| record.sequence >= sequence)
                {
                    return Err(format!(
                        "rollout history sequence is not strictly increasing: {}",
                        path.display()
                    ));
                }
                history.push(RolloutHistoryRecord {
                    sequence,
                    fingerprint,
                    changes,
                });
            }
            RolloutRecord::ThreadMetadata {
                schema_version,
                session_id: record_session_id,
                thread_id: record_thread_id,
                updated_at_ms,
                previous_content_digest,
                content_digest,
                metadata,
            } => {
                if schema_version != ROLLOUT_SCHEMA_VERSION
                    || record_session_id != session_id
                    || record_thread_id != thread_id
                    || previous_content_digest != metadata_digest
                    || updated_at_ms < initial_thread.updated_at_ms
                    || metadata_content_digest(&session_id, &thread_id, updated_at_ms, &metadata)?
                        != content_digest
                {
                    return Err(format!(
                        "invalid rollout metadata record: {}",
                        path.display()
                    ));
                }
                initial_thread.metadata = metadata;
                initial_thread.updated_at_ms = updated_at_ms;
                metadata_digest = content_digest;
            }
            RolloutRecord::SessionMeta { .. } => {
                return Err(format!(
                    "rollout session metadata may only appear on the first line: {}",
                    path.display()
                ));
            }
        }
    }
    Ok(RolloutScan {
        session_id,
        thread_id,
        created_at_ms,
        rollout_path,
        initial_thread,
        history,
    })
}

fn parse_record(line: &str, path: &Path) -> Result<RolloutRecord, String> {
    serde_json::from_str(line).map_err(|error| {
        format!(
            "invalid rollout JSONL record in {}: {error}",
            path.display()
        )
    })
}

fn history_content_digest(
    session_id: &str,
    thread_id: &str,
    changes: &ThreadHistoryChangeSet,
) -> Result<String, String> {
    let bytes = serde_json::to_vec(&(session_id, thread_id, changes))
        .map_err(|error| format!("failed to encode rollout history digest: {error}"))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn metadata_content_digest(
    session_id: &str,
    thread_id: &str,
    updated_at_ms: i64,
    metadata: &Value,
) -> Result<String, String> {
    let bytes = serde_json::to_vec(&(session_id, thread_id, updated_at_ms, metadata))
        .map_err(|error| format!("failed to encode rollout metadata digest: {error}"))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn validate_metadata_update(expected: &Thread, next: &Thread) -> Result<(), String> {
    let mut allowed = expected.clone();
    allowed.metadata = next.metadata.clone();
    allowed.updated_at_ms = next.updated_at_ms;
    if &allowed == next {
        return Ok(());
    }
    Err(format!(
        "rollout metadata update changed immutable thread fields for {}",
        expected.thread_id
    ))
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn collect_rollout_files(
    root: &Path,
    relative_root: &Path,
    recursive: bool,
    output: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = fs::read_dir(root)
        .map_err(|error| format!("failed to read rollout root {}: {error}", root.display()))?;
    for entry in entries {
        let entry = entry
            .map_err(|error| format!("failed to read rollout root {}: {error}", root.display()))?;
        let file_name = entry.file_name();
        let path = entry.path();
        let metadata = fs::symlink_metadata(&path).map_err(|error| {
            format!("failed to inspect rollout path {}: {error}", path.display())
        })?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "rollout collection must not contain symlinks: {}",
                path.display()
            ));
        }
        let relative_path = relative_root.join(&file_name);
        if metadata.is_dir() {
            if recursive {
                collect_rollout_files(&path, &relative_path, true, output)?;
            }
            continue;
        }
        let is_rollout = file_name
            .to_str()
            .is_some_and(|name| name.starts_with("rollout-") && name.ends_with(".jsonl"));
        if metadata.is_file() && is_rollout {
            output.push(relative_path);
        }
    }
    Ok(())
}

fn write_synced_record(file: &mut File, path: &Path, record: &RolloutRecord) -> Result<(), String> {
    let mut encoded = serde_json::to_vec(record)
        .map_err(|error| format!("failed to encode rollout record: {error}"))?;
    encoded.push(b'\n');
    file.write_all(&encoded)
        .map_err(|error| format!("failed to write rollout file {}: {error}", path.display()))?;
    file.flush()
        .map_err(|error| format!("failed to flush rollout file {}: {error}", path.display()))?;
    file.sync_data()
        .map_err(|error| format!("failed to sync rollout file {}: {error}", path.display()))
}

fn validate_thread_id_for_file(thread_id: &str) -> Result<(), String> {
    if thread_id.is_empty()
        || thread_id.chars().count() > MAX_THREAD_ID_FILE_CHARS
        || !thread_id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
    {
        return Err(format!(
            "thread id is not safe for a rollout filename: {thread_id:?}"
        ));
    }
    Ok(())
}

fn validate_scoped_rollout_path(path: &Path, scope: &str) -> Result<(), String> {
    if path.is_absolute()
        || path.components().next() != Some(Component::Normal(scope.as_ref()))
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(format!(
            "rollout path must remain under {scope}: {}",
            path.display()
        ));
    }
    Ok(())
}

fn active_storage_path(path: &Path) -> Result<String, String> {
    validate_scoped_rollout_path(path, SESSIONS_DIR_NAME)?;
    normalized_storage_path(path)
}

fn location_storage_path(path: &Path) -> Result<String, String> {
    let scope = path
        .components()
        .next()
        .and_then(|component| match component {
            Component::Normal(value) => value.to_str(),
            _ => None,
        })
        .ok_or_else(|| format!("invalid rollout location: {}", path.display()))?;
    if !matches!(scope, SESSIONS_DIR_NAME | ARCHIVED_SESSIONS_DIR_NAME) {
        return Err(format!("invalid rollout location: {}", path.display()));
    }
    validate_scoped_rollout_path(path, scope)?;
    normalized_storage_path(path)
}

fn normalized_storage_path(path: &Path) -> Result<String, String> {
    Ok(path
        .components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/"))
}

fn active_path_from_archived(path: &Path) -> Result<PathBuf, String> {
    validate_scoped_rollout_path(path, ARCHIVED_SESSIONS_DIR_NAME)?;
    if path.components().count() != 2 {
        return Err(format!(
            "archived rollout must be a direct child of archived_sessions: {}",
            path.display()
        ));
    }
    let file_name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| format!("invalid archived rollout filename: {}", path.display()))?;
    let date = file_name
        .strip_prefix("rollout-")
        .and_then(|value| value.get(..10))
        .ok_or_else(|| format!("rollout filename has no date: {file_name}"))?;
    let bytes = date.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes
            .iter()
            .enumerate()
            .any(|(index, value)| !matches!(index, 4 | 7) && !value.is_ascii_digit())
    {
        return Err(format!("invalid rollout filename date: {file_name}"));
    }
    Ok(PathBuf::from(SESSIONS_DIR_NAME)
        .join(&date[0..4])
        .join(&date[5..7])
        .join(&date[8..10])
        .join(file_name))
}

fn reject_symlink(path: &Path) -> Result<(), String> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(format!(
            "rollout path must not traverse a symlink: {}",
            path.display()
        )),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(format!(
            "failed to inspect rollout path {}: {error}",
            path.display()
        )),
    }
}
