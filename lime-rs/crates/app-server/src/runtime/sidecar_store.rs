use chrono::{SecondsFormat, Utc};
use hex::encode as hex_encode;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

pub(crate) const SIDECAR_READ_CANCELED: &str = "sidecar read canceled";
const SIDECAR_READ_CHUNK_BYTES: usize = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SidecarRef {
    #[serde(rename = "ref")]
    pub ref_id: String,
    pub kind: String,
    pub relative_path: String,
    pub bytes: u64,
    pub sha256: String,
    pub content_status: String,
    pub created_at: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarWriteRequest {
    pub session_id: String,
    pub kind: String,
    pub logical_id: String,
    pub relative_path: String,
    pub content: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarBytesWriteRequest {
    pub session_id: String,
    pub kind: String,
    pub logical_id: String,
    pub relative_path: String,
    pub content: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarReadBytesResult {
    pub bytes: Vec<u8>,
    pub sha256: String,
    pub total_bytes: u64,
    pub offset: u64,
    pub length: u64,
    pub has_more: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarStore {
    root: PathBuf,
}

impl SidecarStore {
    pub fn new(root: impl AsRef<Path>) -> Result<Self, String> {
        let root = root.as_ref().to_path_buf();
        fs::create_dir_all(&root)
            .map_err(|error| format!("无法创建 sidecar 根目录 {}: {error}", root.display()))?;
        Ok(Self { root })
    }

    pub fn write_text(&self, request: &SidecarWriteRequest) -> Result<SidecarRef, String> {
        self.write_bytes(&SidecarBytesWriteRequest {
            session_id: request.session_id.clone(),
            kind: request.kind.clone(),
            logical_id: request.logical_id.clone(),
            relative_path: request.relative_path.clone(),
            content: request.content.as_bytes().to_vec(),
        })
    }

    pub fn write_bytes(&self, request: &SidecarBytesWriteRequest) -> Result<SidecarRef, String> {
        let relative_path = normalize_sidecar_relative_path(&request.relative_path)?;
        let path = self
            .root
            .join(relative_path_to_platform_path(&relative_path));
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("无法创建 sidecar 目录 {}: {error}", parent.display()))?;
        }

        fs::write(&path, &request.content)
            .map_err(|error| format!("无法写入 sidecar 文件 {}: {error}", path.display()))?;

        let bytes = request.content.len() as u64;
        let sha256 = sha256_prefixed(&request.content);
        let actual = sha256_prefixed(
            &fs::read(&path)
                .map_err(|error| format!("无法读取 sidecar 文件 {}: {error}", path.display()))?,
        );
        if actual != sha256 {
            return Err(format!(
                "sidecar 文件校验失败 {}: expected {sha256}, actual {actual}",
                path.display()
            ));
        }

        Ok(SidecarRef {
            ref_id: sidecar_ref_id(&request.kind, &request.session_id, &request.logical_id),
            kind: request.kind.trim().to_string(),
            relative_path,
            bytes,
            sha256,
            content_status: "available".to_string(),
            created_at: Utc::now().to_rfc3339_opts(SecondsFormat::Millis, true),
        })
    }

    pub fn read_text(&self, relative_path: &str) -> Option<String> {
        let relative_path = normalize_sidecar_relative_path(relative_path).ok()?;
        fs::read_to_string(
            self.root
                .join(relative_path_to_platform_path(&relative_path)),
        )
        .ok()
    }

    pub fn read_bytes_verified(
        &self,
        relative_path: &str,
        expected_sha256: Option<&str>,
        max_bytes: u64,
    ) -> Result<Option<SidecarReadBytesResult>, String> {
        let relative_path = normalize_sidecar_relative_path(relative_path)?;
        let path = self
            .root
            .join(relative_path_to_platform_path(&relative_path));
        let total_bytes = match sidecar_file_len(&path)? {
            Some(total_bytes) => total_bytes,
            None => return Ok(None),
        };
        if total_bytes > max_bytes {
            return Err(format!(
                "sidecar 文件超过读取上限 {}: {} > {} bytes",
                path.display(),
                total_bytes,
                max_bytes
            ));
        }
        self.read_bytes_window_verified(&path, expected_sha256, 0, total_bytes, total_bytes)
            .map(Some)
    }

    pub fn read_bytes_range_verified(
        &self,
        relative_path: &str,
        expected_sha256: Option<&str>,
        offset: u64,
        length: u64,
        max_bytes: u64,
    ) -> Result<Option<SidecarReadBytesResult>, String> {
        self.read_bytes_range_verified_with_cancel(
            relative_path,
            expected_sha256,
            offset,
            length,
            max_bytes,
            &|| false,
        )
    }

    pub(crate) fn read_bytes_range_verified_with_cancel(
        &self,
        relative_path: &str,
        expected_sha256: Option<&str>,
        offset: u64,
        length: u64,
        max_bytes: u64,
        is_canceled: &impl Fn() -> bool,
    ) -> Result<Option<SidecarReadBytesResult>, String> {
        fail_if_canceled(is_canceled)?;
        let relative_path = normalize_sidecar_relative_path(relative_path)?;
        let path = self
            .root
            .join(relative_path_to_platform_path(&relative_path));
        if length == 0 {
            return Err("sidecar range length must be positive".to_string());
        }
        if max_bytes == 0 {
            return Err("sidecar range max bytes must be positive".to_string());
        }
        if length > max_bytes {
            return Err(format!(
                "sidecar range exceeds read window: {length} > {max_bytes} bytes"
            ));
        }
        let total_bytes = match sidecar_file_len(&path)? {
            Some(total_bytes) => total_bytes,
            None => return Ok(None),
        };
        if offset > total_bytes {
            return Err(format!(
                "sidecar range offset exceeds file size {}: {} > {} bytes",
                path.display(),
                offset,
                total_bytes
            ));
        }
        let window_length = length.min(total_bytes.saturating_sub(offset));
        self.read_bytes_window_verified_with_cancel(
            &path,
            expected_sha256,
            offset,
            window_length,
            total_bytes,
            is_canceled,
        )
        .map(Some)
    }

    fn read_bytes_window_verified(
        &self,
        path: &Path,
        expected_sha256: Option<&str>,
        offset: u64,
        length: u64,
        total_bytes: u64,
    ) -> Result<SidecarReadBytesResult, String> {
        self.read_bytes_window_verified_with_cancel(
            path,
            expected_sha256,
            offset,
            length,
            total_bytes,
            &|| false,
        )
    }

    fn read_bytes_window_verified_with_cancel(
        &self,
        path: &Path,
        expected_sha256: Option<&str>,
        offset: u64,
        length: u64,
        total_bytes: u64,
        is_canceled: &impl Fn() -> bool,
    ) -> Result<SidecarReadBytesResult, String> {
        fail_if_canceled(is_canceled)?;
        let mut file = fs::File::open(path)
            .map_err(|error| format!("无法读取 sidecar 文件 {}: {error}", path.display()))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| format!("无法定位 sidecar 文件 {}: {error}", path.display()))?;
        let buffer_len = usize::try_from(length)
            .map_err(|_| format!("sidecar range too large to allocate: {length} bytes"))?;
        let mut bytes = Vec::with_capacity(buffer_len);
        let mut remaining = buffer_len;
        let mut buffer = vec![0_u8; SIDECAR_READ_CHUNK_BYTES.min(buffer_len.max(1))];
        while remaining > 0 {
            fail_if_canceled(is_canceled)?;
            let read_len = buffer.len().min(remaining);
            file.read_exact(&mut buffer[..read_len])
                .map_err(|error| format!("无法读取 sidecar 文件 {}: {error}", path.display()))?;
            fail_if_canceled(is_canceled)?;
            bytes.extend_from_slice(&buffer[..read_len]);
            remaining -= read_len;
        }
        let sha256 = sha256_file_with_cancel(path, is_canceled)?;
        if let Some(expected_sha256) = expected_sha256
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            if sha256 != expected_sha256 {
                return Err(format!(
                    "sidecar 文件校验失败 {}: expected {expected_sha256}, actual {sha256}",
                    path.display()
                ));
            }
        }
        Ok(SidecarReadBytesResult {
            bytes,
            sha256,
            total_bytes,
            offset,
            length,
            has_more: offset.saturating_add(length) < total_bytes,
        })
    }

    pub fn clear_session(&self, session_id: &str) -> Result<(), String> {
        let session_id = session_id.trim();
        if session_id.is_empty() {
            return Err("sidecar session id must not be empty".to_string());
        }
        let relative_path =
            normalize_sidecar_relative_path(&format!("sessions/{}", safe_file_stem(session_id)))?;
        let path = self
            .root
            .join(relative_path_to_platform_path(&relative_path));
        match fs::remove_dir_all(&path) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(format!(
                "无法删除 sidecar 会话目录 {}: {error}",
                path.display()
            )),
        }
    }
}

fn fail_if_canceled(is_canceled: &impl Fn() -> bool) -> Result<(), String> {
    if is_canceled() {
        Err(SIDECAR_READ_CANCELED.to_string())
    } else {
        Ok(())
    }
}

fn sidecar_file_len(path: &Path) -> Result<Option<u64>, String> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(format!(
                "无法读取 sidecar 文件元数据 {}: {error}",
                path.display()
            ));
        }
    };
    Ok(Some(metadata.len()))
}

fn sha256_file_with_cancel(path: &Path, is_canceled: &impl Fn() -> bool) -> Result<String, String> {
    fail_if_canceled(is_canceled)?;
    let mut file = fs::File::open(path)
        .map_err(|error| format!("无法读取 sidecar 文件 {}: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; SIDECAR_READ_CHUNK_BYTES];
    loop {
        fail_if_canceled(is_canceled)?;
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("无法读取 sidecar 文件 {}: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        fail_if_canceled(is_canceled)?;
    }
    Ok(format!("sha256:{}", hex_encode(hasher.finalize())))
}

pub fn normalize_sidecar_relative_path(path: &str) -> Result<String, String> {
    let trimmed = path.trim().replace('\\', "/");
    if trimmed.is_empty() {
        return Err("sidecar relative path must not be empty".to_string());
    }
    if trimmed.starts_with('/') || trimmed.contains("://") {
        return Err(format!("sidecar relative path must be relative: {path}"));
    }

    let mut parts = Vec::new();
    for component in Path::new(&trimmed).components() {
        match component {
            Component::Normal(value) => {
                let value = value
                    .to_str()
                    .ok_or_else(|| format!("sidecar path contains non-utf8 component: {path}"))?;
                if value.is_empty() {
                    continue;
                }
                parts.push(value.to_string());
            }
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("sidecar path must stay inside root: {path}"));
            }
        }
    }

    if parts.is_empty() {
        Err("sidecar relative path must include a file name".to_string())
    } else {
        Ok(parts.join("/"))
    }
}

pub fn session_scoped_relative_path(session_id: &str, file_name: &str) -> String {
    format!("sessions/{}/{}", safe_file_stem(session_id), file_name)
}

fn relative_path_to_platform_path(relative_path: &str) -> PathBuf {
    relative_path
        .split('/')
        .filter(|part| !part.is_empty())
        .collect()
}

fn sidecar_ref_id(kind: &str, session_id: &str, logical_id: &str) -> String {
    format!(
        "sidecar://{}/{}",
        safe_ref_part(kind),
        hex_encode(Sha256::digest(
            format!("{session_id}:{logical_id}").as_bytes()
        ))
    )
}

fn safe_file_stem(value: &str) -> String {
    let stem = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>();
    let stem = stem.trim_matches('_');
    if stem.is_empty() {
        "unknown".to_string()
    } else {
        stem.to_string()
    }
}

fn safe_ref_part(value: &str) -> String {
    let part = value
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    if part.is_empty() {
        "content".to_string()
    } else {
        part
    }
}

fn sha256_prefixed(bytes: &[u8]) -> String {
    format!("sha256:{}", hex_encode(Sha256::digest(bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[test]
    fn write_text_persists_relative_file_and_ref_metadata() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = SidecarStore::new(temp.path()).expect("store");

        let reference = store
            .write_text(&SidecarWriteRequest {
                session_id: "sess-a".to_string(),
                kind: "tool_output".to_string(),
                logical_id: "output-a".to_string(),
                relative_path: "sessions/sess-a/runtime-outputs/output-a.txt".to_string(),
                content: "hello".to_string(),
            })
            .expect("write");

        assert_eq!(
            reference.relative_path,
            "sessions/sess-a/runtime-outputs/output-a.txt"
        );
        assert_eq!(reference.bytes, 5);
        assert!(reference.sha256.starts_with("sha256:"));
        assert_eq!(
            store.read_text(&reference.relative_path).as_deref(),
            Some("hello")
        );
    }

    #[test]
    fn write_bytes_and_read_bytes_verified_enforces_digest_and_size() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = SidecarStore::new(temp.path()).expect("store");

        let reference = store
            .write_bytes(&SidecarBytesWriteRequest {
                session_id: "sess-a".to_string(),
                kind: "media".to_string(),
                logical_id: "image-a".to_string(),
                relative_path: "sessions/sess-a/media/image-a.png".to_string(),
                content: vec![0x89, b'P', b'N', b'G'],
            })
            .expect("write");

        let read = store
            .read_bytes_verified(&reference.relative_path, Some(&reference.sha256), 16)
            .expect("read")
            .expect("available");
        assert_eq!(read.bytes, vec![0x89, b'P', b'N', b'G']);
        assert_eq!(read.sha256, reference.sha256);
        assert_eq!(read.total_bytes, 4);
        assert_eq!(read.offset, 0);
        assert_eq!(read.length, 4);
        assert!(!read.has_more);
        assert!(store
            .read_bytes_verified(&reference.relative_path, Some("sha256:bad"), 16)
            .expect_err("digest mismatch")
            .contains("校验失败"));
        assert!(store
            .read_bytes_verified(&reference.relative_path, Some(&reference.sha256), 2)
            .expect_err("too large")
            .contains("超过读取上限"));
    }

    #[test]
    fn read_bytes_range_verified_reads_window_and_keeps_full_digest() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = SidecarStore::new(temp.path()).expect("store");
        let reference = store
            .write_bytes(&SidecarBytesWriteRequest {
                session_id: "sess-a".to_string(),
                kind: "media".to_string(),
                logical_id: "image-a".to_string(),
                relative_path: "sessions/sess-a/media/image-a.png".to_string(),
                content: vec![0x89, b'P', b'N', b'G'],
            })
            .expect("write");

        let read = store
            .read_bytes_range_verified(&reference.relative_path, Some(&reference.sha256), 1, 2, 2)
            .expect("read range")
            .expect("available");

        assert_eq!(read.bytes, vec![b'P', b'N']);
        assert_eq!(read.sha256, reference.sha256);
        assert_eq!(read.total_bytes, 4);
        assert_eq!(read.offset, 1);
        assert_eq!(read.length, 2);
        assert!(read.has_more);
        assert!(store
            .read_bytes_range_verified(&reference.relative_path, Some(&reference.sha256), 0, 3, 2)
            .expect_err("range too large")
            .contains("exceeds read window"));
    }

    #[test]
    fn read_bytes_range_verified_with_cancel_stops_before_file_io() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = SidecarStore::new(temp.path()).expect("store");
        let reference = store
            .write_bytes(&SidecarBytesWriteRequest {
                session_id: "sess-a".to_string(),
                kind: "media".to_string(),
                logical_id: "image-a".to_string(),
                relative_path: "sessions/sess-a/media/image-a.png".to_string(),
                content: vec![0x89, b'P', b'N', b'G'],
            })
            .expect("write");
        let canceled = AtomicBool::new(true);

        let error = store
            .read_bytes_range_verified_with_cancel(
                &reference.relative_path,
                Some(&reference.sha256),
                0,
                4,
                4,
                &|| canceled.load(Ordering::SeqCst),
            )
            .expect_err("canceled");

        assert_eq!(error, SIDECAR_READ_CANCELED);
    }

    #[test]
    fn rejects_paths_outside_sidecar_root() {
        assert!(normalize_sidecar_relative_path("../secret.txt").is_err());
        assert!(normalize_sidecar_relative_path("/tmp/secret.txt").is_err());
        assert!(normalize_sidecar_relative_path("file://secret.txt").is_err());
    }

    #[test]
    fn clear_session_removes_session_scoped_sidecars() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = SidecarStore::new(temp.path()).expect("store");
        let reference = store
            .write_text(&SidecarWriteRequest {
                session_id: "sess-a".to_string(),
                kind: "conversation_import_runtime_events".to_string(),
                logical_id: "normalized-runtime-events".to_string(),
                relative_path: session_scoped_relative_path(
                    "sess-a",
                    "conversation-import/runtime-events.jsonl",
                ),
                content: "line".to_string(),
            })
            .expect("write");

        assert_eq!(
            store.read_text(&reference.relative_path).as_deref(),
            Some("line")
        );
        store.clear_session("sess-a").expect("clear");
        assert!(store.read_text(&reference.relative_path).is_none());
        store.clear_session("sess-a").expect("clear missing");
    }
}
