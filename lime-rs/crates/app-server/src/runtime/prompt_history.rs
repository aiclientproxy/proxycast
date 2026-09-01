use app_server_protocol::protocol::v2::{
    PromptHistoryAppendResponse, PromptHistoryEntry, PromptHistoryReadResponse,
};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_MAX_BYTES: usize = 4 * 1024 * 1024;
const MAX_READ_LIMIT: usize = 100;

#[derive(Clone, Deserialize, Serialize)]
struct StoredPromptHistoryEntry {
    session_id: String,
    ts: u64,
    text: String,
}

#[derive(Clone)]
pub(crate) struct PromptHistoryStore {
    path: PathBuf,
    max_bytes: usize,
}

impl PromptHistoryStore {
    pub(crate) fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    pub(crate) fn read(
        &self,
        cursor: Option<&str>,
        limit: Option<u32>,
        expected_log_id: Option<&str>,
    ) -> std::io::Result<PromptHistoryReadResponse> {
        let Some((log_id, rows)) = self.read_rows()? else {
            return Ok(PromptHistoryReadResponse {
                log_id: "0".to_string(),
                entry_count: 0,
                data: Vec::new(),
                next_cursor: None,
            });
        };
        let log_id_string = log_id.to_string();
        if expected_log_id.is_some_and(|expected| expected != log_id_string) {
            return Ok(PromptHistoryReadResponse {
                log_id: log_id_string,
                entry_count: rows.len() as u64,
                data: Vec::new(),
                next_cursor: None,
            });
        }
        let end = cursor
            .map(|value| value.parse::<usize>())
            .transpose()
            .map_err(|error| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("invalid prompt history cursor: {error}"),
                )
            })?
            .unwrap_or(rows.len())
            .min(rows.len());
        let limit = limit
            .map(|value| value as usize)
            .unwrap_or(MAX_READ_LIMIT)
            .clamp(1, MAX_READ_LIMIT);
        let mut data = Vec::with_capacity(limit);
        for index in (0..end).rev() {
            let Some(entry) = rows[index].clone() else {
                continue;
            };
            data.push(PromptHistoryEntry {
                offset: index as u64,
                session_id: entry.session_id,
                ts: entry.ts,
                text: entry.text,
            });
            if data.len() == limit {
                break;
            }
        }
        let next_cursor = data
            .last()
            .and_then(|entry| (entry.offset > 0).then(|| entry.offset.to_string()));
        Ok(PromptHistoryReadResponse {
            log_id: log_id_string,
            entry_count: rows.len() as u64,
            data,
            next_cursor,
        })
    }

    pub(crate) fn append(
        &self,
        session_id: &str,
        text: &str,
    ) -> std::io::Result<PromptHistoryAppendResponse> {
        if text.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "prompt history text must not be empty",
            ));
        }
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&self.path)?;
        file.lock_exclusive()?;
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| std::io::Error::other(format!("system clock before epoch: {error}")))?
            .as_secs();
        let stored = StoredPromptHistoryEntry {
            session_id: session_id.to_string(),
            ts,
            text: text.to_string(),
        };
        let mut line = serde_json::to_vec(&stored)?;
        line.push(b'\n');
        file.seek(SeekFrom::End(0))?;
        file.write_all(&line)?;
        file.flush()?;
        enforce_limit(&mut file, self.max_bytes)?;
        let log_id = file_identity(&file)?;
        let entry_count = count_rows(&mut file)?;
        file.unlock()?;
        let entry = PromptHistoryEntry {
            offset: entry_count.saturating_sub(1),
            session_id: stored.session_id,
            ts: stored.ts,
            text: stored.text,
        };
        Ok(PromptHistoryAppendResponse {
            entry,
            log_id: log_id.to_string(),
            entry_count,
        })
    }

    fn read_rows(&self) -> std::io::Result<Option<(u64, Vec<Option<StoredPromptHistoryEntry>>)>> {
        let mut file = match OpenOptions::new().read(true).open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        file.lock_shared()?;
        let identity = file_identity(&file)?;
        let rows = read_rows_locked(&mut file)?;
        file.unlock()?;
        Ok(Some((identity, rows)))
    }
}

fn read_rows_locked(file: &mut File) -> std::io::Result<Vec<Option<StoredPromptHistoryEntry>>> {
    file.seek(SeekFrom::Start(0))?;
    let reader = BufReader::new(file.try_clone()?);
    let mut rows = Vec::new();
    for line in reader.lines() {
        rows.push(serde_json::from_str(&line?).ok());
    }
    Ok(rows)
}

fn count_rows(file: &mut File) -> std::io::Result<u64> {
    file.seek(SeekFrom::Start(0))?;
    Ok(BufReader::new(file.try_clone()?).lines().count() as u64)
}

fn enforce_limit(file: &mut File, max_bytes: usize) -> std::io::Result<()> {
    if file.metadata()?.len() <= max_bytes as u64 {
        return Ok(());
    }
    let mut reader = BufReader::new(file.try_clone()?);
    let mut retained = VecDeque::new();
    let mut retained_bytes = 0usize;
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = reader.read_until(b'\n', &mut line)?;
        if read == 0 {
            break;
        }
        retained_bytes = retained_bytes.saturating_add(read);
        retained.push_back(line.clone());
        while retained_bytes > max_bytes && retained.len() > 1 {
            if let Some(oldest) = retained.pop_front() {
                retained_bytes = retained_bytes.saturating_sub(oldest.len());
            }
        }
    }
    file.set_len(0)?;
    file.seek(SeekFrom::Start(0))?;
    for row in retained {
        file.write_all(&row)?;
    }
    file.flush()
}

fn file_identity(file: &File) -> std::io::Result<u64> {
    let metadata = file.metadata()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        return Ok(metadata.ino());
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        return Ok(metadata.creation_time());
    }
    #[cfg(not(any(unix, windows)))]
    {
        Ok(metadata.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_and_reads_newest_first_with_cursor() {
        let temp = tempfile::tempdir().expect("tempdir");
        let store = PromptHistoryStore::new(temp.path().join("prompt_history.jsonl"));
        store.append("thread-a", "first").expect("append first");
        store.append("thread-b", "second").expect("append second");
        let page = store.read(None, Some(1), None).expect("read page");
        assert_eq!(page.data[0].text, "second");
        assert_eq!(page.next_cursor.as_deref(), Some("1"));
        let page = store
            .read(page.next_cursor.as_deref(), Some(10), None)
            .expect("read older");
        assert_eq!(page.data[0].text, "first");
    }

    #[test]
    fn malformed_rows_are_skipped_without_changing_offsets() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("prompt_history.jsonl");
        std::fs::write(
            &path,
            b"not-json\n{\"session_id\":\"s\",\"ts\":1,\"text\":\"ok\"}\n",
        )
        .expect("write fixture");
        let store = PromptHistoryStore::new(path);
        let page = store.read(None, Some(10), None).expect("read");
        assert_eq!(page.entry_count, 2);
        assert_eq!(page.data[0].offset, 1);
    }
}
