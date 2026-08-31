use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use serde_json::Value;

use super::{scan_rollout, validate_rollout_prefix, ROLLOUT_TAIL_READ_CHUNK_BYTES};

pub(super) fn repair_crash_tail(path: &Path) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| format!("failed to open rollout file {}: {error}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|error| format!("failed to inspect rollout file {}: {error}", path.display()))?
        .len();
    if file_len == 0 || final_byte(&mut file, path)? == b'\n' {
        return Ok(());
    }

    let tail_start = find_tail_start(&mut file, file_len, path)?;
    file.seek(SeekFrom::Start(tail_start))
        .map_err(|error| format!("failed to seek rollout file {}: {error}", path.display()))?;
    let mut tail = Vec::new();
    file.read_to_end(&mut tail)
        .map_err(|error| format!("failed to read rollout file {}: {error}", path.display()))?;
    drop(file);

    if serde_json::from_slice::<Value>(&tail).is_ok() {
        // A complete record that only lost its delimiter is durable. Validate
        // the entire chain before preserving it so identity or digest drift
        // cannot masquerade as crash repair.
        scan_rollout(path)?;
        append_newline(path, file_len)?;
        return Ok(());
    }

    if tail_start == 0 {
        return Err(format!(
            "rollout first record is an incomplete crash tail: {}",
            path.display()
        ));
    }

    // Validate the complete prefix before mutation. A malformed middle line,
    // sequence regression, identity mismatch, or digest divergence remains a
    // hard error and leaves the source file untouched.
    validate_rollout_prefix(path, tail_start)?;
    truncate_tail(path, file_len, tail_start)?;
    scan_rollout(path).map(|_| ())
}

fn final_byte(file: &mut File, path: &Path) -> Result<u8, String> {
    file.seek(SeekFrom::End(-1))
        .map_err(|error| format!("failed to seek rollout file {}: {error}", path.display()))?;
    let mut byte = [0_u8; 1];
    file.read_exact(&mut byte)
        .map_err(|error| format!("failed to read rollout file {}: {error}", path.display()))?;
    Ok(byte[0])
}

fn find_tail_start(file: &mut File, file_len: u64, path: &Path) -> Result<u64, String> {
    let mut cursor = file_len;
    while cursor > 0 {
        let chunk_start = cursor.saturating_sub(ROLLOUT_TAIL_READ_CHUNK_BYTES as u64);
        let chunk_len = usize::try_from(cursor - chunk_start)
            .map_err(|_| format!("rollout read range is too large: {}", path.display()))?;
        let mut chunk = vec![0_u8; chunk_len];
        file.seek(SeekFrom::Start(chunk_start))
            .and_then(|_| file.read_exact(&mut chunk))
            .map_err(|error| format!("failed to read rollout file {}: {error}", path.display()))?;
        if let Some(index) = chunk.iter().rposition(|byte| *byte == b'\n') {
            return Ok(chunk_start + index as u64 + 1);
        }
        cursor = chunk_start;
    }
    Ok(0)
}

fn append_newline(path: &Path, expected_len: u64) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .append(true)
        .open(path)
        .map_err(|error| format!("failed to open rollout file {}: {error}", path.display()))?;
    ensure_unchanged_len(&file, path, expected_len)?;
    file.write_all(b"\n")
        .and_then(|_| file.flush())
        .and_then(|_| file.sync_data())
        .map_err(|error| {
            format!(
                "failed to terminate rollout tail {}: {error}",
                path.display()
            )
        })
}

fn truncate_tail(path: &Path, expected_len: u64, tail_start: u64) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|error| format!("failed to open rollout file {}: {error}", path.display()))?;
    ensure_unchanged_len(&file, path, expected_len)?;
    file.set_len(tail_start)
        .and_then(|_| file.sync_data())
        .map_err(|error| {
            format!(
                "failed to truncate rollout crash tail {} to {tail_start}: {error}",
                path.display()
            )
        })
}

fn ensure_unchanged_len(file: &File, path: &Path, expected_len: u64) -> Result<(), String> {
    let actual_len = file
        .metadata()
        .map_err(|error| format!("failed to inspect rollout file {}: {error}", path.display()))?
        .len();
    if actual_len == expected_len {
        return Ok(());
    }
    Err(format!(
        "rollout file changed during crash-tail repair: {} (expected {expected_len} bytes, found {actual_len})",
        path.display()
    ))
}
