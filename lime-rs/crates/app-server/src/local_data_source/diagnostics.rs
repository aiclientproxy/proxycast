use app_server_protocol::LogArtifactEntry;
use app_server_protocol::LogEntry;
use app_server_protocol::LogStorageDiagnosticsResponse;
use app_server_protocol::WindowsStartupCheck;
use app_server_protocol::WindowsStartupDiagnosticsResponse;
use chrono::DateTime;
use chrono::NaiveDateTime;
use chrono::Utc;
use flate2::read::GzDecoder;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::UNIX_EPOCH;

mod support_bundle;

pub(crate) use support_bundle::export_support_bundle;

fn parse_persisted_log_line(line: &str) -> Option<LogEntry> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }

    if let Some((timestamp, rest)) = trimmed.split_once(" [") {
        if let Some((level, message)) = rest.split_once("] ") {
            return Some(LogEntry {
                timestamp: timestamp.trim().to_string(),
                level: level.trim().to_lowercase(),
                message: message.trim().to_string(),
            });
        }
    }

    Some(LogEntry {
        timestamp: Utc::now().to_rfc3339(),
        level: "info".to_string(),
        message: trimmed.to_string(),
    })
}

pub(super) fn to_rfc3339(system_time: std::time::SystemTime) -> String {
    DateTime::<Utc>::from(system_time).to_rfc3339()
}

fn parse_rotated_log_timestamp(current_log_path: &Path, candidate: &Path) -> Option<i64> {
    let current_name = current_log_path.file_name()?.to_str()?;
    let candidate_name = candidate.file_name()?.to_str()?;

    if candidate_name == current_name {
        return Some(i64::MAX);
    }

    let prefix = format!("{current_name}.");
    let suffix = candidate_name.strip_prefix(&prefix)?;
    let suffix = suffix.strip_suffix(".gz").unwrap_or(suffix);
    let parsed = NaiveDateTime::parse_from_str(suffix, "%Y%m%d-%H%M%S").ok()?;
    parsed.and_utc().timestamp_nanos_opt()
}

fn path_sort_key(current_log_path: &Path, path: &Path) -> (i64, String) {
    let logical_ts = parse_rotated_log_timestamp(current_log_path, path).unwrap_or_else(|| {
        fs::metadata(path)
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| DateTime::<Utc>::from(modified).timestamp_nanos_opt())
            .unwrap_or_else(|| {
                DateTime::<Utc>::from(UNIX_EPOCH)
                    .timestamp_nanos_opt()
                    .unwrap_or(0)
            })
    });

    (logical_ts, path.to_string_lossy().to_string())
}

fn collect_related_log_paths(current_log_path: &Path) -> Vec<PathBuf> {
    let Some(log_dir) = current_log_path.parent() else {
        return Vec::new();
    };
    let Some(file_name) = current_log_path.file_name().and_then(|name| name.to_str()) else {
        return Vec::new();
    };
    let prefix = format!("{file_name}.");
    let mut candidates = Vec::new();

    if current_log_path.exists() {
        candidates.push(current_log_path.to_path_buf());
    }

    let Ok(entries) = fs::read_dir(log_dir) else {
        return candidates;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|value| value.to_str()) else {
            continue;
        };
        if (name == file_name || name.starts_with(&prefix))
            && !candidates.iter().any(|candidate| candidate == &path)
        {
            candidates.push(path);
        }
    }

    candidates.sort_by_key(|path| path_sort_key(current_log_path, path));
    candidates
}

fn collect_raw_response_paths(current_log_path: &Path) -> Vec<PathBuf> {
    let Some(log_dir) = current_log_path.parent() else {
        return Vec::new();
    };
    let Ok(entries) = fs::read_dir(log_dir) else {
        return Vec::new();
    };

    let mut candidates: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.starts_with("raw_response_") && name.ends_with(".txt"))
        })
        .collect();
    candidates.sort_by_key(|path| path_sort_key(current_log_path, path));
    candidates.reverse();
    candidates
}

fn read_log_file_content(path: &Path) -> Option<String> {
    if path.extension().and_then(|ext| ext.to_str()) == Some("gz") {
        let file = fs::File::open(path).ok()?;
        let mut decoder = GzDecoder::new(file);
        let mut content = String::new();
        decoder.read_to_string(&mut content).ok()?;
        return Some(content);
    }

    fs::read_to_string(path).ok()
}

pub(super) fn read_persisted_logs_tail_from_path(
    current_log_path: &Path,
    lines: usize,
) -> Vec<LogEntry> {
    let safe_limit = lines.clamp(1, 1_000);
    let related_paths = collect_related_log_paths(current_log_path);
    if related_paths.is_empty() {
        return Vec::new();
    }

    let mut parsed = Vec::new();
    for log_path in related_paths.into_iter().rev() {
        let remaining = safe_limit.saturating_sub(parsed.len());
        if remaining == 0 {
            break;
        }

        let Some(content) = read_log_file_content(&log_path) else {
            continue;
        };

        parsed.extend(
            content
                .lines()
                .rev()
                .take(remaining)
                .filter_map(parse_persisted_log_line),
        );
    }

    parsed.reverse();
    parsed
}

fn build_log_artifact_entry(path: &Path) -> Option<LogArtifactEntry> {
    let metadata = fs::metadata(path).ok()?;
    let modified_at = metadata.modified().ok().map(to_rfc3339);
    let file_name = path.file_name()?.to_string_lossy().to_string();

    Some(LogArtifactEntry {
        file_name,
        path: path.to_string_lossy().to_string(),
        size_bytes: metadata.len(),
        modified_at,
        compressed: path.extension().and_then(|ext| ext.to_str()) == Some("gz"),
    })
}

pub(super) fn read_log_storage_diagnostics_from_path(
    current_log_path: &Path,
    in_memory_log_count: usize,
) -> LogStorageDiagnosticsResponse {
    let current_log_exists = current_log_path.exists();
    let current_log_size_bytes = fs::metadata(current_log_path)
        .ok()
        .map(|metadata| metadata.len());
    let log_directory = current_log_path
        .parent()
        .map(|path| path.to_string_lossy().to_string());
    let related_log_files = collect_related_log_paths(current_log_path)
        .into_iter()
        .rev()
        .take(12)
        .filter_map(|path| build_log_artifact_entry(&path))
        .collect();
    let raw_response_files = collect_raw_response_paths(current_log_path)
        .into_iter()
        .take(12)
        .filter_map(|path| build_log_artifact_entry(&path))
        .collect();

    LogStorageDiagnosticsResponse {
        log_directory,
        current_log_path: Some(current_log_path.to_string_lossy().to_string()),
        current_log_exists,
        current_log_size_bytes,
        in_memory_log_count,
        related_log_files,
        raw_response_files,
    }
}

pub(super) fn legacy_data_dir_guess() -> Option<PathBuf> {
    dirs::home_dir().map(|home| home.join(".lime"))
}

fn path_to_string(path: Option<PathBuf>) -> Option<String> {
    path.map(|path| path.to_string_lossy().to_string())
}

fn env_value(key: &str) -> Option<String> {
    std::env::var(key).ok().filter(|value| !value.is_empty())
}

fn ok_startup_check(key: &str, message: String) -> WindowsStartupCheck {
    WindowsStartupCheck {
        key: key.to_string(),
        status: "ok".to_string(),
        message,
        detail: None,
    }
}

fn warn_startup_check(key: &str, message: String, detail: Option<String>) -> WindowsStartupCheck {
    WindowsStartupCheck {
        key: key.to_string(),
        status: "warning".to_string(),
        message,
        detail,
    }
}

fn error_startup_check(key: &str, message: String, detail: Option<String>) -> WindowsStartupCheck {
    WindowsStartupCheck {
        key: key.to_string(),
        status: "error".to_string(),
        message,
        detail,
    }
}

fn ensure_dir_writable(path: &Path) -> Result<(), String> {
    fs::create_dir_all(path)
        .map_err(|error| format!("创建目录失败 {}: {error}", path.display()))?;
    let probe_path = path.join(".app-server-startup-write-test");
    fs::write(&probe_path, b"ok")
        .map_err(|error| format!("写入探测文件失败 {}: {error}", probe_path.display()))?;
    fs::remove_file(&probe_path)
        .map_err(|error| format!("删除探测文件失败 {}: {error}", probe_path.display()))
}

pub(crate) fn read_windows_startup_diagnostics(
    app_data_root: &Path,
    database_path: &Path,
) -> Result<WindowsStartupDiagnosticsResponse, String> {
    // 两者都由组合根注入：诊断必须报告实际生效的根与实际打开的库，
    // 而不是重新猜出来的平台默认值。
    let app_data_dir = Some(app_data_root.to_path_buf());
    let legacy_data_dir = legacy_data_dir_guess();
    let db_path = Some(database_path.to_path_buf());
    let current_exe = std::env::current_exe().ok();
    let current_dir = std::env::current_dir().ok();
    let home_dir = dirs::home_dir();
    let shell_env = env_value("SHELL");
    let comspec_env = env_value("COMSPEC");
    let resolved_terminal_shell = shell_env.clone().or_else(|| comspec_env.clone());
    let mut checks = Vec::new();

    match app_data_dir.as_deref() {
        Some(path) => match ensure_dir_writable(path) {
            Ok(()) => checks.push(ok_startup_check(
                "app_data_dir",
                format!("应用数据目录可写: {}", path.display()),
            )),
            Err(error) => checks.push(warn_startup_check(
                "app_data_dir",
                format!("应用数据目录不可写: {}", path.display()),
                Some(error),
            )),
        },
        None => checks.push(warn_startup_check(
            "app_data_dir",
            "无法解析应用数据目录".to_string(),
            None,
        )),
    }

    match current_dir.as_deref() {
        Some(path) if path.exists() => checks.push(ok_startup_check(
            "current_dir",
            format!("当前工作目录存在: {}", path.display()),
        )),
        Some(path) => checks.push(warn_startup_check(
            "current_dir",
            format!("当前工作目录不存在: {}", path.display()),
            None,
        )),
        None => checks.push(warn_startup_check(
            "current_dir",
            "无法解析当前工作目录".to_string(),
            None,
        )),
    }

    if db_path.as_ref().is_some_and(|path| path.exists()) {
        checks.push(ok_startup_check("db_path", "数据库路径已存在".to_string()));
    } else if db_path.is_some() {
        checks.push(warn_startup_check(
            "db_path",
            "数据库路径尚未创建".to_string(),
            None,
        ));
    } else {
        checks.push(error_startup_check(
            "db_path",
            "无法解析数据库路径".to_string(),
            None,
        ));
    }

    let has_blocking_issues = checks.iter().any(|check| check.status == "error");
    let has_warnings = checks.iter().any(|check| check.status == "warning");
    let summary_message = if has_blocking_issues {
        Some("App Server 启动环境存在阻塞问题。".to_string())
    } else if has_warnings {
        Some("App Server 启动环境存在需要关注的警告。".to_string())
    } else {
        Some("App Server 启动环境自检通过。".to_string())
    };

    Ok(WindowsStartupDiagnosticsResponse {
        platform: std::env::consts::OS.to_string(),
        app_data_dir: path_to_string(app_data_dir),
        legacy_lime_dir: path_to_string(legacy_data_dir),
        db_path: path_to_string(db_path),
        webview2_version: None,
        current_exe: path_to_string(current_exe),
        current_dir: path_to_string(current_dir),
        resource_dir: None,
        home_dir: path_to_string(home_dir),
        shell_env,
        comspec_env,
        resolved_terminal_shell,
        installation_kind_guess: Some("app-server".to_string()),
        checks,
        has_blocking_issues,
        has_warnings,
        summary_message,
    })
}

fn truncate_current_log_file(current_log_path: &Path) -> Result<(), String> {
    if let Some(parent) = current_log_path.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            format!(
                "创建日志目录失败（{}）: {}",
                parent.to_string_lossy(),
                error
            )
        })?;
    }

    fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(current_log_path)
        .and_then(|mut file| file.write_all(b""))
        .map_err(|error| {
            format!(
                "清空当前日志文件失败（{}）: {}",
                current_log_path.to_string_lossy(),
                error
            )
        })
}

pub(super) fn clear_diagnostic_log_artifacts_from_path(
    current_log_path: &Path,
) -> Result<(), String> {
    for path in collect_related_log_paths(current_log_path) {
        if path == current_log_path || !path.exists() {
            continue;
        }
        fs::remove_file(&path).map_err(|error| {
            format!(
                "删除历史日志文件失败（{}）: {}",
                path.to_string_lossy(),
                error
            )
        })?;
    }

    for path in collect_raw_response_paths(current_log_path) {
        if !path.exists() {
            continue;
        }
        fs::remove_file(&path).map_err(|error| {
            format!(
                "删除原始响应文件失败（{}）: {}",
                path.to_string_lossy(),
                error
            )
        })?;
    }

    Ok(())
}

pub(super) fn clear_persisted_log_artifacts_from_path(
    current_log_path: &Path,
) -> Result<(), String> {
    truncate_current_log_file(current_log_path)?;
    clear_diagnostic_log_artifacts_from_path(current_log_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn startup_diagnostics_reports_injected_storage_roots() {
        let temp = TempDir::new().expect("temp dir");
        let app_data_root = temp.path().join("machine-data");
        let database_path = temp.path().join("portable-agent").join("lime.db");
        fs::create_dir_all(database_path.parent().expect("database parent"))
            .expect("create database parent");
        fs::write(&database_path, []).expect("create database file");

        let diagnostics = read_windows_startup_diagnostics(&app_data_root, &database_path)
            .expect("startup diagnostics");

        assert_eq!(
            diagnostics.app_data_dir.as_deref(),
            Some(app_data_root.to_string_lossy().as_ref())
        );
        assert_eq!(
            diagnostics.db_path.as_deref(),
            Some(database_path.to_string_lossy().as_ref())
        );
        assert!(diagnostics
            .checks
            .iter()
            .any(|check| check.key == "app_data_dir" && check.status == "ok"));
        assert!(diagnostics
            .checks
            .iter()
            .any(|check| check.key == "db_path" && check.status == "ok"));
    }

    #[test]
    fn log_helpers_read_tail_and_clear_persisted_artifacts() {
        let temp = TempDir::new().expect("temp dir");
        let log_path = temp.path().join("lime.log");
        let rotated_path = temp.path().join("lime.log.20260313-010000");
        let raw_response_path = temp.path().join("raw_response_request-1.txt");
        fs::write(
            &rotated_path,
            "2026-03-13 01:00:00.000 [INFO] rotated one\nfallback line\n",
        )
        .expect("write rotated log");
        fs::write(&log_path, "2026-03-13 01:01:00.000 [WARN] current two\n")
            .expect("write current log");
        fs::write(&raw_response_path, "raw").expect("write raw response");

        let entries = read_persisted_logs_tail_from_path(&log_path, 3);
        assert_eq!(
            entries
                .iter()
                .map(|entry| (entry.level.as_str(), entry.message.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("info", "rotated one"),
                ("info", "fallback line"),
                ("warn", "current two"),
            ]
        );

        clear_persisted_log_artifacts_from_path(&log_path).expect("clear persisted logs");

        assert_eq!(fs::read_to_string(&log_path).expect("read current log"), "");
        assert!(!rotated_path.exists());
        assert!(!raw_response_path.exists());
    }
}
