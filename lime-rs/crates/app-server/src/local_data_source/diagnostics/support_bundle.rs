use super::legacy_data_dir_guess;
use super::read_log_storage_diagnostics_from_path;
use super::read_persisted_logs_tail_from_path;
use super::to_rfc3339;
use crate::summarize_trace_event_store;
use crate::TRACE_EVENT_MAX_FILES_PER_SESSION;
use app_server_protocol::LogStorageDiagnosticsResponse;
use app_server_protocol::SupportBundleExportParams;
use app_server_protocol::SupportBundleExportResponse;
use chrono::Utc;
use serde::Serialize;
use std::fs;
use std::io;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use zip::write::FileOptions;
use zip::CompressionMethod;
use zip::ZipWriter;

mod trace_attachment;

use trace_attachment::write_selected_trace_export;
use trace_attachment::SupportBundleTraceExportManifest;

const SUPPORT_BUNDLE_OUTPUT_DIR_ENV: &str = "LIME_SUPPORT_BUNDLE_OUTPUT_DIR";

#[derive(Debug, Clone, Serialize)]
struct SupportBundlePathMetadata {
    path: String,
    exists: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_write_time: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct SupportBundleTreeEntry {
    relative_path: String,
    is_directory: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modified_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SupportBundleManifest {
    generated_at: String,
    app_version: String,
    platform: String,
    arch: String,
    username: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    app_data_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    legacy_data_dir: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    database_path: Option<String>,
    path_checks: Vec<SupportBundlePathMetadata>,
    log_storage_diagnostics: LogStorageDiagnosticsResponse,
    trace_store_summary: SupportBundleTraceStoreSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace_export: Option<SupportBundleTraceExportManifest>,
    persisted_log_tail_lines: usize,
    included_sections: Vec<String>,
    omitted_sections: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct SupportBundleTraceStoreSummary {
    available: bool,
    root_exists: bool,
    max_files: usize,
    trace_count: usize,
    raw_trace_events_included: bool,
    redaction_mode: String,
    traces: Vec<SupportBundleTraceFileSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct SupportBundleTraceFileSummary {
    relative_path: String,
    size_bytes: u64,
    event_count: u64,
    parse_error_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    first_wall_time_unix_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_wall_time_unix_ms: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    modified_at: Option<String>,
}

fn default_support_bundle_output_dir() -> PathBuf {
    if let Some(path) = std::env::var_os(SUPPORT_BUNDLE_OUTPUT_DIR_ENV)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
    {
        return path;
    }
    dirs::download_dir()
        .or_else(dirs::desktop_dir)
        .unwrap_or_else(std::env::temp_dir)
}

fn support_bundle_username() -> String {
    std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "unknown".to_string())
}

fn config_path_guess() -> Option<PathBuf> {
    dirs::config_dir().map(|dir| dir.join("lime").join("config.yaml"))
}

fn collect_support_path_metadata(path: Option<&Path>) -> SupportBundlePathMetadata {
    let Some(path) = path else {
        return SupportBundlePathMetadata {
            path: String::new(),
            exists: false,
            kind: None,
            last_write_time: None,
            size_bytes: None,
        };
    };

    let path_string = path.to_string_lossy().to_string();
    let Ok(metadata) = fs::metadata(path) else {
        return SupportBundlePathMetadata {
            path: path_string,
            exists: false,
            kind: None,
            last_write_time: None,
            size_bytes: None,
        };
    };

    let kind = if metadata.is_dir() {
        Some("directory".to_string())
    } else if metadata.is_file() {
        Some("file".to_string())
    } else {
        Some("other".to_string())
    };

    SupportBundlePathMetadata {
        path: path_string,
        exists: true,
        kind,
        last_write_time: metadata.modified().ok().map(to_rfc3339),
        size_bytes: if metadata.is_dir() {
            None
        } else {
            Some(metadata.len())
        },
    }
}

fn should_exclude_support_listing(relative_path: &Path) -> bool {
    relative_path
        .components()
        .next()
        .and_then(|component| component.as_os_str().to_str())
        .is_some_and(|name| matches!(name, "credentials" | "auth"))
}

fn collect_directory_tree_entries(root: &Path) -> Vec<SupportBundleTreeEntry> {
    fn walk(base: &Path, current: &Path, entries: &mut Vec<SupportBundleTreeEntry>) {
        let Ok(children) = fs::read_dir(current) else {
            return;
        };

        for child in children.flatten() {
            let path = child.path();
            let Ok(relative_path) = path.strip_prefix(base) else {
                continue;
            };
            if should_exclude_support_listing(relative_path) {
                continue;
            }
            let Ok(metadata) = child.metadata() else {
                continue;
            };
            let is_directory = metadata.is_dir();
            entries.push(SupportBundleTreeEntry {
                relative_path: relative_path.to_string_lossy().to_string(),
                is_directory,
                size_bytes: if is_directory {
                    None
                } else {
                    Some(metadata.len())
                },
                modified_at: metadata.modified().ok().map(to_rfc3339),
            });

            if is_directory {
                walk(base, &path, entries);
            }
        }
    }

    if !root.exists() {
        return Vec::new();
    }

    let mut entries = Vec::new();
    walk(root, root, &mut entries);
    entries
}

fn collect_trace_store_summary(trace_root: Option<&Path>) -> SupportBundleTraceStoreSummary {
    let Some(trace_root) = trace_root else {
        return empty_trace_store_summary(false);
    };
    if !trace_root.exists() {
        return empty_trace_store_summary(false);
    }

    let traces = summarize_trace_event_store(trace_root, TRACE_EVENT_MAX_FILES_PER_SESSION)
        .into_iter()
        .map(|summary| SupportBundleTraceFileSummary {
            relative_path: summary.relative_path,
            size_bytes: summary.size_bytes,
            event_count: summary.event_count,
            parse_error_count: summary.parse_error_count,
            session_id: summary.session_id,
            trace_id: summary.trace_id,
            first_wall_time_unix_ms: summary.first_wall_time_unix_ms,
            last_wall_time_unix_ms: summary.last_wall_time_unix_ms,
            modified_at: summary.modified_at,
        })
        .collect::<Vec<_>>();

    SupportBundleTraceStoreSummary {
        available: true,
        root_exists: true,
        max_files: TRACE_EVENT_MAX_FILES_PER_SESSION,
        trace_count: traces.len(),
        raw_trace_events_included: false,
        redaction_mode: "summary_only".to_string(),
        traces,
    }
}

fn empty_trace_store_summary(root_exists: bool) -> SupportBundleTraceStoreSummary {
    SupportBundleTraceStoreSummary {
        available: root_exists,
        root_exists,
        max_files: TRACE_EVENT_MAX_FILES_PER_SESSION,
        trace_count: 0,
        raw_trace_events_included: false,
        redaction_mode: "summary_only".to_string(),
        traces: Vec::new(),
    }
}

fn write_support_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let content = serde_json::to_vec_pretty(value)
        .map_err(|error| format!("序列化支持包 JSON 失败 {}: {error}", path.display()))?;
    fs::write(path, content)
        .map_err(|error| format!("写入支持包 JSON 失败 {}: {error}", path.display()))
}

fn normalize_archive_path(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn add_directory_to_zip<W: Write + io::Seek>(
    writer: &mut ZipWriter<W>,
    root: &Path,
    current: &Path,
) -> Result<(), String> {
    let file_options = FileOptions::default().compression_method(CompressionMethod::Deflated);
    let dir_options = FileOptions::default().compression_method(CompressionMethod::Stored);

    for entry in fs::read_dir(current)
        .map_err(|error| format!("读取支持包目录失败 {}: {error}", current.display()))?
    {
        let entry = entry
            .map_err(|error| format!("读取支持包目录项失败 {}: {error}", current.display()))?;
        let path = entry.path();
        let relative = path
            .strip_prefix(root)
            .map_err(|error| format!("计算支持包相对路径失败 {}: {error}", path.display()))?;
        let archive_path = normalize_archive_path(relative);

        if entry
            .file_type()
            .map_err(|error| format!("读取支持包文件类型失败 {}: {error}", path.display()))?
            .is_dir()
        {
            writer
                .add_directory(format!("{archive_path}/"), dir_options)
                .map_err(|error| format!("写入 zip 目录失败 {archive_path}: {error}"))?;
            add_directory_to_zip(writer, root, &path)?;
            continue;
        }

        writer
            .start_file(archive_path.clone(), file_options)
            .map_err(|error| format!("写入 zip 文件失败 {archive_path}: {error}"))?;
        let mut file = fs::File::open(&path)
            .map_err(|error| format!("打开支持包文件失败 {}: {error}", path.display()))?;
        io::copy(&mut file, writer)
            .map_err(|error| format!("压缩支持包文件失败 {}: {error}", path.display()))?;
    }

    Ok(())
}

fn create_zip_from_directory(source_dir: &Path, zip_path: &Path) -> Result<(), String> {
    let file = fs::File::create(zip_path)
        .map_err(|error| format!("创建支持包 zip 失败 {}: {error}", zip_path.display()))?;
    let mut writer = ZipWriter::new(file);
    add_directory_to_zip(&mut writer, source_dir, source_dir)?;
    writer
        .finish()
        .map_err(|error| format!("完成支持包压缩失败 {}: {error}", zip_path.display()))?;
    Ok(())
}

fn copy_directory_recursive(source: &Path, destination: &Path) -> Result<(), String> {
    if !source.exists() {
        return Ok(());
    }
    fs::create_dir_all(destination)
        .map_err(|error| format!("创建支持包目录失败 {}: {error}", destination.display()))?;

    for entry in fs::read_dir(source)
        .map_err(|error| format!("读取目录失败 {}: {error}", source.display()))?
    {
        let entry =
            entry.map_err(|error| format!("读取目录项失败 {}: {error}", source.display()))?;
        let path = entry.path();
        let target = destination.join(entry.file_name());
        if entry
            .file_type()
            .map_err(|error| format!("读取文件类型失败 {}: {error}", path.display()))?
            .is_dir()
        {
            copy_directory_recursive(&path, &target)?;
            continue;
        }
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("创建支持包父目录失败 {}: {error}", parent.display()))?;
        }
        fs::copy(&path, &target).map_err(|error| {
            format!(
                "复制支持包文件失败 {} -> {}: {error}",
                path.display(),
                target.display()
            )
        })?;
    }

    Ok(())
}

fn write_support_bundle_readme(
    path: &Path,
    included_sections: &[String],
    omitted_sections: &[String],
) -> Result<(), String> {
    let included = included_sections
        .iter()
        .map(|section| format!("- {section}"))
        .collect::<Vec<_>>()
        .join("\n");
    let omitted = omitted_sections
        .iter()
        .map(|section| format!("- {section}"))
        .collect::<Vec<_>>()
        .join("\n");
    let content = format!("Lime 支持包\n\n已包含：\n{included}\n\n默认未包含：\n{omitted}\n");

    fs::write(path, content)
        .map_err(|error| format!("写入支持包 README 失败 {}: {error}", path.display()))
}

fn export_support_bundle_to(
    output_directory: &Path,
    app_data_root: &Path,
    database_path: &Path,
    current_log_path: &Path,
    params: SupportBundleExportParams,
    trace_store_root: Option<&Path>,
) -> Result<SupportBundleExportResponse, String> {
    fs::create_dir_all(output_directory).map_err(|error| {
        format!(
            "创建支持包输出目录失败 {}: {error}",
            output_directory.display()
        )
    })?;

    let generated_at = Utc::now().to_rfc3339();
    let timestamp = Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let bundle_name = format!("Lime-Support-{timestamp}");
    let temp_dir = tempfile::tempdir().map_err(|error| format!("创建临时目录失败: {error}"))?;
    let bundle_dir = temp_dir.path().join(&bundle_name);
    let meta_dir = bundle_dir.join("meta");
    let logs_dir = bundle_dir.join("logs");
    fs::create_dir_all(&meta_dir)
        .map_err(|error| format!("创建支持包元数据目录失败 {}: {error}", meta_dir.display()))?;

    let log_storage_diagnostics = read_log_storage_diagnostics_from_path(current_log_path, 0);
    let persisted_log_tail = read_persisted_logs_tail_from_path(current_log_path, 200);
    // AppDataRoot 与 Product DB 路径都由组合根注入，不再重新猜平台根或重解析库路径。
    let app_data_dir = Some(app_data_root.to_path_buf());
    let config_path = config_path_guess();
    let legacy_data_dir = legacy_data_dir_guess();
    let database_path = Some(database_path.to_path_buf());
    let effective_logs_dir = current_log_path.parent().map(Path::to_path_buf);
    let trace_store_summary = collect_trace_store_summary(trace_store_root);

    if let Some(log_dir) = effective_logs_dir.as_deref() {
        copy_directory_recursive(log_dir, &logs_dir)?;
    }

    let mut included_sections = vec![
        "meta/manifest.json".to_string(),
        "meta/log-storage-diagnostics.json".to_string(),
        "meta/persisted-log-tail.json".to_string(),
        "meta/trace-store-summary.json".to_string(),
        "meta/appdata-listing.json（如目录存在）".to_string(),
        "logs/".to_string(),
    ];
    let omitted_sections = vec![
        "config 内容".to_string(),
        "数据库内容".to_string(),
        "credentials 目录正文".to_string(),
        "auth 目录正文".to_string(),
        "raw trace event JSONL 原始字节（支持包默认只包含 trace-store-summary；显式 trace export 只包含 summary-only 重序列化 events）".to_string(),
        "Windows 启动诊断（Desktop Host current 待迁移）".to_string(),
    ];
    let trace_export = if let Some(selection) = params.include_trace_export.as_ref() {
        let trace_export =
            write_selected_trace_export(&bundle_dir, trace_store_root, selection, &generated_at)?;
        included_sections.push(trace_export.relative_path.clone());
        Some(trace_export)
    } else {
        None
    };

    let manifest = SupportBundleManifest {
        generated_at: generated_at.clone(),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        username: support_bundle_username(),
        app_data_dir: app_data_dir
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        config_path: config_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        legacy_data_dir: legacy_data_dir
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        database_path: database_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        path_checks: vec![
            collect_support_path_metadata(app_data_dir.as_deref()),
            collect_support_path_metadata(config_path.as_deref()),
            collect_support_path_metadata(legacy_data_dir.as_deref()),
            collect_support_path_metadata(database_path.as_deref()),
            collect_support_path_metadata(effective_logs_dir.as_deref()),
        ],
        log_storage_diagnostics: log_storage_diagnostics.clone(),
        trace_store_summary: trace_store_summary.clone(),
        trace_export,
        persisted_log_tail_lines: persisted_log_tail.len(),
        included_sections: included_sections.clone(),
        omitted_sections: omitted_sections.clone(),
    };

    write_support_json(&meta_dir.join("manifest.json"), &manifest)?;
    write_support_json(
        &meta_dir.join("log-storage-diagnostics.json"),
        &log_storage_diagnostics,
    )?;
    write_support_json(
        &meta_dir.join("persisted-log-tail.json"),
        &persisted_log_tail,
    )?;
    write_support_json(
        &meta_dir.join("trace-store-summary.json"),
        &trace_store_summary,
    )?;
    if let Some(app_data_dir) = app_data_dir.as_deref() {
        write_support_json(
            &meta_dir.join("appdata-listing.json"),
            &collect_directory_tree_entries(app_data_dir),
        )?;
    }
    write_support_bundle_readme(
        &bundle_dir.join("README.txt"),
        &included_sections,
        &omitted_sections,
    )?;

    let bundle_path = output_directory.join(format!("{bundle_name}.zip"));
    create_zip_from_directory(&bundle_dir, &bundle_path)?;

    Ok(SupportBundleExportResponse {
        bundle_path: bundle_path.to_string_lossy().to_string(),
        output_directory: output_directory.to_string_lossy().to_string(),
        generated_at,
        platform: std::env::consts::OS.to_string(),
        included_sections,
        omitted_sections,
    })
}

pub(crate) fn export_support_bundle(
    app_data_root: &Path,
    database_path: &Path,
    current_log_path: &Path,
    params: SupportBundleExportParams,
    trace_store_root: Option<&Path>,
) -> Result<SupportBundleExportResponse, String> {
    export_support_bundle_to(
        &default_support_bundle_output_dir(),
        app_data_root,
        database_path,
        current_log_path,
        params,
        trace_store_root,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    use std::io::Read;
    use zip::ZipArchive;

    #[test]
    fn trace_store_summary_omits_raw_event_payload_text() {
        let temp = tempfile::tempdir().expect("tempdir");
        let trace_root = temp.path().join("runtime").join("traces");
        let session_dir = trace_root.join("sessions").join("session_session-a");
        fs::create_dir_all(&session_dir).expect("session dir");
        fs::write(
            session_dir.join("trace_trace-a.jsonl"),
            r#"{"schema_version":1,"seq":1,"wall_time_unix_ms":1780000000000,"trace_id":"trace-a","run_id":null,"request_id":null,"session_id":"session-a","thread_id":null,"turn_id":null,"event_id":"event-a","event_sequence":1,"event_type":"message.delta","checkpoint":"app_server.message_delta.emitted","metrics":{"text_chars":21},"redaction":{"mode":"summary_only","raw_agent_event_payload":false,"prompt_text":false,"provider_payload":false},"text":"secret assistant text"}"#,
        )
        .expect("trace file");

        let summary = collect_trace_store_summary(Some(&trace_root));
        let summary_json = serde_json::to_string(&summary).expect("summary json");

        assert!(summary.available);
        assert_eq!(summary.trace_count, 1);
        assert!(!summary.raw_trace_events_included);
        assert_eq!(
            summary.traces[0].relative_path,
            "sessions/session_session-a/trace_trace-a.jsonl"
        );
        assert_eq!(summary.traces[0].event_count, 1);
        assert_eq!(summary.traces[0].trace_id.as_deref(), Some("trace-a"));
        assert!(!summary_json.contains("secret assistant text"));
        assert!(!summary_json.contains("\"text\""));
    }

    #[test]
    fn selected_trace_export_is_summary_only_zip() {
        let temp = tempfile::tempdir().expect("tempdir");
        let trace_root = temp.path().join("runtime").join("traces");
        let session_dir = trace_root.join("sessions").join("session_session-a");
        fs::create_dir_all(&session_dir).expect("session dir");
        fs::write(
            session_dir.join("trace_trace-a.jsonl"),
            r#"{"schema_version":1,"seq":1,"wall_time_unix_ms":1780000000000,"trace_id":"trace-a","run_id":null,"request_id":null,"session_id":"session-a","thread_id":null,"turn_id":null,"event_id":"event-a","event_sequence":1,"event_type":"message.delta","checkpoint":"app_server.message_delta.emitted","metrics":{"text_chars":21},"redaction":{"mode":"summary_only","raw_agent_event_payload":false,"prompt_text":false,"provider_payload":false},"text":"secret assistant text"}"#,
        )
        .expect("trace file");

        let bundle_dir = temp.path().join("support");
        let trace_export = write_selected_trace_export(
            &bundle_dir,
            Some(&trace_root),
            &app_server_protocol::SupportBundleTraceExportSelection {
                session_id: "session-a".to_string(),
                trace_id: "trace-a".to_string(),
            },
            "2026-06-27T00:00:00.000Z",
        )
        .expect("trace export");

        assert_eq!(
            trace_export.relative_path,
            "trace-export/claw-trace-session-a-trace-a.zip"
        );
        assert_eq!(trace_export.redaction_mode, "summary_only");
        assert!(trace_export.summary_only_trace_events_included);

        let bundle = fs::File::open(bundle_dir.join(&trace_export.relative_path))
            .expect("open trace export zip");
        let mut archive = ZipArchive::new(bundle).expect("read trace export zip");
        let mut events = String::new();
        archive
            .by_name("trace/events.jsonl")
            .expect("events")
            .read_to_string(&mut events)
            .expect("read events");

        assert!(events.contains("\"checkpoint\":\"app_server.message_delta.emitted\""));
        assert!(!events.contains("secret assistant text"));
        assert!(!events.contains("\"text\""));
    }

    #[test]
    fn support_bundle_uses_supplied_trace_root_for_opt_in_trace_export() {
        let temp = tempfile::tempdir().expect("tempdir");
        let app_data_root = temp.path().join("machine-data");
        let database_path = temp.path().join("portable-agent").join("lime.db");
        fs::create_dir_all(&app_data_root).expect("app data dir");
        fs::create_dir_all(database_path.parent().expect("database parent")).expect("database dir");
        fs::write(&database_path, []).expect("database file");
        let trace_root = temp.path().join("runtime-current").join("traces");
        let session_dir = trace_root.join("sessions").join("session_session-a");
        fs::create_dir_all(&session_dir).expect("session dir");
        fs::write(
            session_dir.join("trace_trace-a.jsonl"),
            r#"{"schema_version":1,"seq":1,"wall_time_unix_ms":1780000000000,"trace_id":"trace-a","run_id":null,"request_id":null,"session_id":"session-a","thread_id":null,"turn_id":null,"event_id":"event-a","event_sequence":1,"event_type":"message.delta","checkpoint":"app_server.message_delta.emitted","metrics":{"text_chars":21},"redaction":{"mode":"summary_only","raw_agent_event_payload":false,"prompt_text":false,"provider_payload":false},"text":"secret assistant text"}"#,
        )
        .expect("trace file");
        let log_path = temp
            .path()
            .join("app-server")
            .join("observability")
            .join("log")
            .join("lime.log");
        fs::create_dir_all(log_path.parent().expect("log parent")).expect("log dir");
        fs::write(&log_path, "2026-07-19 00:00:00.000 [INFO] current log\n").expect("log file");

        let response = export_support_bundle_to(
            &temp.path().join("bundles"),
            &app_data_root,
            &database_path,
            &log_path,
            SupportBundleExportParams {
                include_trace_export: Some(
                    app_server_protocol::SupportBundleTraceExportSelection {
                        session_id: "session-a".to_string(),
                        trace_id: "trace-a".to_string(),
                    },
                ),
            },
            Some(&trace_root),
        )
        .expect("support bundle");

        assert!(response
            .included_sections
            .contains(&"trace-export/claw-trace-session-a-trace-a.zip".to_string()));
        assert!(response
            .omitted_sections
            .iter()
            .any(|section| section.contains("raw trace event JSONL")));

        let bundle = fs::File::open(response.bundle_path).expect("open support bundle");
        let mut archive = ZipArchive::new(bundle).expect("read support bundle");

        let mut manifest = String::new();
        archive
            .by_name("meta/manifest.json")
            .expect("manifest")
            .read_to_string(&mut manifest)
            .expect("read manifest");
        let manifest: serde_json::Value = serde_json::from_str(&manifest).expect("parse manifest");
        assert_eq!(
            manifest["app_data_dir"],
            app_data_root.to_string_lossy().as_ref()
        );
        assert_eq!(
            manifest["database_path"],
            database_path.to_string_lossy().as_ref()
        );

        let mut trace_summary = String::new();
        archive
            .by_name("meta/trace-store-summary.json")
            .expect("trace summary")
            .read_to_string(&mut trace_summary)
            .expect("read trace summary");
        assert!(trace_summary.contains("\"trace_count\": 1"));
        assert!(!trace_summary.contains("secret assistant text"));
        assert!(!trace_summary.contains("\"text\""));

        let mut trace_export_bytes = Vec::new();
        archive
            .by_name("trace-export/claw-trace-session-a-trace-a.zip")
            .expect("trace export")
            .read_to_end(&mut trace_export_bytes)
            .expect("read trace export");
        let mut trace_export =
            ZipArchive::new(Cursor::new(trace_export_bytes)).expect("read nested trace export");

        let mut events = String::new();
        trace_export
            .by_name("trace/events.jsonl")
            .expect("events")
            .read_to_string(&mut events)
            .expect("read events");
        assert!(events.contains("\"checkpoint\":\"app_server.message_delta.emitted\""));
        assert!(!events.contains("secret assistant text"));
        assert!(!events.contains("\"text\""));
    }
}
