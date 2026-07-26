use std::fs;
use std::path::{Path, PathBuf};

const PRODUCT_DB_FILE_NAME: &str = "lime.db";
const MEMORY_DIR_NAME: &str = "memories";
const SESSIONS_DIR_NAME: &str = "sessions";
const ARCHIVED_SESSIONS_DIR_NAME: &str = "archived_sessions";
const RUNTIME_DIR_NAME: &str = "runtime";
const EVENT_LOG_DIR_NAME: &str = "events";
const TRACE_LOG_DIR_NAME: &str = "traces";
const SIDECAR_DIR_NAME: &str = "sidecar";
const SQLITE_DIR_NAME: &str = "sqlite";
const STATE_DB_FILE_NAME: &str = "state.sqlite";
const THREAD_HISTORY_DB_FILE_NAME: &str = "thread_history.sqlite";
const PROJECTION_DB_FILE_NAME: &str = "projection_1.sqlite";
const TELEMETRY_DB_FILE_NAME: &str = "telemetry_1.sqlite";

/// Product DB 的唯一位置只由 AgentRoot 决定，供不持有 `StorageRoots` 的
/// owner 复用同一个契约，避免各自重新解析库路径。
pub fn product_db_path_for_agent_root(data_root: impl AsRef<Path>) -> PathBuf {
    data_root.as_ref().join(PRODUCT_DB_FILE_NAME)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StorageRoots {
    /// 应用管理的机器数据根。只能由组合根显式注入，
    /// 不得从 `data_root.parent()` 推断——portable/E2E override 下会逃逸出受控根。
    pub app_data_root: PathBuf,
    pub data_root: PathBuf,
    pub memory_root: PathBuf,
    pub product_db_path: PathBuf,
    pub sessions_root: PathBuf,
    pub archived_sessions_root: PathBuf,
    pub runtime_root: PathBuf,
    pub event_log_root: PathBuf,
    pub trace_log_root: PathBuf,
    pub sidecar_root: PathBuf,
    pub sqlite_root: PathBuf,
    pub state_db_path: PathBuf,
    pub thread_history_db_path: PathBuf,
    pub projection_db_path: PathBuf,
    pub telemetry_db_path: PathBuf,
}

impl StorageRoots {
    pub fn from_roots(app_data_root: impl AsRef<Path>, data_root: impl AsRef<Path>) -> Self {
        let app_data_root = app_data_root.as_ref().to_path_buf();
        let data_root = data_root.as_ref().to_path_buf();
        let runtime_root = data_root.join(RUNTIME_DIR_NAME);
        let event_log_root = runtime_root.join(EVENT_LOG_DIR_NAME);
        let trace_log_root = runtime_root.join(TRACE_LOG_DIR_NAME);
        let sidecar_root = runtime_root.join(SIDECAR_DIR_NAME);
        let sqlite_root = data_root.join(SQLITE_DIR_NAME);

        Self {
            product_db_path: data_root.join(PRODUCT_DB_FILE_NAME),
            memory_root: data_root.join(MEMORY_DIR_NAME),
            sessions_root: data_root.join(SESSIONS_DIR_NAME),
            archived_sessions_root: data_root.join(ARCHIVED_SESSIONS_DIR_NAME),
            state_db_path: sqlite_root.join(STATE_DB_FILE_NAME),
            thread_history_db_path: sqlite_root.join(THREAD_HISTORY_DB_FILE_NAME),
            projection_db_path: runtime_root.join(PROJECTION_DB_FILE_NAME),
            telemetry_db_path: runtime_root.join(TELEMETRY_DB_FILE_NAME),
            app_data_root,
            data_root,
            runtime_root,
            event_log_root,
            trace_log_root,
            sidecar_root,
            sqlite_root,
        }
    }

    pub fn initialize(
        app_data_root: impl AsRef<Path>,
        data_root: impl AsRef<Path>,
    ) -> Result<Self, String> {
        let roots = Self::from_roots(app_data_root, data_root);
        fs::create_dir_all(&roots.event_log_root).map_err(|error| {
            format!(
                "无法创建 App Server runtime 事件目录 {}: {error}",
                roots.event_log_root.display()
            )
        })?;
        fs::create_dir_all(&roots.sidecar_root).map_err(|error| {
            format!(
                "无法创建 App Server runtime sidecar 目录 {}: {error}",
                roots.sidecar_root.display()
            )
        })?;
        fs::create_dir_all(&roots.trace_log_root).map_err(|error| {
            format!(
                "无法创建 App Server runtime trace 目录 {}: {error}",
                roots.trace_log_root.display()
            )
        })?;
        fs::create_dir_all(&roots.runtime_root).map_err(|error| {
            format!(
                "无法创建 App Server runtime 目录 {}: {error}",
                roots.runtime_root.display()
            )
        })?;
        fs::create_dir_all(&roots.sqlite_root).map_err(|error| {
            format!(
                "无法创建 App Server SQLite 目录 {}: {error}",
                roots.sqlite_root.display()
            )
        })?;

        Ok(roots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialize_derives_runtime_paths_from_data_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let roots =
            StorageRoots::initialize(temp.path(), temp.path().join("app-server")).expect("roots");

        assert_eq!(roots.app_data_root, temp.path());
        assert_eq!(roots.product_db_path, roots.data_root.join("lime.db"));
        assert_eq!(roots.memory_root, roots.data_root.join("memories"));
        assert_eq!(roots.sessions_root, roots.data_root.join("sessions"));
        assert_eq!(
            roots.archived_sessions_root,
            roots.data_root.join("archived_sessions")
        );
        assert_eq!(roots.runtime_root, roots.data_root.join("runtime"));
        assert_eq!(roots.event_log_root, roots.runtime_root.join("events"));
        assert_eq!(roots.trace_log_root, roots.runtime_root.join("traces"));
        assert_eq!(roots.sidecar_root, roots.runtime_root.join("sidecar"));
        assert_eq!(roots.sqlite_root, roots.data_root.join("sqlite"));
        assert_eq!(roots.state_db_path, roots.sqlite_root.join("state.sqlite"));
        assert_eq!(
            roots.thread_history_db_path,
            roots.sqlite_root.join("thread_history.sqlite")
        );
        assert_eq!(
            roots.projection_db_path,
            roots.runtime_root.join("projection_1.sqlite")
        );
        assert_eq!(
            roots.telemetry_db_path,
            roots.runtime_root.join("telemetry_1.sqlite")
        );
        assert!(roots.event_log_root.is_dir());
        assert!(roots.trace_log_root.is_dir());
        assert!(roots.sidecar_root.is_dir());
        assert!(roots.sqlite_root.is_dir());
    }

    #[test]
    fn app_data_root_is_injected_not_derived_from_agent_root_parent() {
        let temp = tempfile::tempdir().expect("tempdir");
        let app_data_root = temp.path().join("machine-data");
        let portable_agent_root = temp.path().join("portable/elsewhere/agent");
        let roots = StorageRoots::from_roots(&app_data_root, &portable_agent_root);

        assert_eq!(roots.app_data_root, app_data_root);
        assert_eq!(roots.data_root, portable_agent_root);
        assert_ne!(
            roots.app_data_root,
            portable_agent_root
                .parent()
                .expect("agent root parent")
                .to_path_buf()
        );
    }
}
