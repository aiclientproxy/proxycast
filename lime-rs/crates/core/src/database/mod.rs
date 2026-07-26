pub mod agent_session_repository;
pub mod dao;
pub mod migration;
mod migration_support;
pub mod migration_v2;
pub mod migration_v4;
pub mod migration_v5;
pub mod model_control_migration;
pub mod schema;
mod startup_migrations;
pub mod system_providers;

use crate::app_paths;
use rusqlite::Connection;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

const SQLITE_WAL_PRAGMAS: &str = r#"
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;
PRAGMA temp_store = MEMORY;
"#;

const SQLITE_FALLBACK_PRAGMAS: &str = r#"
PRAGMA journal_mode = DELETE;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;
PRAGMA temp_store = MEMORY;
"#;

/// 进程内共享的 SQLite 连接。
///
/// 注意：
/// - `DbConnection` 底层是单个 `Mutex<Connection>`，不支持同调用链重入获取锁。
/// - 持有 `lock_db()` / `db.lock()` 返回的 guard 时，不要再调用会再次依赖
///   `DbConnection` 的 manager / service / wrapper。
/// - 如果调用方已经拿到了 `&Connection`，优先沿用该连接向下传递。
pub type DbConnection = Arc<Mutex<Connection>>;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConversationWindowSummary {
    pub session_count: i64,
    pub message_count: i64,
    pub content_chars: i64,
}

/// 获取数据库连接锁（自动处理 poisoned lock）
pub fn lock_db(db: &DbConnection) -> Result<std::sync::MutexGuard<'_, Connection>, String> {
    match db.lock() {
        Ok(guard) => Ok(guard),
        Err(poisoned) => {
            tracing::warn!("[数据库] 检测到数据库锁被污染，尝试恢复: {}", poisoned);
            db.clear_poison();
            Ok(poisoned.into_inner())
        }
    }
}

/// 获取数据库文件路径
pub fn get_db_path() -> Result<PathBuf, String> {
    app_paths::resolve_database_path()
}

/// 初始化数据库连接
pub fn init_database() -> Result<DbConnection, String> {
    let db_path = get_db_path()?;
    init_database_at_path(&db_path)
}

/// 在指定数据根下初始化数据库连接。旧库不迁移：路径只解析，不复制、不写 migration manifest。
pub fn init_database_with_data_dir(data_dir: impl AsRef<Path>) -> Result<DbConnection, String> {
    let database_path = app_paths::resolve_database_path_for_data_dir(data_dir)?;
    init_database_at_path(&database_path)
}

/// 在指定路径初始化数据库连接。
pub fn init_database_at_path(db_path: impl AsRef<Path>) -> Result<DbConnection, String> {
    let db_path = db_path.as_ref();
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("无法创建数据库目录 {}: {e}", parent.display()))?;
    }
    let conn = Connection::open(&db_path).map_err(|e| e.to_string())?;

    // 设置 busy_timeout 为 5 秒，避免 "database is locked" 错误
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|e| format!("设置 busy_timeout 失败: {e}"))?;

    // 启用 WAL 模式提升并发性能
    apply_database_pragmas(&conn)?;

    // 创建表结构
    schema::create_tables(&conn).map_err(|e| e.to_string())?;
    migration::migrate_from_json(&conn)?;
    startup_migrations::run_startup_migrations(&conn);

    Ok(Arc::new(Mutex::new(conn)))
}

fn apply_database_pragmas(conn: &Connection) -> Result<(), String> {
    match conn.execute_batch(SQLITE_WAL_PRAGMAS) {
        Ok(()) => {
            tracing::info!("[数据库] 已启用 WAL 模式和性能优化参数");
            Ok(())
        }
        Err(wal_error) => {
            tracing::warn!(
                "[数据库] 启用 WAL 模式失败，将回退到兼容模式: {}",
                wal_error
            );
            conn.execute_batch(SQLITE_FALLBACK_PRAGMAS)
                .map_err(|fallback_error| {
                    format!("设置数据库优化参数失败: WAL={wal_error}; fallback={fallback_error}")
                })?;
            tracing::info!("[数据库] 已回退到 DELETE journal 兼容模式");
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_database_does_not_seed_a_default_playwright_mcp_server() {
        let temp_dir = tempfile::tempdir().expect("create database fixture directory");
        let db = init_database_at_path(temp_dir.path().join("lime.db"))
            .expect("initialize fresh database");
        let count: i64 = lock_db(&db)
            .expect("lock database")
            .query_row(
                "SELECT COUNT(*) FROM mcp_servers WHERE name = 'playwright'",
                [],
                |row| row.get(0),
            )
            .expect("count default playwright MCP rows");

        assert_eq!(count, 0);
    }
}
