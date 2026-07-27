use rusqlite::types::Value;
use rusqlite::{
    params, Connection, OpenFlags, OptionalExtension, Transaction, TransactionBehavior,
};
use std::collections::HashSet;
use std::path::Path;

const MODEL_CONTROL_MIGRATION_KEY: &str = "migration.model_control.v1";

const PROVIDER_COLUMNS: &[&str] = &[
    "id",
    "name",
    "type",
    "api_host",
    "is_system",
    "group_name",
    "enabled",
    "sort_order",
    "api_version",
    "project",
    "location",
    "region",
    "models",
    "prompt_cache_mode",
    "created_at",
    "updated_at",
];

const API_KEY_COLUMNS: &[&str] = &[
    "id",
    "provider_id",
    "api_key_encrypted",
    "alias",
    "enabled",
    "usage_count",
    "error_count",
    "last_used_at",
    "created_at",
];

const PROVIDER_UI_STATE_COLUMNS: &[&str] = &["key", "value"];
const MODEL_PREFERENCE_COLUMNS: &[&str] = &[
    "model_id",
    "is_favorite",
    "is_hidden",
    "custom_alias",
    "usage_count",
    "last_used_at",
    "created_at",
    "updated_at",
];

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelControlMigrationReport {
    pub providers: usize,
    pub api_keys: usize,
    pub provider_ui_state: usize,
    pub model_preferences: usize,
    pub settings: usize,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelControlSourceSignal {
    pub api_keys: usize,
    pub custom_providers: usize,
    pub model_preferences: usize,
    pub provider_ui_state: usize,
    pub settings: usize,
}

impl ModelControlSourceSignal {
    pub fn priority(self) -> (usize, usize, usize, usize, usize) {
        (
            self.api_keys,
            self.custom_providers,
            self.model_preferences,
            self.provider_ui_state,
            self.settings,
        )
    }

    fn is_empty(self) -> bool {
        self.priority() == (0, 0, 0, 0, 0)
    }
}

impl ModelControlMigrationReport {
    pub fn changed(self) -> bool {
        self.providers > 0
            || self.api_keys > 0
            || self.provider_ui_state > 0
            || self.model_preferences > 0
            || self.settings > 0
    }
}

/// 只迁移模型控制面，不复制 Product DB，也不读取会话、日志或缓存表。
pub fn migrate_model_control_data(
    source_path: &Path,
    target_path: &Path,
) -> Result<ModelControlMigrationReport, String> {
    if source_path == target_path || !source_path.exists() {
        return Ok(ModelControlMigrationReport::default());
    }
    let mut target = Connection::open(target_path).map_err(|error| {
        format!(
            "无法打开模型控制面 target {}: {error}",
            target_path.display()
        )
    })?;
    target
        .busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|error| format!("设置模型控制面 target busy_timeout 失败: {error}"))?;

    let already_migrated = target
        .query_row(
            "SELECT 1 FROM settings WHERE key = ?1 LIMIT 1",
            [MODEL_CONTROL_MIGRATION_KEY],
            |_| Ok(()),
        )
        .optional()
        .map_err(|error| format!("读取模型控制面迁移 marker 失败: {error}"))?
        .is_some();
    if already_migrated {
        return Ok(ModelControlMigrationReport::default());
    }

    let mut source = open_model_control_source(source_path)?;
    let source_tx = source
        .transaction_with_behavior(TransactionBehavior::Deferred)
        .map_err(|error| format!("开始模型控制面 source 只读事务失败: {error}"))?;
    let mut report = ModelControlMigrationReport::default();
    if model_control_source_signal(&source_tx)?.is_none() {
        return Ok(report);
    }
    let tx = target
        .transaction()
        .map_err(|error| format!("开始模型控制面迁移事务失败: {error}"))?;
    let custom_provider_predicate = |values: &[Value], columns: &[String]| {
        columns
            .iter()
            .position(|column| column == "is_system")
            .and_then(|index| values.get(index))
            .is_none_or(|value| match value {
                Value::Null => true,
                Value::Integer(value) => *value == 0,
                Value::Real(value) => *value == 0.0,
                Value::Text(value) => value == "0",
                Value::Blob(_) => false,
            })
    };
    report.providers = copy_rows(
        &source_tx,
        &tx,
        "api_key_providers",
        PROVIDER_COLUMNS,
        Some(&custom_provider_predicate),
    )?;
    let provider_ids = target_provider_ids(&tx)?;
    let api_key_predicate = |values: &[Value], columns: &[String]| {
        columns
            .iter()
            .position(|column| column == "provider_id")
            .and_then(|index| values.get(index))
            .and_then(|value| match value {
                Value::Text(value) => Some(value.as_str()),
                _ => None,
            })
            .is_some_and(|provider_id| provider_ids.contains(provider_id))
    };
    report.api_keys = copy_rows(
        &source_tx,
        &tx,
        "api_keys",
        API_KEY_COLUMNS,
        Some(&api_key_predicate),
    )?;
    report.provider_ui_state = copy_rows(
        &source_tx,
        &tx,
        "provider_ui_state",
        PROVIDER_UI_STATE_COLUMNS,
        None,
    )?;
    report.model_preferences = copy_rows(
        &source_tx,
        &tx,
        "user_model_preferences",
        MODEL_PREFERENCE_COLUMNS,
        None,
    )?;
    report.settings = copy_settings(&source_tx, &tx)?;
    tx.execute(
        "INSERT OR REPLACE INTO settings (key, value) VALUES (?1, ?2)",
        params![
            MODEL_CONTROL_MIGRATION_KEY,
            source_path.display().to_string()
        ],
    )
    .map_err(|error| format!("写入模型控制面迁移 marker 失败: {error}"))?;
    tx.commit()
        .map_err(|error| format!("提交模型控制面迁移失败: {error}"))?;

    Ok(report)
}

/// 只读取模型控制面计数，不读取或输出 Provider 凭证内容。
pub fn inspect_model_control_source(
    source_path: &Path,
) -> Result<Option<ModelControlSourceSignal>, String> {
    if !source_path.is_file() {
        return Ok(None);
    }
    let mut source = open_model_control_source(source_path)?;
    let source_tx = source
        .transaction_with_behavior(TransactionBehavior::Deferred)
        .map_err(|error| format!("开始模型控制面 source 只读事务失败: {error}"))?;
    model_control_source_signal(&source_tx)
}

fn open_model_control_source(source_path: &Path) -> Result<Connection, String> {
    let source = Connection::open_with_flags(
        source_path,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_URI
            | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|error| {
        format!(
            "无法只读打开模型控制面 source {}: {error}",
            source_path.display()
        )
    })?;
    source
        .busy_timeout(std::time::Duration::from_secs(5))
        .map_err(|error| format!("设置模型控制面 source busy_timeout 失败: {error}"))?;
    Ok(source)
}

fn table_columns(conn: &Connection, table: &str) -> Result<HashSet<String>, String> {
    let mut statement = conn
        .prepare(&format!("PRAGMA table_info({table})"))
        .map_err(|error| format!("读取 {table} schema 失败: {error}"))?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(1))
        .map_err(|error| format!("查询 {table} schema 失败: {error}"))?
        .collect::<Result<HashSet<_>, _>>()
        .map_err(|error| format!("解析 {table} schema 失败: {error}"))?;
    Ok(rows)
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool, String> {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1",
        [table],
        |_| Ok(true),
    )
    .optional()
    .map(|value| value.is_some())
    .map_err(|error| format!("检查模型控制面表 {table} 失败: {error}"))
}

fn copy_rows(
    source: &Connection,
    target: &Transaction<'_>,
    table: &str,
    allowed_columns: &[&str],
    predicate: Option<&dyn Fn(&[Value], &[String]) -> bool>,
) -> Result<usize, String> {
    if !table_exists(source, table)? {
        return Ok(0);
    }
    let source_columns = table_columns(source, table)?;
    let target_columns = table_columns(target, table)?;
    let columns = allowed_columns
        .iter()
        .filter(|column| source_columns.contains(**column) && target_columns.contains(**column))
        .map(|column| (*column).to_string())
        .collect::<Vec<_>>();
    if columns.is_empty() {
        return Ok(0);
    }
    let projection = columns.join(", ");
    let mut statement = source
        .prepare(&format!("SELECT {projection} FROM {table}"))
        .map_err(|error| format!("读取模型控制面表 {table} 失败: {error}"))?;
    let mut rows = statement
        .query([])
        .map_err(|error| format!("遍历模型控制面表 {table} 失败: {error}"))?;
    let placeholders = (1..=columns.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    let insert_sql =
        format!("INSERT OR IGNORE INTO {table} ({projection}) VALUES ({placeholders})");
    let mut copied = 0;
    while let Some(row) = rows
        .next()
        .map_err(|error| format!("读取模型控制面表 {table} 行失败: {error}"))?
    {
        let values = (0..columns.len())
            .map(|index| row.get::<_, Value>(index))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| format!("解析模型控制面表 {table} 行失败: {error}"))?;
        if predicate
            .as_ref()
            .is_some_and(|predicate| !predicate(&values, &columns))
        {
            continue;
        }
        copied += target
            .execute(&insert_sql, rusqlite::params_from_iter(values))
            .map_err(|error| format!("写入模型控制面表 {table} 失败: {error}"))?;
    }
    Ok(copied)
}

fn model_control_source_signal(
    source: &Connection,
) -> Result<Option<ModelControlSourceSignal>, String> {
    let custom_providers = if table_exists(source, "api_key_providers")? {
        let columns = table_columns(source, "api_key_providers")?;
        if columns.contains("is_system") {
            count_rows(
                source,
                "api_key_providers",
                "WHERE COALESCE(is_system, 0) = 0",
            )?
        } else {
            count_rows(source, "api_key_providers", "")?
        }
    } else {
        0
    };
    let signal = ModelControlSourceSignal {
        api_keys: count_rows_if_table_exists(source, "api_keys", "")?,
        custom_providers,
        model_preferences: count_rows_if_table_exists(source, "user_model_preferences", "")?,
        provider_ui_state: count_rows_if_table_exists(source, "provider_ui_state", "")?,
        settings: count_rows_if_table_exists(
            source,
            "settings",
            "WHERE key = 'providers.active_tab'",
        )?,
    };
    Ok((!signal.is_empty()).then_some(signal))
}

fn count_rows_if_table_exists(
    conn: &Connection,
    table: &str,
    predicate: &str,
) -> Result<usize, String> {
    if !table_exists(conn, table)? {
        return Ok(0);
    }
    count_rows(conn, table, predicate)
}

fn count_rows(conn: &Connection, table: &str, predicate: &str) -> Result<usize, String> {
    let count: i64 = conn
        .query_row(
            &format!("SELECT COUNT(*) FROM {table} {predicate}"),
            [],
            |row| row.get(0),
        )
        .map_err(|error| format!("检查模型控制面数据 {table} 失败: {error}"))?;
    usize::try_from(count).map_err(|_| format!("模型控制面表 {table} 行数无效: {count}"))
}

fn target_provider_ids(target: &Transaction<'_>) -> Result<HashSet<String>, String> {
    let mut statement = target
        .prepare("SELECT id FROM api_key_providers")
        .map_err(|error| format!("读取目标 provider id 失败: {error}"))?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|error| format!("查询目标 provider id 失败: {error}"))?
        .collect::<Result<HashSet<_>, _>>()
        .map_err(|error| format!("解析目标 provider id 失败: {error}"))?;
    Ok(rows)
}

fn copy_settings(source: &Connection, target: &Transaction<'_>) -> Result<usize, String> {
    if !table_exists(source, "settings")? {
        return Ok(0);
    }
    let value = source
        .query_row(
            "SELECT value FROM settings WHERE key = 'providers.active_tab' LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(|error| format!("读取 provider active tab 失败: {error}"))?;
    let Some(value) = value else {
        return Ok(0);
    };
    target
        .execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES ('providers.active_tab', ?1)",
            [value],
        )
        .map_err(|error| format!("写入 provider active tab 失败: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::{init_database_at_path, lock_db};
    use tempfile::tempdir;

    #[test]
    fn migrates_model_control_tables_without_copying_sessions_or_caches() {
        let temp = tempdir().expect("tempdir");
        let source_path = temp.path().join("legacy/lime.db");
        let target_path = temp.path().join("agent-root/lime.db");
        let source = init_database_at_path(&source_path).expect("source db");
        {
            let conn = lock_db(&source).expect("source lock");
            conn.execute(
                "INSERT INTO api_key_providers
                 (id, name, type, api_host, group_name, enabled, sort_order, created_at, updated_at)
                 VALUES ('custom', 'Custom', 'openai', 'https://example.invalid/v1', 'custom', 1, 1, '2026-07-19', '2026-07-19')",
                [],
            )
            .expect("provider");
            conn.execute(
                "INSERT INTO api_keys (id, provider_id, api_key_encrypted, created_at)
                 VALUES ('key-1', 'custom', 'ciphertext', '2026-07-19')",
                [],
            )
            .expect("api key");
            conn.execute(
                "INSERT INTO api_key_providers
                 (id, name, type, api_host, is_system, group_name, enabled, sort_order, created_at, updated_at)
                 VALUES ('openai', 'Source OpenAI', 'openai', 'https://source.invalid/v1', 1, 'official', 1, 2, '2026-07-19', '2026-07-19'),
                        ('source-only-system', 'Source Only', 'openai', 'https://source-only.invalid/v1', 1, 'official', 1, 3, '2026-07-19', '2026-07-19')",
                [],
            )
            .expect("source system providers");
            conn.execute(
                "INSERT INTO api_keys (id, provider_id, api_key_encrypted, created_at)
                 VALUES ('key-system', 'openai', 'system-ciphertext', '2026-07-19')",
                [],
            )
            .expect("system provider api key");
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('providers.active_tab', 'service-providers')",
                [],
            )
            .expect("provider tab");
            conn.execute(
                "INSERT INTO settings (key, value) VALUES ('request_logs.retention', '7d')",
                [],
            )
            .expect("non-model setting");
            conn.execute(
                "INSERT INTO user_model_preferences (model_id, is_favorite, created_at, updated_at)
                 VALUES ('custom-model', 1, 1, 1)",
                [],
            )
            .expect("model preference");
            conn.execute(
                "INSERT INTO model_registry
                 (id, display_name, provider_id, provider_name, created_at, updated_at)
                 VALUES ('cached-model', 'Cached', 'custom', 'Custom', 1, 1)",
                [],
            )
            .expect("model cache");
        }
        let target = init_database_at_path(&target_path).expect("target db");
        {
            let conn = lock_db(&target).expect("target lock");
            conn.execute(
                "INSERT INTO api_key_providers
                 (id, name, type, api_host, is_system, group_name, enabled, sort_order, created_at, updated_at)
                 VALUES ('openai', 'Target OpenAI', 'openai', 'https://target.invalid/v1', 1, 'official', 1, 1, '2026-07-19', '2026-07-19')",
                [],
            )
            .expect("target system provider");
        }
        let report = migrate_model_control_data(&source_path, &target_path).expect("migrate");
        assert_eq!(report.providers, 1);
        assert_eq!(report.api_keys, 2);
        assert_eq!(report.model_preferences, 1);
        let conn = lock_db(&target).expect("target lock");
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM api_key_providers", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM api_key_providers WHERE id = 'source-only-system'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
        assert_eq!(
            conn.query_row(
                "SELECT name FROM api_key_providers WHERE id = 'openai'",
                [],
                |row| row.get::<_, String>(0)
            )
            .unwrap(),
            "Target OpenAI"
        );
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM api_keys", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(
            conn.query_row("SELECT COUNT(*) FROM model_registry", [], |row| row
                .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM settings WHERE key = 'request_logs.retention'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
        drop(conn);
        drop(target);
        assert!(source_path.exists());
    }

    #[test]
    fn ignores_catalog_only_source_without_user_model_control_state() {
        let temp = tempdir().expect("tempdir");
        let source_path = temp.path().join("catalog/lime.db");
        let source = init_database_at_path(&source_path).expect("source db");
        {
            let conn = lock_db(&source).expect("source lock");
            conn.execute(
                "INSERT INTO api_key_providers
                 (id, name, type, api_host, is_system, group_name, enabled, sort_order, created_at, updated_at)
                 VALUES ('system', 'System', 'openai', 'https://example.invalid/v1', 1, 'official', 0, 1, '2026-07-19', '2026-07-19')",
                [],
            )
            .expect("system provider");
        }

        assert_eq!(
            inspect_model_control_source(&source_path).expect("inspect source"),
            None
        );
    }

    #[test]
    fn reads_selected_rows_from_source_with_live_wal() {
        let temp = tempdir().expect("tempdir");
        let source_path = temp.path().join("legacy/lime.db");
        let target_path = temp.path().join("agent-root/lime.db");
        let source = init_database_at_path(&source_path).expect("source db");
        {
            let conn = lock_db(&source).expect("source lock");
            conn.pragma_update(None, "journal_mode", "WAL")
                .expect("enable WAL");
            conn.execute(
                "INSERT INTO api_key_providers
                 (id, name, type, api_host, group_name, enabled, sort_order, created_at, updated_at)
                 VALUES ('custom-wal', 'Custom WAL', 'openai', 'https://example.invalid/v1', 'custom', 1, 1, '2026-07-19', '2026-07-19')",
                [],
            )
            .expect("provider");
            conn.execute(
                "INSERT INTO api_keys (id, provider_id, api_key_encrypted, created_at)
                 VALUES ('key-wal', 'custom-wal', 'ciphertext-wal', '2026-07-19')",
                [],
            )
            .expect("api key");
        }
        let target = init_database_at_path(&target_path).expect("target db");

        let report = migrate_model_control_data(&source_path, &target_path).expect("migrate");

        assert_eq!(report.providers, 1);
        assert_eq!(report.api_keys, 1);
        let conn = lock_db(&target).expect("target lock");
        assert_eq!(
            conn.query_row(
                "SELECT COUNT(*) FROM api_keys WHERE id = 'key-wal'",
                [],
                |row| row.get::<_, i64>(0)
            )
            .expect("key count"),
            1
        );
    }
}
