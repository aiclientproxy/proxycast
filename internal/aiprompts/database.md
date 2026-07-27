# 数据库层

## 概述

使用 SQLite (rusqlite) 存储 API Key Provider、模型注册表、Agent 会话与流量记录等。旧凭证池表只保留为历史迁移和启动清理边界，运行时不得再读取它选择凭证。

## 目录结构

```
lime-rs/src/database/
├── mod.rs          # 模块入口
├── schema.rs       # 表结构定义
├── migrations.rs   # 数据库迁移
└── dao/            # 数据访问对象
    ├── api_key_provider.rs
    ├── flow_dao.rs
    └── config_dao.rs
```

## 表结构

### api_key_providers

```sql
CREATE TABLE api_key_providers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    api_host TEXT NOT NULL,
    is_system INTEGER NOT NULL DEFAULT 0,
    group_name TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 0,
    sort_order INTEGER NOT NULL DEFAULT 0,
    models TEXT,
    prompt_cache_mode TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
```

`models` 保存 `ProviderModelConfig[]` JSON，每项唯一形态为 `id`、可选 `displayName` 与可选
`capability`。带 `capability` 的记录是 `provider_explicit` 执行事实；只有 `id` 的记录仅用于 catalog hint，
不得授权 RuntimeCore route。旧 `custom_models/customModels` 已删除，不存在兼容读写路径。

### api_keys

```sql
CREATE TABLE api_keys (
    id TEXT PRIMARY KEY,
    provider_id TEXT NOT NULL,
    api_key_encrypted TEXT NOT NULL,
    alias TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    usage_count INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    last_used_at TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (provider_id) REFERENCES api_key_providers(id) ON DELETE CASCADE
);
```

### model_registry

```sql
CREATE TABLE model_registry (
    id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    provider_id TEXT NOT NULL,
    provider_name TEXT NOT NULL,
    capabilities TEXT NOT NULL DEFAULT '{}',
    limits TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'active',
    source TEXT NOT NULL DEFAULT 'local',
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
```

### flow_records

```sql
CREATE TABLE flow_records (
    id TEXT PRIMARY KEY,
    timestamp INTEGER NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT,
    status TEXT NOT NULL,
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at INTEGER NOT NULL
);

CREATE INDEX idx_flow_timestamp ON flow_records(timestamp);
```

## DAO 模式

```rust
pub struct ApiKeyProviderDao;

impl ApiKeyProviderDao {
    pub fn get_all_providers(conn: &Connection) -> Result<Vec<ApiKeyProvider>>;
    pub fn get_provider_by_id(conn: &Connection, id: &str) -> Result<Option<ApiKeyProvider>>;
    pub fn insert_provider(conn: &Connection, provider: &ApiKeyProvider) -> Result<()>;
    pub fn update_provider(conn: &Connection, provider: &ApiKeyProvider) -> Result<()>;
    pub fn delete_provider(conn: &Connection, id: &str) -> Result<bool>;
    pub fn get_enabled_api_keys_by_provider(conn: &Connection, provider_id: &str) -> Result<Vec<ApiKeyEntry>>;
}
```

## 旧凭证池表边界

`provider_pool_credentials` 分类为 `deprecated/read-only migration source`：schema、历史盘点和独立 maintenance 迁移可以引用；正常启动不得删除表行或 `AppDataRoot/credentials` 文件，运行时服务、legacy adapter 命令、前端 API 和旁路统计也不得读取它做凭证选择。

## 数据库迁移

```rust
pub fn run_migrations(conn: &Connection) -> Result<()> {
    let version = get_schema_version(conn)?;

    if version < 1 {
        conn.execute_batch(include_str!("migrations/001_initial.sql"))?;
    }
    run_startup_migrations(conn)?;

    set_schema_version(conn, CURRENT_VERSION)?;
    Ok(())
}
```

## 相关文档

- [services.md](services.md) - 业务服务
- [credential-pool.md](credential-pool.md) - 凭证池退役说明
